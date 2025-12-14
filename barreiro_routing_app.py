# app.py
"""
Capelle route optimization (drive network only, Google geocoding).

Packages:
- persistence.dropbox_store: store addresses_text in Dropbox JSON (App Folder)
- geocoder: Google Geocoding + 2-level cache (in-memory + Dropbox)
- ui.*: Streamlit session_state helpers and widgets
- routing.*: network, optimization, plotting, urls, timing
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from geocoder import GeocodingError, geocode_addresses
from routing.drive_network import (
    assert_all_pairs_reachable,
    build_distance_matrix_networkx,
    load_drive_graph,
    snap_coords_to_nodes,
)
from routing.nav_urls import build_google_maps_url_from_addresses
from routing.route_plot import (
    make_matplotlib_route_map,
    route_nodes_to_edge_geometry_xy_3857,
    snapped_nodes_xy_3857,
)
from routing.timing import timeblock
from routing.tsp_solver import route_length, solve_tsp_or_path_gurobi
from ui.i18n.t import t
from ui.i18n.widgets import language_selector
from ui.ui_state import (
    addresses_text_area,
    drive_buttons_row,
    ensure_addresses_loaded,
    get_addresses_text,
    init_state_if_missing,
)

# Keep this name; it is now a Dropbox path within the App Folder.
STORE_FILENAME: str = 'barreiro_addresses.json'

DATA_DIR: Path = Path('data')
DRIVE_PREFIX: str = 'barreiro_drive'
LOGFILE: str = 'routing_time_log.txt'

MAX_SNAP_DISTANCE_M: float = 5000.0

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOGFILE, mode='a', encoding='utf-8'),
        logging.StreamHandler(),
    ],
)

logging.info('app.py imported')


@st.cache_resource(show_spinner=False)
def _cached_drive_graph() -> tuple[object, object]:
    """Streamlit-cached wrapper around routing.drive_network.load_drive_graph()."""
    return load_drive_graph(data_dir=DATA_DIR, drive_prefix=DRIVE_PREFIX)


def main() -> None:
    """Run the Streamlit app."""
    logging.info('main() called')

    language_selector(default_lang=None)
    st.title(t('app_title'))

    # NOTE: init_state_if_missing/ensure_addresses_loaded/save logic now uses Dropbox store
    # through ui.ui_state (after you update ui.ui_state to import persistence.dropbox_store).
    init_state_if_missing(filename=STORE_FILENAME)

    ui_mode_label = st.radio(t('ui_mode'), [t('ui_simple'), t('ui_full')], index=0, horizontal=True)
    simple_mode = ui_mode_label == t('ui_simple')

    show_road_overlay = False
    if not simple_mode:
        show_road_overlay_label = st.radio(
            t('road_overlay'),
            [t('off'), t('on')],
            index=0,
            horizontal=True,
            key='show_road_overlay_label',
        )
        show_road_overlay = show_road_overlay_label == t('on')

    if not simple_mode:
        st.markdown(t('instructions'))

    default_text = (
        'Spoorlaan 6, Capelle aan den IJssel, Netherlands\n'
        'Erasmus Plaza, Rotterdam, Netherlands\n'
        'DPFC, Rotterdam, Netherlands\n'
        'Voetbal Vereniging SVS, Capelle aan den IJssel, Netherlands\n'
    )

    ensure_addresses_loaded(default_text=default_text, filename=STORE_FILENAME)

    addresses_text_area(
        label=t('addresses_label'),
        height=200,
        key='addresses_text_area',
    )

    drive_buttons_row(
        default_text=default_text,
        width='stretch',
        rerun_after_reload=True,
    )

    route_type_label = st.radio(
        t('route_type'),
        [t('route_closed'), t('route_open')],
        index=0,
    )
    is_closed = route_type_label == t('route_closed')

    if st.button(t('optimize')):
        logs: list[str] = []
        with timeblock('Total optimization run', logs):
            addresses = [a.strip() for a in get_addresses_text().splitlines() if a.strip()]

            if len(addresses) < 2:
                st.error(t('need_two'))
                return

            if (not is_closed) and len(addresses) < 3:
                st.error(t('need_three_open'))
                return

            try:
                with st.spinner(t('loading_network')):
                    with timeblock('Loading drive graph', logs):
                        graph, nodes = _cached_drive_graph()
            except Exception as exc:
                st.error(t('network_load_error', error=str(exc)))
                return

            with timeblock('Computing network bbox (WGS84)', logs):
                nodes_wgs84 = nodes.to_crs('EPSG:4326')
                min_lon, min_lat, max_lon, max_lat = nodes_wgs84.total_bounds

                center_lon = 0.5 * (min_lon + max_lon)
                center_lat = 0.5 * (min_lat + max_lat)

                half_width_lon = 0.10
                half_height_lat = 0.07

                geo_min_lon = max(min_lon, center_lon - half_width_lon)
                geo_max_lon = min(max_lon, center_lon + half_width_lon)
                geo_min_lat = max(min_lat, center_lat - half_height_lat)
                geo_max_lat = min(max_lat, center_lat + half_height_lat)

                geocode_bbox = (float(geo_min_lon), float(geo_min_lat), float(geo_max_lon), float(geo_max_lat))

            try:
                with st.spinner(t('geocoding')):
                    with timeblock('Geocoding addresses', logs):
                        coords = geocode_addresses(
                            addresses=addresses,
                            bbox=geocode_bbox,
                            persist=True,
                            store_filename=STORE_FILENAME,
                            throttle_s=0.0,
                        )
            except GeocodingError as exc:
                st.error(t('geocode_error', error=str(exc)))
                return
            except Exception as exc:
                st.error(t('geocode_unexpected', error=str(exc)))
                return

            try:
                with st.spinner(t('snapping')):
                    with timeblock('Snapping coords to nodes', logs):
                        snapped_node_ids, snapped_distances_m = snap_coords_to_nodes(coords, nodes)
            except Exception as exc:
                st.error(t('snap_error', error=str(exc)))
                return

            if not simple_mode:
                st.subheader(t('diagnostics_title'))

                coord_groups: dict[tuple[float, float], list[int]] = {}
                for i, (lat, lon) in enumerate(coords):
                    key = (round(lat, 6), round(lon, 6))
                    coord_groups.setdefault(key, []).append(i)

                duplicates = [idxs for idxs in coord_groups.values() if len(idxs) > 1]
                if duplicates:
                    st.warning(t('duplicate_coords_warning'))

                for addr, (lat, lon), node_id, dist_m in zip(addresses, coords, snapped_node_ids, snapped_distances_m):
                    gmaps_link = f'https://www.google.com/maps/search/?api=1&query={lat},{lon}'
                    st.markdown(
                        f"**{t('input_address')}** {addr}  \n"
                        f"{t('geocode_line', lat=f'{lat:.6f}', lon=f'{lon:.6f}')}  \n"
                        f"{t('nearest_node_line', node_id=str(node_id))}  \n"
                        f"{t('dist_to_node_line', dist_m=f'{dist_m:.1f}')}  \n"
                        f"[{t('view_in_maps')}]({gmaps_link})"
                    )

            offending_indices = [i for i, d in enumerate(snapped_distances_m) if d > MAX_SNAP_DISTANCE_M]
            if offending_indices:
                st.error(t('too_far_error', km=f'{MAX_SNAP_DISTANCE_M / 1000.0:.1f}'))
                if not simple_mode:
                    for i in offending_indices:
                        st.write(f'- {addresses[i]} ({snapped_distances_m[i] / 1000.0:.2f} km)')
                return

            try:
                with st.spinner(t('dist_matrix')):
                    with timeblock('Computing NetworkX distance matrix', logs):
                        dist_matrix_raw = build_distance_matrix_networkx(snapped_node_ids, graph)
            except Exception as exc:
                st.error(f'Fout tijdens het berekenen van netwerkafstanden: {exc}')
                return

            if not simple_mode:
                st.subheader(t('distance_matrix_title'))
                df_dist = pd.DataFrame(
                    dist_matrix_raw,
                    columns=[str(i + 1) for i in range(len(addresses))],
                    index=[f'{i + 1}: {",".join(addr.split(",")[:-1])}' for i, addr in enumerate(addresses)],
                )
                st.dataframe(df_dist.style.format('{:.1f}'), width='stretch')

            try:
                with timeblock('Checking connectivity of snapped nodes', logs):
                    assert_all_pairs_reachable(dist_matrix_raw)
            except Exception as exc:
                st.error(t('unreachable_error', error=str(exc)))
                return

            with timeblock('Preparing distance matrix for optimization', logs):
                c = np.array(dist_matrix_raw, dtype=float)
                c = 0.5 * (c + c.T)
                np.fill_diagonal(c, 0.0)
                dist_matrix_opt = c.tolist()

            start_idx = 0
            end_idx = None if is_closed else (len(dist_matrix_opt) - 1)

            try:
                with st.spinner(t('gurobi')):
                    with timeblock('Route optimization', logs):
                        route_indices = solve_tsp_or_path_gurobi(
                            dist_matrix_opt,
                            closed=is_closed,
                            start_idx=start_idx,
                            end_idx=end_idx,
                            trace=False,
                        )
            except Exception as exc:
                st.error(f'Fout tijdens de route-optimalisatie: {exc}')
                return

            ordered_addresses = [addresses[i] for i in route_indices]

            if not simple_mode:
                st.subheader(t('order_title'))
                for k, addr in enumerate(ordered_addresses, start=1):
                    st.write(f'{k}. {addr}')

                total_km = route_length(route_indices, dist_matrix_opt, closed=is_closed)
                st.write(t('estimated_total_km', km=f'{total_km:.2f}'))

            if not simple_mode:
                with timeblock('Building route maps', logs):
                    orig_coords = coords[:]
                    opt_coords = [coords[i] for i in route_indices]

                    if is_closed:
                        orig_coords_plot = orig_coords + orig_coords[:1]
                        opt_coords_plot = opt_coords + opt_coords[:1]
                    else:
                        orig_coords_plot = orig_coords
                        opt_coords_plot = opt_coords

                    road_xs_orig: list[float] | None = None
                    road_ys_orig: list[float] | None = None
                    road_xs_opt: list[float] | None = None
                    road_ys_opt: list[float] | None = None
                    snapped_xs_orig: list[float] | None = None
                    snapped_ys_orig: list[float] | None = None
                    snapped_xs_opt: list[float] | None = None
                    snapped_ys_opt: list[float] | None = None

                    if show_road_overlay:
                        orig_route_node_ids = snapped_node_ids[:]
                        opt_route_node_ids = [snapped_node_ids[i] for i in route_indices]

                        if is_closed:
                            orig_route_node_ids_plot = orig_route_node_ids + orig_route_node_ids[:1]
                            opt_route_node_ids_plot = opt_route_node_ids + opt_route_node_ids[:1]
                        else:
                            orig_route_node_ids_plot = orig_route_node_ids
                            opt_route_node_ids_plot = opt_route_node_ids

                        road_xs_orig, road_ys_orig = route_nodes_to_edge_geometry_xy_3857(
                            orig_route_node_ids_plot,
                            graph,
                            nodes,
                        )
                        road_xs_opt, road_ys_opt = route_nodes_to_edge_geometry_xy_3857(
                            opt_route_node_ids_plot,
                            graph,
                            nodes,
                        )

                        snapped_xs_orig, snapped_ys_orig = snapped_nodes_xy_3857(orig_route_node_ids_plot, nodes)
                        snapped_xs_opt, snapped_ys_opt = snapped_nodes_xy_3857(opt_route_node_ids_plot, nodes)

                    fig_orig = make_matplotlib_route_map(
                        orig_coords_plot,
                        title='Oorspronkelijke volgorde (wegennet)',
                        color='blue',
                        road_xs=road_xs_orig,
                        road_ys=road_ys_orig,
                        snapped_xs=snapped_xs_orig,
                        snapped_ys=snapped_ys_orig,
                    )
                    fig_opt = make_matplotlib_route_map(
                        opt_coords_plot,
                        title='Geoptimaliseerde volgorde (wegennet)',
                        color='red',
                        road_xs=road_xs_opt,
                        road_ys=road_ys_opt,
                        snapped_xs=snapped_xs_opt,
                        snapped_ys=snapped_ys_opt,
                    )

                total_km_original = route_length(list(range(len(dist_matrix_opt))), dist_matrix_opt, closed=is_closed)
                total_km_optimized = route_length(route_indices, dist_matrix_opt, closed=is_closed)

                st.subheader(t('maps_compare_title'))
                col_left, col_right = st.columns(2)

                with col_left:
                    st.markdown(f"**{t('orig_order')}**  \n{t('total_distance_km')} **{total_km_original:.2f}**")
                    st.pyplot(fig_orig, width='stretch')

                with col_right:
                    st.markdown(f"**{t('opt_order')}**  \n{t('total_distance_km')} **{total_km_optimized:.2f}**")
                    st.pyplot(fig_opt, width='stretch')

            with timeblock('Building navigation URL', logs):
                maps_addresses = ordered_addresses + [ordered_addresses[0]] if is_closed else ordered_addresses
                try:
                    maps_url = build_google_maps_url_from_addresses(maps_addresses)
                except Exception as exc:
                    st.error(t('maps_url_error', error=str(exc)))
                    maps_url = ''

            if maps_url:
                if not simple_mode:
                    st.subheader(t('open_nav'))
                st.link_button(t('open_in_maps'), maps_url)
                if not simple_mode:
                    st.code(maps_url, language='text')

        if not simple_mode:
            st.subheader(t('timinglog_title'))
            with st.expander(t('timinglog_expander')):
                for line in logs:
                    st.write(line)
            st.caption(t('timinglog_caption'))


if __name__ == '__main__':
    main()
