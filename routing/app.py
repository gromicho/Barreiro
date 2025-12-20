"""
Shared Streamlit routing application.

All routing logic and UI lives here.
City- or instance-specific apps should only provide configuration and call
`run_routing_app(cfg=...)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging

import numpy as np
import pandas as pd
import streamlit as st

from services.geocoding import GeocodingError, geocode_addresses
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
    drive_version_loader,
    ensure_addresses_loaded,
    get_addresses_text,
    init_state_if_missing,
)

LOGFILE_DEFAULT: str = 'routing_time_log.txt'
MAX_SNAP_DISTANCE_M: float = 5000.0


@dataclass(frozen=True)
class RoutingAppConfig:
    """Configuration for a routing app instance."""

    store_filename: str
    drive_prefix: str
    title_name: str
    title_city: str
    data_dir: Path = Path('data')
    logfile: str = LOGFILE_DEFAULT


def _setup_logging(*, logfile: str) -> None:
    """Configure logging once per process."""
    if getattr(_setup_logging, '_configured', False):
        return

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(logfile, mode='a', encoding='utf-8'),
            logging.StreamHandler(),
        ],
    )
    setattr(_setup_logging, '_configured', True)


@st.cache_resource(show_spinner=False)
def _cached_drive_graph(data_dir_str: str, drive_prefix: str) -> tuple[object, object]:
    """Load and cache the drive graph per (data_dir, drive_prefix)."""
    return load_drive_graph(data_dir=Path(data_dir_str), drive_prefix=drive_prefix)


def run_routing_app(*, cfg: RoutingAppConfig) -> None:
    """Run the shared Streamlit routing app for the given configuration."""
    _setup_logging(logfile=cfg.logfile)
    logging.info('Starting routing app: %s', cfg.store_filename)

    language_selector(default_lang=None)
    st.title(t('app_title', name=cfg.title_name, city=cfg.title_city))

    init_state_if_missing(filename=cfg.store_filename)

    ui_mode_label = st.radio(
        t('ui_mode'),
        [t('ui_simple'), t('ui_full')],
        index=0,
        horizontal=True,
    )
    simple_mode = ui_mode_label == t('ui_simple')

    show_road_overlay = False
    if not simple_mode:
        overlay_label = st.radio(
            t('road_overlay'),
            [t('off'), t('on')],
            index=0,
            horizontal=True,
        )
        show_road_overlay = overlay_label == t('on')

    if not simple_mode:
        st.markdown(t('instructions'))

    default_text = ''

    ensure_addresses_loaded(default_text=default_text, filename=cfg.store_filename)

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

    if not simple_mode:
        drive_version_loader(
            default_text=default_text,
            width='stretch',
            rerun_after_load=True,
        )

    route_type_label = st.radio(
        t('route_type'),
        [t('route_closed'), t('route_open')],
        index=0,
    )
    is_closed = route_type_label == t('route_closed')

    if not st.button(t('optimize')):
        return

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
                    graph, nodes = _cached_drive_graph(
                        str(cfg.data_dir),
                        cfg.drive_prefix,
                    )
        except Exception as exc:
            st.error(t('network_load_error', error=str(exc)))
            return

        try:
            with st.spinner(t('geocoding')):
                with timeblock('Geocoding addresses', logs):
                    coords = geocode_addresses(
                        addresses=addresses,
                        bbox=None,
                        persist=True,
                        store_filename=cfg.store_filename,
                        throttle_s=0.0,
                    )

                    if not simple_mode:
                        st.subheader(t('geocoded_title'))

                        coord_groups: dict[tuple[float, float], list[int]] = {}
                        for i, (lat, lon) in enumerate(coords):
                            key = (round(lat, 6), round(lon, 6))
                            coord_groups.setdefault(key, []).append(i)

                        duplicates = [idxs for idxs in coord_groups.values() if len(idxs) > 1]
                        if duplicates:
                            st.warning(t('duplicate_coords_warning'))

                        for addr, (lat, lon) in zip(addresses, coords):
                            gmaps_link = f'https://www.google.com/maps/search/?api=1&query={lat},{lon}'
                            st.markdown(
                                f"**{t('input_address')}** {addr}  \n"
                                f"{t('geocode_line', lat=f'{lat:.6f}', lon=f'{lon:.6f}')}  \n"
                                f"[{t('view_in_maps')}]({gmaps_link})"
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

        offending = [i for i, d in enumerate(snapped_distances_m) if d > MAX_SNAP_DISTANCE_M]
        if offending:
            st.error(t('too_far_error', km=f'{MAX_SNAP_DISTANCE_M / 1000.0:.1f}'))
            if not simple_mode:
                for i in offending:
                    st.write(f'- {addresses[i]} ({snapped_distances_m[i] / 1000.0:.2f} km)')
            return

        try:
            with st.spinner(t('dist_matrix')):
                with timeblock('Computing distance matrix', logs):
                    dist_matrix_raw = build_distance_matrix_networkx(snapped_node_ids, graph)
        except Exception as exc:
            st.error(str(exc))
            return

        try:
            with timeblock('Checking connectivity', logs):
                assert_all_pairs_reachable(dist_matrix_raw)
        except Exception as exc:
            st.error(t('unreachable_error', error=str(exc)))
            return

        c = np.array(dist_matrix_raw, dtype=float)
        c = 0.5 * (c + c.T)
        np.fill_diagonal(c, 0.0)
        dist_matrix = c.tolist()

        start_idx = 0
        end_idx = None if is_closed else len(dist_matrix) - 1

        try:
            with st.spinner(t('gurobi')):
                with timeblock('Route optimization', logs):
                    route_indices = solve_tsp_or_path_gurobi(
                        dist_matrix,
                        closed=is_closed,
                        start_idx=start_idx,
                        end_idx=end_idx,
                        trace=False,
                    )
        except Exception as exc:
            st.error(str(exc))
            return

        ordered_addresses = [addresses[i] for i in route_indices]

        if not simple_mode:
            st.subheader(t('order_title'))
            for k, addr in enumerate(ordered_addresses, start=1):
                st.write(f'{k}. {addr}')

        if not simple_mode:
            orig_coords = coords[:]
            opt_coords = [coords[i] for i in route_indices]

            orig_node_ids = snapped_node_ids[:]
            opt_node_ids = [snapped_node_ids[i] for i in route_indices]

            # Close the displayed loop if needed (avoid double-close if already closed).
            if is_closed:
                if not (len(orig_coords) >= 2 and orig_coords[0] == orig_coords[-1]):
                    orig_coords = orig_coords + [orig_coords[0]]
                if not (len(opt_coords) >= 2 and opt_coords[0] == opt_coords[-1]):
                    opt_coords = opt_coords + [opt_coords[0]]

                if not (len(orig_node_ids) >= 2 and orig_node_ids[0] == orig_node_ids[-1]):
                    orig_node_ids = orig_node_ids + [orig_node_ids[0]]
                if not (len(opt_node_ids) >= 2 and opt_node_ids[0] == opt_node_ids[-1]):
                    opt_node_ids = opt_node_ids + [opt_node_ids[0]]

            road_xs_orig: list[float] | None = None
            road_ys_orig: list[float] | None = None
            road_xs_opt: list[float] | None = None
            road_ys_opt: list[float] | None = None
            snapped_xs_orig: list[float] | None = None
            snapped_ys_orig: list[float] | None = None
            snapped_xs_opt: list[float] | None = None
            snapped_ys_opt: list[float] | None = None

            if show_road_overlay:
                with timeblock('Building road overlay geometries', logs):
                    road_xs_orig, road_ys_orig = route_nodes_to_edge_geometry_xy_3857(
                        orig_node_ids,
                        graph,
                        nodes,
                    )
                    road_xs_opt, road_ys_opt = route_nodes_to_edge_geometry_xy_3857(
                        opt_node_ids,
                        graph,
                        nodes,
                    )

                    snapped_xs_orig, snapped_ys_orig = snapped_nodes_xy_3857(orig_node_ids, nodes)
                    snapped_xs_opt, snapped_ys_opt = snapped_nodes_xy_3857(opt_node_ids, nodes)

            total_km_original = route_length(
                list(range(len(dist_matrix))),
                dist_matrix,
                closed=is_closed,
            )
            total_km_optimized = route_length(
                route_indices,
                dist_matrix,
                closed=is_closed,
            )

            fig_orig = make_matplotlib_route_map(
                orig_coords,
                title=t('orig_order'),
                color='blue',
                road_xs=road_xs_orig,
                road_ys=road_ys_orig,
                snapped_xs=snapped_xs_orig,
                snapped_ys=snapped_ys_orig,
            )
            fig_opt = make_matplotlib_route_map(
                opt_coords,
                title=t('opt_order'),
                color='red',
                road_xs=road_xs_opt,
                road_ys=road_ys_opt,
                snapped_xs=snapped_xs_opt,
                snapped_ys=snapped_ys_opt,
            )

            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown(f"**{t('orig_order')}**  \n{t('total_distance_km')} **{total_km_original:.2f}**")
                st.pyplot(fig_orig, width='stretch')
            with col_r:
                st.markdown(f"**{t('opt_order')}**  \n{t('total_distance_km')} **{total_km_optimized:.2f}**")
                st.pyplot(fig_opt, width='stretch')

        try:
            with timeblock('Building navigation URL', logs):
                maps_addresses = ordered_addresses + [ordered_addresses[0]] if is_closed else ordered_addresses
                maps_url = build_google_maps_url_from_addresses(maps_addresses)
        except Exception as exc:
            st.error(t('maps_url_error', error=str(exc)))
            return

        st.link_button(t('open_in_maps'), maps_url)

    if not simple_mode:
        with st.expander(t('timinglog_expander')):
            for line in logs:
                st.write(line)
