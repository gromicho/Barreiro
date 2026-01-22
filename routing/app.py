from __future__ import annotations

from dataclasses import dataclass
import difflib
import logging
from pathlib import Path
import re
from urllib.parse import quote_plus

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
from ui.drive_handlers import ensure_addresses_loaded
from ui.i18n.t import t
from ui.i18n.widgets import language_selector
from ui.state_accessors import get_addresses_text
from ui.state_keys import init_state_if_missing
from ui.widgets import (
    addresses_text_area,
    camera_ocr_widget,
    drive_buttons_row,
    drive_version_loader,
)

LOGFILE_DEFAULT: str = 'routing_time_log.txt'
MAX_SNAP_DISTANCE_M_DEFAULT: float = 5000.0
_NEAR_DUPLICATE_THRESHOLD: float = 0.92


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


def get_drive_graph_once_per_session(*, data_dir: Path | str, drive_prefix: str) -> tuple[object, object]:
    """
    Return the (graph, nodes) for the drive network, loading it at most once per Streamlit session.

    Streamlit reruns the script on every widget interaction. If the drive graph load is ever
    retriggered (cache misses due to path differences, etc.), the UI becomes very slow.

    This function:
    - canonicalizes data_dir to an absolute resolved path string (stable cache key)
    - stores the loaded (graph, nodes) in st.session_state (session singleton)

    Args:
        data_dir: Directory containing drive network data (Path or str).
        drive_prefix: Dataset prefix for the drive network.

    Returns:
        (graph, nodes) as returned by load_drive_graph().
    """
    data_dir_key = str(Path(data_dir).resolve())
    drive_prefix_key = str(drive_prefix)

    state_key = f'_drive_graph:{data_dir_key}:{drive_prefix_key}'
    cached = st.session_state.get(state_key)

    if isinstance(cached, tuple) and len(cached) == 2 and cached[0] is not None and cached[1] is not None:
        return cached[0], cached[1]

    graph, nodes = _cached_drive_graph(data_dir_key, drive_prefix_key)
    st.session_state[state_key] = (graph, nodes)
    return graph, nodes


@dataclass(frozen=True)
class RoutingAppConfig:
    """Configuration for a routing app instance."""

    store_filename: str
    drive_prefix: str
    title_name: str
    title_city: str
    home_address: str

    # Optional ROI to constrain geocoding and (optional) coverage map
    # Tuple is (min_lat, min_lon, max_lat, max_lon) in WGS84 degrees.
    roi_bbox_wgs84: tuple[float, float, float, float] | None = None
    roi_name: str | None = None

    # Concave hull control for coverage map (no UI slider; set per instance)
    # ratio in (0, 1]. Smaller => more concave. 1.0 ~ convex hull.
    coverage_concavity_ratio: float = 0.25

    # Whether to clip coverage polygon to land (water removal is handled elsewhere if enabled)
    clip_coverage_to_land: bool = True

    data_dir: Path = Path('data')
    logfile: str = LOGFILE_DEFAULT
    max_snap_distance_m: float = MAX_SNAP_DISTANCE_M_DEFAULT


@dataclass(frozen=True)
class OptimizationResult:
    """Results of one optimization run."""

    is_closed: bool
    addresses: list[str]
    coords: list[tuple[float, float]]

    snapped_node_ids: list[int]
    snapped_distances_m: list[float]

    dist_matrix: list[list[float]]
    route_indices: list[int]
    ordered_addresses: list[str]

    total_km_original: float
    total_km_optimized: float

    maps_url: str
    logs: list[str]


@dataclass(frozen=True)
class AddressRow:
    """One address candidate with provenance and normalization."""

    captured: str
    final: str
    include: bool
    note: str = ''


def _parse_addresses(text: str) -> list[str]:
    """Parse non-empty address lines from a text blob."""
    return [line.strip() for line in text.splitlines() if line.strip()]


def _summarize_address_label(address: str) -> str:
    """Produce a compact address label for UI."""
    parts = [p.strip() for p in address.split(',') if p.strip()]
    if not parts:
        return address.strip()

    trailing_countries = {
        'portugal',
        'the netherlands',
        'netherlands',
        'nederland',
        'belgium',
        'belgië',
        'spain',
        'españa',
        'france',
        'germany',
        'deutschland',
        'luxembourg',
        'luxemburg',
    }

    if parts[-1].casefold() in trailing_countries and len(parts) >= 2:
        parts = parts[:-1]

    return ', '.join(parts)


def _build_input_addresses(*, cfg: RoutingAppConfig, addresses_text: str) -> list[str]:
    """
    Build the final address list used for routing.

    Rules:
        - Parse non-empty lines.
        - Prepend home exactly once (always at index 0 if set).
        - Remove exact duplicates while preserving order.
    """
    raw = _parse_addresses(addresses_text)

    home = cfg.home_address.strip()
    combined: list[str] = []
    if home:
        combined.append(home)
    combined.extend(raw)

    seen: set[str] = set()
    out: list[str] = []
    for a in combined:
        key = a.strip()
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(key)

    return out if out else ([home] if home else [])


def _ensure_closed(items: list[object]) -> list[object]:
    """
    Ensure a route list is closed by repeating the first element at the end.

    Args:
        items: Sequence representing a route.

    Returns:
        Closed route sequence.
    """
    if len(items) >= 2 and items[0] == items[-1]:
        return items
    if not items:
        return items
    return items + [items[0]]


def _distance_matrix_to_km(dist_matrix: list[list[float]]) -> np.ndarray:
    """
    Convert a distance matrix to kilometers using a heuristic for units.

    If the median non-zero value is large (>200), we assume meters and divide by 1000.
    """
    a = np.array(dist_matrix, dtype=float)
    nonzero = a[a > 0]
    if nonzero.size == 0:
        return a

    med = float(np.median(nonzero))
    if med > 200.0:
        return a / 1000.0
    return a


def _build_distance_matrix_df_km(dist_matrix_raw_units: list[list[float]], addresses: list[str]) -> pd.DataFrame:
    """
    Build a DataFrame to display a distance matrix in Streamlit (km).
    """
    n = len(dist_matrix_raw_units)
    row_labels = [f'{i}. {_summarize_address_label(a)}' for i, a in enumerate(addresses, start=1)]
    col_labels = [str(i) for i in range(1, n + 1)]
    km = _distance_matrix_to_km(dist_matrix_raw_units).round(1)
    return pd.DataFrame(km, index=row_labels, columns=col_labels)


def _normalize_address_line(line: str) -> str:
    """
    Normalize a single OCR/text line to reduce junk.

    Removes bullets/numbering, collapses whitespace, trims punctuation.
    """
    s = line.strip()
    s = re.sub(r'^\s*([-\u2022*]|(\d+\s*[\).\:-]))\s*', '', s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip(' ,;')


def _similar(a: str, b: str) -> float:
    """Return similarity ratio in [0, 1] using stdlib difflib."""
    return difflib.SequenceMatcher(None, a.casefold(), b.casefold()).ratio()


def _reconcile_addresses(lines: list[str], *, home_address: str) -> list[AddressRow]:
    """
    Build a reconciliation list from raw lines.

    - Normalizes each line
    - Excludes empty lines
    - Excludes 'home-like' entries
    - Excludes near-duplicates (keeps first occurrence)
    """
    cleaned = [_normalize_address_line(x) for x in lines]
    cleaned = [c for c in cleaned if c]

    rows: list[AddressRow] = []
    home = home_address.strip()

    for c in cleaned:
        include = True
        note = ''
        if home and _similar(c, home) >= _NEAR_DUPLICATE_THRESHOLD:
            include = False
            note = t('reconcile_note_home_excluded')
        rows.append(AddressRow(captured=c, final=c, include=include, note=note))

    for i in range(len(rows)):
        if not rows[i].include:
            continue
        for j in range(i + 1, len(rows)):
            if not rows[j].include:
                continue
            if _similar(rows[i].final, rows[j].final) >= _NEAR_DUPLICATE_THRESHOLD:
                rows[j] = AddressRow(
                    captured=rows[j].captured,
                    final=rows[j].final,
                    include=False,
                    note=t('reconcile_note_near_duplicate', row=i + 1),
                )

    return rows


def _set_addresses_text_in_state(text: str) -> None:
    """
    Best-effort: write address text back into session state.

    Defensive because `get_addresses_text()` may read from a different key than the
    visible textarea widget, depending on your UI helpers.
    """
    st.session_state['addresses_text_area'] = text
    for k in ('addresses_text', 'addresses', 'addresses_input'):
        if k in st.session_state:
            st.session_state[k] = text


def _load_graph_and_nodes(*, cfg: RoutingAppConfig, logs: list[str]) -> tuple[object, object]:
    """Load drive graph + nodes once per session; record timing logs."""
    with st.spinner(t('loading_network')):
        with timeblock('Loading drive graph', logs):
            graph, nodes = get_drive_graph_once_per_session(
                data_dir=cfg.data_dir,
                drive_prefix=cfg.drive_prefix,
            )
    return graph, nodes


def _compute_optimization(
    *,
    cfg: RoutingAppConfig,
    routing_text: str,
    is_closed: bool,
) -> OptimizationResult:
    """
    Run the end-to-end optimization pipeline.

    Raises:
        ValueError: for user-facing validation errors.
        GeocodingError: geocoding failures.
        Exception: unexpected errors.
    """
    logs: list[str] = []

    with timeblock('Total optimization run', logs):
        addresses = _build_input_addresses(cfg=cfg, addresses_text=routing_text)

        if len(addresses) < 2:
            raise ValueError(t('need_two'))
        if (not is_closed) and len(addresses) < 3:
            raise ValueError(t('need_three_open'))

        graph, nodes = _load_graph_and_nodes(cfg=cfg, logs=logs)

        with st.spinner(t('geocoding')):
            with timeblock('Geocoding addresses', logs):
                coords = geocode_addresses(
                    addresses=addresses,
                    bbox=cfg.roi_bbox_wgs84,
                    persist=True,
                    store_filename=cfg.store_filename,
                    throttle_s=0.0,
                )

        with st.spinner(t('snapping')):
            with timeblock('Snapping coords to nodes', logs):
                snapped_node_ids, snapped_distances_m = snap_coords_to_nodes(coords, nodes)

        offending = [i for i, d in enumerate(snapped_distances_m) if d > cfg.max_snap_distance_m]
        if offending:
            raise ValueError(t('too_far_error', km=f'{cfg.max_snap_distance_m / 1000.0:.1f}'))

        with st.spinner(t('dist_matrix')):
            with timeblock('Computing distance matrix', logs):
                dist_matrix_raw = build_distance_matrix_networkx(snapped_node_ids, graph)

        with timeblock('Checking connectivity', logs):
            assert_all_pairs_reachable(dist_matrix_raw)

        c = np.array(dist_matrix_raw, dtype=float)
        c = 0.5 * (c + c.T)
        np.fill_diagonal(c, 0.0)
        dist_matrix = c.tolist()

        start_idx = 0
        end_idx = None if is_closed else len(dist_matrix) - 1

        with st.spinner(t('gurobi')):
            with timeblock('Route optimization', logs):
                route_indices = solve_tsp_or_path_gurobi(
                    dist_matrix,
                    closed=is_closed,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    trace=False,
                )

        ordered_addresses = [addresses[i] for i in route_indices]

        total_km_original = route_length(list(range(len(dist_matrix))), dist_matrix, closed=is_closed)
        total_km_optimized = route_length(route_indices, dist_matrix, closed=is_closed)

        with timeblock('Building navigation URL', logs):
            maps_addresses = ordered_addresses + [ordered_addresses[0]] if is_closed else ordered_addresses
            maps_url = build_google_maps_url_from_addresses(maps_addresses)

    return OptimizationResult(
        is_closed=is_closed,
        addresses=addresses,
        coords=coords,
        snapped_node_ids=snapped_node_ids,
        snapped_distances_m=snapped_distances_m,
        dist_matrix=dist_matrix,
        route_indices=route_indices,
        ordered_addresses=ordered_addresses,
        total_km_original=total_km_original,
        total_km_optimized=total_km_optimized,
        maps_url=maps_url,
        logs=logs,
    )


def _render_quick_preflight(*, cfg: RoutingAppConfig, routing_text: str) -> None:
    """Show a compact preview of what will be optimized (always cheap)."""
    addresses = _build_input_addresses(cfg=cfg, addresses_text=routing_text)
    if not addresses:
        st.info(t('need_two'))
        return

    st.caption(t('input_preview_caption'))
    preview_df = pd.DataFrame(
        {
            t('col_num'): list(range(1, len(addresses) + 1)),
            t('col_address'): [_summarize_address_label(a) for a in addresses],
        }
    )
    st.dataframe(preview_df, width='stretch', hide_index=True)


def _render_validation_tables(*, result: OptimizationResult, advanced_ui: bool) -> None:
    """Render geocoding and snapping validation tables."""
    snap_km = np.array(result.snapped_distances_m, dtype=float) / 1000.0

    if not advanced_ui:
        worst: float = float(np.max(snap_km)) if snap_km.size else 0.0
        st.caption(t('max_snap_distance_caption', km=f'{worst:.2f}'))
        return

    st.subheader(t('geocoded_title'))
    with st.expander(t('snapping_overview_title'), expanded=False):
        col_num = t('col_num')
        col_addr = t('col_address')
        col_lat = t('col_lat')
        col_lon = t('col_lon')
        col_snap = t('col_snap_dist_km')

        df = pd.DataFrame(
            {
                col_num: list(range(1, len(result.addresses) + 1)),
                col_addr: [_summarize_address_label(a) for a in result.addresses],
                col_lat: [round(lat, 6) for lat, _ in result.coords],
                col_lon: [round(lon, 6) for _, lon in result.coords],
                col_snap: np.round(snap_km, 2),
            }
        )

        try:
            st.dataframe(
                df,
                width='stretch',
                hide_index=True,
                column_config={
                    col_snap: st.column_config.NumberColumn(
                        t('snap_dist_km_col'),
                        format='%.2f km',
                    ),
                },
            )
        except Exception:
            st.dataframe(df, width='stretch', hide_index=True)


def _build_google_maps_url_from_latlon(*, lat: float, lon: float, zoom: int = 16) -> str:
    """Open Google Maps at a WGS84 (lat, lon) coordinate."""
    return f'https://www.google.com/maps/search/?api=1&query={lat:.6f},{lon:.6f}&zoom={int(zoom)}'


def _build_google_maps_url_from_address(*, address: str) -> str:
    """Fallback: search an address string in Google Maps."""
    return f'https://www.google.com/maps/search/?api=1&query={quote_plus(address)}'


def _render_results(
    *,
    cfg: RoutingAppConfig,
    result: OptimizationResult,
    advanced_ui: bool,
    show_road_overlay: bool,
) -> None:
    """Render the optimized order and optional plots/matrix/logs."""
    st.subheader(t('order_title') if advanced_ui else t('order_title_simple'))

    # Display a "closed" route by repeating the first stop at the end.
    display_addresses = result.ordered_addresses
    display_indices = result.route_indices
    if result.is_closed and result.ordered_addresses:
        display_addresses = result.ordered_addresses + [result.ordered_addresses[0]]
        display_indices = result.route_indices + [result.route_indices[0]]

    for k, (addr, original_idx) in enumerate(zip(display_addresses, display_indices), start=1):
        url: str
        if 0 <= original_idx < len(result.coords):
            lat, lon = result.coords[original_idx]  # geocoded coords are WGS84 (lat, lon)
            url = _build_google_maps_url_from_latlon(lat=float(lat), lon=float(lon))
        else:
            url = _build_google_maps_url_from_address(address=addr)

        st.markdown(f'{k}. [{addr}]({url})')

    st.markdown(
        f'**{t("total_distance_km")}**  \n'
        f'- {t("orig_order")}: **{result.total_km_original:.2f}**  \n'
        f'- {t("opt_order")}: **{result.total_km_optimized:.2f}**'
    )

    st.link_button(t('open_in_maps'), result.maps_url)

    if not advanced_ui:
        return

    with st.expander(t('dist_matrix_expander'), expanded=False):
        dist_df = _build_distance_matrix_df_km(result.dist_matrix, result.addresses)
        try:
            st.dataframe(
                dist_df,
                width='stretch',
                column_config={col: st.column_config.NumberColumn(col, format='%.1f km') for col in dist_df.columns},
            )
        except Exception:
            st.dataframe(dist_df, width='stretch')

    with st.expander(t('maps_plots_expander'), expanded=False):
        with timeblock('Route display', result.logs):
            graph, nodes = get_drive_graph_once_per_session(data_dir=cfg.data_dir, drive_prefix=cfg.drive_prefix)

            orig_coords = result.coords[:]
            opt_coords = [result.coords[i] for i in result.route_indices]

            orig_node_ids = result.snapped_node_ids[:]
            opt_node_ids = [result.snapped_node_ids[i] for i in result.route_indices]

            if result.is_closed:
                orig_coords = list(_ensure_closed(orig_coords))  # type: ignore[assignment]
                opt_coords = list(_ensure_closed(opt_coords))  # type: ignore[assignment]
                orig_node_ids = list(_ensure_closed(orig_node_ids))  # type: ignore[assignment]
                opt_node_ids = list(_ensure_closed(opt_node_ids))  # type: ignore[assignment]

            road_xs_orig: list[float] | None = None
            road_ys_orig: list[float] | None = None
            road_xs_opt: list[float] | None = None
            road_ys_opt: list[float] | None = None
            snapped_xs_orig: list[float] | None = None
            snapped_ys_orig: list[float] | None = None
            snapped_xs_opt: list[float] | None = None
            snapped_ys_opt: list[float] | None = None

            if show_road_overlay:
                with timeblock('Building road overlay geometries', result.logs):
                    road_xs_orig, road_ys_orig = route_nodes_to_edge_geometry_xy_3857(orig_node_ids, graph, nodes)
                    road_xs_opt, road_ys_opt = route_nodes_to_edge_geometry_xy_3857(opt_node_ids, graph, nodes)
                    snapped_xs_orig, snapped_ys_orig = snapped_nodes_xy_3857(orig_node_ids, nodes)
                    snapped_xs_opt, snapped_ys_opt = snapped_nodes_xy_3857(opt_node_ids, nodes)

            fig_orig = make_matplotlib_route_map(
                orig_coords,
                title=f'{result.total_km_original:.2f} km',
                color='blue',
                road_xs=road_xs_orig,
                road_ys=road_ys_orig,
                snapped_xs=snapped_xs_orig,
                snapped_ys=snapped_ys_orig,
            )
            fig_opt = make_matplotlib_route_map(
                opt_coords,
                title=f'{result.total_km_optimized:.2f} km',
                color='red',
                road_xs=road_xs_opt,
                road_ys=road_ys_opt,
                snapped_xs=snapped_xs_opt,
                snapped_ys=snapped_ys_opt,
            )

            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown(f'**{t("orig_order")}**')
                st.pyplot(fig_orig, width='stretch')
            with col_r:
                st.markdown(f'**{t("opt_order")}**')
                st.pyplot(fig_opt, width='stretch')

    with st.expander(t('timinglog_expander'), expanded=False):
        for line in result.logs:
            st.write(line)


def _inject_mobile_fullwidth_css() -> None:
    """Reduce Streamlit side gutters on small screens to give widgets more horizontal space."""
    st.markdown(
        """
        <style>
        @media (max-width: 768px) {
          .appview-container .main .block-container {
            max-width: 100vw !important;
            padding-left: 0.25rem !important;
            padding-right: 0.25rem !important;
          }

          /* Optional: reduce vertical whitespace a bit */
          .appview-container .main .block-container > div {
            padding-top: 0.25rem;
          }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def run_routing_app(*, cfg: RoutingAppConfig) -> None:
    """
    Run the Streamlit routing app.

    Structure:
        1) Mode + optional advanced toggles
        2) Optional coverage map
        3) Addresses (OCR + textarea + drive buttons)
        4) Optional reconciliation (advanced)
        5) Optimize + results
    """
    _setup_logging(logfile=cfg.logfile)
    logging.info('Starting routing app: %s', cfg.store_filename)

    # If you want: st.set_page_config(layout='wide')
    _inject_mobile_fullwidth_css()

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
    advanced_ui = not simple_mode

    show_road_overlay = False
    if advanced_ui:
        overlay_label = st.radio(
            t('road_overlay'),
            [t('off'), t('on')],
            index=0,
            horizontal=True,
        )
        show_road_overlay = overlay_label == t('on')

    with st.expander(t('graph_coverage_title'), expanded=False):
        show_cov = st.checkbox(t('show_coverage_map'), value=False)
        if show_cov:
            from routing.coverage_map import render_coverage_map

            subtitle = (
                t('graph_coverage_subtitle_roi', roi=cfg.roi_name) if cfg.roi_name else t('graph_coverage_subtitle')
            )

            data_dir_key = str(cfg.data_dir.resolve())
            render_coverage_map(
                data_dir=data_dir_key,
                drive_prefix=cfg.drive_prefix,
                roi_bbox_wgs84=cfg.roi_bbox_wgs84,
                concavity_ratio=cfg.coverage_concavity_ratio,
                clip_to_land=cfg.clip_coverage_to_land,
                roi_name=cfg.roi_name,
                title=t('graph_coverage_title'),
                map_title=t('graph_coverage_map_title'),
                subtitle=subtitle,
            )

    default_text = ''
    ensure_addresses_loaded(default_text=default_text, filename=cfg.store_filename)

    st.header(t('step1_addresses'))

    if simple_mode:
        camera_ocr_widget(
            filename=cfg.store_filename,
            model='gpt-4.1-mini',
            key_prefix=f'camera_ocr.{cfg.store_filename}',
            home_address=cfg.home_address,
            overwrite=True,
            show_debug=False,
            duplicate_first_on_overwrite=False,
            allow_upload_fallback=False,
        )
    else:
        with st.expander(t('camera_ocr_expander'), expanded=False):
            camera_ocr_widget(
                filename=cfg.store_filename,
                model='gpt-4.1-mini',
                key_prefix=f'camera_ocr.{cfg.store_filename}',
                home_address=cfg.home_address,
                overwrite=True,
                show_debug=True,
                duplicate_first_on_overwrite=False,
                allow_upload_fallback=True,
            )

    addresses_text_area(label=t('addresses_label'), height=200, key='addresses_text_area')

    drive_buttons_row(default_text=default_text, width='stretch', rerun_after_reload=True)

    if advanced_ui:
        drive_version_loader(default_text=default_text, width='stretch', rerun_after_load=True)

    routing_text = get_addresses_text()
    _render_quick_preflight(cfg=cfg, routing_text=routing_text)

    if advanced_ui and routing_text.strip():
        st.header(t('step2_cleanup_optional'))
        with st.expander(t('reconcile_expander'), expanded=False):
            raw_lines = _parse_addresses(routing_text)
            rows = _reconcile_addresses(raw_lines, home_address=cfg.home_address)

            col_use = t('reconcile_col_use')
            col_captured = t('reconcile_col_captured')
            col_final = t('reconcile_col_final')
            col_note = t('reconcile_col_note')

            rec_df = pd.DataFrame(
                {
                    col_use: [r.include for r in rows],
                    col_captured: [r.captured for r in rows],
                    col_final: [r.final for r in rows],
                    col_note: [r.note for r in rows],
                }
            )

            edited = st.data_editor(
                rec_df,
                width='stretch',
                column_config={col_use: st.column_config.CheckboxColumn(col_use)},
                disabled=[col_captured, col_note],
                hide_index=True,
            )

            final_lines = [
                str(a).strip()
                for use, a in zip(edited[col_use].tolist(), edited[col_final].tolist())
                if bool(use) and str(a).strip()
            ]
            reconciled_text = '\n'.join(final_lines)

            cols = st.columns([1, 1, 3])
            with cols[0]:
                if st.button(t('apply_to_input'), width='stretch'):
                    _set_addresses_text_in_state(reconciled_text)
                    st.rerun()
            with cols[1]:
                if st.button(t('use_for_this_run_only'), width='stretch'):
                    routing_text = reconciled_text
            with cols[2]:
                st.caption(t('reconcile_tip_apply_persists'))

    st.header(t('step3_optimize'))
    route_type_label = st.radio(t('route_type'), [t('route_closed'), t('route_open')], index=0)
    is_closed = route_type_label == t('route_closed')

    optimize_clicked = st.button(t('optimize'), type='primary')
    if not optimize_clicked:
        return

    try:
        result = _compute_optimization(cfg=cfg, routing_text=routing_text, is_closed=is_closed)
    except GeocodingError as exc:
        st.error(t('geocode_error', error=str(exc)))
        return
    except ValueError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        st.error(t('geocode_unexpected', error=str(exc)))
        return

    st.header(t('step4_results'))
    _render_validation_tables(result=result, advanced_ui=advanced_ui)
    _render_results(cfg=cfg, result=result, advanced_ui=advanced_ui, show_road_overlay=show_road_overlay)


__all__ = ['RoutingAppConfig', 'run_routing_app']
