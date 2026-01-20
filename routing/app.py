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


LOGFILE_DEFAULT: str = "routing_time_log.txt"
MAX_SNAP_DISTANCE_M_DEFAULT: float = 5000.0


@dataclass(frozen=True)
class RoutingAppConfig:
    """Configuration for a routing app instance."""

    store_filename: str
    drive_prefix: str
    title_name: str
    title_city: str

    home_address: str

    # Optional ROI to constrain geocoding and graph coverage.
    # Tuple is (min_lat, min_lon, max_lat, max_lon) in WGS84 degrees.
    roi_bbox_wgs84: tuple[float, float, float, float] | None = None
    roi_name: str | None = None

    # Concave hull control (no UI slider; set per instance)
    # ratio in (0, 1]. Smaller => more concave. 1.0 ~ convex hull.
    coverage_concavity_ratio: float = 0.25

    # Whether to remove water by clipping coverage polygon to land
    clip_coverage_to_land: bool = True

    data_dir: Path = Path("data")
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


def _set_addresses_text_in_state(text: str) -> None:
    """
    Best-effort: write address text back into session state.

    This is defensive because `get_addresses_text()` may read from a different key than
    the visible textarea widget, depending on your UI helpers.
    """
    st.session_state["addresses_text_area"] = text

    # Common alternate keys (safe no-op if unused)
    for k in ("addresses_text", "addresses", "addresses_input"):
        if k in st.session_state:
            st.session_state[k] = text


def _load_graph_and_nodes(*, cfg: RoutingAppConfig, logs: list[str]) -> tuple[object, object]:
    """Load drive graph + nodes using cache; record timing logs."""
    with st.spinner(t("loading_network")):
        with timeblock("Loading drive graph", logs):
            graph, nodes = _cached_drive_graph(str(cfg.data_dir), cfg.drive_prefix)
    return graph, nodes


def _compute_optimization(
    *,
    cfg: RoutingAppConfig,
    routing_text: str,
    is_closed: bool,
    show_road_overlay: bool,
    advanced_ui: bool,
) -> OptimizationResult:
    """
    Run the end-to-end optimization pipeline.

    Raises exceptions for the caller to surface appropriately.
    """
    logs: list[str] = []

    with timeblock("Total optimization run", logs):
        addresses = _build_input_addresses(cfg=cfg, addresses_text=routing_text)

        if len(addresses) < 2:
            raise ValueError(t("need_two"))
        if (not is_closed) and len(addresses) < 3:
            raise ValueError(t("need_three_open"))

        graph, nodes = _load_graph_and_nodes(cfg=cfg, logs=logs)

        with st.spinner(t("geocoding")):
            with timeblock("Geocoding addresses", logs):
                coords = geocode_addresses(
                    addresses=addresses,
                    bbox=cfg.roi_bbox_wgs84,
                    persist=True,
                    store_filename=cfg.store_filename,
                    throttle_s=0.0,
                )

        with st.spinner(t("snapping")):
            with timeblock("Snapping coords to nodes", logs):
                snapped_node_ids, snapped_distances_m = snap_coords_to_nodes(coords, nodes)

        offending = [i for i, d in enumerate(snapped_distances_m) if d > cfg.max_snap_distance_m]
        if offending:
            # Caller can show the per-address detail in advanced UI; here we return a clear error.
            raise ValueError(t("too_far_error", km=f"{cfg.max_snap_distance_m / 1000.0:.1f}"))

        with st.spinner(t("dist_matrix")):
            with timeblock("Computing distance matrix", logs):
                dist_matrix_raw = build_distance_matrix_networkx(snapped_node_ids, graph)

        with timeblock("Checking connectivity", logs):
            assert_all_pairs_reachable(dist_matrix_raw)

        # Symmetrize + ensure diagonal is 0
        c = np.array(dist_matrix_raw, dtype=float)
        c = 0.5 * (c + c.T)
        np.fill_diagonal(c, 0.0)
        dist_matrix = c.tolist()

        start_idx = 0
        end_idx = None if is_closed else len(dist_matrix) - 1

        with st.spinner(t("gurobi")):
            with timeblock("Route optimization", logs):
                route_indices = solve_tsp_or_path_gurobi(
                    dist_matrix,
                    closed=is_closed,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    trace=False,
                )

        ordered_addresses = [addresses[i] for i in route_indices]

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

        with timeblock("Building navigation URL", logs):
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
        st.info(t("need_two"))
        return

    st.caption("Input preview (after home + de-dup)")  # TODO: i18n
    preview_df = pd.DataFrame(
        {
            "#": list(range(1, len(addresses) + 1)),
            "Address": [_summarize_address_label(a) for a in addresses],
        }
    )
    st.dataframe(preview_df, use_container_width=True, hide_index=True)


def _render_validation_tables(
    *,
    cfg: RoutingAppConfig,
    result: OptimizationResult,
    advanced_ui: bool,
) -> None:
    """Render geocoding and snapping validation tables."""
    if not advanced_ui:
        # Still give *some* trust signal in non-advanced UI.
        snap_km = (np.array(result.snapped_distances_m, dtype=float) / 1000.0).round(2)
        worst = float(np.max(snap_km)) if len(snap_km) else 0.0
        st.caption(f"Max snap distance: {worst:.2f} km")  # TODO: i18n
        return

    st.subheader(t("geocoded_title"))
    geo_df = pd.DataFrame(
        {
            "#": list(range(1, len(result.addresses) + 1)),
            "Address": [_summarize_address_label(a) for a in result.addresses],
            "Lat": [round(lat, 6) for (lat, _lon) in result.coords],
            "Lon": [round(lon, 6) for (_lat, lon) in result.coords],
        }
    )
    st.dataframe(geo_df, use_container_width=True, hide_index=True)

    st.subheader(t("snapping_overview_title"))
    snap_km = np.array(result.snapped_distances_m, dtype=float) / 1000.0
    snap_df = pd.DataFrame(
        {
            "#": list(range(1, len(result.addresses) + 1)),
            "Address": [_summarize_address_label(a) for a in result.addresses],
            "Snap dist (km)": np.round(snap_km, 2),
        }
    )
    try:
        st.dataframe(
            snap_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Snap dist (km)": st.column_config.NumberColumn(
                    t("snap_dist_km_col"),
                    format="%.2f km",
                ),
            },
        )
    except Exception:
        st.dataframe(snap_df, use_container_width=True, hide_index=True)


def _render_results(
    *,
    cfg: RoutingAppConfig,
    result: OptimizationResult,
    advanced_ui: bool,
    show_road_overlay: bool,
) -> None:
    """Render the optimized order and optional plots/matrix/logs."""
    st.subheader(t("order_title") if advanced_ui else "Optimized order")  # TODO: i18n for fallback label

    for k, addr in enumerate(result.ordered_addresses, start=1):
        st.write(f"{k}. {addr}")

    st.markdown(
        f"**{t('total_distance_km')}**  \n"
        f"- {t('orig_order')}: **{result.total_km_original:.2f}**  \n"
        f"- {t('opt_order')}: **{result.total_km_optimized:.2f}**"
    )

    st.link_button(t("open_in_maps"), result.maps_url)

    if not advanced_ui:
        return

    # Optional distance matrix
    with st.expander(t("dist_matrix_expander"), expanded=False):
        dist_df = _build_distance_matrix_df_km(result.dist_matrix, result.addresses)
        try:
            st.dataframe(
                dist_df,
                use_container_width=True,
                column_config={col: st.column_config.NumberColumn(col, format="%.1f km") for col in dist_df.columns},
            )
        except Exception:
            st.dataframe(dist_df, use_container_width=True)

    # Optional plots (require graph+nodes)
    with st.expander("Maps / plots", expanded=False):  # TODO: i18n
        graph, nodes = _cached_drive_graph(str(cfg.data_dir), cfg.drive_prefix)

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
            with timeblock("Building road overlay geometries", result.logs):
                road_xs_orig, road_ys_orig = route_nodes_to_edge_geometry_xy_3857(orig_node_ids, graph, nodes)
                road_xs_opt, road_ys_opt = route_nodes_to_edge_geometry_xy_3857(opt_node_ids, graph, nodes)
                snapped_xs_orig, snapped_ys_orig = snapped_nodes_xy_3857(orig_node_ids, nodes)
                snapped_xs_opt, snapped_ys_opt = snapped_nodes_xy_3857(opt_node_ids, nodes)

        fig_orig = make_matplotlib_route_map(
            orig_coords,
            title=t("orig_order"),
            color="blue",
            road_xs=road_xs_orig,
            road_ys=road_ys_orig,
            snapped_xs=snapped_xs_orig,
            snapped_ys=snapped_ys_orig,
        )
        fig_opt = make_matplotlib_route_map(
            opt_coords,
            title=t("opt_order"),
            color="red",
            road_xs=road_xs_opt,
            road_ys=road_ys_opt,
            snapped_xs=snapped_xs_opt,
            snapped_ys=snapped_ys_opt,
        )

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown(f"**{t('orig_order')}**")
            st.pyplot(fig_orig, width="stretch")
        with col_r:
            st.markdown(f"**{t('opt_order')}**")
            st.pyplot(fig_opt, width="stretch")

    with st.expander(t("timinglog_expander"), expanded=False):
        for line in result.logs:
            st.write(line)


def run_routing_app(*, cfg: RoutingAppConfig) -> None:
    """
    Run the Streamlit routing app.

    Re-engineering goals implemented here:
    - Keep "simple vs full" selector (so you don't break translations),
      but upgrade simple mode to include *essential trust signals*.
    - Move heavy/optional UI into expanders (progressive disclosure).
    - Separate: input -> optional reconcile -> optimize -> results.
    - Add a typed OptimizationResult to simplify rendering.
    """
    _setup_logging(logfile=cfg.logfile)
    logging.info("Starting routing app: %s", cfg.store_filename)

    language_selector(default_lang=None)
    st.title(t("app_title", name=cfg.title_name, city=cfg.title_city))

    init_state_if_missing(filename=cfg.store_filename)

    # Keep existing mode selector for compatibility with your i18n keys
    ui_mode_label = st.radio(
        t("ui_mode"),
        [t("ui_simple"), t("ui_full")],
        index=0,
        horizontal=True,
    )
    simple_mode = ui_mode_label == t("ui_simple")
    advanced_ui = not simple_mode

    # Advanced-only: optional road overlay
    show_road_overlay = False
    if advanced_ui:
        overlay_label = st.radio(
            t("road_overlay"),
            [t("off"), t("on")],
            index=0,
            horizontal=True,
        )
        show_road_overlay = overlay_label == t("on")

    # Inputs
    default_text = ""
    ensure_addresses_loaded(default_text=default_text, filename=cfg.store_filename)

    st.header("1) Addresses")  # TODO: i18n

    # OCR: always available, but detailed debug only in advanced UI
    if simple_mode:
        camera_ocr_widget(
            filename=cfg.store_filename,
            model="gpt-4.1-mini",
            key_prefix=f"camera_ocr.{cfg.store_filename}",
            home_address=cfg.home_address,
            overwrite=True,
            show_debug=False,
            duplicate_first_on_overwrite=False,
        )
    else:
        with st.expander(t("camera_ocr_expander"), expanded=False):
            camera_ocr_widget(
                filename=cfg.store_filename,
                model="gpt-4.1-mini",
                key_prefix=f"camera_ocr.{cfg.store_filename}",
                home_address=cfg.home_address,
                overwrite=True,
                show_debug=True,
                duplicate_first_on_overwrite=False,
            )

    addresses_text_area(
        label=t("addresses_label"),
        height=200,
        key="addresses_text_area",
    )

    drive_buttons_row(
        default_text=default_text,
        width="stretch",
        rerun_after_reload=True,
    )

    if advanced_ui:
        drive_version_loader(
            default_text=default_text,
            width="stretch",
            rerun_after_load=True,
        )

    routing_text = get_addresses_text()

    _render_quick_preflight(cfg=cfg, routing_text=routing_text)

    # Optional: reconciliation (advanced only), but with an "Apply" button to persist edits
    if advanced_ui and routing_text.strip():
        st.header("2) Clean up (optional)")  # TODO: i18n
        with st.expander("OCR → routing reconciliation", expanded=False):  # TODO: i18n
            raw_lines = _parse_addresses(routing_text)
            rows = _reconcile_addresses(raw_lines, home_address=cfg.home_address)

            rec_df = pd.DataFrame(
                {
                    "Use": [r.include for r in rows],
                    "Captured": [r.captured for r in rows],
                    "Final (editable)": [r.final for r in rows],
                    "Note": [r.note for r in rows],
                }
            )

            edited = st.data_editor(
                rec_df,
                use_container_width=True,
                column_config={"Use": st.column_config.CheckboxColumn("Use")},
                disabled=["Captured", "Note"],
                hide_index=True,
            )

            final_lines = [
                str(a).strip()
                for use, a in zip(
                    edited["Use"].tolist(),
                    edited["Final (editable)"].tolist(),
                )
                if bool(use) and str(a).strip()
            ]
            reconciled_text = "\n".join(final_lines)

            cols = st.columns([1, 1, 3])
            with cols[0]:
                if st.button("Apply to input", use_container_width=True):  # TODO: i18n
                    _set_addresses_text_in_state(reconciled_text)
                    st.rerun()
            with cols[1]:
                if st.button("Use for this run only", use_container_width=True):  # TODO: i18n
                    routing_text = reconciled_text
            with cols[2]:
                st.caption("Tip: ‘Apply’ updates the textarea so your edits persist.")  # TODO: i18n

    # Optional: coverage map (lazy)
    st.header("3) Network coverage (optional)")  # TODO: i18n
    with st.expander(t("graph_coverage_title"), expanded=False):
        show_cov = st.checkbox("Show coverage map", value=False)  # TODO: i18n
        if show_cov:
            graph = None
            nodes = None
            roi_3857 = _roi_bbox_3857(cfg)

            try:
                with st.spinner(t("loading_network")):
                    graph, nodes = _cached_drive_graph(str(cfg.data_dir), cfg.drive_prefix)

                poly = _graph_coverage_polygon_xy_3857(
                    graph=graph,
                    nodes=nodes,
                    roi_bbox_3857=roi_3857,
                    concavity_ratio=float(cfg.coverage_concavity_ratio),
                )

                if poly is None:
                    st.warning(t("graph_coverage_failed"))
                else:
                    if cfg.clip_coverage_to_land:
                        try:
                            land_union = _cached_land_union_3857()

                            water_union = None
                            if roi_3857 is not None:
                                water_union = _cached_water_union_3857(
                                    Polygon.from_bounds(*roi_3857)
                                )

                            poly = _clip_polygon_to_land_and_remove_water(
                                poly,
                                land_union_3857=land_union,
                                water_union_3857=water_union,
                            )
                        except Exception as exc:
                            st.warning(t("graph_coverage_landclip_error", error=str(exc)))

                    if cfg.roi_name:
                        st.caption(t("graph_coverage_subtitle_roi", roi=cfg.roi_name))
                    else:
                        st.caption(t("graph_coverage_subtitle"))

                    fig_cov = _make_tiled_coverage_figure(
                        poly_3857=poly,
                        title=t("graph_coverage_map_title"),
                        graph=graph,
                        nodes=nodes,
                        roi_bbox_3857=roi_3857,
                    )
                    st.pyplot(fig_cov, width="stretch")
            except Exception as exc:
                st.warning(t("graph_coverage_error", error=str(exc)))

    # Optimization controls
    st.header("4) Optimize")  # TODO: i18n

    route_type_label = st.radio(
        t("route_type"),
        [t("route_closed"), t("route_open")],
        index=0,
    )
    is_closed = route_type_label == t("route_closed")

    optimize_clicked = st.button(t("optimize"), type="primary")
    if not optimize_clicked:
        return

    # Run optimization and render results
    try:
        result = _compute_optimization(
            cfg=cfg,
            routing_text=routing_text,
            is_closed=is_closed,
            show_road_overlay=show_road_overlay,
            advanced_ui=advanced_ui,
        )
    except GeocodingError as exc:
        st.error(t("geocode_error", error=str(exc)))
        return
    except ValueError as exc:
        # For preflight / validation failures where we intentionally raised ValueError with a user message
        st.error(str(exc))
        if advanced_ui and "too_far_error" in str(exc):
            st.caption("Some locations are too far from the road network nodes.")  # TODO: i18n
        return
    except Exception as exc:
        # Generic unexpected
        st.error(t("geocode_unexpected", error=str(exc)))
        return

    # Validation (trust signals) + results
    st.header("5) Results")  # TODO: i18n
    _render_validation_tables(cfg=cfg, result=result, advanced_ui=advanced_ui)
    _render_results(cfg=cfg, result=result, advanced_ui=advanced_ui, show_road_overlay=show_road_overlay)
