from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import math

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import MultiPoint, Polygon
from shapely.ops import unary_union

# Shapely (latest) concave hull API (keep a robust fallback)
try:
    from shapely import concave_hull as _shapely_concave_hull  # shapely >= 2.0
except Exception:  # pragma: no cover
    _shapely_concave_hull = None

# Basemap tiles
try:
    import contextily as ctx  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    ctx = None

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

    data_dir: Path = Path('data')
    logfile: str = LOGFILE_DEFAULT
    max_snap_distance_m: float = MAX_SNAP_DISTANCE_M_DEFAULT


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


DriveGraph = object
DriveNodes = object


@st.cache_resource(show_spinner=False)
def _cached_drive_graph(data_dir_str: str, drive_prefix: str) -> tuple[DriveGraph, DriveNodes]:
    """Load and cache the drive graph per (data_dir, drive_prefix)."""
    return load_drive_graph(data_dir=Path(data_dir_str), drive_prefix=drive_prefix)


@st.cache_resource(show_spinner=False)
def _cached_water_union_3857(roi_polygon_3857: Polygon) -> object:
    """
    Load OSM water polygons (rivers, estuaries, sea) inside ROI and union them.
    """
    import osmnx as ox

    roi_wgs84 = gpd.GeoSeries([roi_polygon_3857], crs=3857).to_crs(epsg=4326).iloc[0]

    tags = {
        'natural': ['water'],
        'waterway': ['riverbank'],
    }

    gdf = ox.geometries_from_polygon(roi_wgs84, tags=tags)

    if gdf.empty:
        return None

    gdf_3857 = gdf.to_crs(epsg=3857)
    return unary_union(gdf_3857.geometry)

@st.cache_resource(show_spinner=False)
def _cached_land_union_3857() -> object:
    """
    Load a land polygon union in EPSG:3857.

    Uses GeoPandas' built-in Natural Earth lowres dataset (offline, no extra packages).
    Returns a Shapely geometry (often a MultiPolygon) suitable for intersections.
    """
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    land_3857 = world[['geometry']].to_crs(epsg=3857)
    return unary_union(land_3857.geometry)


def _clip_polygon_to_land_and_remove_water(
    poly_3857: Polygon,
    *,
    land_union_3857: object,
    water_union_3857: object | None,
) -> Polygon:
    """
    Keep land, remove water (rivers, estuaries, sea).
    """
    try:
        geom = poly_3857.intersection(land_union_3857)
        if water_union_3857 is not None:
            geom = geom.difference(water_union_3857)
    except Exception:
        return poly_3857

    if geom.is_empty:
        return poly_3857

    if geom.geom_type == 'Polygon':
        return geom

    if geom.geom_type == 'MultiPolygon':
        return max(geom.geoms, key=lambda g: g.area)

    return poly_3857


def _parse_addresses(text: str) -> list[str]:
    """Parse non-empty address lines from a text blob."""
    return [line.strip() for line in text.splitlines() if line.strip()]


def _summarize_address_label(address: str) -> str:
    """
    Produce a compact address label for UI.

    Args:
        address: Full address string.

    Returns:
        Shortened label suitable for UI labels.
    """
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


def _distance_matrix_to_km(dist_matrix: list[list[float]]) -> np.ndarray:
    """
    Convert a distance matrix to kilometers using a robust heuristic for units.

    Args:
        dist_matrix: Square matrix of distances in unknown units (meters or km).

    Returns:
        NumPy array with distances in kilometers.
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
    Build a DataFrame to display a distance matrix in Streamlit.

    Args:
        dist_matrix_raw_units: Distance matrix in unknown units (meters or km).
        addresses: Addresses in the same order as the matrix rows.

    Returns:
        DataFrame with km values (float) rounded to 1 decimal.
    """
    n = len(dist_matrix_raw_units)
    row_labels = [f'{i}. {_summarize_address_label(a)}' for i, a in enumerate(addresses, start=1)]
    col_labels = [str(i) for i in range(1, n + 1)]

    km = _distance_matrix_to_km(dist_matrix_raw_units).round(1)
    return pd.DataFrame(km, index=row_labels, columns=col_labels)


def _build_input_addresses(*, cfg: RoutingAppConfig, addresses_text: str, ocr_used: bool) -> list[str]:
    """
    Build the final address list used for routing.

    Args:
        cfg: Routing app configuration (includes home_address).
        addresses_text: Text from the addresses textarea.
        ocr_used: Whether OCR was used to populate the textarea.

    Returns:
        Address list for routing.
    """
    addresses = _parse_addresses(addresses_text)

    if not addresses:
        return [cfg.home_address]

    if ocr_used and len(addresses) >= 1:
        addresses = [addresses[0]] + addresses

    home = cfg.home_address.strip()
    if home and (not addresses or addresses[0] != home):
        addresses = [home] + addresses

    return addresses


def _ensure_closed[T](items: list[T]) -> list[T]:
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


def _latlon_to_webmercator_xy(lat: float, lon: float) -> tuple[float, float]:
    """
    Convert WGS84 lat/lon to EPSG:3857 x/y (meters).

    Args:
        lat: Latitude in degrees.
        lon: Longitude in degrees.

    Returns:
        (x, y) in Web Mercator meters.
    """
    r = 6378137.0
    x = r * math.radians(float(lon))
    lat_clamped = max(min(float(lat), 89.999999), -89.999999)
    y = r * math.log(math.tan(math.pi / 4.0 + math.radians(lat_clamped) / 2.0))
    return x, y


def _roi_bbox_3857(cfg: RoutingAppConfig) -> tuple[float, float, float, float] | None:
    """
    Convert config ROI bbox from WGS84 to EPSG:3857.

    Returns:
        (min_x, min_y, max_x, max_y) in meters or None.
    """
    if cfg.roi_bbox_wgs84 is None:
        return None

    min_lat, min_lon, max_lat, max_lon = cfg.roi_bbox_wgs84
    x1, y1 = _latlon_to_webmercator_xy(min_lat, min_lon)
    x2, y2 = _latlon_to_webmercator_xy(max_lat, max_lon)
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def _iter_node_ids_from_graph(graph: object) -> list[int]:
    """
    Extract node ids from a NetworkX-like graph.

    Args:
        graph: Drive graph (expected to behave like a NetworkX graph).

    Returns:
        List of node ids (as ints when possible).
    """
    try:
        node_ids = list(graph.nodes)  # type: ignore[attr-defined]
    except Exception:
        node_ids = list(graph)  # type: ignore[arg-type]

    out: list[int] = []
    for nid in node_ids:
        try:
            out.append(int(nid))
        except Exception:
            continue
    return out


def _estimate_edge_length_m(graph: object, *, sample_size: int = 5000) -> float | None:
    """
    Estimate a typical edge length from the graph (meters), if available.

    Looks for edge attribute 'length' (common in OSMnx / road graphs).

    Args:
        graph: Drive graph (NetworkX-like).
        sample_size: Max number of edges to sample.

    Returns:
        Median edge length in meters, or None if not found.
    """
    lengths: list[float] = []
    try:
        edges_iter = graph.edges(data=True)  # type: ignore[attr-defined]
    except Exception:
        return None

    for k, (_u, _v, data) in enumerate(edges_iter):
        if k >= sample_size:
            break
        if not isinstance(data, dict):
            continue
        val = data.get('length')
        if val is None:
            continue
        try:
            f = float(val)
        except Exception:
            continue
        if np.isfinite(f) and f > 0:
            lengths.append(f)

    if not lengths:
        return None

    return float(np.median(np.array(lengths, dtype=float)))


def _concave_or_convex_hull(points: MultiPoint, *, ratio: float) -> Polygon | None:
    """
    Compute a concave hull (preferred) with a safe convex fallback.

    Args:
        points: MultiPoint in EPSG:3857.
        ratio: Concavity ratio in (0, 1]. Lower => more concave. 1 => convex-ish.

    Returns:
        Polygon or None.
    """
    if points.is_empty:
        return None

    r = float(ratio)
    if not (0.0 < r <= 1.0):
        r = 0.25

    if _shapely_concave_hull is not None:
        try:
            geom = _shapely_concave_hull(points, r)
            if geom.is_empty:
                return None
            if geom.geom_type == 'Polygon':
                return geom
            if geom.geom_type == 'MultiPolygon':
                return max(geom.geoms, key=lambda g: g.area)
        except Exception:
            pass

    hull = points.convex_hull
    if hull.is_empty:
        return None
    if hull.geom_type == 'Polygon':
        return hull
    return None


def _graph_coverage_polygon_xy_3857(
    *,
    graph: object,
    nodes: DriveNodes,
    roi_bbox_3857: tuple[float, float, float, float] | None,
    concavity_ratio: float,
) -> Polygon | None:
    """
    Build a coverage polygon for the graph based on its nodes and edges.

    Strategy:
    1) Take node coordinates (EPSG:3857).
    2) If ROI is provided, restrict to nodes inside ROI.
    3) Compute concave hull (preferred) else convex hull.
    4) Buffer by an automatically-derived distance using median edge length.

    Args:
        graph: Drive graph.
        nodes: Nodes lookup compatible with snapped_nodes_xy_3857.
        roi_bbox_3857: Optional ROI bbox in EPSG:3857 (min_x, min_y, max_x, max_y).
        concavity_ratio: Concave hull ratio in (0, 1].

    Returns:
        A Shapely Polygon in EPSG:3857, or None if it cannot be constructed.
    """
    node_ids = _iter_node_ids_from_graph(graph)
    if not node_ids:
        return None

    xs, ys = snapped_nodes_xy_3857(node_ids, nodes)
    if not xs or not ys:
        return None

    if roi_bbox_3857 is not None:
        min_x, min_y, max_x, max_y = roi_bbox_3857
        keep_ids: list[int] = []
        for nid, x, y in zip(node_ids, xs, ys):
            if (min_x <= float(x) <= max_x) and (min_y <= float(y) <= max_y):
                keep_ids.append(nid)

        if not keep_ids:
            return None

        xs, ys = snapped_nodes_xy_3857(keep_ids, nodes)

    pts = MultiPoint([(float(x), float(y)) for x, y in zip(xs, ys)])
    poly = _concave_or_convex_hull(pts, ratio=concavity_ratio)
    if poly is None:
        return None

    edge_med_m = _estimate_edge_length_m(graph)
    if edge_med_m is None:
        buffer_m = 250.0
    else:
        buffer_m = float(np.clip(0.75 * edge_med_m, 75.0, 1500.0))

    buffered = poly.buffer(buffer_m)

    if buffered.geom_type == 'Polygon':
        return buffered
    if buffered.geom_type == 'MultiPolygon':
        return max(buffered.geoms, key=lambda g: g.area)
    return None


def _make_tiled_coverage_figure(
    *,
    poly_3857: Polygon,
    title: str,
    graph: object,
    nodes: DriveNodes,
    roi_bbox_3857: tuple[float, float, float, float] | None,
    max_scatter_points: int = 4000,
) -> object:
    """
    Create a map with basemap tiles and overlay the coverage polygon + nodes.

    Uses EPSG:3857 so it matches typical web tiles.

    Args:
        poly_3857: Coverage polygon in EPSG:3857.
        title: Figure title.
        graph: Drive graph.
        nodes: Nodes lookup.
        roi_bbox_3857: Optional ROI bbox to set a stable viewport.
        max_scatter_points: Downsample node scatter for speed.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_title(title)

    minx, miny, maxx, maxy = poly_3857.bounds
    pad_x = 0.05 * (maxx - minx) if maxx > minx else 250.0
    pad_y = 0.05 * (maxy - miny) if maxy > miny else 250.0

    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)

    if roi_bbox_3857 is not None:
        rminx, rminy, rmaxx, rmaxy = roi_bbox_3857
        ax.set_xlim(rminx, rmaxx)
        ax.set_ylim(rminy, rmaxy)

    if ctx is not None:
        try:
            ctx.add_basemap(
                ax,
                crs='EPSG:3857',
                source=ctx.providers.OpenStreetMap.Mapnik,
                attribution=False,
            )
        except Exception:
            pass

    node_ids = _iter_node_ids_from_graph(graph)
    if node_ids:
        if len(node_ids) > max_scatter_points:
            step = max(1, len(node_ids) // max_scatter_points)
            node_ids = node_ids[::step]

        xs, ys = snapped_nodes_xy_3857(node_ids, nodes)
        if xs and ys:
            ax.scatter(xs, ys, s=2, alpha=0.25)

    x, y = poly_3857.exterior.xy
    ax.plot(x, y, linewidth=2.0, alpha=0.9)

    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    return fig


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

    if simple_mode:
        ocr_used = camera_ocr_widget(
            filename=cfg.store_filename,
            model='gpt-4.1-mini',
            key_prefix=f'camera_ocr.{cfg.store_filename}',
            home_address=cfg.home_address,
            overwrite=True,
            show_debug=False,
            duplicate_first_on_overwrite=False,
        )
    else:
        with st.expander(t('camera_ocr_expander'), expanded=False):
            ocr_used = camera_ocr_widget(
                filename=cfg.store_filename,
                model='gpt-4.1-mini',
                key_prefix=f'camera_ocr.{cfg.store_filename}',
                home_address=cfg.home_address,
                overwrite=True,
                show_debug=True,
                duplicate_first_on_overwrite=False,
            )

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

    # --- Graph coverage map FIRST (tiles + concave hull + land clipping) ---
    graph: DriveGraph | None = None
    nodes: DriveNodes | None = None
    roi_3857 = _roi_bbox_3857(cfg)

    try:
        with st.spinner(t('loading_network')):
            graph, nodes = _cached_drive_graph(str(cfg.data_dir), cfg.drive_prefix)

        poly = _graph_coverage_polygon_xy_3857(
            graph=graph,
            nodes=nodes,
            roi_bbox_3857=roi_3857,
            concavity_ratio=float(cfg.coverage_concavity_ratio),
        )

        if poly is None:
            st.warning(t('graph_coverage_failed'))
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
                    st.warning(t('graph_coverage_landclip_error', error=str(exc)))

            st.subheader(t('graph_coverage_title'))
            if cfg.roi_name:
                st.caption(t('graph_coverage_subtitle_roi', roi=cfg.roi_name))
            else:
                st.caption(t('graph_coverage_subtitle'))

            fig_cov = _make_tiled_coverage_figure(
                poly_3857=poly,
                title=t('graph_coverage_map_title'),
                graph=graph,
                nodes=nodes,
                roi_bbox_3857=roi_3857,
            )
            st.pyplot(fig_cov, width='stretch')
    except Exception as exc:
        st.warning(t('graph_coverage_error', error=str(exc)))

    if not st.button(t('optimize')):
        return

    logs: list[str] = []
    with timeblock('Total optimization run', logs):
        addresses = _build_input_addresses(
            cfg=cfg,
            addresses_text=get_addresses_text(),
            ocr_used=bool(ocr_used),
        )

        if len(addresses) < 2:
            st.error(t('need_two'))
            return

        if (not is_closed) and len(addresses) < 3:
            st.error(t('need_three_open'))
            return

        try:
            if graph is None or nodes is None:
                with st.spinner(t('loading_network')):
                    with timeblock('Loading drive graph', logs):
                        graph, nodes = _cached_drive_graph(str(cfg.data_dir), cfg.drive_prefix)
        except Exception as exc:
            st.error(t('network_load_error', error=str(exc)))
            return

        try:
            with st.spinner(t('geocoding')):
                with timeblock('Geocoding addresses', logs):
                    coords = geocode_addresses(
                        addresses=addresses,
                        bbox=cfg.roi_bbox_wgs84,
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
                                f'**{t("input_address")}** {addr}  \n'
                                f'{t("geocode_line", lat=f"{lat:.6f}", lon=f"{lon:.6f}")}  \n'
                                f'[{t("view_in_maps")}]({gmaps_link})'
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
            st.subheader(t('snapping_overview_title'))
            snap_km = np.array(snapped_distances_m, dtype=float) / 1000.0
            overview_df = pd.DataFrame(
                {
                    '#': list(range(1, len(addresses) + 1)),
                    'Address': [_summarize_address_label(a) for a in addresses],
                    'Lat': [round(lat, 6) for (lat, _lon) in coords],
                    'Lon': [round(lon, 6) for (_lat, lon) in coords],
                    'Snap dist (km)': np.round(snap_km, 2),
                }
            )
            try:
                st.dataframe(
                    overview_df,
                    use_container_width=True,
                    column_config={
                        'Snap dist (km)': st.column_config.NumberColumn(
                            t('snap_dist_km_col'),
                            format='%.2f km',
                        ),
                    },
                )
            except Exception:
                st.dataframe(overview_df, use_container_width=True)

        offending = [i for i, d in enumerate(snapped_distances_m) if d > cfg.max_snap_distance_m]
        if offending:
            st.error(t('too_far_error', km=f'{cfg.max_snap_distance_m / 1000.0:.1f}'))
            if not simple_mode:
                st.write(t('too_far_list_title'))
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

        if not simple_mode:
            with st.expander(t('dist_matrix_expander'), expanded=False):
                dist_df = _build_distance_matrix_df_km(dist_matrix, addresses)
                try:
                    st.dataframe(
                        dist_df,
                        use_container_width=True,
                        column_config={
                            col: st.column_config.NumberColumn(col, format='%.1f km') for col in dist_df.columns
                        },
                    )
                except Exception:
                    st.dataframe(dist_df, use_container_width=True)

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

            if is_closed:
                orig_coords = _ensure_closed(orig_coords)
                opt_coords = _ensure_closed(opt_coords)
                orig_node_ids = _ensure_closed(orig_node_ids)
                opt_node_ids = _ensure_closed(opt_node_ids)

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
                st.markdown(
                    f'**{t("orig_order")}**  \n{t("total_distance_km")} **{total_km_original:.2f}**'
                )
                st.pyplot(fig_orig, width='stretch')
            with col_r:
                st.markdown(
                    f'**{t("opt_order")}**  \n{t("total_distance_km")} **{total_km_optimized:.2f}**'
                )
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
