# routing/coverage_map.py
from __future__ import annotations

import math

import numpy as np
import streamlit as st

# Heavy geo stack lives here (so routing/app.py stays light)
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import MultiPoint, Polygon
from shapely.ops import unary_union

from routing.drive_network import load_drive_graph
from routing.route_plot import snapped_nodes_xy_3857

# Shapely (latest) concave hull API (keep a robust fallback)
try:
    from shapely import concave_hull as _shapely_concave_hull  # shapely >= 2.0
except Exception:  # pragma: no cover
    _shapely_concave_hull = None

# Basemap tiles (optional)
try:
    import contextily as ctx  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    ctx = None


def _latlon_to_webmercator_xy(lat: float, lon: float) -> tuple[float, float]:
    """Convert WGS84 lat/lon to EPSG:3857 x/y (meters)."""
    r = 6378137.0
    x = r * math.radians(float(lon))
    lat_clamped = max(min(float(lat), 89.999999), -89.999999)
    y = r * math.log(math.tan(math.pi / 4.0 + math.radians(lat_clamped) / 2.0))
    return x, y


def _roi_bbox_3857(roi_bbox_wgs84: tuple[float, float, float, float] | None) -> tuple[float, float, float, float] | None:
    """
    Convert ROI bbox from WGS84 to EPSG:3857.

    Args:
        roi_bbox_wgs84: (min_lat, min_lon, max_lat, max_lon) in degrees.

    Returns:
        (min_x, min_y, max_x, max_y) in meters or None.
    """
    if roi_bbox_wgs84 is None:
        return None

    min_lat, min_lon, max_lat, max_lon = roi_bbox_wgs84
    x1, y1 = _latlon_to_webmercator_xy(min_lat, min_lon)
    x2, y2 = _latlon_to_webmercator_xy(max_lat, max_lon)
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def _iter_node_ids_from_graph(graph: object) -> list[int]:
    """Extract node ids from a NetworkX-like graph."""
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
        val = data.get("length")
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
            if geom.geom_type == "Polygon":
                return geom
            if geom.geom_type == "MultiPolygon":
                return max(geom.geoms, key=lambda g: g.area)
        except Exception:
            pass

    hull = points.convex_hull
    if hull.is_empty:
        return None
    if hull.geom_type == "Polygon":
        return hull
    return None


def _graph_coverage_polygon_xy_3857(
    *,
    graph: object,
    nodes: object,
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
    buffer_m = 250.0 if edge_med_m is None else float(np.clip(0.75 * edge_med_m, 75.0, 1500.0))

    buffered = poly.buffer(buffer_m)
    if buffered.geom_type == "Polygon":
        return buffered
    if buffered.geom_type == "MultiPolygon":
        return max(buffered.geoms, key=lambda g: g.area)
    return None


@st.cache_resource(show_spinner=False)
def _cached_drive_graph(data_dir_str: str, drive_prefix: str) -> tuple[object, object]:
    """Load and cache the drive graph per (data_dir, drive_prefix)."""
    return load_drive_graph(data_dir=gpd.GeoSeries([0]).index._data.__class__.__mro__[1]("data") if False else __import__("pathlib").Path(data_dir_str),  # noqa: E501
                           drive_prefix=drive_prefix)
