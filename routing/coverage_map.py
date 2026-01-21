# routing/coverage_map.py
from __future__ import annotations

import math
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import streamlit as st

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from shapely.geometry import MultiPoint, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from routing.drive_network import load_drive_graph
from routing.route_plot import snapped_nodes_xy_3857

# Shapely concave hull (shapely >= 2.0). Fallback to convex hull if absent.
try:
    from shapely import concave_hull as _shapely_concave_hull  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    _shapely_concave_hull = None

# Basemap tiles (optional)
try:
    import contextily as ctx  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    ctx = None


@st.cache_resource(show_spinner=False)
def _cached_drive_graph(data_dir_str: str, drive_prefix: str) -> tuple[object, object]:
    """
    Load and cache the drive graph per (data_dir, drive_prefix).

    Args:
        data_dir_str: Path to the data directory as a string (Streamlit cache key friendly).
        drive_prefix: Dataset prefix for the drive network.

    Returns:
        A tuple (graph, nodes) as returned by `load_drive_graph`.
    """
    return load_drive_graph(data_dir=Path(data_dir_str), drive_prefix=drive_prefix)


@st.cache_resource(show_spinner=False)
def _cached_ne_land_shapefile_path_110m() -> Path:
    """
    Ensure Natural Earth 110m land shapefile exists locally and return its .shp path.

    This avoids `geopandas.datasets` (removed in GeoPandas 1.0) and works well in Streamlit:
    download once into a stable temp directory, then reuse on reruns.

    Returns:
        Path to `ne_110m_land.shp`.
    """
    url = "https://naturalearth.s3.amazonaws.com/110m_physical/ne_110m_land.zip"

    base_dir = Path(tempfile.gettempdir()) / "natural_earth" / "ne_110m_land"
    shp_path = base_dir / "ne_110m_land.shp"
    if shp_path.exists():
        return shp_path

    base_dir.mkdir(parents=True, exist_ok=True)
    zip_path = base_dir / "ne_110m_land.zip"

    urllib.request.urlretrieve(url, zip_path)  # noqa: S310 (URL is fixed, trusted source)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(base_dir)

    if not shp_path.exists():
        raise FileNotFoundError(f"Expected land shapefile not found after extract: {shp_path}")

    return shp_path


@st.cache_resource(show_spinner=False)
def _cached_land_union_3857() -> BaseGeometry:
    """
    Load a land polygon union in EPSG:3857.

    Uses Natural Earth 110m land data downloaded and cached locally.

    Returns:
        Unioned land geometry in EPSG:3857.
    """
    land_path = _cached_ne_land_shapefile_path_110m()
    land = gpd.read_file(land_path)
    if land.empty:
        raise ValueError(f"Natural Earth land dataset is empty: {land_path}")

    land_3857 = land[["geometry"]].to_crs(epsg=3857)
    return unary_union(land_3857.geometry)


def _clip_polygon_to_land(
    poly_3857: Polygon,
    *,
    land_union_3857: BaseGeometry,
) -> Polygon:
    """
    Clip coverage polygon to land where possible.

    Args:
        poly_3857: Coverage polygon in EPSG:3857.
        land_union_3857: Unioned land geometry in EPSG:3857.

    Returns:
        A Polygon in EPSG:3857 (falls back to the original polygon on failure).
    """
    try:
        geom = poly_3857.intersection(land_union_3857)
    except Exception:
        return poly_3857

    if geom.is_empty:
        return poly_3857

    if geom.geom_type == "Polygon":
        return geom  # type: ignore[return-value]
    if geom.geom_type == "MultiPolygon":
        return max(geom.geoms, key=lambda g: g.area)  # type: ignore[return-value]

    return poly_3857


def _latlon_to_webmercator_xy(lat: float, lon: float) -> tuple[float, float]:
    """
    Convert WGS84 latitude/longitude to EPSG:3857 x/y meters.

    Args:
        lat: Latitude in degrees.
        lon: Longitude in degrees.

    Returns:
        (x, y) in EPSG:3857 meters.
    """
    r = 6378137.0
    x = r * math.radians(float(lon))
    lat_clamped = max(min(float(lat), 89.999999), -89.999999)
    y = r * math.log(math.tan(math.pi / 4.0 + math.radians(lat_clamped) / 2.0))
    return x, y


def _roi_bbox_3857(
    roi_bbox_wgs84: tuple[float, float, float, float] | None,
) -> tuple[float, float, float, float] | None:
    """
    Convert ROI bounding box from WGS84 to EPSG:3857.

    Args:
        roi_bbox_wgs84: (min_lat, min_lon, max_lat, max_lon) in WGS84 degrees.

    Returns:
        (minx, miny, maxx, maxy) in EPSG:3857 meters, or None.
    """
    if roi_bbox_wgs84 is None:
        return None
    min_lat, min_lon, max_lat, max_lon = roi_bbox_wgs84
    x1, y1 = _latlon_to_webmercator_xy(min_lat, min_lon)
    x2, y2 = _latlon_to_webmercator_xy(max_lat, max_lon)
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def _iter_node_ids_from_graph(graph: object) -> list[int]:
    """
    Extract node IDs from a NetworkX-like graph.

    Args:
        graph: A graph with `.nodes` or iterable node IDs.

    Returns:
        List of node IDs coercible to int.
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
    Estimate typical edge length (meters) from edge attribute 'length'.

    Args:
        graph: A graph with `.edges(data=True)` yielding edge attribute dicts.
        sample_size: Max number of edges to sample.

    Returns:
        Median edge length in meters, or None if unavailable.
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
    Compute concave hull (preferred) with convex hull fallback.

    Args:
        points: MultiPoint geometry (EPSG:3857).
        ratio: Concavity ratio in (0, 1]; smaller usually means "more concave".

    Returns:
        Polygon hull, or None if hull cannot be constructed.
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
                return geom  # type: ignore[return-value]
            if geom.geom_type == "MultiPolygon":
                return max(geom.geoms, key=lambda g: g.area)  # type: ignore[return-value]
        except Exception:
            pass

    hull = points.convex_hull
    if hull.is_empty:
        return None
    if hull.geom_type == "Polygon":
        return hull  # type: ignore[return-value]
    return None


def _graph_coverage_polygon_xy_3857(
    *,
    graph: object,
    nodes: object,
    roi_bbox_3857: tuple[float, float, float, float] | None,
    concavity_ratio: float,
) -> Polygon | None:
    """
    Build a buffered coverage polygon from node locations (EPSG:3857).

    Args:
        graph: Drive graph.
        nodes: Node lookup structure expected by `snapped_nodes_xy_3857`.
        roi_bbox_3857: Optional ROI bounds (minx, miny, maxx, maxy) in EPSG:3857.
        concavity_ratio: Concavity ratio for concave hull.

    Returns:
        Coverage polygon (EPSG:3857), or None if not constructible.
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
        return buffered  # type: ignore[return-value]
    if buffered.geom_type == "MultiPolygon":
        return max(buffered.geoms, key=lambda g: g.area)  # type: ignore[return-value]
    return None


def _make_tiled_coverage_figure(
    *,
    poly_3857: Polygon,
    title: str,
    roi_bbox_3857: tuple[float, float, float, float] | None,
    fill_alpha: float = 0.12,
) -> Figure:
    """
    Create a map with basemap tiles and overlay the coverage polygon (EPSG:3857).

    Notes:
        - Nodes are intentionally not plotted to keep rendering fast.
        - The polygon area is filled with a transparent (but visible) background.

    Args:
        poly_3857: Coverage polygon in EPSG:3857.
        title: Plot title.
        roi_bbox_3857: Optional ROI bounds to set axes limits.
        fill_alpha: Alpha for the polygon fill (0..1).

    Returns:
        Matplotlib Figure.
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
                crs="EPSG:3857",
                source=ctx.providers.OpenStreetMap.Mapnik,
                attribution=False,
            )
        except Exception:
            pass

    x, y = poly_3857.exterior.xy

    coverage_color = "#f57c00"
    ax.plot(
        x,
        y,
        linewidth=2.0,
        alpha=0.9,
        color=coverage_color,
    )
    ax.fill(
        x,
        y,
        alpha=float(np.clip(fill_alpha, 0.0, 1.0)),
        color=coverage_color,
        linewidth=0,
    )

    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    return fig


def render_coverage_map(
    *,
    data_dir: str,
    drive_prefix: str,
    roi_bbox_wgs84: tuple[float, float, float, float] | None,
    concavity_ratio: float,
    clip_to_land: bool,
    roi_name: str | None = None,
    title: str = "Network coverage",
    map_title: str = "Coverage map",
    subtitle: str = "This map shows where the drive network exists.",
    fill_alpha: float = 0.12,
) -> None:
    """
    Render a coverage map section in Streamlit.

    Args:
        data_dir: Directory containing the drive network data.
        drive_prefix: Dataset prefix for the drive network.
        roi_bbox_wgs84: Optional ROI bounds (min_lat, min_lon, max_lat, max_lon) in WGS84.
        concavity_ratio: Concavity ratio for concave hull.
        clip_to_land: Whether to clip polygon to land.
        roi_name: Optional ROI label.
        title: Streamlit section title.
        map_title: Figure title.
        subtitle: Streamlit caption text.
        fill_alpha: Alpha for the polygon fill (0..1).
    """
    roi_3857 = _roi_bbox_3857(roi_bbox_wgs84)
    graph, nodes = _cached_drive_graph(data_dir, drive_prefix)

    poly = _graph_coverage_polygon_xy_3857(
        graph=graph,
        nodes=nodes,
        roi_bbox_3857=roi_3857,
        concavity_ratio=float(concavity_ratio),
    )

    if poly is None:
        st.warning("Could not construct coverage polygon.")
        return

    if clip_to_land:
        try:
            land_union = _cached_land_union_3857()
            poly = _clip_polygon_to_land(poly, land_union_3857=land_union)
        except Exception as exc:
            st.warning(f"Land clipping failed: {exc}")

    st.subheader(title)
    st.caption(f"{subtitle} ROI: {roi_name}" if roi_name else subtitle)

    fig = _make_tiled_coverage_figure(
        poly_3857=poly,
        title=map_title,
        roi_bbox_3857=roi_3857,
        fill_alpha=float(fill_alpha),
    )
    st.pyplot(fig, width='stretch')
