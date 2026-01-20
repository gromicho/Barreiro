# routing/coverage_map.py
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import streamlit as st

import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import MultiPoint, Polygon
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
    """Load and cache the drive graph per (data_dir, drive_prefix)."""
    return load_drive_graph(data_dir=Path(data_dir_str), drive_prefix=drive_prefix)


@st.cache_resource(show_spinner=False)
def _cached_land_union_3857() -> object:
    """
    Load a land polygon union in EPSG:3857.

    Uses GeoPandas' built-in Natural Earth lowres dataset (offline).
    """
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    land_3857 = world[["geometry"]].to_crs(epsg=3857)
    return unary_union(land_3857.geometry)


@st.cache_resource(show_spinner=False)
def _cached_water_union_3857(roi_polygon_3857: Polygon) -> object | None:
    """
    Load OSM water polygons inside ROI and union them.

    Requires osmnx; returns None if unavailable or errors.
    """
    try:
        import osmnx as ox
    except Exception:
        return None

    roi_wgs84 = gpd.GeoSeries([roi_polygon_3857], crs=3857).to_crs(epsg=4326).iloc[0]
    tags = {"natural": "water", "waterway": "riverbank"}

    try:
        gdf = ox.features_from_polygon(roi_wgs84, tags=tags)
    except Exception:
        return None

    if gdf.empty:
        return None

    gdf_3857 = gdf.to_crs(epsg=3857)
    return unary_union(gdf_3857.geometry)


def _clip_polygon_to_land_and_remove_water(
    poly_3857: Polygon,
    *,
    land_union_3857: object,
    water_union_3857: object | None,
) -> Polygon:
    """Keep land, remove water when possible."""
    try:
        geom = poly_3857.intersection(land_union_3857)
        if water_union_3857 is not None:
            geom = geom.difference(water_union_3857)
    except Exception:
        return poly_3857

    if geom.is_empty:
        return poly_3857

    if geom.geom_type == "Polygon":
        return geom
    if geom.geom_type == "MultiPolygon":
        return max(geom.geoms, key=lambda g: g.area)

    return poly_3857


def _latlon_to_webmercator_xy(lat: float, lon: float) -> tuple[float, float]:
    """Convert WGS84 lat/lon to EPSG:3857 x/y (meters)."""
    r = 6378137.0
    x = r * math.radians(float(lon))
    lat_clamped = max(min(float(lat), 89.999999), -89.999999)
    y = r * math.log(math.tan(math.pi / 4.0 + math.radians(lat_clamped) / 2.0))
    return x, y


def _roi_bbox_3857(
    roi_bbox_wgs84: tuple[float, float, float, float] | None,
) -> tuple[float, float, float, float] | None:
    """Convert ROI bbox from WGS84 to EPSG:3857."""
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
    """Estimate typical edge length (meters) from edge attribute 'length'."""
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
    """Compute concave hull (preferred) with convex fallback."""
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
    """Build a buffered coverage polygon from node locations."""
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


def _make_tiled_coverage_figure(
    *,
    poly_3857: Polygon,
    title: str,
    graph: object,
    nodes: object,
    roi_bbox_3857: tuple[float, float, float, float] | None,
    max_scatter_points: int = 4000,
) -> object:
    """Create a map with basemap tiles and overlay the coverage polygon + nodes."""
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
) -> None:
    """Render a coverage map section in Streamlit."""
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
            water_union = None
            if roi_3857 is not None:
                water_union = _cached_water_union_3857(Polygon.from_bounds(*roi_3857))
            poly = _clip_polygon_to_land_and_remove_water(
                poly,
                land_union_3857=land_union,
                water_union_3857=water_union,
            )
        except Exception as exc:
            st.warning(f"Land/water clipping failed: {exc}")

    st.subheader(title)
    st.caption(f"{subtitle} ROI: {roi_name}" if roi_name else subtitle)

    fig = _make_tiled_coverage_figure(
        poly_3857=poly,
        title=map_title,
        graph=graph,
        nodes=nodes,
        roi_bbox_3857=roi_3857,
    )
    st.pyplot(fig, width="stretch")
