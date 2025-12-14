# routing/route_plot.py

import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
from pyproj import Transformer


def _transform_xy_lists(
    xs: list[float],
    ys: list[float],
    src_crs: str,
    dst_crs: str = 'EPSG:3857',
) -> tuple[list[float], list[float]]:
    """
    Transform coordinate lists between CRS.

    Args:
        xs: X coordinates in src CRS.
        ys: Y coordinates in src CRS.
        src_crs: Source CRS string.
        dst_crs: Destination CRS string.

    Returns:
        (xs_transformed, ys_transformed)
    """
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    xs_t: list[float] = []
    ys_t: list[float] = []
    for x, y in zip(xs, ys):
        x2, y2 = transformer.transform(x, y)
        xs_t.append(float(x2))
        ys_t.append(float(y2))
    return xs_t, ys_t


def route_nodes_to_edge_geometry_xy_3857(
    route_node_ids: list[int],
    graph: nx.Graph,
    nodes: gpd.GeoDataFrame,
) -> tuple[list[float], list[float]]:
    """
    Expand a route of node ids into a polyline using edge geometries (where present),
    returned as (x,y) in EPSG:3857 for plotting.

    Args:
        route_node_ids: Node ids in visit order (may include repeated first node for closed plots).
        graph: NetworkX graph with 'length' edges and optional 'geometry'.
        nodes: Nodes GeoDataFrame with x/y columns and CRS.

    Returns:
        (xs_3857, ys_3857)
    """
    if nodes.crs is None:
        raise RuntimeError('Nodes GeoDataFrame has no CRS.')

    if len(route_node_ids) < 2:
        return [], []

    xs_path: list[float] = []
    ys_path: list[float] = []

    for a, b in zip(route_node_ids[:-1], route_node_ids[1:]):
        path_nodes: list[int] = nx.shortest_path(graph, int(a), int(b), weight='length')

        for u, v in zip(path_nodes[:-1], path_nodes[1:]):
            edge_data = graph.get_edge_data(int(u), int(v)) or {}
            geom = edge_data.get('geometry')

            if geom is not None and hasattr(geom, 'coords'):
                coords = list(geom.coords)
                x_list = [float(x) for x, _y in coords]
                y_list = [float(y) for _x, y in coords]
            else:
                row_u = nodes.loc[int(u)]
                row_v = nodes.loc[int(v)]
                x_list = [float(row_u['x']), float(row_v['x'])]
                y_list = [float(row_u['y']), float(row_v['y'])]

            if xs_path and x_list and y_list:
                x_list = x_list[1:]
                y_list = y_list[1:]

            xs_path.extend(x_list)
            ys_path.extend(y_list)

    return _transform_xy_lists(xs_path, ys_path, src_crs=str(nodes.crs), dst_crs='EPSG:3857')


def snapped_nodes_xy_3857(
    snapped_node_ids: list[int],
    nodes: gpd.GeoDataFrame,
) -> tuple[list[float], list[float]]:
    """
    Get snapped node coordinates in EPSG:3857 for plotting.

    Args:
        snapped_node_ids: Node ids.
        nodes: Nodes GeoDataFrame.

    Returns:
        (xs_3857, ys_3857)
    """
    if nodes.crs is None:
        raise RuntimeError('Nodes GeoDataFrame has no CRS.')

    xs: list[float] = []
    ys: list[float] = []
    for nid in snapped_node_ids:
        row = nodes.loc[int(nid)]
        xs.append(float(row['x']))
        ys.append(float(row['y']))

    return _transform_xy_lists(xs, ys, src_crs=str(nodes.crs), dst_crs='EPSG:3857')


def make_matplotlib_route_map(
    coords: list[tuple[float, float]],
    *,
    title: str,
    color: str,
    road_xs: list[float] | None,
    road_ys: list[float] | None,
    snapped_xs: list[float] | None,
    snapped_ys: list[float] | None,
) -> plt.Figure:
    """
    Make a basemap-backed plot (EPSG:3857) comparing straight lines vs road overlay.

    Args:
        coords: List of (lat, lon) in WGS84 (already ordered for plotting).
        title: Plot title.
        color: Line color for the main line.
        road_xs: Optional road polyline xs (EPSG:3857).
        road_ys: Optional road polyline ys (EPSG:3857).
        snapped_xs: Optional snapped node xs (EPSG:3857).
        snapped_ys: Optional snapped node ys (EPSG:3857).

    Returns:
        Matplotlib Figure.
    """
    transformer = Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True)

    xs: list[float] = []
    ys: list[float] = []
    for lat, lon in coords:
        x, y = transformer.transform(lon, lat)
        xs.append(float(x))
        ys.append(float(y))

    fig, ax = plt.subplots(figsize=(8, 13))
    ax.set_title(title)

    ax.plot(xs, ys, '-o', markersize=6, linewidth=2, color=color)

    if road_xs and road_ys:
        ax.plot(road_xs, road_ys, linewidth=3)

    if snapped_xs and snapped_ys and len(snapped_xs) == len(xs):
        ax.scatter(snapped_xs, snapped_ys, marker='x', s=60)

    for i, (x, y) in enumerate(zip(xs, ys), start=1):
        ax.text(x, y, str(i), fontsize=10, color='black')

    cx.add_basemap(ax, crs='EPSG:3857', source=cx.providers.OpenStreetMap.Mapnik, zoom=14)

    all_xs = list(xs)
    all_ys = list(ys)
    if road_xs and road_ys:
        all_xs.extend(float(v) for v in road_xs)
        all_ys.extend(float(v) for v in road_ys)
    if snapped_xs and snapped_ys:
        all_xs.extend(float(v) for v in snapped_xs)
        all_ys.extend(float(v) for v in snapped_ys)

    x_min = min(all_xs)
    x_max = max(all_xs)
    y_min = min(all_ys)
    y_max = max(all_ys)

    dx = x_max - x_min
    dy = y_max - y_min

    cx0 = 0.5 * (x_min + x_max)
    cy0 = 0.5 * (y_min + y_max)

    half_range = 0.5 * max(dx, dy)
    half_range = max(half_range, 300.0)
    pad = max(0.10 * (2.0 * half_range), 50.0)

    ax.set_xlim(cx0 - half_range - pad, cx0 + half_range + pad)
    ax.set_ylim(cy0 - half_range - pad, cy0 + half_range + pad)

    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    return fig
