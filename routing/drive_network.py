# routing/drive_network.py

import logging
import math
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
from shapely.geometry import Point


def load_drive_graph(
    *,
    data_dir: Path,
    drive_prefix: str,
) -> tuple[nx.Graph, gpd.GeoDataFrame]:
    """
    Load the drive network from Parquet and build a NetworkX graph.

    Expects:
        {data_dir}/{drive_prefix}_nodes.parquet
        {data_dir}/{drive_prefix}_edges.parquet

    Args:
        data_dir: Directory containing the parquet files.
        drive_prefix: Prefix used for nodes/edges parquet files.

    Returns:
        (graph, nodes) where:
            graph is an undirected graph with edge weight 'length' in meters,
            nodes is a GeoDataFrame indexed by node_id (OSM node ids).
    """
    nodes_path = data_dir / f'{drive_prefix}_nodes.parquet'
    edges_path = data_dir / f'{drive_prefix}_edges.parquet'

    if not nodes_path.exists() or not edges_path.exists():
        raise FileNotFoundError(
            f'Expected {nodes_path} and {edges_path} to exist. '
            'Run the OSMnx drive export script first.'
        )

    nodes = gpd.read_parquet(nodes_path)
    edges = gpd.read_parquet(edges_path)

    if 'node_id' in nodes.columns:
        nodes = nodes.set_index('node_id').copy()
    else:
        nodes = nodes.copy()

    required_edge_cols = {'u', 'v', 'length'}
    missing_cols = required_edge_cols - set(edges.columns)
    if missing_cols:
        raise ValueError(f'drive_edges.parquet is missing required columns: {missing_cols}')

    used_nodes = set(edges['u']).union(set(edges['v']))
    nodes_used = nodes.loc[nodes.index.intersection(used_nodes)].copy()

    mask_valid = edges['u'].isin(nodes_used.index) & edges['v'].isin(nodes_used.index)
    edges_used = edges.loc[mask_valid].copy()

    graph = nx.Graph()

    for node_id, row in nodes_used.iterrows():
        attrs: dict[str, object] = {}
        if 'x' in row:
            attrs['x'] = float(row['x'])
        if 'y' in row:
            attrs['y'] = float(row['y'])
        if 'geometry' in row:
            attrs['geometry'] = row['geometry']
        graph.add_node(int(node_id), **attrs)

    for _, row in edges_used.iterrows():
        u = int(row['u'])
        v = int(row['v'])
        if u == v:
            continue
        attrs: dict[str, object] = {'length': float(row['length'])}
        if 'geometry' in row:
            attrs['geometry'] = row['geometry']
        graph.add_edge(u, v, **attrs)

    logging.info(
        'Loaded drive graph with %d nodes and %d edges',
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )
    return graph, nodes_used


def snap_coords_to_nodes(
    coords: list[tuple[float, float]],
    nodes: gpd.GeoDataFrame,
) -> tuple[list[int], list[float]]:
    """
    Snap geographic coordinates to nearest drive network nodes.

    Args:
        coords: List of (lat, lon) in WGS84.
        nodes: Nodes GeoDataFrame with projected CRS (meters) and a geometry column.

    Returns:
        (snapped_node_ids, distances_m)
    """
    if nodes.crs is None:
        raise RuntimeError('Nodes GeoDataFrame has no CRS.')

    if 'geometry' not in nodes.columns:
        raise RuntimeError('Nodes GeoDataFrame has no geometry column.')

    if not coords:
        return [], []

    points_wgs84 = gpd.GeoSeries(
        [Point(lon, lat) for lat, lon in coords],
        crs='EPSG:4326',
    )
    points_proj = points_wgs84.to_crs(nodes.crs)

    node_geom = nodes['geometry']

    # Shapely 2.x STRtree: nearest() returns the INDEX of the nearest geometry.
    try:
        from shapely.strtree import STRtree
    except Exception as exc:
        raise RuntimeError('Shapely STRtree is required for fast snapping.') from exc

    geoms = list(node_geom.values)
    if not geoms:
        raise RuntimeError('Nodes GeoDataFrame has empty geometry.')

    tree = STRtree(geoms)

    node_ids = list(nodes.index)
    snapped_ids: list[int] = []
    snapped_distances_m: list[float] = []

    for pt in points_proj:
        nearest_idx = tree.nearest(pt)
        if nearest_idx is None:
            raise RuntimeError('Failed to find nearest node for a point.')

        node_id = int(node_ids[int(nearest_idx)])
        dist_m = float(node_geom.loc[node_id].distance(pt))

        snapped_ids.append(node_id)
        snapped_distances_m.append(dist_m)

    return snapped_ids, snapped_distances_m

def build_distance_matrix_networkx(
    snapped_node_ids: list[int],
    graph: nx.Graph,
) -> list[list[float]]:
    """
    Pairwise distance matrix (km) using NetworkX shortest paths.

    Args:
        snapped_node_ids: Node ids (one per address).
        graph: NetworkX graph with 'length' in meters.

    Returns:
        Square matrix distances in km, math.inf if unreachable.
    """
    n = len(snapped_node_ids)
    if n < 2:
        raise ValueError('At least two coordinates are required.')

    missing = [nid for nid in snapped_node_ids if nid not in graph.nodes]
    if missing:
        raise RuntimeError('Some snapped node ids are not present in the drive graph.')

    matrix: list[list[float]] = [[math.inf] * n for _ in range(n)]

    for i, source in enumerate(snapped_node_ids):
        lengths: dict[int, float] = nx.single_source_dijkstra_path_length(graph, source, weight='length')
        for j, target in enumerate(snapped_node_ids):
            dist_m = lengths.get(target)
            if dist_m is None:
                continue
            matrix[i][j] = float(dist_m) / 1000.0

    return matrix


def assert_all_pairs_reachable(distance: list[list[float]]) -> None:
    """
    Raise if any off-diagonal distance is not finite.

    Args:
        distance: Square matrix.

    Raises:
        RuntimeError: If unreachable pairs exist.
    """
    c = np.array(distance, dtype=float)
    if c.shape[0] != c.shape[1]:
        raise ValueError('Distance matrix must be square.')

    offdiag = ~np.eye(c.shape[0], dtype=bool)
    if not np.all(np.isfinite(c[offdiag])):
        raise RuntimeError(
            'Sommige locaties zijn onderling niet bereikbaar in het wegennet. '
            'Bekijk de afstandsmatrix en pas de adressen of de wegennetdata aan.'
        )
