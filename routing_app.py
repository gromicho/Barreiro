# app.py
# Barreiro route-optimalisatie (alleen wegennet, Google-geocodering)
# Ondersteunt:
# - Gesloten rondrit: start en eindig op het eerste adres (bijvoorbeeld ziekenhuis)
# - Open traject: start op het eerste adres, eindig op het laatste adres (bijvoorbeeld station)
# Biedt:
# - Eenvoudige en volledige interface-modi
# - Links voor Google Maps

# https://console.cloud.google.com/google/maps-apis/metrics?project=streamlit-logger-475213

import os
import logging
import math
import time
import urllib.parse
from contextlib import contextmanager
from pathlib import Path

import requests
import contextily as cx
import geopandas as gpd
import gurobipy as gp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
from gurobipy import GRB
from pyproj import Transformer
from shapely.geometry import Point


DATA_DIR: Path = Path('data')
DRIVE_PREFIX: str = 'drive'
LOGFILE: str = 'routing_time_log.txt'

# Max toegestane afstand tussen geocodeerde locatie en netwerk-node (in meter)
MAX_SNAP_DISTANCE_M: float = 5000.0


# -----------------------
# Logging and timing
# -----------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOGFILE, mode='a', encoding='utf-8'),
        logging.StreamHandler(),
    ],
)

logging.info('routing_app module imported')


@contextmanager
def timeblock(label: str, log_list: list[str]) -> None:
    """
    Measure and log the execution time of a code block.

    Args:
        label: Text label describing the block.
        log_list: List that collects human-readable timing lines.
    """
    start: float = time.perf_counter()
    logging.info('START: %s', label)
    try:
        yield
    finally:
        end: float = time.perf_counter()
        delta: float = end - start
        msg: str = f'{label}: {delta:.3f} seconds'
        logging.info('END:   %s', msg)
        log_list.append(msg)


# -----------------------
# Google geocoding (forward and reverse)
# -----------------------

def get_google_maps_api_key() -> str:
    """
    Retrieve the Google Maps API key from Streamlit secrets or
    the environment.

    Returns:
        Google Maps API key string.

    Raises:
        RuntimeError: If no key can be found.
    """
    api_key: str = ''

    try:
        api_key = st.secrets.get('GOOGLE_MAPS_API_KEY', '')
    except Exception:
        api_key = ''

    if not api_key:
        api_key = os.environ.get('GOOGLE_MAPS_API_KEY', '')

    if not api_key:
        raise RuntimeError(
            'Google Maps API key not found. Set GOOGLE_MAPS_API_KEY in '
            '.streamlit/secrets.toml or as an environment variable.',
        )

    return api_key


def geocode_address_google(
    address: str,
    bbox: tuple[float, float, float, float] | None = None,
) -> tuple[float, float]:
    """
    Geocode an address into latitude and longitude using the Google
    Geocoding API.

    Args:
        address:
            Free text address string.
        bbox:
            Optional (min_lon, min_lat, max_lon, max_lat) bounding box
            in WGS84 that is passed as a bounds parameter to bias the
            geocoder towards the network region.

    Returns:
        Tuple (lat, lon) in WGS84.

    Raises:
        RuntimeError: If the API response status is not OK or if no
            results are returned.
    """
    api_key: str = get_google_maps_api_key()
    cleaned_address: str = address.strip()

    url: str = 'https://maps.googleapis.com/maps/api/geocode/json'
    params: dict[str, str] = {
        'address': cleaned_address,
        'key': api_key,
    }

    if bbox is not None:
        min_lon, min_lat, max_lon, max_lat = bbox
        sw: str = f'{min_lat},{min_lon}'
        ne: str = f'{max_lat},{max_lon}'
        params['bounds'] = f'{sw}|{ne}'

    try:
        response = requests.get(url, params=params, timeout=10)
    except Exception as exc:
        raise RuntimeError(
            f'Google geocoding request failed for "{cleaned_address}": {exc}',
        ) from exc

    response.raise_for_status()
    data: dict = response.json()

    status: str | None = data.get('status')
    if status != 'OK':
        error_message: str = data.get('error_message', '')
        raise RuntimeError(
            f'Google geocoding failed for "{cleaned_address}" '
            f'with status {status}, error_message: {error_message}',
        )

    results: list[dict] = data.get('results', [])
    if not results:
        raise RuntimeError(
            f'Google geocoding returned no results for "{cleaned_address}".',
        )

    location: dict = results[0]['geometry']['location']
    lat: float = float(location['lat'])
    lon: float = float(location['lng'])

    formatted_address: str = results[0].get('formatted_address', '')
    logging.info(
        'Google geocode "%s" -> lat=%.6f, lon=%.6f, formatted="%s"',
        cleaned_address,
        lat,
        lon,
        formatted_address,
    )

    return lat, lon


def geocode_addresses(
    addresses: list[str],
    bbox: tuple[float, float, float, float],
) -> list[tuple[float, float]]:
    """
    Geocode a list of addresses with Google within a given bounding box.

    Args:
        addresses:
            List of address strings.
        bbox:
            (min_lon, min_lat, max_lon, max_lat) bounding box of the
            drive network in WGS84, used to bias the Google geocoding.

    Returns:
        List of (latitude, longitude) tuples in the same order as the
        input addresses.

    Raises:
        RuntimeError: If any address cannot be geocoded.
    """
    coords: list[tuple[float, float]] = []
    for addr in addresses:
        lat, lon = geocode_address_google(addr, bbox)
        coords.append((lat, lon))
    return coords

# -----------------------
# Drive network and distances
# -----------------------

@st.cache_resource(show_spinner=False)
def load_drive_graph() -> tuple[nx.Graph, gpd.GeoDataFrame]:
    """
    Load the drive network from Parquet and build a NetworkX graph.

    Expects:
        data/drive_nodes.parquet
        data/drive_edges.parquet

    Returns:
        (graph, nodes) where:
            graph is an undirected NetworkX Graph with edge weight "length"
            in meters,
            nodes is a GeoDataFrame indexed by node_id (OSM node ids).
    """
    nodes_path: Path = DATA_DIR / f'{DRIVE_PREFIX}_nodes.parquet'
    edges_path: Path = DATA_DIR / f'{DRIVE_PREFIX}_edges.parquet'

    if not nodes_path.exists() or not edges_path.exists():
        raise FileNotFoundError(
            f'Expected {nodes_path} and {edges_path} to exist. '
            'Run the OSMnx drive export script first.',
        )

    nodes: gpd.GeoDataFrame = gpd.read_parquet(nodes_path)
    edges: gpd.GeoDataFrame = gpd.read_parquet(edges_path)

    if 'node_id' in nodes.columns:
        nodes = nodes.set_index('node_id').copy()
    else:
        nodes = nodes.copy()

    required_edge_cols: set[str] = {'u', 'v', 'length'}
    missing_cols: set[str] = required_edge_cols - set(edges.columns)
    if missing_cols:
        raise ValueError(
            f'drive_edges.parquet is missing required columns: {missing_cols}',
        )

    used_nodes: set[int] = set(edges['u']).union(set(edges['v']))
    nodes_used = nodes.loc[nodes.index.intersection(used_nodes)].copy()

    mask_valid = edges['u'].isin(nodes_used.index) & edges['v'].isin(nodes_used.index)
    edges_used = edges.loc[mask_valid].copy()

    graph = nx.Graph()

    for node_id, row in nodes_used.iterrows():
        attrs: dict = {}
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
        length = float(row['length'])
        attrs = {'length': length}
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
    Snap geographic coordinates to the nearest drive network nodes.

    Args:
        coords: List of (latitude, longitude) tuples in WGS84.
        nodes: Nodes GeoDataFrame with projected CRS and geometry,
            indexed by node_id. The CRS is assumed to use meters
            as the distance unit.

    Returns:
        (snapped_node_ids, distances_m) where:
            snapped_node_ids is a list of node ids corresponding to the
                nearest nodes,
            distances_m is a list of Euclidean distances in meters
                between each geocoded point and its snapped node.

    Raises:
        RuntimeError: If the nodes GeoDataFrame has no CRS.
    """
    if nodes.crs is None:
        raise RuntimeError('Nodes GeoDataFrame has no CRS.')

    points_wgs84: gpd.GeoSeries = gpd.GeoSeries(
        [Point(lon, lat) for lat, lon in coords],
        crs='EPSG:4326',
    )
    points_proj: gpd.GeoSeries = points_wgs84.to_crs(nodes.crs)

    snapped_ids: list[int] = []
    snapped_distances_m: list[float] = []

    node_geom: gpd.GeoSeries = nodes['geometry']

    for point in points_proj:
        distances = node_geom.distance(point)
        idx = distances.idxmin()
        snapped_ids.append(int(idx))
        snapped_distances_m.append(float(distances.loc[idx]))

    logging.info('Snapped node ids: %s', snapped_ids)
    logging.info('Snapping distances (m): %s', snapped_distances_m)

    return snapped_ids, snapped_distances_m


def build_distance_matrix_networkx(
    snapped_node_ids: list[int],
    graph: nx.Graph,
) -> list[list[float]]:
    """
    Build a pairwise distance matrix in kilometers using NetworkX shortest paths.

    Args:
        snapped_node_ids: List of node ids, one for each coordinate.
        graph: NetworkX graph with edge weight "length" in meters.

    Returns:
        Square matrix d[i][j] of distances in kilometers. Unreachable pairs
        get math.inf.

    Raises:
        RuntimeError: If a snapped node id does not exist in the graph.
        ValueError: If fewer than two coordinates are supplied.
    """
    n: int = len(snapped_node_ids)
    if n < 2:
        raise ValueError('At least two coordinates are required.')

    missing = [nid for nid in snapped_node_ids if nid not in graph.nodes]
    if missing:
        logging.error('Snapped node ids not in graph: %s', missing)
        raise RuntimeError(
            'Some snapped node ids are not present in the drive graph. '
            'Check drive_nodes.parquet and drive_edges.parquet consistency.',
        )

    matrix: list[list[float]] = [[math.inf] * n for _ in range(n)]

    for i, source in enumerate(snapped_node_ids):
        lengths: dict[int, float] = nx.single_source_dijkstra_path_length(
            graph,
            source,
            weight='length',
        )
        for j, target in enumerate(snapped_node_ids):
            dist_m = lengths.get(target)
            if dist_m is None:
                continue
            matrix[i][j] = float(dist_m) / 1000.0

    return matrix


def assert_all_pairs_reachable(distance: list[list[float]]) -> None:
    """
    Verify that all pairs of distinct nodes have finite distances.

    Args:
        distance: Square distance matrix.

    Raises:
        RuntimeError: If some off-diagonal entry is infinite or NaN.
        ValueError: If the matrix is not square.
    """
    C = np.array(distance, dtype=float)
    if C.shape[0] != C.shape[1]:
        raise ValueError('Distance matrix must be square.')

    n = C.shape[0]
    offdiag = ~np.eye(n, dtype=bool)

    if not np.all(np.isfinite(C[offdiag])):
        raise RuntimeError(
            'Sommige locaties zijn onderling niet bereikbaar in het wegennet. '
            'Bekijk de afstandsmatrix hierboven en pas de adressen of de '
            'wegennetdata aan.',
        )


# -----------------------
# Exact route via symmetric TSP (Gurobi)
# -----------------------

def solve_tsp_or_path_gurobi(
    distance: list[list[float]],
    closed: bool,
    start_idx: int,
    end_idx: int | None = None,
    trace: bool = False,
) -> list[int]:
    """
    Solve an exact route with Gurobi using a symmetric TSP model.

    If closed is True:
        Solve the symmetric TSP on all nodes and return a cycle starting
        at start_idx (without repeating start_idx).

    If closed is False:
        Solve the symmetric TSP with an additional constraint that the
        undirected edge (start_idx, end_idx) is used. Remove that edge
        from the cycle to obtain an optimal Hamiltonian path from start_idx
        to end_idx.

    Args:
        distance: Square matrix of pairwise distances d[i][j], finite and symmetric.
        closed: If True, solve a closed tour. If False, solve an open path.
        start_idx: Index of the starting node in the distance matrix.
        end_idx: Index of the ending node in the distance matrix, required for
            closed is False.
        trace: Enable Gurobi solver output if True.

    Returns:
        Route as a list of node indices.

    Raises:
        RuntimeError: If the solver does not reach OPTIMAL or TIME_LIMIT, or
            if route reconstruction fails.
        ValueError: If the distance matrix is not square, or end_idx is missing
            in the open case.
    """
    C = np.array(distance, dtype=float)
    if C.shape[0] != C.shape[1]:
        raise ValueError('Distance matrix must be square.')

    n: int = C.shape[0]
    nodes = range(n)

    if not closed and end_idx is None:
        raise ValueError('end_idx must be provided when closed is False.')

    model = gp.Model('symmetric_tsp')
    model.Params.OutputFlag = 1 if trace else 0
    model.Params.LazyConstraints = 1

    edges = [(i, j) for i in nodes for j in nodes if i < j]
    x = model.addVars(edges, vtype=GRB.BINARY, name='x')

    model.setObjective(
        gp.quicksum(C[i, j] * x[i, j] for i, j in edges),
        GRB.MINIMIZE,
    )

    for i in nodes:
        model.addConstr(
            gp.quicksum(x[min(i, j), max(i, j)] for j in nodes if j != i) == 2,
            name=f'deg_{i}',
        )

    if not closed and end_idx is not None:
        i_forced = min(start_idx, end_idx)
        j_forced = max(start_idx, end_idx)
        model.addConstr(
            x[i_forced, j_forced] == 1,
            name='force_start_end_edge',
        )

    def find_smallest_component(
        selected_edges: list[tuple[int, int]],
    ) -> list[int]:
        """
        Find the smallest connected component induced by selected edges.

        Args:
            selected_edges: List of undirected edges (i, j) selected in the
                current solution.

        Returns:
            List of node indices for the smallest connected component.
        """
        unvisited = list(nodes)
        best_comp = list(nodes)

        while unvisited:
            current = unvisited[0]
            this_comp: list[int] = []
            stack: list[int] = [current]
            while stack:
                node = stack.pop()
                if node not in unvisited:
                    continue
                unvisited.remove(node)
                this_comp.append(node)
                neighbors = [
                    j for i2, j in selected_edges if i2 == node and j in unvisited
                ] + [
                    i2 for i2, j in selected_edges if j == node and i2 in unvisited
                ]
                stack.extend(neighbors)
            if 0 < len(this_comp) < len(best_comp):
                best_comp = this_comp

        return best_comp

    def subtour_callback(model_cb: gp.Model, where: int) -> None:
        """
        Gurobi lazy constraint callback that eliminates extra components
        (subtours or disconnected parts).

        Args:
            model_cb: The Gurobi model object within the callback context.
            where: Callback location code.
        """
        if where == GRB.Callback.MIPSOL:
            vals = model_cb.cbGetSolution(model_cb._x)
            selected = [
                (i, j)
                for i, j in model_cb._x.keys()
                if vals[i, j] > 0.5
            ]
            comp = find_smallest_component(selected)
            if len(comp) < n:
                expr = 0.0
                for i_idx in range(len(comp)):
                    for j_idx in range(i_idx + 1, len(comp)):
                        a = comp[i_idx]
                        b = comp[j_idx]
                        expr += model_cb._x[min(a, b), max(a, b)]
                model_cb.cbLazy(expr <= len(comp) - 1)

    model._x = x
    model.optimize(subtour_callback)

    if model.status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        raise RuntimeError(f'Gurobi TSP solver ended with status {model.status}')

    vals = model.getAttr('x', x)
    selected_edges = [(i, j) for i, j in x.keys() if vals[i, j] > 0.5]

    adjacency: dict[int, list[int]] = {i: [] for i in nodes}
    for i, j in selected_edges:
        adjacency[i].append(j)
        adjacency[j].append(i)

    if closed:
        tour: list[int] = [start_idx]
        current: int = start_idx
        prev: int = -1
        while True:
            neighbors = adjacency[current]
            candidates = [v for v in neighbors if v != prev]
            if not candidates:
                break
            nxt = candidates[0]
            if nxt == start_idx:
                break
            tour.append(nxt)
            prev = current
            current = nxt

        if len(tour) != n:
            raise RuntimeError(
                f'Closed tour reconstruction failed, expected {n} nodes, '
                f'got {len(tour)}.',
            )

        return tour

    if end_idx is None:
        raise RuntimeError('end_idx is required for open path reconstruction.')

    if end_idx not in adjacency[start_idx] or start_idx not in adjacency[end_idx]:
        raise RuntimeError(
            'Forced edge (start_idx, end_idx) is not present in the TSP solution.',
        )

    adjacency_path: dict[int, list[int]] = {
        i: list(neighs) for i, neighs in adjacency.items()
    }
    adjacency_path[start_idx].remove(end_idx)
    adjacency_path[end_idx].remove(start_idx)

    path: list[int] = [start_idx]
    current = start_idx
    prev = -1

    while True:
        neighbors = adjacency_path[current]
        candidates = [v for v in neighbors if v != prev]
        if not candidates:
            break
        nxt = candidates[0]
        path.append(nxt)
        prev = current
        current = nxt

    if current != end_idx or len(path) != n:
        raise RuntimeError(
            f'Hamiltonian path reconstruction failed, expected {n} nodes, '
            f'got {len(path)} (end node {current}, expected {end_idx}).',
        )

    return path


def route_length(
    route: list[int],
    distance: list[list[float]],
    closed: bool,
) -> float:
    """
    Compute the total length of a route.

    Args:
        route: Route as a list of node indices.
        distance: Distance matrix d[i][j].
        closed: If True, include the leg from last back to first
            (closed tour). If False, treat the route as an open path.

    Returns:
        Total route length in the same units as the distance matrix.
    """
    total: float = 0.0
    if len(route) < 2:
        return 0.0

    for i in range(len(route) - 1):
        total += distance[route[i]][route[i + 1]]

    if closed:
        total += distance[route[-1]][route[0]]

    return total


# -----------------------
# Matplotlib route maps
# -----------------------

def make_matplotlib_route_map(
    coords: list[tuple[float, float]],
    title: str,
    color: str,
) -> plt.Figure:
    """
    Build a Matplotlib figure showing a route on top of web map tiles.

    Args:
        coords: List of (lat, lon) in visiting order.
        title: Plot title.
        color: Line and marker color.

    Returns:
        Matplotlib Figure instance.
    """
    transformer = Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True)

    xs: list[float] = []
    ys: list[float] = []
    for lat, lon in coords:
        x, y = transformer.transform(lon, lat)
        xs.append(x)
        ys.append(y)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(title)

    ax.plot(xs, ys, '-o', markersize=6, linewidth=2, color=color)

    for i, (x, y) in enumerate(zip(xs, ys), start=1):
        ax.text(x, y, str(i), fontsize=10, color='black')

    cx.add_basemap(
        ax,
        crs='EPSG:3857',
        source=cx.providers.OpenStreetMap.Mapnik,
        zoom=14,
    )

    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)

    dx = x_max - x_min
    dy = y_max - y_min

    pad_x = max(dx * 0.1, 50.0)
    pad_y = max(dy * 0.1, 50.0)

    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_ylim(y_min - pad_y, y_max + pad_y)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

    return fig


# -----------------------
# Navigation URLs
# -----------------------

def build_google_maps_url_from_addresses(addresses: list[str]) -> str:
    """
    Build a Google Maps directions URL for a sequence of addresses.

    Args:
        addresses: Ordered list of addresses.

    Returns:
        Google Maps URL for multi stop directions.

    Raises:
        ValueError: If fewer than two addresses are supplied.
    """
    if len(addresses) < 2:
        raise ValueError('At least origin and destination are required.')

    origin: str = urllib.parse.quote(addresses[0])
    destination: str = urllib.parse.quote(addresses[-1])
    intermediates: list[str] = addresses[1:-1]

    base: str = 'https://www.google.com/maps/dir/?api=1'
    url: str = f'{base}&origin={origin}&destination={destination}'

    if intermediates:
        waypoints: str = '|'.join(urllib.parse.quote(a) for a in intermediates)
        url += f'&waypoints={waypoints}'

    return url


# -----------------------
# Streamlit UI
# -----------------------

def main() -> None:
    """
    Run the Streamlit app for drive-only route optimization in Barreiro.

    Simple mode:
        - Uses Google geocoder.
        - Computes the full solution but only shows the "Open in Google Maps"
          button (plus any error messages).

    Full mode:
        - Shows explanatory text.
        - Shows geocoding and snapping diagnostics.
        - Shows distance matrix, route information, maps, and a timing log.
    """
    logging.info('main() called')
    st.title('Bezoekroute-optimalisatie in de regio Barreiro')

    ui_mode_label: str = st.radio(
        'Interfacemodus',
        ['Eenvoudig', 'Volledig'],
        index=0,
        horizontal=True,
    )
    simple_mode: bool = ui_mode_label == 'Eenvoudig'

    if not simple_mode:
        st.markdown(
            'Voer een adres per regel in binnen de regio Barreiro.\n\n'
            '- Het eerste adres wordt gezien als de startlocatie '
            '(bijvoorbeeld het ziekenhuis).\n'
            '- Bij een open traject wordt het laatste adres gezien als de '
            'eindlocatie (bijvoorbeeld jouw huis).\n\n'
            'De app zal:\n'
            '1. De adressen geocoderen\n'
            '2. Ze koppelen aan een eigen mini-wegennet\n'
            '3. Een afstandsmatrix (km) berekenen binnen het wegennet\n'
            '4. Een exacte route oplossen met Gurobi '
            '(gesloten rondrit of open traject)\n'
            '5. Een link genereren voor Google Maps.\n\n'
            'In de volledige modus toont de app ook de afstandsmatrix, '
            'kaarten en een timinglog.',
        )

    default_text: str = (
        'Hospital Nossa Senhora do Rosario, Barreiro, Portugal\n'
        'Forum Barreiro, Barreiro, Portugal\n'
        'Estacao Barreiro A, Barreiro, Portugal\n'
        'Pastelaria Prestigio, Barreiro, Portugal\n'
        'Parque Catarina Eufemia, Barreiro, Portugal\n'
        'Avenida Escola dos Fuzileiros Navais, Barreiro, Portugal'
    )

    addresses_input: str = st.text_area(
        'Adressen (een per regel):',
        value=default_text,
        height=200,
    )

    route_type_label: str = st.radio(
        'Routetype',
        [
            'Gesloten rondrit (start en einde bij het eerste adres)',
            'Open traject (start bij het eerste adres, einde bij het laatste adres)',
        ],
        index=0,
    )
    is_closed: bool = route_type_label.startswith('Gesloten')

    if st.button('Optimaliseer route'):
        logs: list[str] = []
        with timeblock('Total optimization run', logs):
            addresses: list[str] = [
                a.strip() for a in addresses_input.splitlines() if a.strip()
            ]

            if len(addresses) < 2:
                st.error('Geef minstens twee adressen op.')
                return

            if not is_closed and len(addresses) < 3:
                st.error(
                    'Voor een open traject zijn minstens drie adressen nodig '
                    '(start, minimaal een tussenadres, einde).',
                )
                return

            # Load drive network first, to obtain bounding box for geocoding
            try:
                with st.spinner('Wegennetwerk laden...'):
                    with timeblock('Loading drive graph', logs):
                        graph, nodes = load_drive_graph()
            except Exception as exc:
                st.error(f'Fout tijdens het laden van het netwerk: {exc}')
                return

            # Compute network bounding box in WGS84 for Google geocoding
            with timeblock('Computing network bounding box (WGS84)', logs):
                nodes_wgs84 = nodes.to_crs('EPSG:4326')
                min_lon, min_lat, max_lon, max_lat = nodes_wgs84.total_bounds
                network_bbox: tuple[float, float, float, float] = (
                    float(min_lon),
                    float(min_lat),
                    float(max_lon),
                    float(max_lat),
                )

                center_lon: float = 0.5 * (min_lon + max_lon)
                center_lat: float = 0.5 * (min_lat + max_lat)

                # Shrink bbox around the network center to focus on Barreiro
                # 0.07 degrees latitude ~ 7.8 km, 0.10 degrees longitude ~ 8–9 km here.
                half_width_lon: float = 0.10
                half_height_lat: float = 0.07

                geo_min_lon: float = max(min_lon, center_lon - half_width_lon)
                geo_max_lon: float = min(max_lon, center_lon + half_width_lon)
                geo_min_lat: float = max(min_lat, center_lat - half_height_lat)
                geo_max_lat: float = min(max_lat, center_lat + half_height_lat)

                geocode_bbox: tuple[float, float, float, float] = (
                    geo_min_lon,
                    geo_min_lat,
                    geo_max_lon,
                    geo_max_lat,
                )

                logging.info(
                    'Network bbox WGS84: min_lon=%.6f, min_lat=%.6f, '
                    'max_lon=%.6f, max_lat=%.6f',
                    min_lon,
                    min_lat,
                    max_lon,
                    max_lat,
                )
                logging.info(
                    'Geocode bbox WGS84: min_lon=%.6f, min_lat=%.6f, '
                    'max_lon=%.6f, max_lat=%.6f',
                    geo_min_lon,
                    geo_min_lat,
                    geo_max_lon,
                    geo_max_lat,
                )

            # Geocoding (Google, restricted to network area)
            try:
                with st.spinner('Adressen geocoderen binnen het netwerkgebied...'):
                    with timeblock('Geocoding addresses', logs):
                        coords: list[tuple[float, float]] = geocode_addresses(
                            addresses,
                            geocode_bbox,
                        )
            except Exception as exc:
                st.error(f'Fout tijdens het geocoderen: {exc}')
                return

            # Snap geocoded points to drive network nodes
            try:
                with st.spinner('Adressen koppelen aan het wegennet...'):
                    with timeblock('Snapping coordinates to drive nodes', logs):
                        snapped_node_ids, snapped_distances_m = snap_coords_to_nodes(
                            coords,
                            nodes,
                        )
            except Exception as exc:
                st.error(
                    'Fout tijdens het koppelen van adressen aan het wegennet: '
                    f'{exc}',
                )
                return

            # Diagnostics in full mode
            if not simple_mode:
                st.subheader('Geocoding en snapping (diagnostiek)')

                # Detect duplicate geocoded coordinates
                coord_groups: dict[tuple[float, float], list[int]] = {}
                for i, (lat, lon) in enumerate(coords):
                    key = (round(lat, 6), round(lon, 6))
                    coord_groups.setdefault(key, []).append(i)

                duplicates: list[list[int]] = [
                    idxs for idxs in coord_groups.values() if len(idxs) > 1
                ]

                if duplicates:
                    st.warning(
                        'Let op: meerdere adressen zijn naar exact dezelfde '
                        'coördinaat gegeocoderd. Dit kan duiden op onduidelijke '
                        'of niet-herkende adressen door de geocoder.'
                    )

                # Show diagnostics per address (no reverse geocoding)
                for addr, (lat, lon), node_id, dist_m in zip(
                    addresses,
                    coords,
                    snapped_node_ids,
                    snapped_distances_m,
                ):
                    gmaps_link: str = (
                        'https://www.google.com/maps/search/'
                        f'?api=1&query={lat},{lon}'
                    )
                    st.markdown(
                        f'**Invoeradres:** {addr}  \n'
                        f'↳ Geocode: lat `{lat:.6f}`, lon `{lon:.6f}`  \n'
                        f'↳ Dichtstbijzijnde netwerk-node: `{node_id}`  \n'
                        f'↳ Euclidische afstand naar node: `{dist_m:.1f}` meter  \n'
                        f'[Bekijk punt in Google Maps]({gmaps_link})'
                    )
            # Quality check on snapping distances (all modes)
            offending_indices: list[int] = [
                i
                for i, d in enumerate(snapped_distances_m)
                if d > MAX_SNAP_DISTANCE_M
            ]
            if offending_indices:
                st.error(
                    'Minstens een adres ligt te ver van het beschikbare '
                    'wegennet (meer dan '
                    f'{MAX_SNAP_DISTANCE_M / 1000.0:.1f} km). Pas de adressen '
                    'aan of controleer de geocoding.',
                )
                if not simple_mode:
                    st.write('Probleemadressen:')
                    for i in offending_indices:
                        st.write(
                            f'- {addresses[i]} (afstand tot netwerk: '
                            f'{snapped_distances_m[i] / 1000.0:.2f} km)',
                        )
                return

            # Distance matrix on the network
            try:
                with st.spinner('Afstandsmatrix in het wegennet berekenen...'):
                    with timeblock('Computing NetworkX distance matrix', logs):
                        dist_matrix_raw: list[list[float]] = (
                            build_distance_matrix_networkx(
                                snapped_node_ids,
                                graph,
                            )
                        )
            except Exception as exc:
                st.error(
                    'Fout tijdens het berekenen van de netwerkafstanden: '
                    f'{exc}',
                )
                return

            if not simple_mode:
                st.subheader('Afstandsmatrix in het wegennet (km)')
                df_dist = pd.DataFrame(
                    dist_matrix_raw,
                    columns=[
                        f'{i + 1}: {addr}' for i, addr in enumerate(addresses)
                    ],
                    index=[
                        f'{i + 1}: {addr}' for i, addr in enumerate(addresses)
                    ],
                )
                st.dataframe(
                    df_dist.style.format('{:.3f}'),
                    width='stretch',
                )

                C_raw = np.array(dist_matrix_raw, dtype=float)
                n_raw = C_raw.shape[0]
                offdiag_mask = ~np.eye(n_raw, dtype=bool)
                finite_vals = C_raw[offdiag_mask][np.isfinite(C_raw[offdiag_mask])]
                if finite_vals.size > 0:
                    min_dist = float(finite_vals.min())
                    max_dist = float(finite_vals.max())
                    st.caption(
                        'Netwerkafstanden tussen punten: '
                        f'minimaal {min_dist:.3f} km, maximaal {max_dist:.3f} km.',
                    )

            # Connectivity check
            try:
                with timeblock('Checking connectivity of snapped nodes', logs):
                    assert_all_pairs_reachable(dist_matrix_raw)
            except Exception as exc:
                st.error(f'Niet-bereikbare locaties gedetecteerd: {exc}')
                return

            # Prepare matrix for optimization
            with timeblock('Preparing distance matrix for optimization', logs):
                C = np.array(dist_matrix_raw, dtype=float)
                C = 0.5 * (C + C.T)
                np.fill_diagonal(C, 0.0)
                dist_matrix_opt: list[list[float]] = C.tolist()

            start_idx: int = 0
            end_idx: int | None
            if is_closed:
                end_idx = None
            else:
                end_idx = len(dist_matrix_opt) - 1

            # Route optimization via Gurobi
            try:
                label = (
                    'Route optimization (closed tour)'
                    if is_closed
                    else 'Route optimization (open path)'
                )
                with st.spinner('Route optimaal oplossen met Gurobi...'):
                    with timeblock(label, logs):
                        route_indices: list[int] = solve_tsp_or_path_gurobi(
                            dist_matrix_opt,
                            closed=is_closed,
                            start_idx=start_idx,
                            end_idx=end_idx,
                            trace=False,
                        )
            except Exception as exc:
                st.error(f'Fout tijdens de route-optimalisatie: {exc}')
                return

            ordered_addresses: list[str] = [addresses[i] for i in route_indices]

            if not simple_mode:
                st.subheader('Geoptimaliseerde bezoekvolgorde (wegennet)')
                if is_closed:
                    st.write(
                        'Gesloten rondrit: start en einde bij het eerste adres '
                        'in deze lijst.',
                    )
                else:
                    st.write(
                        'Open traject: start bij het eerste adres en eindig bij '
                        'het laatste adres in deze lijst.',
                    )

                for k, addr in enumerate(ordered_addresses, start=1):
                    st.write(f'{k}. {addr}')

                total_km: float = route_length(
                    route_indices,
                    dist_matrix_opt,
                    closed=is_closed,
                )
                if is_closed:
                    st.write(
                        'Geschatte totale lengte van de gesloten rondrit (km): '
                        f'{total_km:.2f}',
                    )
                else:
                    st.write(
                        'Geschatte totale lengte van het open traject (km): '
                        f'{total_km:.2f}',
                    )

            # Route maps in full mode
            if not simple_mode:
                with timeblock('Building route maps', logs):
                    orig_coords: list[tuple[float, float]] = coords[:]
                    opt_coords: list[tuple[float, float]] = [
                        coords[i] for i in route_indices
                    ]

                    if is_closed:
                        orig_coords_plot = orig_coords + orig_coords[:1]
                        opt_coords_plot = opt_coords + opt_coords[:1]
                    else:
                        orig_coords_plot = orig_coords
                        opt_coords_plot = opt_coords

                    fig_orig = make_matplotlib_route_map(
                        coords=orig_coords_plot,
                        title='Oorspronkelijke volgorde (wegennet)',
                        color='blue',
                    )
                    fig_opt = make_matplotlib_route_map(
                        coords=opt_coords_plot,
                        title='Geoptimaliseerde volgorde (wegennet)',
                        color='red',
                    )

                st.subheader('Route op kaart (oorspronkelijke volgorde)')
                st.pyplot(fig_orig)

                st.subheader('Route op kaart (geoptimaliseerde volgorde)')
                st.pyplot(fig_opt)

            # Build Google Maps URL
            maps_url: str = ''
            with timeblock('Building navigation URLs', logs):
                if is_closed:
                    maps_addresses: list[str] = (
                        ordered_addresses + [ordered_addresses[0]]
                    )
                else:
                    maps_addresses = ordered_addresses

                try:
                    maps_url = build_google_maps_url_from_addresses(maps_addresses)
                except Exception as exc:
                    st.error(
                        'Fout bij het opbouwen van de Google Maps URL: '
                        f'{exc}',
                    )
                    maps_url = ''

            if maps_url:
                if not simple_mode:
                    st.subheader('Open in navigatie-app')
                st.link_button('Open in Google Maps', maps_url)
                if not simple_mode:
                    st.code(maps_url, language='text')

        if not simple_mode:
            st.subheader('Timinglog voor deze run')
            with st.expander('Toon gedetailleerde timing'):
                for line in logs:
                    st.write(line)
            st.caption(
                'De volledige geschiedenis wordt ook weggeschreven naar '
                'routing_time_log.txt in de app-map.',
            )


if __name__ == '__main__':
    main()
