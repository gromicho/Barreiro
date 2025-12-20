# routing/tsp_solver.py

import gurobipy as gp
import numpy as np
from gurobipy import GRB


def solve_tsp_or_path_gurobi(
    distance: list[list[float]],
    *,
    closed: bool,
    start_idx: int,
    end_idx: int | None,
    trace: bool = False,
) -> list[int]:
    """
    Solve an exact route with Gurobi using a symmetric TSP model.

    If closed=True:
        Return a cycle starting at start_idx (without repeating start_idx).

    If closed=False:
        Force the undirected edge (start_idx, end_idx) in the cycle, then remove it,
        yielding an optimal Hamiltonian path from start_idx to end_idx.

    Args:
        distance: Square matrix (km), symmetric, finite off-diagonal.
        closed: Closed tour or open path.
        start_idx: Start index.
        end_idx: End index (required if closed=False).
        trace: Gurobi output flag.

    Returns:
        Visiting order as indices into the distance matrix.
    """
    c = np.array(distance, dtype=float)
    if c.shape[0] != c.shape[1]:
        raise ValueError('Distance matrix must be square.')

    n = int(c.shape[0])

    # Trivial cases: the undirected degree-2 cycle model is infeasible for n <= 2.
    if n <= 0:
        raise ValueError('Distance matrix is empty.')

    start = int(start_idx)
    if start < 0 or start >= n:
        raise ValueError('start_idx is out of range.')

    if n == 1:
        return [start]

    if n == 2:
        other = 1 - start
        if closed:
            return [start, other]

        if end_idx is None:
            raise ValueError('end_idx must be provided when closed is False.')

        end = int(end_idx)
        if end < 0 or end >= n:
            raise ValueError('end_idx is out of range.')

        if start == end:
            return [start, other]

        return [start, end]

    nodes = range(n)


    if (not closed) and end_idx is None:
        raise ValueError('end_idx must be provided when closed is False.')

    model = gp.Model('symmetric_tsp')
    model.Params.OutputFlag = 1 if trace else 0
    model.Params.LazyConstraints = 1

    edges = [(i, j) for i in nodes for j in nodes if i < j]
    x = model.addVars(edges, vtype=GRB.BINARY, name='x')

    model.setObjective(gp.quicksum(c[i, j] * x[i, j] for i, j in edges), GRB.MINIMIZE)

    for i in nodes:
        model.addConstr(
            gp.quicksum(x[min(i, j), max(i, j)] for j in nodes if j != i) == 2,
            name=f'deg_{i}',
        )

    if (not closed) and end_idx is not None:
        i_forced = min(start_idx, end_idx)
        j_forced = max(start_idx, end_idx)
        model.addConstr(x[i_forced, j_forced] == 1, name='force_start_end_edge')

    def find_smallest_component(selected_edges: list[tuple[int, int]]) -> list[int]:
        unvisited = list(nodes)
        best_comp = list(nodes)

        while unvisited:
            current = unvisited[0]
            stack = [current]
            comp: list[int] = []

            while stack:
                node = stack.pop()
                if node not in unvisited:
                    continue
                unvisited.remove(node)
                comp.append(node)

                neighbors = (
                    [j for i2, j in selected_edges if i2 == node and j in unvisited]
                    + [i2 for i2, j in selected_edges if j == node and i2 in unvisited]
                )
                stack.extend(neighbors)

            if 0 < len(comp) < len(best_comp):
                best_comp = comp

        return best_comp

    def subtour_callback(model_cb: gp.Model, where: int) -> None:
        if where != GRB.Callback.MIPSOL:
            return

        vals = model_cb.cbGetSolution(model_cb._x)
        selected = [(i, j) for i, j in model_cb._x.keys() if vals[i, j] > 0.5]
        comp = find_smallest_component(selected)
        if len(comp) >= n:
            return

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
        tour = [start_idx]
        current = start_idx
        prev = -1
        while True:
            candidates = [v for v in adjacency[current] if v != prev]
            if not candidates:
                break
            nxt = candidates[0]
            if nxt == start_idx:
                break
            tour.append(nxt)
            prev = current
            current = nxt

        if len(tour) != n:
            raise RuntimeError(f'Closed tour reconstruction failed, expected {n}, got {len(tour)}.')
        return tour

    if end_idx is None:
        raise RuntimeError('end_idx is required for open path reconstruction.')

    if end_idx not in adjacency[start_idx] or start_idx not in adjacency[end_idx]:
        raise RuntimeError('Forced edge (start_idx, end_idx) is not present in the TSP solution.')

    adjacency_path = {i: list(neighs) for i, neighs in adjacency.items()}
    adjacency_path[start_idx].remove(end_idx)
    adjacency_path[end_idx].remove(start_idx)

    path = [start_idx]
    current = start_idx
    prev = -1
    while True:
        candidates = [v for v in adjacency_path[current] if v != prev]
        if not candidates:
            break
        nxt = candidates[0]
        path.append(nxt)
        prev = current
        current = nxt

    if current != end_idx or len(path) != n:
        raise RuntimeError(f'Path reconstruction failed, expected {n}, got {len(path)}.')
    return path


def route_length(route: list[int], distance: list[list[float]], *, closed: bool) -> float:
    """
    Compute total length of a route.

    Args:
        route: Indices.
        distance: Matrix.
        closed: Include last->first.

    Returns:
        Total length in same units as matrix (km).
    """
    if len(route) < 2:
        return 0.0

    total = 0.0
    for i in range(len(route) - 1):
        total += distance[route[i]][route[i + 1]]

    if closed:
        total += distance[route[-1]][route[0]]

    return total
