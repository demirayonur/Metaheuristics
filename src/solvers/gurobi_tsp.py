"""
Exact solver for TSP using Gurobi (Miller–Tucker–Zemlin formulation).
"""

from __future__ import annotations

import gurobipy as gp
from gurobipy import GRB

from ..problems.tsp import TSPData


def solve_tsp_gurobi(data: TSPData) -> tuple[float, list[int]]:
    """
    Solve the TSP to optimality using the MTZ formulation.

    Parameters
    ----------
    data : TSPData
        Problem instance.

    Returns
    -------
    optimal_distance : float
    tour : list[int]
        Ordered list of city indices.
    """
    n = data.n_cities
    d = data.dist_matrix

    m = gp.Model("TSP")
    m.setParam("OutputFlag", 0)
    m.setParam("TimeLimit", 120)

    x = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                x[i, j] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

    u = m.addVars(n, lb=0, ub=n - 1, vtype=GRB.CONTINUOUS, name="u")

    m.setObjective(
        gp.quicksum(d[i, j] * x[i, j] for i, j in x), GRB.MINIMIZE
    )

    for i in range(n):
        m.addConstr(gp.quicksum(x[i, j] for j in range(n) if j != i) == 1)
        m.addConstr(gp.quicksum(x[j, i] for j in range(n) if j != i) == 1)

    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                m.addConstr(u[i] - u[j] + n * x[i, j] <= n - 1)

    m.optimize()

    if m.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        raise RuntimeError(f"Gurobi failed with status {m.Status}")

    tour = [0]
    current = 0
    for _ in range(n - 1):
        for j in range(n):
            if j != current and x[current, j].X > 0.5:
                tour.append(j)
                current = j
                break
    return m.ObjVal, tour
