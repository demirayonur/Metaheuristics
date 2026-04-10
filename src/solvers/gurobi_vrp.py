"""
Exact solver for CVRP using Gurobi (two-index vehicle-flow + MTZ).
"""

from __future__ import annotations

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from ..problems.vrp import VRPData


def solve_vrp_gurobi(data: VRPData) -> tuple[float, list[list[int]]]:
    """
    Solve the Capacitated VRP to optimality.

    Parameters
    ----------
    data : VRPData
        Problem instance.

    Returns
    -------
    optimal_distance : float
    routes : list[list[int]]
        Each inner list is a sequence of customer indices for one vehicle.
    """
    n = data.n_customers + 1  # 0 = depot
    d = data.dist_matrix
    q = data.demands
    Q = data.vehicle_capacity
    K = int(np.ceil(q[1:].sum() / Q)) + 1

    m = gp.Model("CVRP")
    m.setParam("OutputFlag", 0)
    m.setParam("TimeLimit", 120)

    x = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                x[i, j] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

    u = m.addVars(range(1, n), lb=0, ub=Q, vtype=GRB.CONTINUOUS, name="u")

    m.setObjective(
        gp.quicksum(d[i, j] * x[i, j] for i, j in x), GRB.MINIMIZE
    )

    for i in range(1, n):
        m.addConstr(gp.quicksum(x[j, i] for j in range(n) if j != i) == 1)
        m.addConstr(gp.quicksum(x[i, j] for j in range(n) if j != i) == 1)

    m.addConstr(gp.quicksum(x[0, j] for j in range(1, n)) <= K)
    m.addConstr(gp.quicksum(x[j, 0] for j in range(1, n)) <= K)

    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                m.addConstr(u[i] - u[j] + Q * x[i, j] <= Q - q[j])
    for i in range(1, n):
        m.addConstr(u[i] >= q[i])
        m.addConstr(u[i] <= Q)

    m.optimize()

    if m.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        raise RuntimeError(f"Gurobi failed: status {m.Status}")

    routes: list[list[int]] = []
    for j in range(1, n):
        if x[0, j].X > 0.5:
            route = []
            current = j
            while current != 0:
                route.append(current)
                for k in range(n):
                    if k != current and x[current, k].X > 0.5:
                        current = k
                        break
            routes.append(route)

    return m.ObjVal, routes
