"""
Exact solver for 0/1 Knapsack using Gurobi MIP.
"""

from __future__ import annotations

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from ..problems.knapsack import KnapsackData


def solve_knapsack_gurobi(data: KnapsackData) -> tuple[float, np.ndarray]:
    """
    Solve the 0/1 Knapsack Problem to optimality.

    Parameters
    ----------
    data : KnapsackData
        Problem instance.

    Returns
    -------
    optimal_value : float
        Maximum total profit.
    selection : numpy.ndarray
        Binary vector indicating which items are packed.
    """
    m = gp.Model("Knapsack")
    m.setParam("OutputFlag", 0)

    x = m.addVars(data.n_items, vtype=GRB.BINARY, name="x")

    m.setObjective(
        gp.quicksum(data.values[i] * x[i] for i in range(data.n_items)),
        GRB.MAXIMIZE,
    )
    m.addConstr(
        gp.quicksum(data.weights[i] * x[i] for i in range(data.n_items))
        <= data.capacity
    )

    m.optimize()

    opt_val = m.ObjVal
    sel = np.array([x[i].X for i in range(data.n_items)])
    return opt_val, sel
