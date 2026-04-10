"""
Capacitated Vehicle Routing Problem (CVRP)
==========================================

Given a depot (node 0), *n* customers with known demands, and a fleet
of vehicles with identical capacity, find a set of routes that serves
every customer exactly once while minimising total travel distance.

Representation
--------------
List of routes, where each route is a list of customer indices (the
depot is implicit at the start and end of every route).

Objective
---------
Total distance over all routes.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from ..core.solution import Solution
from ..core.objective import ObjectiveFunction


# ── Data ────────────────────────────────────────────────────────────

@dataclass
class VRPData:
    """
    Capacitated VRP instance.

    Node 0 is the **depot**; nodes 1 … *n* are customers.

    Attributes
    ----------
    n_customers : int
        Number of customers (excludes depot).
    coords : numpy.ndarray
        Shape ``(n+1, 2)`` — coordinates of depot + customers.
    demands : numpy.ndarray
        Shape ``(n+1,)`` — ``demands[0] = 0`` (depot).
    vehicle_capacity : float
        Maximum load per vehicle.
    dist_matrix : numpy.ndarray
        Shape ``(n+1, n+1)`` — pairwise Euclidean distances.
    """

    n_customers: int
    coords: np.ndarray
    demands: np.ndarray
    vehicle_capacity: float
    dist_matrix: np.ndarray

    @staticmethod
    def generate_random(n: int = 15, seed: int = 42) -> "VRPData":
        """
        Generate a random CVRP instance.

        The depot is placed at the centre (50, 50); customers are
        scattered randomly.  Vehicle capacity is set so that roughly
        3–4 vehicles are needed.

        Parameters
        ----------
        n : int
            Number of customers.
        seed : int
            Random seed.
        """
        rng = np.random.default_rng(seed)
        coords = np.zeros((n + 1, 2))
        coords[0] = [50, 50]
        coords[1:] = rng.random((n, 2)) * 100

        demands = np.zeros(n + 1)
        demands[1:] = rng.integers(5, 20, size=n).astype(float)

        cap = float(int(demands[1:].sum() / 3.5))

        dist = np.zeros((n + 1, n + 1))
        for i in range(n + 1):
            for j in range(i + 1, n + 1):
                d = np.linalg.norm(coords[i] - coords[j])
                dist[i, j] = dist[j, i] = d
        return VRPData(n, coords, demands, cap, dist)


# ── Solution ────────────────────────────────────────────────────────

class VRPSolution(Solution):
    """
    A VRP solution as a list of routes.

    Each route is a list of **customer** indices (no depot — the depot
    is implicit at the start and end).

    Attributes
    ----------
    routes : list[list[int]]
        E.g. ``[[3, 7, 2], [5, 1, 8]]`` — two vehicles.
    """

    def __init__(self, routes: list[list[int]]):
        self.routes = [list(r) for r in routes]

    def copy(self) -> "VRPSolution":
        return VRPSolution([r[:] for r in self.routes])

    def all_customers(self) -> set[int]:
        """Return the set of all served customers."""
        return {c for r in self.routes for c in r}

    def remove_customer(self, customer: int) -> None:
        """Remove *customer* from whichever route contains it."""
        for route in self.routes:
            if customer in route:
                route.remove(customer)
                return

    def remove_empty_routes(self) -> None:
        """Drop routes that have no customers."""
        self.routes = [r for r in self.routes if r]

    def __repr__(self) -> str:
        total = sum(len(r) for r in self.routes)
        return f"VRPSolution({len(self.routes)} routes, {total} customers)"


# ── Objective ───────────────────────────────────────────────────────

class VRPObjective(ObjectiveFunction):
    """
    Total distance of all routes (depot → customers → depot).

    Parameters
    ----------
    data : VRPData
        Problem instance.
    """

    def __init__(self, data: VRPData):
        self.data = data

    def evaluate(self, solution: VRPSolution) -> float:
        """Sum of distances over all routes."""
        d = self.data.dist_matrix
        total = 0.0
        for route in solution.routes:
            if not route:
                continue
            total += d[0, route[0]]
            for i in range(len(route) - 1):
                total += d[route[i], route[i + 1]]
            total += d[route[-1], 0]
        return total

    def route_load(self, route: list[int]) -> float:
        """Total demand served by a single route."""
        return sum(self.data.demands[c] for c in route)

    def is_feasible(self, solution: VRPSolution) -> bool:
        """Return ``True`` if all routes respect the vehicle capacity."""
        for route in solution.routes:
            if self.route_load(route) > self.data.vehicle_capacity:
                return False
        return True
