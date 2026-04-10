"""
Travelling Salesman Problem (TSP)
=================================

Given *n* cities with pairwise distances, find the shortest tour that
visits every city exactly once and returns to the starting city.

Representation
--------------
Permutation of city indices ``[c_0, c_1, …, c_{n-1}]``.

Objective
---------
Total tour length (sum of edge weights + return edge).
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from ..core.solution import Solution
from ..core.objective import ObjectiveFunction


# ── Data ────────────────────────────────────────────────────────────

@dataclass
class TSPData:
    """
    Euclidean TSP instance.

    Attributes
    ----------
    n_cities : int
        Number of cities.
    coords : numpy.ndarray
        Shape ``(n, 2)`` — *(x, y)* for each city.
    dist_matrix : numpy.ndarray
        Precomputed pairwise Euclidean distances — shape ``(n, n)``.
    """

    n_cities: int
    coords: np.ndarray
    dist_matrix: np.ndarray

    @staticmethod
    def generate_random(n: int = 20, seed: int = 42) -> "TSPData":
        """
        Generate *n* random cities in a 100 × 100 square.

        Parameters
        ----------
        n : int
            Number of cities.
        seed : int
            Random seed.
        """
        rng = np.random.default_rng(seed)
        coords = rng.random((n, 2)) * 100
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(coords[i] - coords[j])
                dist[i, j] = dist[j, i] = d
        return TSPData(n_cities=n, coords=coords, dist_matrix=dist)


# ── Solution ────────────────────────────────────────────────────────

class TSPSolution(Solution):
    """
    A TSP tour as an ordered list of city indices.

    Attributes
    ----------
    tour : list[int]
        Permutation of ``[0, 1, …, n-1]``.
    """

    def __init__(self, tour: list[int]):
        self.tour = list(tour)

    def copy(self) -> "TSPSolution":
        return TSPSolution(self.tour[:])

    def __repr__(self) -> str:
        return f"TSPSolution(n={len(self.tour)}, tour={self.tour[:5]}…)"


# ── Objective ───────────────────────────────────────────────────────

class TSPObjective(ObjectiveFunction):
    """
    Total tour distance (minimisation).

    Parameters
    ----------
    data : TSPData
        Problem instance.
    """

    def __init__(self, data: TSPData):
        self.data = data

    def evaluate(self, solution: TSPSolution) -> float:
        """Sum of edge weights along the tour, including return edge."""
        d = self.data.dist_matrix
        tour = solution.tour
        total = sum(d[tour[i], tour[i + 1]] for i in range(len(tour) - 1))
        total += d[tour[-1], tour[0]]
        return total
