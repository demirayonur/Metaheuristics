"""
0/1 Knapsack Problem
====================

Given *n* items, each with a value and a weight, select a subset that
maximises total value without exceeding the knapsack capacity.

Representation
--------------
Binary vector ``x in {0, 1}^n`` where ``x_i = 1`` means item *i* is packed.

Objective (minimisation convention)
-----------------------------------
``f(x) = -sum(v_i * x_i)  +  penalty * max(0, sum(w_i * x_i) - C)``

Neighbourhood
-------------
Single bit flip — pack or unpack one item.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from ..core.solution import Solution
from ..core.objective import ObjectiveFunction
from ..core.neighbourhood import Neighbourhood


# ── Data ────────────────────────────────────────────────────────────

@dataclass
class KnapsackData:
    """
    Data container for a 0/1 Knapsack instance.

    Attributes
    ----------
    n_items : int
        Number of items.
    values : numpy.ndarray
        Profit of each item — shape ``(n,)``.
    weights : numpy.ndarray
        Weight of each item — shape ``(n,)``.
    capacity : float
        Knapsack weight capacity.
    """

    n_items: int
    values: np.ndarray
    weights: np.ndarray
    capacity: float

    @staticmethod
    def generate_random(n: int = 50, seed: int = 42) -> "KnapsackData":
        """
        Generate a random knapsack instance.

        Capacity is set to approx 40% of total weight so the problem is
        neither trivially easy nor infeasible.
        """
        rng = np.random.default_rng(seed)
        values = rng.integers(10, 100, size=n).astype(float)
        weights = rng.integers(5, 50, size=n).astype(float)
        capacity = float(int(0.4 * weights.sum()))
        return KnapsackData(n_items=n, values=values,
                            weights=weights, capacity=capacity)


# ── Solution ────────────────────────────────────────────────────────

class KnapsackSolution(Solution):
    """
    Binary vector encoding for the 0/1 Knapsack.

    Attributes
    ----------
    selection : numpy.ndarray
        Binary array of length ``n_items``.
    """

    def __init__(self, selection: np.ndarray):
        self.selection = selection.copy()

    def copy(self) -> "KnapsackSolution":
        return KnapsackSolution(self.selection)

    def __repr__(self) -> str:
        packed = int(self.selection.sum())
        return f"KnapsackSolution({packed}/{len(self.selection)} items packed)"


# ── Objective ───────────────────────────────────────────────────────

class KnapsackObjective(ObjectiveFunction):
    """
    Penalised objective for 0/1 Knapsack (minimisation).

    f(x) = -sum(v_i * x_i) + penalty * max(0, sum(w_i * x_i) - C)

    Parameters
    ----------
    data : KnapsackData
        Problem instance.
    penalty : float
        Penalty coefficient per unit of excess weight.
    """

    def __init__(self, data: KnapsackData, penalty: float = 100.0):
        self.data = data
        self.penalty = penalty

    def evaluate(self, solution: KnapsackSolution) -> float:
        """Return negative profit + penalty for constraint violation."""
        total_value = float(self.data.values @ solution.selection)
        total_weight = float(self.data.weights @ solution.selection)
        excess = max(0.0, total_weight - self.data.capacity)
        return -total_value + self.penalty * excess

    def get_value_and_weight(
        self, solution: KnapsackSolution
    ) -> tuple[float, float]:
        """Return (total_value, total_weight) without penalty."""
        v = float(self.data.values @ solution.selection)
        w = float(self.data.weights @ solution.selection)
        return v, w


# ── Neighbourhood ───────────────────────────────────────────────────

class KnapsackFlipNeighbourhood(Neighbourhood):
    """
    Flip exactly one bit — pack or unpack one item.

    Parameters
    ----------
    n_items : int
        Length of the binary vector.
    """

    def __init__(self, n_items: int):
        self.n_items = n_items

    def get_random_neighbour(
        self, solution: KnapsackSolution, rng: np.random.Generator
    ) -> KnapsackSolution:
        neighbour = solution.copy()
        idx = rng.integers(0, self.n_items)
        neighbour.selection[idx] = 1 - neighbour.selection[idx]
        return neighbour
