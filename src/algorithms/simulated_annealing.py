"""
Simulated Annealing (SA)
========================

A single-solution metaheuristic that escapes local optima by accepting
worse solutions with a probability that decreases over time (cooling).

Algorithm sketch
----------------

1. Start from an initial solution *s*.
2. Generate a random neighbour *s'*.
3. If *s'* is better → accept.
4. If *s'* is worse  → accept with probability ``exp(-Δf / T)``.
5. Cool the temperature: ``T ← α · T``.
6. Repeat until termination.

This module applies SA to the **0/1 Knapsack** problem.
"""

from __future__ import annotations

import numpy as np

from ..core.metaheuristic import Metaheuristic
from ..core.acceptance import MetropolisAcceptance
from ..core.termination import TerminationCriteria
from ..problems.knapsack import (
    KnapsackData,
    KnapsackSolution,
    KnapsackObjective,
    KnapsackFlipNeighbourhood,
)


class SimulatedAnnealing(Metaheuristic):
    """
    Simulated Annealing with geometric cooling schedule.

    Parameters
    ----------
    objective : KnapsackObjective
        Penalised knapsack objective.
    neighbourhood : KnapsackFlipNeighbourhood
        Bit-flip neighbourhood.
    termination : TerminationCriteria
        Stopping conditions.
    data : KnapsackData
        Problem instance (used for greedy initial solution).
    initial_temp : float
        Starting temperature *T₀*.
    cooling_rate : float
        Multiplicative cooling factor *α*: ``T_{k+1} = α · T_k``.
        Typical values: 0.99 – 0.9995.
    min_temp : float
        Stop cooling below this temperature.
    reheat_interval : int or None
        If set, partially reheat every *k* non-improving iterations.
    """

    def __init__(
        self,
        objective: KnapsackObjective,
        neighbourhood: KnapsackFlipNeighbourhood,
        termination: TerminationCriteria,
        data: KnapsackData,
        initial_temp: float = 100.0,
        cooling_rate: float = 0.995,
        min_temp: float = 0.01,
        reheat_interval: int | None = None,
    ):
        super().__init__(objective, termination)
        self.neighbourhood = neighbourhood
        self.acceptance = MetropolisAcceptance()
        self.data = data
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.reheat_interval = reheat_interval
        self.temperature = initial_temp

    # ── hooks ───────────────────────────────────────────────────────

    def _build_initial(self, rng: np.random.Generator) -> KnapsackSolution:
        """
        Greedy initial solution: sort items by value-to-weight ratio
        (descending) and pack greedily.
        """
        data = self.data
        ratios = data.values / data.weights
        order = np.argsort(-ratios)
        sel = np.zeros(data.n_items, dtype=float)
        total_w = 0.0
        for i in order:
            if total_w + data.weights[i] <= data.capacity:
                sel[i] = 1.0
                total_w += data.weights[i]
        return KnapsackSolution(sel)

    def _iterate(self, iteration: int, rng: np.random.Generator) -> None:
        """One SA step: propose → evaluate → accept/reject → cool."""
        candidate = self.neighbourhood.get_random_neighbour(
            self.current_solution, rng
        )
        candidate_cost = self.objective.evaluate(candidate)

        if self.acceptance.accept(
            self.current_cost,
            candidate_cost,
            temperature=self.temperature,
            rng=rng,
        ):
            self.current_solution = candidate
            self.current_cost = candidate_cost

        # Update global best
        if self.current_cost < self.best_cost:
            self.best_solution = self.current_solution.copy()
            self.best_cost = self.current_cost

        # Cool
        if self.temperature > self.min_temp:
            self.temperature *= self.cooling_rate

        # Optional reheat
        if (
            self.reheat_interval
            and iteration % self.reheat_interval == 0
            and self.best_cost < self.current_cost
        ):
            self.temperature = self.initial_temp * 0.5
