"""
Abstract skeleton for a single-solution metaheuristic.

Subclasses only need to implement two methods:

* :meth:`_build_initial` — construct or receive a starting solution.
* :meth:`_iterate` — execute **one** iteration of the search.

The run loop, timing, history recording, and termination logic are all
handled by :meth:`run` in the base class.
"""

from __future__ import annotations
import time
from abc import ABC, abstractmethod

import numpy as np

from .solution import Solution
from .objective import ObjectiveFunction
from .termination import TerminationCriteria
from .history import SearchHistory


class Metaheuristic(ABC):
    """
    Abstract base class for single-solution metaheuristics.

    Parameters
    ----------
    objective : ObjectiveFunction
        The function to minimise.
    termination : TerminationCriteria
        When to stop searching.

    Attributes
    ----------
    best_solution : Solution or None
        Best solution found so far.
    best_cost : float
        Objective value of the best solution.
    current_solution : Solution or None
        The working solution in the current iteration.
    current_cost : float
        Objective value of the working solution.
    history : SearchHistory
        Iteration-level trajectory log.
    """

    def __init__(
        self, objective: ObjectiveFunction, termination: TerminationCriteria
    ):
        self.objective = objective
        self.termination = termination
        self.history = SearchHistory()
        self.best_solution: Solution | None = None
        self.best_cost: float = np.inf
        self.current_solution: Solution | None = None
        self.current_cost: float = np.inf

    # ── abstract hooks ──────────────────────────────────────────────

    @abstractmethod
    def _build_initial(self, rng: np.random.Generator) -> Solution:
        """Construct or receive an initial solution."""
        ...

    @abstractmethod
    def _iterate(self, iteration: int, rng: np.random.Generator) -> None:
        """
        Execute **one** iteration of the search.

        Must update ``self.current_solution`` / ``self.current_cost``,
        and, when an improvement is found, also
        ``self.best_solution`` / ``self.best_cost``.
        """
        ...

    # ── main run loop ───────────────────────────────────────────────

    def run(self, seed: int = 42) -> tuple[Solution, float, SearchHistory]:
        """
        Execute the metaheuristic.

        Parameters
        ----------
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        best_solution : Solution
        best_cost : float
        history : SearchHistory
        """
        rng = np.random.default_rng(seed)

        self.current_solution = self._build_initial(rng)
        self.current_cost = self.objective.evaluate(self.current_solution)
        self.best_solution = self.current_solution.copy()
        self.best_cost = self.current_cost

        self.history = SearchHistory()
        start = time.time()
        no_improve = 0

        for it in range(1, (self.termination.max_iterations or 10**9) + 1):
            elapsed = time.time() - start
            prev_best = self.best_cost

            self._iterate(it, rng)

            # Track improvement
            no_improve = 0 if self.best_cost < prev_best else no_improve + 1

            # Record history
            self.history.record(
                iteration=it,
                current_cost=self.current_cost,
                best_cost=self.best_cost,
                temperature=getattr(self, "temperature", 0.0),
                elapsed=elapsed,
            )

            # Check termination
            if self.termination.should_stop(
                it, elapsed, no_improve, self.best_cost
            ):
                break

        return self.best_solution, self.best_cost, self.history