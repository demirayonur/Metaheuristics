"""
Base class for objective (fitness) functions.

Convention
----------
We **minimise** throughout this codebase.  If the original problem is a
maximisation problem (e.g. Knapsack profit), negate the value so that
lower = better.
"""

from __future__ import annotations

from .solution import Solution


class ObjectiveFunction:
    """
    Evaluates a :class:`Solution` and returns a scalar cost.

    Lower values are better (minimisation convention).

    Subclasses should override :meth:`evaluate`.
    """

    def evaluate(self, solution: Solution) -> float:
        """
        Compute the objective value of *solution*.

        Parameters
        ----------
        solution : Solution
            The candidate to evaluate.

        Returns
        -------
        float
            Objective value (lower is better).
        """
        raise NotImplementedError("Subclasses must implement evaluate()")
