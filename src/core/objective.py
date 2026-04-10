"""
Abstract base class for objective (fitness) functions.
 
Convention
----------
We **minimize** throughout this codebase.  If the original problem is a
maximization problem (e.g. Knapsack profit), negate the value so that
lower = better.
"""


from __future__ import annotations
from abc import ABC, abstractmethod
 
from .solution import Solution


class ObjectiveFunction(ABC):
    """
    Evaluates a :class:`Solution` and returns a scalar cost.
 
    Lower values are better (minimization convention).
    """
 
    @abstractmethod
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
        ...