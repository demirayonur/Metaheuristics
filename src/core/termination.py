"""
Termination criteria bundle the conditions under which a metaheuristic
should stop.

Any **single** condition being met triggers termination.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class TerminationCriteria:
    """
    Composite stopping condition.

    Parameters
    ----------
    max_iterations : int or None
        Stop after this many iterations.
    max_time_seconds : float or None
        Stop after this many wall-clock seconds.
    max_no_improve : int or None
        Stop after this many consecutive non-improving iterations.
    target_cost : float or None
        Stop if the objective reaches (or goes below) this value.
    """

    max_iterations: int | None = 1000
    max_time_seconds: float | None = None
    max_no_improve: int | None = None
    target_cost: float | None = None

    def should_stop(
        self,
        iteration: int,
        elapsed: float,
        no_improve_count: int,
        best_cost: float,
    ) -> bool:
        """
        Evaluate all stopping conditions.

        Returns ``True`` if **any** condition is met.
        """
        if self.max_iterations is not None and iteration >= self.max_iterations:
            return True
        if self.max_time_seconds is not None and elapsed >= self.max_time_seconds:
            return True
        if self.max_no_improve is not None and no_improve_count >= self.max_no_improve:
            return True
        if self.target_cost is not None and best_cost <= self.target_cost:
            return True
        return False