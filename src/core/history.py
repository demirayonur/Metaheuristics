"""
Search history recorder.

:class:`SearchHistory` stores iteration-level snapshots of the
metaheuristic run so you can later plot convergence curves,
temperature schedules, and other diagnostics.
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class SearchHistory:
    """
    Trajectory log for a single metaheuristic run.

    Attributes
    ----------
    iterations : list[int]
        Iteration indices.
    current_costs : list[float]
        Objective of the *current* solution at each recorded iteration.
    best_costs : list[float]
        Best-so-far objective at each recorded iteration.
    temperatures : list[float]
        Temperature parameter (SA-specific; empty for other methods).
    timestamps : list[float]
        Wall-clock seconds elapsed since the start of the run.
    """

    iterations: list[int] = field(default_factory=list)
    current_costs: list[float] = field(default_factory=list)
    best_costs: list[float] = field(default_factory=list)
    temperatures: list[float] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)

    def record(
        self,
        iteration: int,
        current_cost: float,
        best_cost: float,
        temperature: float = 0.0,
        elapsed: float = 0.0,
    ) -> None:
        """Append one snapshot to the history."""
        self.iterations.append(iteration)
        self.current_costs.append(current_cost)
        self.best_costs.append(best_cost)
        self.temperatures.append(temperature)
        self.timestamps.append(elapsed)