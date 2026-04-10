"""
Acceptance criteria decide whether to move from the current solution
to a candidate neighbour.

Provided implementations:

* :class:`GreedyAcceptance` — accept only strict improvements.
* :class:`MetropolisAcceptance` — accept worse solutions with
  probability ``exp(-delta / T)``, used by Simulated Annealing.
"""

from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np


class AcceptanceCriterion(ABC):
    """
    Abstract acceptance criterion.

    Subclasses implement the :meth:`accept` predicate.
    """

    @abstractmethod
    def accept(
        self, current_cost: float, candidate_cost: float, **kwargs
    ) -> bool:
        """
        Decide whether to accept the candidate.

        Parameters
        ----------
        current_cost : float
            Objective value of the current solution.
        candidate_cost : float
            Objective value of the proposed neighbour.
        **kwargs
            Algorithm-specific parameters (e.g. ``temperature``, ``rng``).

        Returns
        -------
        bool
            ``True`` if the candidate should replace the current solution.
        """
        ...


class GreedyAcceptance(AcceptanceCriterion):
    """Accept **only** strictly improving moves (minimization)."""

    def accept(
        self, current_cost: float, candidate_cost: float, **kwargs
    ) -> bool:
        return candidate_cost < current_cost


class MetropolisAcceptance(AcceptanceCriterion):
    """
    Metropolis criterion (Simulated Annealing).

    * Improvements are **always** accepted.
    * Worse solutions are accepted with probability

      .. math::

          P = \\exp\\!\\left(\\frac{-(\\Delta f)}{T}\\right)

      where *Δf = candidate − current > 0* and *T* is the temperature.

    Required ``**kwargs``:

    * ``temperature`` (float) — current SA temperature.
    * ``rng`` (numpy.random.Generator) — RNG instance.
    """

    def accept(
        self, current_cost: float, candidate_cost: float, **kwargs
    ) -> bool:
        temperature: float = kwargs.get("temperature", 1.0)
        rng: np.random.Generator = kwargs.get("rng", np.random.default_rng())

        delta = candidate_cost - current_cost
        if delta < 0:
            return True  # improvement — always accept
        if temperature <= 0:
            return False
        return float(rng.random()) < np.exp(-delta / temperature)