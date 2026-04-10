"""
Base class for neighbourhood structures.

The neighbourhood defines which solutions are *one move* away from the
current solution.  This is arguably the most important design decision
in any local-search-based metaheuristic.

Examples of neighbourhood moves:

* **Bit flip** — toggle one decision variable (Knapsack).
* **2-opt** — reverse a sub-path in a tour (TSP).
* **Relocate / swap** — move a customer between routes (VRP).
"""

from __future__ import annotations

import numpy as np

from .solution import Solution


class Neighbourhood:
    """
    Defines how to generate neighbours of a given solution.

    Subclasses should override :meth:`get_random_neighbour`.
    """

    def get_random_neighbour(
        self, solution: Solution, rng: np.random.Generator
    ) -> Solution:
        """
        Return **one** random neighbour of *solution*.

        Parameters
        ----------
        solution : Solution
            The current solution.
        rng : numpy.random.Generator
            Random number generator (for reproducibility).

        Returns
        -------
        Solution
            A new, independent neighbour solution.
        """
        raise NotImplementedError(
            "Subclasses must implement get_random_neighbour()"
        )
