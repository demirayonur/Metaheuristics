"""
Base class for candidate solutions.

A :class:`Solution` encapsulates one candidate answer to an optimisation
problem.  Concrete subclasses override :meth:`copy` and :meth:`__repr__`.

Representation is problem-specific:

* **Binary vector** — e.g. which items to pack (Knapsack).
* **Permutation** — e.g. the order in which to visit cities (TSP).
* **List of routes** — e.g. vehicle assignments and sequences (VRP).
"""

from __future__ import annotations


class Solution:
    """
    Base class for a candidate solution.

    Subclasses should override:

    * :meth:`copy` — return a deep, independent clone.
    * :meth:`__repr__` — human-readable summary.
    """

    def copy(self) -> "Solution":
        """Return a deep, independent copy of this solution."""
        raise NotImplementedError("Subclasses must implement copy()")

    def __repr__(self) -> str:
        """Return a human-readable summary string."""
        return f"{self.__class__.__name__}()"
