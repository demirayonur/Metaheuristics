"""
Abstract base class for candidate solutions.
 
A :class:`Solution` encapsulates one candidate answer to an optimisation
problem.  Concrete subclasses define the *representation* — the data
structure that encodes the answer:
 
* **Binary vector** — e.g. which items to pack (Knapsack).
* **Permutation** — e.g. the order in which to visit cities (TSP).
* **List of routes** — e.g. vehicle assignments and sequences (VRP).
 
Design rule
-----------
Every ``Solution`` must be cheaply *copyable* so that the search can
branch without corrupting the incumbent.
"""


from __future__ import annotations
from abc import ABC, abstractmethod
 
 
class Solution(ABC):
    """
    Abstract base class for a candidate solution.
 
    Subclasses **must** implement:
 
    * :meth:`copy` — return a deep, independent clone.
    * :meth:`__repr__` — human-readable summary.
    """
 
    @abstractmethod
    def copy(self) -> "Solution":
        """Return a deep, independent copy of this solution."""
        ...
 
    @abstractmethod
    def __repr__(self) -> str:
        """Return a human-readable summary string."""
        ...
