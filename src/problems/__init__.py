"""
Problem definitions — data containers, solutions, objectives, and
neighbourhoods for each benchmark problem.
"""

from .knapsack import KnapsackData, KnapsackSolution, KnapsackObjective, KnapsackFlipNeighbourhood
from .tsp import TSPData, TSPSolution, TSPObjective
from .vrp import VRPData, VRPSolution, VRPObjective

__all__ = [
    "KnapsackData", "KnapsackSolution", "KnapsackObjective", "KnapsackFlipNeighbourhood",
    "TSPData", "TSPSolution", "TSPObjective",
    "VRPData", "VRPSolution", "VRPObjective",
]
