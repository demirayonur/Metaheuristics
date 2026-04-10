"""
Smoke tests — verify that each algorithm runs without crashing
and produces feasible solutions.
"""

import pytest
import numpy as np


def test_knapsack_sa():
    """SA should find a feasible knapsack solution."""
    from src.problems.knapsack import (
        KnapsackData, KnapsackObjective, KnapsackFlipNeighbourhood,
    )
    from src.core.termination import TerminationCriteria
    from src.algorithms.simulated_annealing import SimulatedAnnealing

    data = KnapsackData.generate_random(n=20, seed=0)
    obj = KnapsackObjective(data, penalty=200.0)
    nbhd = KnapsackFlipNeighbourhood(data.n_items)
    term = TerminationCriteria(max_iterations=500)

    sa = SimulatedAnnealing(
        objective=obj, neighbourhood=nbhd, termination=term, data=data,
        initial_temp=50.0, cooling_rate=0.99,
    )
    sol, cost, history = sa.run(seed=0)
    val, wt = obj.get_value_and_weight(sol)
    assert wt <= data.capacity, "SA produced infeasible knapsack solution"
    assert val > 0, "SA should pack at least one item"


def test_tsp_ga():
    """GA should find a valid TSP tour."""
    from src.problems.tsp import TSPData
    from src.algorithms.genetic_algorithm import GeneticAlgorithmTSP

    data = TSPData.generate_random(n=10, seed=0)
    ga = GeneticAlgorithmTSP(data=data, pop_size=20, n_generations=50)
    sol, dist = ga.run(seed=0)
    assert len(sol.tour) == data.n_cities
    assert len(set(sol.tour)) == data.n_cities, "Tour has duplicate cities"
    assert dist > 0


def test_vrp_alns():
    """ALNS should find a feasible VRP solution."""
    from src.problems.vrp import VRPData, VRPObjective
    from src.algorithms.alns import ALNS

    data = VRPData.generate_random(n=8, seed=0)
    alns = ALNS(data=data, max_iterations=200)
    sol, dist = alns.run(seed=0)
    obj = VRPObjective(data)
    assert obj.is_feasible(sol), "ALNS produced infeasible VRP solution"
    assert sol.all_customers() == set(range(1, data.n_customers + 1))
