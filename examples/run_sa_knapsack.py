#!/usr/bin/env python3
"""
Example — Module 4: Simulated Annealing for 0/1 Knapsack.

Solves a 50-item knapsack with SA, compares to Gurobi's exact solution,
and generates a four-panel analysis figure.

Usage:
    python -m examples.run_sa_knapsack
"""

import time
import numpy as np

from src.problems.knapsack import (
    KnapsackData, KnapsackObjective, KnapsackFlipNeighbourhood,
)
from src.core.termination import TerminationCriteria
from src.algorithms.simulated_annealing import SimulatedAnnealing
from src.solvers.gurobi_knapsack import solve_knapsack_gurobi
from src.visualization.plot_sa import plot_sa_results


def main():
    print("=" * 70)
    print("  Module 4: Simulated Annealing for 0/1 Knapsack")
    print("=" * 70)

    data = KnapsackData.generate_random(n=50, seed=42)
    print(f"\n  Instance : {data.n_items} items, capacity = {data.capacity}")
    print(f"  Total val: {data.values.sum():.0f}, total wt: {data.weights.sum():.0f}")

    # ── Gurobi ──
    print("\n  [Gurobi] solving …")
    t0 = time.time()
    grb_val, grb_sel = solve_knapsack_gurobi(data)
    print(f"  [Gurobi] optimal value = {grb_val:.0f}  ({time.time()-t0:.3f}s)")

    # ── SA ──
    obj = KnapsackObjective(data, penalty=200.0)
    nbhd = KnapsackFlipNeighbourhood(data.n_items)
    term = TerminationCriteria(max_iterations=10_000, max_no_improve=2000)

    sa = SimulatedAnnealing(
        objective=obj, neighbourhood=nbhd, termination=term, data=data,
        initial_temp=100.0, cooling_rate=0.9985, min_temp=0.001,
        reheat_interval=1500,
    )
    print("  [SA] solving …")
    t0 = time.time()
    best_sol, best_cost, history = sa.run(seed=42)
    sa_val, sa_wt = obj.get_value_and_weight(best_sol)
    print(f"  [SA] value = {sa_val:.0f}, weight = {sa_wt:.0f}  ({time.time()-t0:.3f}s)")

    gap = 100 * (1 - sa_val / grb_val) if grb_val > 0 else 0
    print(f"\n  >>> Optimality gap = {gap:.2f}%")

    plot_sa_results(history, sa_val, grb_val, data, best_sol, grb_sel,
                    "docs/figures/module4_sa_knapsack.png")


if __name__ == "__main__":
    main()
