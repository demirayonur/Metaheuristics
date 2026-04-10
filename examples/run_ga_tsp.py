#!/usr/bin/env python3
"""
Example — Module 5: Genetic Algorithm for TSP.

Solves a 15-city Euclidean TSP with a GA, compares to Gurobi's exact
MTZ solution, and generates a six-panel analysis figure.

Usage:
    python -m examples.run_ga_tsp
"""

import time

from src.problems.tsp import TSPData
from src.algorithms.genetic_algorithm import GeneticAlgorithmTSP
from src.solvers.gurobi_tsp import solve_tsp_gurobi
from src.visualization.plot_ga import plot_ga_results


def main():
    print("=" * 70)
    print("  Module 5: Genetic Algorithm for TSP")
    print("=" * 70)

    data = TSPData.generate_random(n=15, seed=42)
    print(f"\n  Instance : {data.n_cities} cities")

    # ── Gurobi ──
    print("\n  [Gurobi] solving …")
    t0 = time.time()
    grb_dist, grb_tour = solve_tsp_gurobi(data)
    print(f"  [Gurobi] optimal distance = {grb_dist:.2f}  ({time.time()-t0:.3f}s)")

    # ── GA ──
    ga = GeneticAlgorithmTSP(
        data=data, pop_size=80, n_generations=300,
        crossover_rate=0.9, mutation_rate=0.35,
        tournament_size=5, elite_count=2, local_search_iters=30,
    )
    print("  [GA] solving …")
    t0 = time.time()
    ga_sol, ga_dist = ga.run(seed=42)
    print(f"  [GA] distance = {ga_dist:.2f}  ({time.time()-t0:.3f}s)")

    gap = 100 * (ga_dist / grb_dist - 1) if grb_dist > 0 else 0
    print(f"\n  >>> Optimality gap = {gap:.2f}%")

    plot_ga_results(ga, ga_sol, ga_dist, grb_tour, grb_dist, data,
                    "docs/figures/module5_ga_tsp.png")


if __name__ == "__main__":
    main()
