#!/usr/bin/env python3
"""
Example — Module 6: ALNS for Capacitated VRP.

Solves a 12-customer CVRP with ALNS, compares to Gurobi's exact
solution, and generates a six-panel analysis figure.

Usage:
    python -m examples.run_alns_vrp
"""

import time

from src.problems.vrp import VRPData
from src.algorithms.alns import ALNS
from src.solvers.gurobi_vrp import solve_vrp_gurobi
from src.visualization.plot_alns import plot_alns_results


def main():
    print("=" * 70)
    print("  Module 6: ALNS for Capacitated VRP")
    print("=" * 70)

    data = VRPData.generate_random(n=12, seed=42)
    print(f"\n  Instance : {data.n_customers} customers, "
          f"capacity = {data.vehicle_capacity}")

    # ── Gurobi ──
    print("\n  [Gurobi] solving …")
    t0 = time.time()
    grb_dist, grb_routes = solve_vrp_gurobi(data)
    print(f"  [Gurobi] optimal distance = {grb_dist:.2f}, "
          f"{len(grb_routes)} vehicles  ({time.time()-t0:.3f}s)")

    # ── ALNS ──
    alns = ALNS(
        data=data, max_iterations=3000, segment_size=100,
        removal_fraction=(0.2, 0.4),
        initial_temp=50.0, cooling_rate=0.9993,
    )
    print("  [ALNS] solving …")
    t0 = time.time()
    alns_sol, alns_dist = alns.run(seed=42)
    print(f"  [ALNS] distance = {alns_dist:.2f}, "
          f"{len(alns_sol.routes)} vehicles  ({time.time()-t0:.3f}s)")
    print(f"  [ALNS] feasible = {alns.objective.is_feasible(alns_sol)}")

    gap = 100 * (alns_dist / grb_dist - 1) if grb_dist > 0 else 0
    print(f"\n  >>> Optimality gap = {gap:.2f}%")

    plot_alns_results(alns, alns_sol, alns_dist, grb_routes, grb_dist, data,
                      "docs/figures/module6_alns_vrp.png")


if __name__ == "__main__":
    main()
