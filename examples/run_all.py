#!/usr/bin/env python3
"""
Run all examples end-to-end and generate every figure.

Usage:
    python -m examples.run_all
"""

import time
from examples import run_local_search, run_sa_knapsack, run_ga_tsp, run_alns_vrp


def main():
    print("=" * 70)
    print("  METAHEURISTICS TUTORIAL — Running All Examples")
    print("=" * 70)
    t0 = time.time()

    run_local_search.main()
    print()
    run_sa_knapsack.main()
    print()
    run_ga_tsp.main()
    print()
    run_alns_vrp.main()

    print(f"\n{'=' * 70}")
    print(f"  All done in {time.time() - t0:.1f} seconds.")
    print(f"  Figures saved in docs/figures/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
