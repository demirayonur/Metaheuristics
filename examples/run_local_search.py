#!/usr/bin/env python3
"""
Example — Modules 1 & 2: Local Search and Motivation.

Demonstrates local search on the 1-D Rastrigin function, shows how it
gets trapped, and motivates multi-start / metaheuristic approaches.

Usage:
    python -m examples.run_local_search
"""

from src.visualization.plot_local_search import plot_landscape_and_search


def main():
    print("=" * 70)
    print("  Modules 1 & 2: Local Search & Why We Need Metaheuristics")
    print("=" * 70)
    plot_landscape_and_search("docs/figures/module1_2_local_search.png")


if __name__ == "__main__":
    main()
