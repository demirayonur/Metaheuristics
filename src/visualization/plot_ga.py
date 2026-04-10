"""
Visualisation for Module 5 — Genetic Algorithm results.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from ..problems.tsp import TSPData, TSPSolution
from ..algorithms.genetic_algorithm import GeneticAlgorithmTSP
from .style import COLORS, apply_style


def _draw_tour(ax, coords, tour, color, title, distance, lw=1.5):
    """Draw a TSP tour on an axes."""
    n = len(tour)
    for i in range(n):
        c1, c2 = coords[tour[i]], coords[tour[(i + 1) % n]]
        ax.plot([c1[0], c2[0]], [c1[1], c2[1]], "-", color=color,
                lw=lw, alpha=0.7)
    ax.scatter(coords[:, 0], coords[:, 1], s=60, color=color,
               edgecolors="k", zorder=5, linewidths=0.6)
    for i, (cx, cy) in enumerate(coords):
        ax.text(cx + 1.2, cy + 1.2, str(i), fontsize=6, color="#333")
    ax.set_title(f"{title}\nDistance = {distance:.2f}")
    ax.set_aspect("equal")


def plot_ga_results(
    ga: GeneticAlgorithmTSP,
    ga_tour: TSPSolution,
    ga_dist: float,
    gurobi_tour: list[int],
    gurobi_dist: float,
    data: TSPData,
    save_path: str = "docs/figures/module5_ga_tsp.png",
) -> None:
    """
    Six-panel GA analysis figure.

    (a) GA best tour.
    (b) Gurobi optimal tour.
    (c) Convergence with population spread.
    (d) Population diversity over generations.
    (e) Distance bar comparison.
    (f) Edge overlap analysis.
    """
    apply_style()
    C = COLORS
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)
    fig.suptitle("Module 5 — Genetic Algorithm for TSP",
                 fontsize=16, fontweight="bold", y=0.98)

    # (a) GA tour
    ax = fig.add_subplot(gs[0, 0])
    _draw_tour(ax, data.coords, ga_tour.tour, C["ga"], "(a) GA Best Tour", ga_dist)

    # (b) Gurobi tour
    ax = fig.add_subplot(gs[0, 1])
    _draw_tour(ax, data.coords, gurobi_tour, C["gurobi"], "(b) Gurobi Optimal", gurobi_dist)

    # (c) Convergence
    ax = fig.add_subplot(gs[0, 2])
    gens = range(len(ga.best_per_gen))
    ax.fill_between(gens, ga.worst_per_gen, ga.best_per_gen,
                    alpha=0.15, color=C["ga"])
    ax.plot(gens, ga.avg_per_gen, color=C["current"], lw=1, alpha=0.7,
            label="Avg fitness")
    ax.plot(gens, ga.best_per_gen, color=C["ga"], lw=2, label="Best fitness")
    ax.axhline(gurobi_dist, color=C["gurobi"], ls="--", lw=2,
               label=f"Optimal = {gurobi_dist:.1f}")
    ax.set_xlabel("Generation"); ax.set_ylabel("Tour Distance")
    ax.set_title("(c) GA Convergence"); ax.legend(fontsize=7)

    # (d) Diversity
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(gens, ga.diversity_per_gen, color=C["purple"], lw=1.5)
    ax.set_xlabel("Generation"); ax.set_ylabel("Avg Edge-Set Difference")
    ax.set_title("(d) Population Diversity"); ax.set_ylim(0, 1.05)

    # (e) Bar comparison
    ax = fig.add_subplot(gs[1, 1])
    vals = [ga_dist, gurobi_dist]
    bars = ax.bar(["GA", "Gurobi\n(Optimal)"], vals,
                  color=[C["ga"], C["gurobi"]], edgecolor="k",
                  linewidth=0.8, width=0.45)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{v:.1f}", ha="center", va="bottom", fontweight="bold", fontsize=12)
    gap = 100 * (ga_dist / gurobi_dist - 1) if gurobi_dist > 0 else 0
    ax.set_title(f"(e) GA vs Gurobi  (gap = {gap:.2f}%)")
    ax.set_ylabel("Tour Distance"); ax.set_ylim(0, max(vals) * 1.2)

    # (f) Edge overlap
    ax = fig.add_subplot(gs[1, 2])

    def edge_set(tour):
        return {(min(tour[k], tour[(k + 1) % len(tour)]),
                 max(tour[k], tour[(k + 1) % len(tour)]))
                for k in range(len(tour))}

    ga_edges = edge_set(ga_tour.tour)
    grb_edges = edge_set(gurobi_tour)
    shared = ga_edges & grb_edges
    coords = data.coords
    for a, b in shared:
        ax.plot([coords[a, 0], coords[b, 0]], [coords[a, 1], coords[b, 1]],
                "-", color=C["gurobi"], lw=2.5, alpha=0.8, zorder=2)
    for a, b in ga_edges - grb_edges:
        ax.plot([coords[a, 0], coords[b, 0]], [coords[a, 1], coords[b, 1]],
                "-", color=C["ga"], lw=1.5, alpha=0.5, zorder=1)
    for a, b in grb_edges - ga_edges:
        ax.plot([coords[a, 0], coords[b, 0]], [coords[a, 1], coords[b, 1]],
                "--", color=C["current"], lw=1.5, alpha=0.5, zorder=1)
    ax.scatter(coords[:, 0], coords[:, 1], s=40, color="#333",
               edgecolors="k", zorder=5, linewidths=0.5)
    ax.set_title(f"(f) Edge Overlap: {len(shared)}/{data.n_cities} shared")
    ax.set_aspect("equal")
    ax.legend(handles=[
        Line2D([0], [0], color=C["gurobi"], lw=2.5, label="Shared"),
        Line2D([0], [0], color=C["ga"], lw=1.5, label="GA only"),
        Line2D([0], [0], color=C["current"], lw=1.5, ls="--", label="Gurobi only"),
    ], fontsize=7, loc="lower right")

    fig.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[+] Saved {save_path}")
