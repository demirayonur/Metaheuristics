"""
Visualisation for Module 3 — Metaheuristic Architecture Diagram.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from .style import apply_style


def plot_components_diagram(
    save_path: str = "docs/figures/module3_components.png",
) -> None:
    """Create a block diagram of the six core metaheuristic components."""
    apply_style()
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14); ax.set_ylim(0, 9); ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(7, 8.6, "Module 3 — Anatomy of a Metaheuristic",
            ha="center", va="center", fontsize=18, fontweight="bold",
            color="#2c3e50")

    components = [
        (0.5, 6.0, 3.0, 1.6, "Solution\nRepresentation",
         "Binary vector, permutation,\nroute list, ...", "#3498db"),
        (5.5, 6.0, 3.0, 1.6, "Objective\nFunction",
         "Evaluate quality:\ncost, profit, distance", "#e74c3c"),
        (10.5, 6.0, 3.0, 1.6, "Neighbourhood\nDefinition",
         "Flip, swap, 2-opt,\nor-opt, relocate, ...", "#2ecc71"),
        (0.5, 3.0, 3.0, 1.6, "Initial Solution\nConstruction",
         "Random, greedy,\nheuristic-based", "#9b59b6"),
        (5.5, 3.0, 3.0, 1.6, "Acceptance\nCriterion",
         "Greedy, Metropolis,\nthreshold, ...", "#f39c12"),
        (10.5, 3.0, 3.0, 1.6, "Termination\nCriteria",
         "Max iter, time limit,\nno-improve, target", "#1abc9c"),
        (3.5, 0.3, 7.0, 1.6, "Search History & Logging",
         "Track current cost, best cost, temperature, elapsed time",
         "#34495e"),
    ]

    for x, y, w, h, label, sub, color in components:
        rect = plt.Rectangle((x, y), w, h, facecolor=color, alpha=0.15,
                              edgecolor=color, linewidth=2.5)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h * 0.65, label, ha="center", va="center",
                fontsize=12, fontweight="bold", color=color)
        ax.text(x + w / 2, y + h * 0.22, sub, ha="center", va="center",
                fontsize=8, color="#555", style="italic")

    arrow_kw = dict(arrowstyle="-|>", color="#7f8c8d", lw=1.8,
                    connectionstyle="arc3,rad=0.15")
    for (x1, y1), (x2, y2) in [
        ((2.0, 6.0), (5.5, 7.0)),
        ((8.5, 7.0), (10.5, 7.0)),
        ((2.0, 4.6), (2.0, 6.0)),
        ((8.5, 3.8), (10.5, 3.8)),
        ((7.0, 6.0), (7.0, 4.6)),
        ((7.0, 3.0), (7.0, 1.9)),
    ]:
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), **arrow_kw))

    bbox = FancyBboxPatch((4.5, 4.85), 5.0, 0.8, boxstyle="round,pad=0.2",
                          facecolor="#ecf0f1", edgecolor="#2c3e50", linewidth=2)
    ax.add_patch(bbox)
    ax.text(7.0, 5.25, "METAHEURISTIC LOOP", ha="center", va="center",
            fontsize=13, fontweight="bold", color="#2c3e50")

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[+] Saved {save_path}")
