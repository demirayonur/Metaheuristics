"""
Visualisation for Module 4 — Simulated Annealing results.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..core.history import SearchHistory
from ..problems.knapsack import KnapsackData, KnapsackSolution
from .style import COLORS, apply_style


def plot_sa_results(
    history: SearchHistory,
    sa_value: float,
    gurobi_value: float,
    data: KnapsackData,
    sa_solution: KnapsackSolution,
    gurobi_selection: np.ndarray,
    save_path: str = "docs/figures/module4_sa_knapsack.png",
) -> None:
    """
    Four-panel SA analysis figure.

    (a) Convergence — current and best profit over iterations.
    (b) Cooling schedule (log scale).
    (c) SA vs Gurobi bar chart with optimality gap.
    (d) Item selection comparison heatmap.
    """
    apply_style()
    C = COLORS
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Module 4 — Simulated Annealing for 0/1 Knapsack",
                 fontsize=16, fontweight="bold", y=0.98)

    iters = history.iterations
    current_profit = [-c for c in history.current_costs]
    best_profit = [-c for c in history.best_costs]

    # (a) Convergence
    ax = axes[0, 0]
    ax.plot(iters, current_profit, color=C["current"], alpha=0.3, lw=0.5,
            label="Current profit")
    ax.plot(iters, best_profit, color=C["best"], lw=2, label="Best profit")
    ax.axhline(gurobi_value, color=C["gurobi"], ls="--", lw=2,
               label=f"Gurobi optimal = {gurobi_value:.0f}")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Total Profit")
    ax.set_title("(a) Convergence of SA"); ax.legend(fontsize=8)

    # (b) Temperature
    ax = axes[0, 1]
    ax.plot(iters, history.temperatures, color=C["temperature"], lw=1.5)
    ax.set_xlabel("Iteration"); ax.set_ylabel("Temperature")
    ax.set_title("(b) Cooling Schedule"); ax.set_yscale("log")

    # (c) Comparison
    ax = axes[1, 0]
    methods = ["Simulated\nAnnealing", "Gurobi\n(Optimal)"]
    vals = [sa_value, gurobi_value]
    bars = ax.bar(methods, vals, color=[C["best"], C["gurobi"]],
                  edgecolor="k", linewidth=0.8, width=0.45)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{v:.0f}", ha="center", va="bottom", fontweight="bold", fontsize=13)
    gap = 100 * (1 - sa_value / gurobi_value) if gurobi_value > 0 else 0
    ax.set_title(f"(c) SA vs Gurobi  (gap = {gap:.2f}%)")
    ax.set_ylabel("Total Profit"); ax.set_ylim(0, max(vals) * 1.15)

    # (d) Heatmap
    ax = axes[1, 1]
    comparison = np.vstack([sa_solution.selection, gurobi_selection])
    im = ax.imshow(comparison, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["SA", "Gurobi"])
    ax.set_xlabel("Item Index"); ax.set_title("(d) Item Selection Comparison")
    plt.colorbar(im, ax=ax, fraction=0.03, label="Selected")
    agree = int((sa_solution.selection == gurobi_selection).sum())
    ax.text(data.n_items / 2, -0.7,
            f"Agreement: {agree}/{data.n_items} items ({100*agree/data.n_items:.1f}%)",
            ha="center", fontsize=9, style="italic")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[+] Saved {save_path}")
