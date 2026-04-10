"""
Visualisation for Modules 1 & 2 — Local Search and Motivation.
"""

from __future__ import annotations
from typing import Callable

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .style import COLORS, apply_style


def rastrigin_1d(x: float) -> float:
    """1-D Rastrigin function. Global minimum at x = 0, f(0) = 0."""
    A = 10
    return A + x**2 - A * np.cos(2 * np.pi * x)


class _LocalSearch:
    """Simple hill-climbing for 1-D functions (internal helper)."""

    def __init__(self, objective: Callable, step_size: float = 0.05):
        self.objective = objective
        self.step_size = step_size
        self.trajectory: list[tuple[float, float]] = []

    def run(self, x0: float, max_iter: int = 500) -> tuple[float, float]:
        x, fx = x0, self.objective(x0)
        self.trajectory = [(x, fx)]
        for _ in range(max_iter):
            neighbours = [x - self.step_size, x + self.step_size]
            best_nb = min(neighbours, key=self.objective)
            f_nb = self.objective(best_nb)
            if f_nb < fx:
                x, fx = best_nb, f_nb
                self.trajectory.append((x, fx))
            else:
                break
        return x, fx


def plot_landscape_and_search(save_path: str = "docs/figures/module1_2_local_search.png") -> None:
    """
    Four-panel figure:

    (a) Rastrigin landscape with global optimum.
    (b) Single local search getting trapped.
    (c) Multi-start local search (30 restarts).
    (d) Bar chart comparing final values.
    """
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Modules 1 & 2 — Local Search and the Need for Metaheuristics",
        fontsize=16, fontweight="bold", y=0.98,
    )

    xs = np.linspace(-5.12, 5.12, 1000)
    ys = np.array([rastrigin_1d(x) for x in xs])

    C = COLORS

    # (a) Landscape
    ax = axes[0, 0]
    ax.fill_between(xs, ys, alpha=0.08, color=C["landscape"])
    ax.plot(xs, ys, color=C["landscape"], lw=1.5)
    ax.scatter([0], [0], s=120, color=C["global_opt"], zorder=5,
               edgecolors="k", linewidths=0.8, label="Global optimum")
    ax.set_title("(a) Rastrigin Landscape")
    ax.set_xlabel("x"); ax.set_ylabel("f(x)")
    ax.legend(loc="upper right", fontsize=9)

    # (b) Single LS
    ax = axes[0, 1]
    ax.plot(xs, ys, color=C["landscape"], lw=1.0, alpha=0.5)
    ax.fill_between(xs, ys, alpha=0.05, color=C["landscape"])
    ls = _LocalSearch(rastrigin_1d, 0.05)
    best_x, best_f = ls.run(3.5)
    tx = [p[0] for p in ls.trajectory]
    ty = [p[1] for p in ls.trajectory]
    ax.plot(tx, ty, "-o", color=C["trajectory"], markersize=4, lw=1.5,
            label=f"Trajectory (start=3.5)", zorder=4)
    ax.scatter([tx[0]], [ty[0]], s=100, marker="^", color=C["trajectory"],
               edgecolors="k", zorder=5, label="Start")
    ax.scatter([best_x], [best_f], s=100, marker="*", color=C["local_opt"],
               edgecolors="k", zorder=5, label=f"Local opt f={best_f:.2f}")
    ax.scatter([0], [0], s=80, color=C["global_opt"], edgecolors="k",
               zorder=5, label="Global opt f=0.00")
    ax.set_title("(b) Local Search Gets Trapped!")
    ax.set_xlabel("x")
    ax.legend(loc="upper left", fontsize=8)

    # (c) Multi-start
    ax = axes[1, 0]
    ax.plot(xs, ys, color=C["landscape"], lw=1.0, alpha=0.5)
    ax.fill_between(xs, ys, alpha=0.05, color=C["landscape"])
    rng = np.random.default_rng(42)
    all_results = []
    for _ in range(30):
        x0 = rng.uniform(-5.12, 5.12)
        rx, rf = _LocalSearch(rastrigin_1d, 0.05).run(x0)
        all_results.append((rx, rf))
    ms_x, ms_f = min(all_results, key=lambda p: p[1])
    for rx, rf in all_results:
        ax.scatter(rx, rf, s=30, color=C["multi_start"], alpha=0.6, zorder=4)
    ax.scatter([ms_x], [ms_f], s=140, marker="*", color=C["global_opt"],
               edgecolors="k", linewidths=1.2, zorder=6,
               label=f"Best found f={ms_f:.4f}")
    ax.scatter([0], [0], s=80, color=C["global_opt"], edgecolors="k",
               zorder=5, marker="D", label="Global opt f=0.00")
    ax.set_title("(c) Multi-Start Local Search (30 restarts)")
    ax.set_xlabel("x")
    ax.legend(loc="upper right", fontsize=8)

    # (d) Comparison
    ax = axes[1, 1]
    ls_lucky = _LocalSearch(rastrigin_1d, 0.05)
    _, f_lucky = ls_lucky.run(0.3)
    methods = ["Single LS\n(worst start)", "Single LS\n(lucky start)",
               "Multi-Start\n(30 runs)", "Global\nOptimum"]
    vals = [best_f, f_lucky, ms_f, 0.0]
    colors = [C["trajectory"], C["local_opt"], C["multi_start"], C["global_opt"]]
    bars = ax.bar(methods, vals, color=colors, edgecolor="k", linewidth=0.8,
                  width=0.55)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                f"{v:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax.set_ylabel("Objective Value f(x)")
    ax.set_title("(d) Why We Need Metaheuristics")
    ax.set_ylim(0, max(vals) * 1.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[+] Saved {save_path}")
