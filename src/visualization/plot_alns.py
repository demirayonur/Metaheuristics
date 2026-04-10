"""
Visualisation for Module 6 — ALNS results.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..problems.vrp import VRPData, VRPSolution
from ..algorithms.alns import ALNS
from .style import COLORS, apply_style


def _draw_routes(ax, coords, routes, title, distance, cmap_name="tab10"):
    """Draw VRP routes on an axes."""
    cmap = plt.colormaps[cmap_name]
    ax.scatter(*coords[0], s=200, marker="s", color="k", zorder=10, label="Depot")
    for ri, route in enumerate(routes):
        color = cmap(ri % 10)
        pts = [coords[0]] + [coords[c] for c in route] + [coords[0]]
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        ax.plot(xs, ys, "-o", color=color, lw=1.8, markersize=6,
                markeredgecolor="k", markeredgewidth=0.5,
                label=f"Route {ri+1}", zorder=4)
        for c in route:
            ax.text(coords[c, 0] + 1.5, coords[c, 1] + 1.5,
                    str(c), fontsize=7, color="#333")
    ax.set_title(f"{title}\nTotal distance = {distance:.2f}")
    ax.legend(fontsize=7, loc="upper right"); ax.set_aspect("equal")


def plot_alns_results(
    alns: ALNS,
    alns_sol: VRPSolution,
    alns_dist: float,
    gurobi_routes: list[list[int]],
    gurobi_dist: float,
    data: VRPData,
    save_path: str = "docs/figures/module6_alns_vrp.png",
) -> None:
    """
    Six-panel ALNS analysis figure.

    (a) ALNS routes.
    (b) Gurobi optimal routes.
    (c) Convergence.
    (d) Operator weight evolution.
    (e) Distance bar comparison.
    (f) SA acceptance temperature (log scale).
    """
    apply_style()
    C = COLORS
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.38, wspace=0.28)
    fig.suptitle("Module 6 — ALNS for Capacitated VRP",
                 fontsize=16, fontweight="bold", y=0.98)

    # (a) ALNS routes
    ax = fig.add_subplot(gs[0, 0])
    _draw_routes(ax, data.coords, alns_sol.routes, "(a) ALNS Solution", alns_dist)

    # (b) Gurobi routes
    ax = fig.add_subplot(gs[0, 1])
    _draw_routes(ax, data.coords, gurobi_routes, "(b) Gurobi Optimal", gurobi_dist)

    # (c) Convergence
    ax = fig.add_subplot(gs[1, 0])
    iters = range(len(alns.history_best))
    ax.plot(iters, alns.history_current, color=C["current"], alpha=0.2,
            lw=0.5, label="Current")
    ax.plot(iters, alns.history_best, color=C["alns"], lw=2, label="Best")
    ax.axhline(gurobi_dist, color=C["gurobi"], ls="--", lw=2,
               label=f"Optimal = {gurobi_dist:.1f}")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Total Distance")
    ax.set_title("(c) ALNS Convergence"); ax.legend(fontsize=8)

    # (d) Operator weights
    ax = fig.add_subplot(gs[1, 1])
    if alns.d_weight_history:
        segs = range(len(alns.d_weight_history))
        d_cols = ["#e74c3c", "#f39c12", "#9b59b6"]
        for k, op in enumerate(alns.destroy_ops):
            ws = [h[k] for h in alns.d_weight_history]
            ax.plot(segs, ws, "-o", markersize=3, color=d_cols[k % 3],
                    label=f"D: {op.name}", lw=1.5)
        r_cols = ["#2ecc71", "#3498db"]
        for k, op in enumerate(alns.repair_ops):
            ws = [h[k] for h in alns.r_weight_history]
            ax.plot(segs, ws, "-s", markersize=3, color=r_cols[k % 2],
                    label=f"R: {op.name}", lw=1.5)
    ax.set_xlabel("Segment"); ax.set_ylabel("Operator Weight")
    ax.set_title("(d) Adaptive Operator Weights"); ax.legend(fontsize=7)

    # (e) Bar comparison
    ax = fig.add_subplot(gs[2, 0])
    vals = [alns_dist, gurobi_dist]
    bars = ax.bar(["ALNS", "Gurobi\n(Optimal)"], vals,
                  color=[C["alns"], C["gurobi"]], edgecolor="k",
                  linewidth=0.8, width=0.45)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{v:.1f}", ha="center", va="bottom", fontweight="bold", fontsize=12)
    gap = 100 * (alns_dist / gurobi_dist - 1) if gurobi_dist > 0 else 0
    ax.set_title(f"(e) ALNS vs Gurobi  (gap = {gap:.2f}%)")
    ax.set_ylabel("Total Distance"); ax.set_ylim(0, max(vals) * 1.2)

    # (f) Temperature
    ax = fig.add_subplot(gs[2, 1])
    ax.plot(range(len(alns.history_temps)), alns.history_temps,
            color=C["temperature"], lw=1)
    ax.set_xlabel("Iteration"); ax.set_ylabel("Temperature")
    ax.set_title("(f) SA Acceptance Temperature"); ax.set_yscale("log")

    fig.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[+] Saved {save_path}")
