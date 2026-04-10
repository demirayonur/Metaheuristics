"""
Shared plotting configuration and colour palette.
"""

import matplotlib.pyplot as plt

# ── Colour palette ──────────────────────────────────────────────────
COLORS = {
    "landscape": "#2c3e50",
    "trajectory": "#e74c3c",
    "global_opt": "#2ecc71",
    "local_opt": "#f39c12",
    "multi_start": "#3498db",
    "sa": "#e74c3c",
    "ga": "#e74c3c",
    "alns": "#e74c3c",
    "gurobi": "#2ecc71",
    "current": "#3498db",
    "best": "#e74c3c",
    "temperature": "#f39c12",
    "purple": "#9b59b6",
    "teal": "#1abc9c",
    "dark": "#34495e",
    "grey": "#7f8c8d",
}


def apply_style() -> None:
    """Apply the tutorial's global matplotlib style."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "figure.dpi": 180,
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
    })
