"""Shared figure style (static PNGs for the report and the defence slides, inline in notebooks)."""
import sys

import matplotlib

if "ipykernel" not in sys.modules:  # scripts: draw off-screen; notebooks keep the inline backend
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
# ordinal blue ramp (steps 250 -> 700) for ordered risk groups
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
CRITICAL = "#d03b3b"

MODEL_COLORS = {"IPSS-R": SERIES[1], "IPSS-M": SERIES[0], "Cox (IPSS-M variables, refit)": SERIES[2],
                "Cox (extended)": SERIES[6], "Boosted trees (extended)": SERIES[3]}


def setup():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 10,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2, "legend.frameon": False, "legend.fontsize": 9,
        "text.color": INK, "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def ramp(n):
    """n ordered colours from the blue ramp, light to dark."""
    if n <= len(RAMP):
        idx = [round(i * (len(RAMP) - 1) / max(n - 1, 1)) for i in range(n)]
        return [RAMP[i] for i in idx]
    return RAMP + [RAMP[-1]] * (n - len(RAMP))
