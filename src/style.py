"""Shared chart style: one palette for every chart in the project."""
import matplotlib.pyplot as plt

# Colour-blind-checked categorical palette (fixed order — never cycled)
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE = CATEGORICAL[0]
ORANGE = CATEGORICAL[1]
TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.major.size": 0, "ytick.major.size": 0,
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlecolor": TEXT,
        "axes.titlepad": 12, "font.size": 10,
        "axes.prop_cycle": plt.cycler(color=CATEGORICAL),
    })
