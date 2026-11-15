"""Shared matplotlib style."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = {"NMOS_ON": "#1b6ca8", "NMOS_OFF": "#2a9d8f", "PMOS_ON": "#c8553d", "PMOS_OFF": "#8e5ea2"}


def style():
    plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 9, "legend.frameon": False})
    return plt
