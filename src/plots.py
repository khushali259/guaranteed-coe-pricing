"""Shared chart style: validated categorical palette, thin marks, recessive axes."""
import matplotlib as mpl
import matplotlib.pyplot as plt

from .config import FIG

# fixed categorical order (validated palette, light mode); colour follows the entity
CAT_COLORS = {"A": "#2a78d6", "B": "#eb6834", "E": "#1baf7a"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
RULE_COLORS = {
    "R0 No buffer": "#52514e",
    "R1 Fixed 5%": "#e34948",
    "R2 Rolling percentile": SERIES[0],
    "R3 Newsvendor SARIMAX": SERIES[3],
    "R4 Newsvendor QR": SERIES[1],
    "R5 GBM (stretch)": SERIES[6],
    "R6 Combo M1+M3": SERIES[2],
    "R7 Vol-scaled": SERIES[4],
    "R8 M1 adaptive": SERIES[5],
}
MODEL_COLORS = {"M0": "#52514e", "M1": SERIES[0], "M2": SERIES[3], "M3": SERIES[1],
                "M4": SERIES[6], "M5": SERIES[2], "M6": SERIES[4], "M7": SERIES[5]}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def set_style():
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelsize": 9,
        "ytick.labelsize": 9, "lines.linewidth": 2, "legend.frameon": False,
        "legend.fontsize": 9, "font.size": 10,
    })


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    path = FIG / name
    fig.savefig(path)
    return path


def sgd(x, _=None):
    return f"S${x/1000:,.0f}k"


set_style()
_ = plt  # re-exported for notebooks
