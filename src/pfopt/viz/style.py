"""Shared matplotlib style, colour palette and figure saving.

One style for every figure: 6.3 in wide at most (fits A4 with margins), readable at report scale,
colour-blind-safe Okabe-Ito palette, one fixed colour per method across all figures, 300 dpi PNG
plus vector PDF.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

FIG_WIDTH_IN = 6.3
DPI = 300

# Okabe-Ito colour-blind-safe palette.
PALETTE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#000000",
    "grey": "#7F7F7F",
    "wine": "#882255",  # Tol palette, added for the two Markowitz strategies
    "teal": "#44AA99",
}

# One colour per method, identical in every figure. Extension methods are added when needed.
METHOD_COLORS = {
    "EW": PALETTE["black"],
    "GMV-SAMPLE": PALETTE["vermillion"],
    "GMV-LW": PALETTE["orange"],
    "GMV-RMT": PALETTE["blue"],
    "GMV-PCA": PALETTE["green"],
    "HRP": PALETTE["purple"],
    "HRP-RMT": PALETTE["sky"],
    "MV-SAMPLE": PALETTE["wine"],
    "MV-RMT": PALETTE["teal"],
    "SPX": PALETTE["grey"],
}


# Covariance estimators carry the colour of the GMV strategy built on them.
ESTIMATOR_COLORS = {
    "Sample": METHOD_COLORS["GMV-SAMPLE"],
    "Ledoit-Wolf": METHOD_COLORS["GMV-LW"],
    "RMT": METHOD_COLORS["GMV-RMT"],
    "PCA": METHOD_COLORS["GMV-PCA"],
}


def apply_style() -> None:
    """Apply the project-wide matplotlib rc settings."""
    mpl.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.2,
            "axes.prop_cycle": mpl.cycler(color=list(PALETTE.values())),
            "figure.constrained_layout.use": True,
            "savefig.dpi": DPI,
            "pdf.fonttype": 42,
        }
    )


def save_figure(fig: Figure, name: str, out_dir: str | Path) -> list[Path]:
    """Save ``fig`` as 300 dpi PNG and vector PDF under ``out_dir``, then close it.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save; its width must not exceed :data:`FIG_WIDTH_IN`.
    name : str
        File stem, following ``<stage>_<name>`` (for example ``eda_mp_spectrum``).
    out_dir : str or Path
        Target directory (created if needed).

    Returns
    -------
    list of Path
        The PNG and PDF paths.

    Raises
    ------
    ValueError
        If the figure is wider than the allowed width.
    """
    if fig.get_figwidth() > FIG_WIDTH_IN + 1e-9:
        raise ValueError(f"figure width {fig.get_figwidth():.2f} in exceeds {FIG_WIDTH_IN} in")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"{name}.png", out / f"{name}.pdf"]
    for path in paths:
        fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return paths
