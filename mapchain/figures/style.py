"""Matplotlib defaults and the orthographic camera shared by the 3D figures."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "text.usetex": False,
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "axes.linewidth": 0.6,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "lines.linewidth": 1.0,
        "figure.dpi": 120,
        "savefig.dpi": 600,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})
MORPH_CAMERA_POSITION = np.array([2.6, -3.0, 4.4])
MORPH_CAMERA_FOCUS = np.full(3, 0.5)
_MORPH_VIEW = MORPH_CAMERA_POSITION - MORPH_CAMERA_FOCUS
MORPH_ELEVATION = float(np.degrees(np.arctan2(_MORPH_VIEW[2], np.hypot(*_MORPH_VIEW[:2]))))
MORPH_AZIMUTH = float(np.degrees(np.arctan2(_MORPH_VIEW[1], _MORPH_VIEW[0])))


def set_morph_camera(ax) -> None:
    """Match the orthographic teaser camera, with z pointing upward."""
    ax.view_init(elev=MORPH_ELEVATION, azim=MORPH_AZIMUTH)
    ax.set_proj_type("ortho")
