"""Colours shared by every figure.

Roles:
- PATH: a single tool path drawn in one colour (blue, far from the fault red).
- TIME: traversal time t, light (early) to dark (late), wherever t is shown.
- spread(n): n categorical roles that overlap (sheets, pieces), far apart in hue.
- RED: faults only (resets, excursions, interruptions); no other role uses a red hue.
- DARK: symbols (start, end, target, base), zoom boxes and connectors.
- GRAY and the neutrals: outlines, edges, grids, obstacles, domains.
"""

from __future__ import annotations

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Categorical, ordered by hue (moderately saturated).
ORANGE = "#C55A18"
OCHRE = "#B48511"
OLIVE = "#6B8C1D"
GREEN = "#229344"
TEAL = "#138E84"
CYAN = "#1586B1"
BLUE = "#3265BD"
PURPLE = "#7849AE"
MAGENTA = "#AE3F8B"

CATEGORICAL = (ORANGE, OCHRE, OLIVE, GREEN, TEAL, CYAN, BLUE, PURPLE, MAGENTA)
# Hues next to the fault red (orange, magenta) are left out unless more roles need them.
AWAY_FROM_RED = (OCHRE, OLIVE, GREEN, TEAL, CYAN, BLUE, PURPLE)

# Reserved and neutral.
RED = "#C6455A"
GRAY = "#6B7280"
# Markers and frames: start/end symbols, targets, zoom boxes and their connectors.
DARK = "#374151"

# Neutrals: text and outlines, leaders, mesh edges, light grids, obstacle and domain fills.
INK, MID, EDGE, FAINT = ("0.25", "0.45", "0.7", "0.86")
OBSTACLE, OBSTACLE_EDGE, DOMAIN, DOMAIN_EDGE = ("#8c8c8c", "#5c5c5c", "#e4e4e4", "#9a9a9a")

# Sequential: light cyan to dark blue. The light end is still visible on white, and the
# lightness range is wide enough to read early from late on shaded 3D tubes.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "CyanBlue", ["#83D6EC", CYAN, BLUE, "#294366"], N=256
)

PATH = BLUE
TIME = SEQUENTIAL


def categorical(i):
    """The i-th categorical colour in hue order, cycling."""
    return CATEGORICAL[int(i) % len(CATEGORICAL)]


def spread(n):
    """n colours for roles that overlap: evenly spaced along the hue order, then ordered
    by a stride near n / 2 so that consecutive roles (neighbouring pieces, adjacent
    sheets) are also far apart in hue. Uses AWAY_FROM_RED while it has enough colours;
    cycles past len(CATEGORICAL)."""
    colours = AWAY_FROM_RED if n <= len(AWAY_FROM_RED) else CATEGORICAL
    k = len(colours)
    m = min(n, k)
    picked = np.round(np.arange(m) * k / m).astype(int) % k
    stride = next(s for s in range(max(1, m // 2), m + 1) if np.gcd(s, m) == 1)
    order = [colours[picked[i * stride % m]] for i in range(m)]
    return tuple(order[i % m] for i in range(n))


def piece_shades(j, n, length):
    """RGB of `length` points of piece j of n: the piece's spread() colour, from a light
    tint at the piece's start to a dark shade at its end, so pieces differ by hue and
    traversal time reads as lightness within each."""
    from matplotlib.colors import to_rgb

    base = np.array(to_rgb(spread(n)[j]))
    t = np.linspace(0, 1, length)[:, None]
    light, dark = (base + 0.45 * (1 - base), 0.55 * base)
    return np.where(t < 0.5, light + 2 * t * (base - light), base + (2 * t - 1) * (dark - base))
