"""Cube, order 2: the layered raster stops between layers, the transported path does not."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from mapchain.figures.draw3d import RED, axes3d, edges, load, segs3
from mapchain.volumetric.parts import geometry


def box_raster(side, lo, hi):
    """Layered raster on the side^3 cell centres of a box: boustrophedon rows, alternate layers reversed (each layer change is one vertical step)."""
    rows = []
    for y in range(side):
        rows += [(x, y) for x in (range(side) if y % 2 == 0 else range(side - 1, -1, -1))]
    cells = np.array(
        [(x, y, k) for k in range(side) for x, y in (rows if k % 2 == 0 else rows[::-1])], float
    )
    return lo + (cells + 0.5) * (np.subtract(hi, lo) / side)


def draw():
    """Cube (models/box.stl), order 2: layered raster vs the computed transported path (results/evaluation/whole_part/box_m2)."""
    u, T = load("box", 2)
    assert u["passed"] and u["metrics"]["exterior_runs"]["count"] == 0
    mesh = geometry("box")[0]
    lo, hi = mesh.bounds
    P = box_raster(4, lo, hi)
    TEAL, NAVY = ("#00897b", "#2f5f8a")
    fig = plt.figure(figsize=(6.3, 3.3))
    for n, (Q, title) in enumerate(((P, "(a) Layered raster"), (T, "(b) Volumetric path"))):
        ax = axes3d(fig, 121 + n, lo, hi, zoom=1.1, view=(10, -62))
        edges(ax, mesh, colors="0.7", linewidths=0.6)
        vertical = np.abs(np.diff(Q[:, 2])) > np.hypot(*np.diff(Q[:, :2], axis=0).T)
        segs3(ax, Q, ~vertical, colors=NAVY, linewidths=1.3, zorder=2)
        if n == 0:
            segs3(
                ax,
                Q,
                vertical,
                colors=RED,
                linewidths=2,
                linestyles=(0, (2, 1.5)),
                zorder=3,
            )
        else:
            segs3(ax, Q, vertical, colors=TEAL, linewidths=2.2, zorder=3)
        ax.scatter(*Q.T, color=NAVY, s=7, depthshade=False, zorder=4)
        ax.scatter(
            *Q[0], s=45, facecolor="white", edgecolor="k", linewidth=1.2, depthshade=False, zorder=5
        )
        ax.scatter(*Q[-1], s=30, marker="s", color="k", depthshade=False, zorder=5)
        ax.set_title(title)
    fig.subplots_adjust(left=0, right=1, wspace=0, top=0.92, bottom=0)
    return fig
