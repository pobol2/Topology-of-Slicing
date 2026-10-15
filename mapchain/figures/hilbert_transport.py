"""Two-dimensional sketch of a Hilbert traversal carried by a computed optimal-transport map."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.path import Path as MplPath

from mapchain.figures.palette import DARK, FAINT, INK, MID, TIME
from mapchain.volumetric.hilbert import hilbert_2d

GRID, OUTLINE, LEADER = (FAINT, INK, MID)
SIDE = 4.4
CENTRE = np.array([10.65, 2.2])


def target_boundary(samples=720):
    """Convex target of unit area: a perturbed, sheared disk."""
    theta = np.linspace(0, 2 * np.pi, samples, endpoint=False)
    r = 1 + 0.075 * np.cos(3 * theta) + 0.045 * np.sin(2 * theta)
    boundary = np.c_[r * np.cos(theta) + 0.12 * r * np.sin(theta), 0.925 * r * np.sin(theta)]
    x, y = boundary.T
    area = 0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    return boundary / np.sqrt(area) + 0.5


def transport_map(n=40, refine=3, eps_final=8e-4, levels=6, iterations=60):
    """Entropic OT map from the uniform unit square to the uniform target.

    Sinkhorn between n x n source cell centres and target cell centres on a grid refined
    `refine` times, with eps annealed geometrically. The map is the soft Laguerre assignment
    x -> E[y | x] at temperature eps / 4: sharper than the entropic plan, so the image of the
    square reaches the target boundary (gap below 0.012), yet still smooth."""
    boundary = target_boundary()
    source = (np.stack(np.meshgrid(np.arange(n), np.arange(n)), -1).reshape(-1, 2) + 0.5) / n
    lo, hi = boundary.min(0), boundary.max(0)
    h = 1 / (refine * n)
    grid = np.stack(np.meshgrid(*(np.arange(a, b, h) for a, b in zip(lo, hi))), -1)
    grid = grid.reshape(-1, 2) + h / 2
    target = grid[MplPath(boundary).contains_points(grid)]
    a, b = np.full(len(source), 1 / len(source)), np.full(len(target), 1 / len(target))
    cost = 0.5 * ((source[:, None] - target[None]) ** 2).sum(-1)
    f, g = np.zeros(len(source)), np.zeros(len(target))
    for eps in np.geomspace(0.03, eps_final, levels):
        kernel = np.exp((f[:, None] + g[None] - cost) / eps)
        u, v = np.ones(len(source)), np.ones(len(target))
        for _ in range(iterations):
            u = a / (kernel @ v)
            v = b / (kernel.T @ u)
        f, g = f + eps * np.log(u), g + eps * np.log(v)

    def phi(points):
        points = np.atleast_2d(points)
        w = (g[None] - 0.5 * ((points[:, None] - target[None]) ** 2).sum(-1)) / (eps / 4)
        w = np.exp(w - w.max(1, keepdims=True))
        return (w @ target) / w.sum(1, keepdims=True)

    return phi


def draw():
    phi = transport_map()

    def left(p):
        return SIDE * np.asarray(p)

    def right(p):
        return CENTRE + SIDE * (phi(p) - 0.5)

    fig = plt.figure(figsize=(6.3, 4.5))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    ax.set_aspect("equal")
    s = np.linspace(0, 1, 160)
    for v in (0.25, 0.5, 0.75):
        for line in (np.c_[s, np.full_like(s, v)], np.c_[np.full_like(s, v), s]):
            ax.plot(*left(line).T, color=GRID, lw=0.45, zorder=1)
            ax.plot(*right(line).T, color=GRID, lw=0.45, zorder=1)
    edge = np.concatenate(
        [np.c_[s, 0 * s], np.c_[1 + 0 * s, s], np.c_[1 - s, 1 + 0 * s], np.c_[0 * s, 1 - s]]
    )
    target = CENTRE + SIDE * (target_boundary() - 0.5)
    for outline in (left(edge), np.r_[target, target[:1]]):
        ax.plot(*outline.T, color=OUTLINE, lw=0.9, zorder=2)
    corners = hilbert_2d(2)
    path = np.concatenate(
        [np.linspace(p, q, 40, endpoint=False) for p, q in zip(corners[:-1], corners[1:])]
        + [corners[-1:]]
    )
    t = np.linspace(0, 1, len(path) - 1)[:, None]
    colours = TIME(t[:, 0])
    for side in (left, right):
        p = side(path)
        ax.add_collection(
            LineCollection(
                np.stack([p[:-1], p[1:]], 1),
                colors=colours,
                lw=2.1,
                capstyle="round",
                joinstyle="round",
                zorder=3,
            )
        )
    marker = dict(mec=DARK, mew=0.8, zorder=5, ls="none")
    for cell, shape, size in (((0.5, 1.5), "o", 5), ((1.5, 2.5), "s", 5), ((3.5, 2.5), "D", 3.6)):
        for side in (left, right):
            ax.plot(*side(np.array([cell]) / 4).T, marker=shape, ms=size, mfc="white", **marker)
    for side, label in ((left, r"$\bm{\upxi}(t)$"), (right, r"$\Phi(\bm{\upxi}(t))$")):
        p = side(np.array([[1.5, 2.5]]) / 4)[0]
        ax.plot(*np.c_[p + (0.08, -0.1), p + (0.25, -0.38)], color=LEADER, lw=0.5)
        ax.text(
            *(p + (0.3, -0.4)),
            label,
            ha="center",
            va="top",
            zorder=6,
            bbox=dict(fc="white", ec="none", pad=1.0),
        )
    km = min(0.0, target[:, 1].min()) - 0.05
    for cell, name, fill in (((0.5, 0.5), "Start", DARK), ((3.5, 0.5), "End", "white")):
        for side in (left, right):
            p = side(np.array([cell]) / 4)[0]
            ax.plot(*p, marker="o", ms=6, mfc=fill, **marker)
            ax.plot([p[0], p[0]], [p[1] - 0.12, km], color=LEADER, lw=0.5, zorder=1)
            ax.text(p[0], km - 0.1, name, ha="center", va="top")
    for x, title in ((2.2, "Source"), (CENTRE[0], "Target")):
        ax.text(x, 5.05, rf"\textbf{{{title}}}", ha="center", va="center")
    ax.annotate(
        "",
        xy=(7.65, 2.2),
        xytext=(5.15, 2.2),
        arrowprops=dict(
            arrowstyle="-|>,head_length=0.6,head_width=0.25",
            color=INK,
            lw=0.9,
            shrinkA=0,
            shrinkB=0,
        ),
    )
    ax.text(6.4, 2.6, r"$\Phi$", ha="center", va="center")
    ax.text(6.4, 1.8, "Geometry map", ha="center", va="center")
    ax.text(2.2, km - 0.8, r"$\bm{\upxi}$", ha="center", va="center")
    ax.text(CENTRE[0], km - 0.8, r"$\bm{\mathrm{x}}=\Phi\circ\bm{\upxi}$", ha="center", va="center")
    ax.set(xlim=(-0.1, target[:, 0].max() + 0.1), ylim=(km - 1.1, 5.3))
    return fig
