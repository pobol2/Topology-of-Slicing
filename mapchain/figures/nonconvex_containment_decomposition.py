"""Non-convex parts: the whole-part path leaves the L-bracket and the torus; decomposed pieces stay inside."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, Point

from mapchain.figures.draw3d import BLUE, L_PLAN, RED, axes3d, load, segs3, solid
from mapchain.volumetric.decomposition import OUT as DECOMPOSITION
from mapchain.volumetric.decomposition import colours
from mapchain.volumetric.parts import geometry


def clip_mesh(mesh, lo, hi):
    """3D axes do not clip to their limits: cut the solid to the close-up box (x, y)."""
    for axis in (0, 1):
        n = np.eye(3)[axis]
        mesh = mesh.slice_plane([lo[0], lo[1], 0], n, cap=True).slice_plane(
            [hi[0], hi[1], 0], -n, cap=True
        )
    return mesh


def box3d(ax, lo, hi, z):
    """Outline of a close-up region, drawn on top of the overview."""
    x0, y0, x1, y1 = (lo[0], lo[1], hi[0], hi[1])
    ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], [z] * 5, color="k", lw=1.2, zorder=7)


def in_box(P, lo, hi):
    ok = np.all((P[:, :2] >= lo[:2]) & (P[:, :2] <= hi[:2]), 1)
    return ok[:-1] & ok[1:]


def closeup(fig, spec, overview, mesh, W, ext, k, inside, zr, faint, margin=0.03, square=False):
    """Close-up of path segment k: box marked on the overview; the segment drawn thick, blue inside the part, red outside."""
    k = np.atleast_1d(k)
    ends = np.r_[k, k[-1] + 1]
    zl, zh = (W[ends].min(0) - margin, W[ends].max(0) + margin)
    if square:
        c, r = ((zl + zh) / 2, (zh - zl)[:2].max() / 2)
        zl[:2], zh[:2] = (c[:2] - r, c[:2] + r)
    box3d(overview, zl, zh, zr[1])
    ax = axes3d(fig, spec, (zl[0], zl[1], zr[0]), (zh[0], zh[1], zr[1]), 1.2)
    box = in_box(W, zl, zh)
    solid(ax, clip_mesh(mesh, zl, zh), 0.1)
    segs3(ax, W, ~ext & box, colors=BLUE, **faint)
    segs3(ax, W, ext & box, colors=RED, linewidths=0.7, alpha=0.5, zorder=2)
    u = np.linspace(0, 1, 300)[:, None]
    for j in k:
        seg = W[j] * (1 - u) + W[j + 1] * u
        ins = np.array([inside(q) for q in seg])
        for run, col in ((ins, BLUE), (~ins, RED)):
            q = seg.copy()
            q[~run] = np.nan
            ax.plot(*q.T, color=col, lw=2.4, zorder=5)
    ax.scatter(
        *W[ends].T,
        color=BLUE,
        s=22,
        edgecolor="k",
        linewidth=0.5,
        depthshade=False,
        zorder=6,
    )
    return (ax, zl, zh)


def draw():
    fig = plt.figure(figsize=(6.3, 4.4))
    g = fig.add_gridspec(2, 3, wspace=0, hspace=0.06)
    cols = colours()
    faint = dict(linewidths=0.5, alpha=0.55, zorder=2)
    lm = geometry("l_box")[0]
    l_box, LW = load("l_box", 4)
    lext = np.array(
        [
            not L_PLAN.buffer(1e-09).contains(LineString([p[:2], q[:2]]))
            for p, q in zip(LW[:-1], LW[1:])
        ]
    )
    assert lext.sum() == l_box["metrics"]["segment_metrics"]["exterior_segment_count"]
    lp = np.load(DECOMPOSITION / "l_box" / "paths.npz")
    pieces = [lp[f"p{j}"] for j in range(int(lp["count"]))]
    assert all(len(lp[f"e{j}"]) == 0 for j in range(len(pieces)))
    lo3, hi3 = ((0.05, 0.05, 0.35), (0.95, 0.95, 0.65))
    a, b = (
        axes3d(fig, g[0, 0], lo3, hi3, 1.2),
        axes3d(fig, g[0, 2], lo3, hi3, 1.2),
    )
    solid(a, lm)
    solid(b, lm)
    segs3(a, LW, ~lext, colors=BLUE, linewidths=0.12, alpha=0.6, zorder=2, rasterized=True)
    segs3(a, LW, lext, colors=RED, linewidths=0.8, zorder=3)
    for j, P in enumerate(pieces):
        segs3(
            b,
            P,
            np.ones(len(P) - 1, bool),
            colors=[cols[j % len(cols)]],
            linewidths=0.12,
            alpha=0.6,
            zorder=2,
            rasterized=True,
        )
    corner = Point(0.35, 0.35)

    def inA(p):
        return p[0] > 0.35 and p[1] < 0.35

    def inB(p):
        return p[0] < 0.35 and p[1] > 0.35

    runs, r = ([], [])
    for j in range(len(lext)):
        if lext[j]:
            r.append(j)
        elif r:
            runs.append(r)
            r = []
    across = [
        r
        for r in runs
        if inB(LW[r[0]]) and inA(LW[r[-1] + 1]) or (inA(LW[r[0]]) and inB(LW[r[-1] + 1]))
    ]
    k = min(
        across,
        key=lambda r: min(LineString([LW[j, :2], LW[j + 1, :2]]).distance(corner) for j in r),
    )
    z, zl, zh = closeup(
        fig,
        g[0, 1],
        a,
        lm,
        LW,
        lext,
        k,
        lambda q: L_PLAN.buffer(1e-09).contains(Point(q[:2])),
        (0.35, 0.65),
        faint,
    )
    tm, tb, _ = geometry("torus")
    tp = np.load(DECOMPOSITION / "torus_m3" / "paths.npz")
    tpieces = [tp[f"p{j}"] for j in range(int(tp["count"]))]
    assert sum(map(len, tpieces)) == 4096

    def torus_ext(P):
        v = np.linspace(0, 1, 60)[None, :, None]
        q = P[:-1, None] * (1 - v) + P[1:, None] * v
        return ~tb.inside_fn(q.reshape(-1, 3)).reshape(len(q), -1).all(1)

    lo3, hi3 = ((0.05, 0.05, 0.275), (0.95, 0.95, 0.725))
    torus, W4 = load("torus", 4)
    u = np.linspace(0, 1, 60)[None, :, None]
    s4 = W4[:-1, None] * (1 - u) + W4[1:, None] * u
    ext4 = ~tb.inside_fn(s4.reshape(-1, 3)).reshape(len(s4), -1).all(1)
    assert ext4.sum() == torus["metrics"]["segment_metrics"]["exterior_segment_count"]
    h, f4 = (
        axes3d(fig, g[1, 0], lo3, hi3, 1.2),
        axes3d(fig, g[1, 2], lo3, hi3, 1.2),
    )
    solid(h, tm)
    solid(f4, tm)
    segs3(h, W4, ~ext4, colors=BLUE, linewidths=0.2, alpha=0.7, zorder=2, rasterized=True)
    segs3(h, W4, ext4, colors=RED, linewidths=1, zorder=3)
    for jj, P in enumerate(tpieces):
        e = torus_ext(P)
        segs3(
            f4,
            P,
            ~e,
            colors=[cols[jj % len(cols)]],
            linewidths=0.2,
            alpha=0.7,
            zorder=2,
            rasterized=True,
        )
        segs3(f4, P, e, colors=RED, linewidths=1, zorder=3)
        print("torus piece", jj, "segments outside the whole torus:", int(e.sum()))
    closeup(
        fig,
        g[1, 1],
        h,
        tm,
        W4,
        ext4,
        int(np.flatnonzero(ext4)[0]),
        lambda q: tb.inside_fn(q[None])[0],
        (0.275, 0.725),
        faint,
        square=True,
    )
    for ax, t in zip((a, z, b), ("Whole part", "Close-up", "Pieces")):
        ax.set_title(t)
    a.text2D(-0.02, 0.5, "L-bracket", rotation=90, ha="right", va="center", transform=a.transAxes)
    h.text2D(-0.02, 0.5, "Torus", rotation=90, ha="right", va="center", transform=h.transAxes)
    fig.subplots_adjust(left=0.04, right=1, top=0.94, bottom=0)
    return fig
