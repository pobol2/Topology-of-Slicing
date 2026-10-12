"""Planar 3R figure: geometry-first versus kinematics-first posture resets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mapchain import store
from mapchain.kinematics.arm import (
    OBSTACLES,
    RMAX,
    RMIN,
    Collision,
    forward,
    joints,
    workspace,
)
from mapchain.kinematics.sheets import OUT, fibre_seed

FIG = store.FIGURES
BLUE, ORANGE, RED = ("#1f77b4", "#ff7f0e", "#d62728")
FAMILY = (BLUE, ORANGE)
POSTURES = ("Down posture", "Up posture")
DENSITY = "uniform"


def load():
    result = json.loads((OUT / "results.json").read_text())
    order = max(result["orders"])
    row = {(r["method"], r["order"], r["density"]): r for r in result["requests"]}
    data = {
        m: dict(np.load(OUT / f"{m}_m{order}_{DENSITY}.npz")) for m in ("geometry", "kinematics")
    }
    return (result, order, row, data)


def chart(x):
    """Tool positions -> (phi in degrees, radius)."""
    x = np.asarray(x)
    return (np.rad2deg(np.arctan2(x[..., 1], x[..., 0])), np.linalg.norm(x, axis=-1))


def shade(ax, grid, alpha=0.22):
    """Sampled availability classes: blue only, orange only, both (hatched)."""
    from matplotlib.colors import to_rgba

    cls = grid["available"][..., 0] * 1 + grid["available"][..., 1] * 2
    rgba = np.zeros(cls.shape + (4,))
    rgba[cls == 1] = to_rgba(BLUE, alpha)
    rgba[cls == 2] = to_rgba(ORANGE, alpha)
    rgba[cls == 3] = to_rgba("white", 0)
    dphi, dr = (np.diff(grid["phi"][:2])[0], np.diff(grid["radius"][:2])[0])
    ax.imshow(
        rgba,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        zorder=0,
        extent=(-30 - dphi / 2, 30 + dphi / 2, RMIN - dr / 2, RMAX + dr / 2),
    )
    both = (cls == 3).astype(float)
    if both.any():
        import matplotlib as mpl

        with mpl.rc_context({"hatch.linewidth": 0.5}):
            cs = ax.contourf(
                grid["phi"],
                grid["radius"],
                both,
                levels=[0.5, 1.5],
                colors=[to_rgba(BLUE, alpha * 0.6)],
                hatches=["////"],
                zorder=0.5,
            )
            cs.set_edgecolor(to_rgba(ORANGE, 0.55))
    return cls


def stretches(phi, radius, breaks):
    """Continuous index ranges; reset edges are never drawn."""
    cut = np.r_[0, np.flatnonzero(breaks) + 1, len(phi)]
    return list(zip(cut[:-1], cut[1:]))


def panel_a(ax, split):
    from matplotlib.patches import Circle, Wedge

    ax.add_patch(
        Wedge(
            (0, 0),
            RMAX,
            -30,
            30,
            width=RMAX - RMIN,
            fc="#e4e4e4",
            ec="#9a9a9a",
            lw=0.6,
            zorder=0,
        )
    )
    for cx, cy, rad in OBSTACLES:
        ax.add_patch(
            Circle((cx, cy), rad, fc="#8c8c8c", ec="#5c5c5c", lw=0.6, alpha=0.85, zorder=1)
        )
    x = workspace([0.5, (split + 30) / 60])
    collision = Collision(OBSTACLES)
    for family in (0, 1):
        q = fibre_seed(x, family, None, collision)
        pts = joints(q)
        ax.plot(*pts.T, "-", color=FAMILY[family], lw=2.2, solid_capstyle="round", zorder=3)
        ax.plot(
            *pts[1:-1].T,
            "o",
            color="white",
            mec=FAMILY[family],
            ms=3.5,
            mew=1.2,
            zorder=4,
        )
    ax.plot(0, 0, "s", color=".2", ms=4, zorder=5)
    ax.plot(*x, "*", color="k", ms=8, zorder=6)
    ax.set(
        aspect="equal", xlim=(-0.08, 2.0), ylim=(-1.02, 1.0), xticks=[0, 1, 2], yticks=[-1, 0, 1]
    )
    return x


def panel_b(ax, grid, result):
    """Sampled availability (shading, hatched where both), certified map domains (outlines), split."""
    from matplotlib.patches import Rectangle

    shade(ax, grid, alpha=0.2)
    for f, p in enumerate(result["preprocessing"]):
        lo, hi = p["phi_range_degrees"]
        ax.add_patch(
            Rectangle(
                (lo, RMIN),
                hi - lo,
                RMAX - RMIN,
                fill=False,
                ec=FAMILY[f],
                lw=1.1,
                ls="-" if f == 0 else (0, (4, 2)),
                zorder=3,
                clip_on=False,
            )
        )
    ax.axvline(result["split_phi_degrees"], color="k", lw=0.8, ls=(0, (3, 2)), zorder=4)
    ax.spines[:].set_visible(False)
    ax.set(
        xlim=(-30, 30),
        ylim=(RMIN, RMAX),
        xlabel="$\\varphi$ (deg)",
        ylabel="$r$",
        xticks=[-30, 0, 30],
        yticks=[1.6, 1.85],
    )


def path_panel(ax, data, grid, kind, split, connectors=()):
    phi_all, r_all = chart(forward(data["q"]))
    breaks, fam = (data["break_edges"], data["family"])
    if kind == "geometry":
        shade(ax, grid, alpha=0.2)
    else:
        for a, b, f in ((-30, split, 0), (split, 30, 1)):
            ax.axvspan(a, b, color=FAMILY[f], alpha=0.12, lw=0, zorder=0)
    for a, b in stretches(phi_all, r_all, breaks):
        marks = sorted({a, b - 1} | {i for c in connectors for i in c if a <= i < b})
        for u, v in zip(marks[:-1], marks[1:]):
            if any(c[0] <= u and v <= c[1] for c in connectors):
                continue
            dashed = False
            ax.plot(
                phi_all[u : v + 1],
                r_all[u : v + 1],
                color=FAMILY[int(fam[a])],
                lw=0.8 if dashed else 0.7,
                ls=(0, (4, 2)) if dashed else "-",
                zorder=2,
            )
    reset = np.flatnonzero(breaks)
    for k, e in enumerate(reset, 1):
        ax.plot(
            phi_all[e],
            r_all[e],
            marker="X",
            ms=7,
            color=RED,
            mec="white",
            mew=0.6,
            zorder=5,
            ls="none",
            clip_on=False,
        )
        ax.annotate(
            str(k),
            (phi_all[e], r_all[e]),
            (5, -9),
            textcoords="offset points",
            color=RED,
            fontsize=8,
            weight="bold",
            bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.6),
            zorder=6,
        )
    ax.plot(
        phi_all[0], r_all[0], "o", color="k", ms=5, mec="white", mew=0.6, zorder=5, clip_on=False
    )
    ax.plot(
        phi_all[-1], r_all[-1], "s", color="k", ms=5, mec="white", mew=0.6, zorder=5, clip_on=False
    )
    ax.set(
        xlim=(-30, 30),
        ylim=(RMIN, RMAX),
        xlabel="$\\varphi$ (deg)",
        ylabel="$r$",
        xticks=[-30, 0, 30],
        yticks=[1.6, 1.85],
    )
    ax.set_yticklabels(["1.60", "1.85"])
    ax.tick_params(labelsize=7)
    return len(reset)


def draw(destination=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.titlelocation": "left",
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
        }
    )
    destination = Path(destination or FIG / "three_link_posture_comparison.pdf")
    result, order, row, data = load()
    grid = dict(np.load(OUT / "availability_grid.npz"))
    split = result["split_phi_degrees"]
    g, k = (
        row["geometry", order, DENSITY],
        row["kinematics", order, DENSITY],
    )
    fig = plt.figure(figsize=(7.2, 7.0), layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0.04, 1, 0.96))
    outer = fig.add_gridspec(2, 2, height_ratios=[1, 1])
    ax_a = fig.add_subplot(outer[0, 0])
    panel_a(ax_a, split)
    ax_a.set_title("(a) Robot and task")
    ax_b = fig.add_subplot(outer[0, 1])
    panel_b(ax_b, grid, result)
    ax_b.set_title("(b) Safe patches")
    ax_c, ax_d = (fig.add_subplot(outer[1, 0]), fig.add_subplot(outer[1, 1], sharex=None))
    nc = path_panel(ax_c, data["geometry"], grid, "geometry", split)
    kd = data["kinematics"]
    ids = kd["waypoint_indices"]
    L = k["lower_waypoint_count"]
    e = int(np.flatnonzero(kd["break_edges"])[0])
    nd = path_panel(
        ax_d, kd, grid, "kinematics", split, connectors=[(int(ids[L - 1]), e), (e + 1, int(ids[L]))]
    )
    assert nc == g["switches"] and nd == k["switches"]
    ax_c.set_title("(c) Geometry-first")
    ax_d.set_title("(d) Kinematics-first")
    handles = [
        Line2D([], [], color=BLUE, lw=2, label=POSTURES[0]),
        Line2D([], [], color=ORANGE, lw=2, label=POSTURES[1]),
        Line2D([], [], marker="X", color=RED, ls="none", ms=7, label="reset"),
        Line2D([], [], marker="o", color="k", ls="none", ms=5, label="start"),
        Line2D([], [], marker="s", color="k", ls="none", ms=5, label="end"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=5,
        fontsize=7.5,
        frameon=False,
        handlelength=1.6,
        columnspacing=1.4,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=300)
    plt.close(fig)
