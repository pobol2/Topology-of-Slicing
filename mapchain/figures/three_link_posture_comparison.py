"""Planar 3R figure: geometry-first versus kinematics-first posture resets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mapchain import store
from mapchain.figures.palette import (
    DARK,
    DOMAIN,
    DOMAIN_EDGE,
    OBSTACLE,
    OBSTACLE_EDGE,
    RED,
    categorical,
    spread,
)
from mapchain.kinematics.arm import (
    DEG,
    OBSTACLES,
    RMAX,
    RMIN,
    Collision,
    band_limits,
    forward,
    joints,
    workspace,
)
from mapchain.kinematics.sheets import OUT, fibre_seed

FIG = store.FIGURES
SHOWN = []  # sheets drawn, in legend order; set by draw()


def colour(sheet):
    """One colour per sheet, shared by all panels and both pipelines; the shown sheets
    take spread() colours in legend order, so adjacent sheets differ most."""
    sheet = int(sheet)
    return spread(len(SHOWN))[SHOWN.index(sheet)] if sheet in SHOWN else categorical(sheet)


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
    """Sampled availability: each sheet's colour where it alone is available, and the mix
    of the sheets' colours where several are."""
    from matplotlib.colors import to_rgb

    available = grid["available"]
    count = available.sum(-1)
    rgb = np.array([to_rgb(colour(f)) for f in range(available.shape[-1])])
    mixed = (available @ rgb) / np.maximum(count, 1)[..., None]
    rgba = np.concatenate((mixed, np.where(count > 0, alpha, 0.0)[..., None]), -1)
    dphi, dr = (np.diff(grid["phi"][:2])[0], np.diff(grid["radius"][:2])[0])
    ax.imshow(
        rgba,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        zorder=0,
        extent=(-DEG - dphi / 2, DEG + dphi / 2, RMIN - dr / 2, RMAX + dr / 2),
    )
    return count


def radial(s):
    """Radius of radial chart coordinate s (workspace() is linear in r squared)."""
    return np.sqrt(RMIN**2 + np.asarray(s) * (RMAX**2 - RMIN**2))


def handoff_lines(result, s):
    """Each handoff curve's angle (degrees) at chart coordinates s."""
    return [np.interp(s, result["handoff_s"], c) for c in result["handoff_curves"]]


def stretches(phi, radius, breaks):
    """Continuous index ranges; reset edges are never drawn."""
    cut = np.r_[0, np.flatnonzero(breaks) + 1, len(phi)]
    return list(zip(cut[:-1], cut[1:]))


def panel_a(ax, splits, sheets):
    from matplotlib.patches import Circle, Wedge

    ax.add_patch(
        Wedge(
            (0, 0),
            RMAX,
            -DEG,
            DEG,
            width=RMAX - RMIN,
            fc=DOMAIN,
            ec=DOMAIN_EDGE,
            lw=0.6,
            zorder=0,
        )
    )
    for cx, cy, rad in OBSTACLES:
        ax.add_patch(
            Circle((cx, cy), rad, fc=OBSTACLE, ec=OBSTACLE_EDGE, lw=0.6, alpha=0.85, zorder=1)
        )
    split = splits[len(splits) // 2]
    x = workspace([0.5, (split + DEG) / (2 * DEG)])
    collision = Collision(OBSTACLES)
    for family in range(len(splits) + 1):
        try:
            q = fibre_seed(x, family, None, collision)
        except RuntimeError:
            continue
        pts = joints(q)
        ax.plot(*pts.T, "-", color=colour(sheets[family]), lw=2.2, solid_capstyle="round", zorder=3)
        ax.plot(
            *pts[1:-1].T,
            "o",
            color="white",
            mec=colour(sheets[family]),
            ms=3.5,
            mew=1.2,
            zorder=4,
        )
    ax.plot(0, 0, "s", color=DARK, ms=4, zorder=5)
    ax.plot(*x, "*", color=DARK, ms=8, zorder=6)
    span = RMAX * np.sin(np.deg2rad(DEG)) + 0.1
    ax.set(ylim=(-span, span), yticks=[-2, -1, 0, 1, 2], xlabel="$x$", ylabel="$y$")
    # Equal scale, but widen the limits rather than shrink the box: same width as (b)-(d).
    ax.set_aspect("equal", adjustable="datalim")
    return x


def panel_b(ax, grid, result):
    """Sampled availability (shading, mixed where several), certified map domains (outlines), split."""
    from matplotlib.patches import Polygon

    shade(ax, grid, alpha=0.3)
    s = np.linspace(0, 1, 60)
    for f, p in enumerate(result["preprocessing"]):
        lo, hi = band_limits(np.array(p["band"]), s)
        outline = np.r_[np.c_[lo, radial(s)], np.c_[hi, radial(s)][::-1]]
        ax.add_patch(
            Polygon(
                outline,
                closed=True,
                fill=False,
                ec=colour(result["chain_sheets"][f]),
                lw=1.1,
                ls="-" if f % 2 == 0 else (0, (4, 2)),
                zorder=3,
                clip_on=False,
            )
        )
    for line in handoff_lines(result, s):
        ax.plot(line, radial(s), color=DARK, lw=0.8, ls=(0, (3, 2)), zorder=4)
    ax.spines[:].set_visible(False)
    ax.set(
        xlim=(-DEG, DEG),
        ylim=(RMIN, RMAX),
        xlabel="$\\varphi$ (deg)",
        ylabel="$r$",
        xticks=[-DEG, 0, DEG],
        yticks=[RMIN, RMAX],
    )


def path_panel(ax, data, grid, kind, result, connectors=()):
    phi_all, r_all = chart(forward(data["q"]))
    breaks, fam = (data["break_edges"], data["family"])
    if kind == "kinematics":
        fam = np.asarray(result["chain_sheets"])[fam]
    if kind == "geometry":
        shade(ax, grid, alpha=0.2)
    else:
        s = np.linspace(0, 1, 60)
        edges = [np.full_like(s, -DEG), *handoff_lines(result, s), np.full_like(s, DEG)]
        for f, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            ax.fill_betweenx(
                radial(s), a, b, color=colour(result["chain_sheets"][f]), alpha=0.12, lw=0, zorder=0
            )
    for a, b in stretches(phi_all, r_all, breaks):
        marks = sorted({a, b - 1} | {i for c in connectors for i in c if a <= i < b})
        for u, v in zip(marks[:-1], marks[1:]):
            if any(c[0] <= u and v <= c[1] for c in connectors):
                continue
            dashed = False
            ax.plot(
                phi_all[u : v + 1],
                r_all[u : v + 1],
                color=colour(fam[a]),
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
            # Near the bottom edge, label above the marker so it clears the tick labels.
            (5, 4) if r_all[e] < RMIN + 0.1 * (RMAX - RMIN) else (5, -9),
            textcoords="offset points",
            color=RED,
            fontsize=8,
            weight="bold",
            bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.6),
            zorder=6,
        )
    ax.plot(
        phi_all[0], r_all[0], "o", color=DARK, ms=5, mec="white", mew=0.6, zorder=5, clip_on=False
    )
    ax.plot(
        phi_all[-1], r_all[-1], "s", color=DARK, ms=5, mec="white", mew=0.6, zorder=5, clip_on=False
    )
    ax.set(
        xlim=(-DEG, DEG),
        ylim=(RMIN, RMAX),
        xlabel="$\\varphi$ (deg)",
        ylabel="$r$",
        xticks=[-DEG, 0, DEG],
        yticks=[RMIN, RMAX],
    )
    ax.set_yticklabels([f"{RMIN:.2f}", f"{RMAX:.2f}"])
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
    splits = result["split_phi_degrees"]
    g, k = (
        row["geometry", order, DENSITY],
        row["kinematics", order, DENSITY],
    )
    # Shade only the sheets the plans use (the legend's); other sheets would mark regions
    # as overlaps although no shown plan can use them there.
    shown = sorted(set(result["chain_sheets"]) | set(data["geometry"]["family"].tolist()))
    SHOWN[:] = shown
    grid = dict(
        grid, available=grid["available"] & np.isin(np.arange(grid["available"].shape[-1]), shown)
    )
    fig = plt.figure(figsize=(7.2, 7.0), layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0.04, 1, 0.96))
    outer = fig.add_gridspec(2, 2, height_ratios=[1, 1])
    ax_a = fig.add_subplot(outer[0, 0])
    panel_a(ax_a, splits, result["chain_sheets"])
    ax_a.set_title("(a) Robot and task")
    ax_b = fig.add_subplot(outer[0, 1])
    panel_b(ax_b, grid, result)
    ax_b.set_title("(b) Safe patches")
    ax_c, ax_d = (fig.add_subplot(outer[1, 0]), fig.add_subplot(outer[1, 1], sharex=None))
    nc = path_panel(ax_c, data["geometry"], grid, "geometry", result)
    kd = data["kinematics"]
    ids = kd["waypoint_indices"]
    connectors = []
    for e in np.flatnonzero(kd["break_edges"]):
        connectors += [(int(ids[ids <= e].max()), int(e)), (int(e) + 1, int(ids[ids > e].min()))]
    nd = path_panel(ax_d, kd, grid, "kinematics", result, connectors=connectors)
    assert nc == g["switches"] + g["in_sheet_resets"] and nd == k["switches"]
    ax_c.set_title("(c) Geometry-first")
    ax_d.set_title("(d) Kinematics-first")
    handles = [
        Line2D([], [], color=colour(f), lw=2, label=f"Sheet {i + 1}") for i, f in enumerate(shown)
    ] + [
        Line2D([], [], marker="X", color=RED, ls="none", ms=7, label="reset"),
        Line2D([], [], marker="o", color=DARK, ls="none", ms=5, label="start"),
        Line2D([], [], marker="s", color=DARK, ls="none", ms=5, label="end"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=len(handles),
        fontsize=7.5,
        frameon=False,
        handlelength=1.6,
        columnspacing=1.4,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=300)
    plt.close(fig)
