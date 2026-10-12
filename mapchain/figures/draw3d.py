"""Shared helpers for the 3D path figures: orthographic axes, solids and segment sets."""

from __future__ import annotations

import json

import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from shapely.geometry import Polygon as Poly

from mapchain.evaluation.whole_part import uniform_path, waypoints

BLUE, RED = ("#1f77b4", "#d62728")
L_PLAN = Poly([(0.05, 0.05), (0.95, 0.05), (0.95, 0.35), (0.35, 0.35), (0.35, 0.95), (0.05, 0.95)])
TOP = (90, -90)  # view from above


def load(part, m):
    """Saved whole-part record and its waypoints."""
    return json.loads(uniform_path(part, m).read_text()), waypoints(part, m)


def axes3d(fig, spec, lo=(0.05, 0.05, 0.05), hi=(0.95, 0.95, 0.95), zoom=1.3, view=TOP):
    ax = fig.add_subplot(spec, projection="3d", computed_zorder=False)
    ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(lo[2], hi[2]))
    aspect = np.subtract(hi, lo)
    if view == TOP:
        aspect[2] = 0.001 * max(aspect[:2])
    ax.set_box_aspect(aspect, zoom=zoom)
    ax.view_init(*view)
    ax.set_proj_type("ortho")
    ax.set_axis_off()
    ax.keep_view = True
    return ax


def solid(ax, mesh, alpha=0.06):
    ax.add_collection3d(
        Poly3DCollection(
            mesh.triangles, facecolor="#999999", edgecolor="none", alpha=alpha, zorder=1
        )
    )


def edges(ax, mesh, **kw):
    e = mesh.vertices[mesh.face_adjacency_edges[mesh.face_adjacency_angles > 0.001]]
    ax.add_collection3d(Line3DCollection(e, **kw))


def segs3(ax, P, keep, **kw):
    k = np.flatnonzero(keep)
    if len(k):
        ax.add_collection3d(Line3DCollection(np.stack([P[k], P[k + 1]], 1), **kw))
