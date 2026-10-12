"""The planar 3R arm: forward kinematics, collision distances, IK, certificates and patches."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

LINKS = (1.0, 0.5, 0.5)


OBSTACLES = np.array([(0.9, np.sqrt(0.19), 0.2), (0.9, -np.sqrt(0.19), 0.2), (1.35, 0.1, 0.1)])


RMIN, RMAX = (1.6, 1.85)


ANGLE = np.pi / 6


WEIGHTS = np.array([2.0, 1.0, 0.5])


MARGIN = 0.001


NR, NT = (16, 128)


def joints(q):
    q = np.asarray(q)
    a = np.cumsum(q, axis=-1)
    p = np.cumsum(np.asarray(LINKS)[:, None] * np.stack((np.cos(a), np.sin(a)), -1), -2)
    return np.concatenate((np.zeros((*q.shape[:-1], 1, 2)), p), axis=-2)


def forward(q):
    return joints(q)[..., -1, :]


class Collision:
    """One query = one pose checked against all three links and every disk."""

    def __init__(self, obstacles=None):
        self.queries = 0
        self.ik_solves = 0
        self.obstacles = np.asarray(OBSTACLES if obstacles is None else obstacles)
        self.radii = (
            self.obstacles[:, 2]
            if self.obstacles.shape[1] == 3
            else np.full(len(self.obstacles), 0.25)
        )
        self.obstacles = self.obstacles[:, :2]

    def distances(self, q):
        q = np.asarray(q)
        self.queries += int(np.prod(q.shape[:-1]))
        p = joints(q)
        a, d = (p[..., :-1, :], np.diff(p, axis=-2))
        v = self.obstacles[:, None, :] - a[..., None, :, :]
        t = np.clip(np.sum(v * d[..., None, :, :], -1) / np.sum(d * d, -1)[..., None, :], 0, 1)
        return (
            np.linalg.norm(v - t[..., None] * d[..., None, :, :], axis=-1) - self.radii[:, None]
        ).reshape(*q.shape[:-1], 3 * len(self.obstacles))

    def clearance(self, q):
        return self.distances(q).min(axis=-1)


def workspace(st, radial_range=None):
    st = np.asarray(st)
    rmin, rmax = (RMIN, RMAX) if radial_range is None else radial_range
    r = np.sqrt(rmin**2 + st[..., 0] * (rmax**2 - rmin**2))
    theta = -ANGLE + 2 * ANGLE * st[..., 1]
    return r[..., None] * np.stack((np.cos(theta), np.sin(theta)), -1)


def jacobian(q):
    a = np.cumsum(q)
    v = np.asarray(LINKS)[:, None] * np.stack((-np.sin(a), np.cos(a)), -1)
    return np.cumsum(v[::-1], axis=0)[::-1].T


def solve_ik(x, seed, collision, margin=MARGIN):
    """Warm-started constrained IK, including every link/disk distance."""
    collision.ik_solves += 1
    constraints = [
        {"type": "eq", "fun": lambda q: forward(q) - x, "jac": jacobian},
        {"type": "ineq", "fun": lambda q: collision.distances(q) - margin},
    ]
    result = minimize(
        lambda q: 0.5 * np.sum((q - seed) ** 2),
        seed,
        jac=lambda q: q - seed,
        method="SLSQP",
        bounds=[(-np.pi, np.pi)] * 3,
        constraints=constraints,
        options={"ftol": 1e-09, "maxiter": 100},
    )
    q = result.x
    if np.linalg.norm(forward(q) - x) > 1e-07 or collision.clearance(q) < margin - 1e-07:
        raise RuntimeError(f"IK failed at {x}: {result.message}")
    return q


def certify(simplices, collision):
    """Guaranteed clearance for every affine simplex, not just its vertices.

    Each arm point moves at most sum((2,1,.5)*abs(dq)). Segment/disk
    distance is Lipschitz under that bound. Convex combinations stay in
    the same joint box. Recursively subdivide only unresolved simplices.
    Works for both line segments and triangular configuration faces.
    """
    pending = np.asarray(simplices)
    lower = np.inf
    for depth in range(15):
        center = pending.mean(axis=1)
        radius = (np.abs(pending - center[:, None, :]) * WEIGHTS).sum(-1).max(-1)
        clearance = collision.clearance(center)
        if np.any(clearance <= 0):
            raise RuntimeError("Connection/patch contains a collision")
        bound = clearance - radius
        good = bound > 1e-09
        if np.any(good):
            lower = min(lower, float(bound[good].min()))
        bad = pending[~good]
        if len(bad) == 0:
            return lower
        if depth == 14 or len(bad) > 200000:
            raise RuntimeError("Connection/patch could not be certified safe")
        if bad.shape[1] == 2:
            mid = bad.mean(axis=1)
            pending = np.concatenate((np.stack((bad[:, 0], mid), 1), np.stack((mid, bad[:, 1]), 1)))
        else:
            a, b, c = (bad[:, 0], bad[:, 1], bad[:, 2])
            ab, bc, ca = ((a + b) / 2, (b + c) / 2, (c + a) / 2)
            pending = np.concatenate(
                [np.stack(t, 1) for t in ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))]
            )


def build_patch(
    phi_range=None,
    obstacles=None,
    seed_function=None,
    margin=MARGIN,
    radial_range=None,
    reverse_theta=False,
):
    """Build a certified component chart over a specified sector-angle range."""
    collision = Collision(obstacles)
    phi_range = (-30.0, 30.0) if phi_range is None else phi_range
    lo, hi = (np.asarray(phi_range) + 30) / 60

    def chart_workspace(st):
        mapped = np.asarray(st).copy()
        mapped[..., 1] = lo + (hi - lo) * mapped[..., 1]
        return workspace(mapped, radial_range)

    s, t = np.meshgrid(np.linspace(0, 1, NR + 1), np.linspace(0, 1, NT + 1), indexing="ij")
    st = np.stack((s, t), -1)
    q = np.empty((*s.shape, 3))
    if seed_function is None:
        raise ValueError("Supply a component-restricted seed function")
    start_j = NT if reverse_theta else 0
    seed = seed_function(chart_workspace(st[0, start_j]), collision)
    for i in range(NR + 1):
        if i:
            seed = seed_function(chart_workspace(st[i, start_j]), collision)
        for j in range(NT, -1, -1) if reverse_theta else range(NT + 1):
            q[i, j] = solve_ik(chart_workspace(st[i, j]), seed, collision, margin=margin)
            seed = q[i, j]
    faces = []
    for i in range(NR):
        for j in range(NT):
            faces.extend(
                ((q[i, j], q[i + 1, j], q[i + 1, j + 1]), (q[i, j], q[i + 1, j + 1], q[i, j + 1]))
            )
    lower = certify(faces, collision)
    if np.max(np.abs(q)) > np.pi:
        raise RuntimeError("Patch violates joint limits")
    ss, tt = np.meshgrid(np.linspace(0, 1, 2 * NR + 1), np.linspace(0, 1, 2 * NT + 1))
    validation_st = np.stack((ss.ravel(), tt.ravel()), -1)
    mapping_error = float(
        np.linalg.norm(
            forward(patch_lookup(validation_st, q)) - chart_workspace(validation_st), axis=1
        ).max()
    )
    if mapping_error > 0.002:
        raise RuntimeError(f"Patch exceeds sampled inverse-map tolerance: {mapping_error:.6g}")
    return (
        q,
        dict(
            preprocessing_collision_queries=collision.queries,
            preprocessing_ik_solves=collision.ik_solves,
            guaranteed_clearance_lower_bound=lower,
            vertices=q.size // 3,
            faces=len(faces),
            sampled_inverse_error=mapping_error,
            sampled_target_points=len(validation_st),
        ),
    )


def patch_lookup(st, patch):
    z = np.asarray(st) * [NR, NT]
    ij = np.minimum(np.floor(z).astype(int), [NR - 1, NT - 1])
    a, b = (z - ij).T
    i, j = ij.T
    q00, q10, q11, q01 = (patch[i, j], patch[i + 1, j], patch[i + 1, j + 1], patch[i, j + 1])
    return np.where(
        (a >= b)[:, None],
        (1 - a)[:, None] * q00 + (a - b)[:, None] * q10 + b[:, None] * q11,
        (1 - b)[:, None] * q00 + a[:, None] * q11 + (b - a)[:, None] * q01,
    )
