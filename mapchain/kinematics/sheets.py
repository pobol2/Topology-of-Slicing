"""Posture sheets and the two construction orders: geometry-first and kinematics-first planning."""

from __future__ import annotations

import argparse
import json
from time import perf_counter

import numpy as np
from scipy.ndimage import label
from scipy.spatial import cKDTree

from mapchain import store
from mapchain.kinematics.arm import (
    ANGLE,
    LINKS,
    MARGIN,
    NR,
    NT,
    OBSTACLES,
    RMAX,
    RMIN,
    WEIGHTS,
    Collision,
    build_patch,
    certify,
    forward,
    joints,
    patch_lookup,
    solve_ik,
    workspace,
)
from mapchain.volumetric.hilbert import hilbert_2d

OUT = store.OUT / "kinematics"


def request(order, density):
    st = hilbert_2d(order).copy()
    if density == "gradient":
        st[:, 1] = -0.5 + np.sqrt(0.25 + 2 * st[:, 1])
    return st


FIRST_LIMIT = np.arctan2(np.sqrt(0.19), 0.9) - np.arcsin(OBSTACLES[0, 2])


FAMILY_COVERAGE = None


def component_of(q, x):
    """Identify the two central families, including their elbow-sign seams.

    Within the overlap use the side of the third disk's tool-directed ray
    containing elbow 2. Outside the measured envelope, only one selected sheet
    exists. This handles safe elbow-sign changes at the sector ends. Tests
    compare this classification with the labelled fibre for all sampled radii.
    """
    global FAMILY_COVERAGE
    if FAMILY_COVERAGE is None:
        saved = json.loads((OUT / "fibre.json").read_text())["radius_reports"]
        FAMILY_COVERAGE = (
            np.array([v["radius"] for v in saved]),
            np.array([v["family_phi_ranges"] for v in saved]),
        )
    phi = np.rad2deg(np.arctan2(np.asarray(x)[..., 1], np.asarray(x)[..., 0]))
    v = np.asarray(x) - OBSTACLES[2, :2]
    e = joints(q)[..., -2, :] - OBSTACLES[2, :2]
    side = v[..., 0] * e[..., 1] - v[..., 1] * e[..., 0]
    radii, ranges = FAMILY_COVERAGE
    radius = np.linalg.norm(x, axis=-1)
    lo = np.interp(radius, radii, ranges[:, 1, 0])
    hi = np.interp(radius, radii, ranges[:, 0, 1])
    fringe = 0.25
    family = np.where(phi < lo - fringe, 0, np.where(phi > hi + fringe, 1, (side > 0).astype(int)))
    return np.where(abs(np.asarray(q)[..., 0]) < FIRST_LIMIT, family, -1)


def safe_fibre(x, collision, samples=801):
    """Dense first-joint fibre, equivalent to the labelled (phi,psi) sheets.

    Use exact reachable q1 limits, rather than missing narrow wrist-grid cusp
    tips. Returned family labels include both wrist elbow signs. Sampling can
    establish feasibility; absence of a sample is not a universal impossibility proof.
    """
    x = np.asarray(x)
    phi = np.arctan2(x[..., 1], x[..., 0])
    reach = np.arccos(np.linalg.norm(x, axis=-1) / 2)
    lo, hi = (np.maximum(phi - reach, -FIRST_LIMIT), np.minimum(phi + reach, FIRST_LIMIT))
    a = lo[..., None] + np.linspace(0, 1, samples) * (hi - lo)[..., None]
    w = x[..., None, :] - np.stack((np.cos(a), np.sin(a)), -1)
    c = (np.sum(w * w, -1) - 0.5) / 0.5
    parts = []
    for sign in (-1, 1):
        b = sign * np.arccos(np.clip(c, -1, 1))
        d = np.arctan2(w[..., 1], w[..., 0]) - a - np.arctan2(np.sin(b), 1 + np.cos(b))
        parts.append(np.stack((a, d, b), -1))
    q = np.concatenate(parts, axis=-2)
    clearance = collision.clearance(q)
    family = component_of(q, x[..., None, :])
    return (q, clearance, family)


def availability(path):
    collision = Collision(OBSTACLES)
    available = []
    for chunk in np.array_split(workspace(path), max(1, int(np.ceil(len(path) / 128)))):
        q, clearance, family = safe_fibre(chunk, collision)
        available.extend(
            np.stack([np.any((family == f) & (clearance >= MARGIN), axis=1) for f in (0, 1)], -1)
        )
    available = np.asarray(available)
    if not available.any(axis=1).all():
        raise RuntimeError("Neither selected family covers a path sample")
    forced = np.where(available.sum(axis=1) == 1, np.argmax(available, axis=1), -1)
    exclusive = forced[forced >= 0]
    minimum = int(np.count_nonzero(np.diff(exclusive)))
    return (available, forced, minimum, collision.queries)


def wrist_ik(x, psi, sign):
    """Analytic closure of the 1,.5 wrist arm for both elbow signs."""
    psi = np.asarray(psi)
    w = np.asarray(x) - 0.5 * np.stack((np.cos(psi), np.sin(psi)), -1)
    c = np.sum(w * w, -1) - 1.25
    q2 = sign * np.arccos(np.clip(c, -1, 1))
    q1 = np.arctan2(w[..., 1], w[..., 0]) - np.arctan2(0.5 * np.sin(q2), 1 + 0.5 * np.cos(q2))
    q3 = (psi - q1 - q2 + np.pi) % (2 * np.pi) - np.pi
    q = np.stack((q1, q2, q3), -1)
    valid = (np.abs(c) <= 1) & (np.abs(q).max(axis=-1) <= np.pi)
    return (q, valid)


class Union:
    def __init__(self, n):
        self.parent = list(range(n))

    def root(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def join(self, a, b):
        self.parent[self.root(b)] = self.root(a)


def fibre_analysis():
    """Label all sampled safe components, merging elbow sheets at safe q2=0.

    Per-radius eight-neighbour grid connectivity is sampled evidence. Safe
    branch joins use exact wrist singular poses and certified short joint edges.
    Additional outer first-link corridors are retained and reported.
    """
    global FAMILY_COVERAGE
    start = perf_counter()
    phi = np.deg2rad(np.linspace(-30, 30, 601))
    psi = np.deg2rad(np.linspace(-180, 180, 1441))
    radii = np.array([RMIN, 1.75, 1.8, RMAX])
    labels = np.zeros((len(radii), 2, len(phi), len(psi)), dtype=np.int32)
    selected = np.zeros((len(radii), 2), dtype=int)
    reports, joins = ([], [])
    collision = Collision(OBSTACLES)
    for ir, r in enumerate(radii):
        qbranches, masks, offset = ([], [], 0)
        x = r * np.stack((np.cos(phi), np.sin(phi)), -1)[:, None, :]
        for sign in (-1, 1):
            q, reachable = wrist_ik(x, psi[None, :], sign)
            safe = reachable & (collision.clearance(q) > 1e-09)
            lab, n = label(safe, np.ones((3, 3)))
            lab[safe] += offset
            labels[ir, len(qbranches)] = lab
            qbranches.append(q)
            masks.append(safe)
            offset += n
        union = Union(offset + 1)
        beta = np.arccos(r - 2 / r)
        for direction in (-1, 1):
            seam_psi = phi + direction * beta
            seam_q, _ = wrist_ik(x[:, 0], seam_psi, -1)
            seam_q[:, 1] = 0
            seam_q[:, 2] = seam_psi - seam_q[:, 0]
            safe_seam = collision.clearance(seam_q) > 1e-08
            for ip in np.flatnonzero(safe_seam):
                jp = int(np.argmin(abs(psi - seam_psi[ip])))
                candidates = []
                for branch in range(2):
                    i0, i1 = (max(0, ip - 1), min(len(phi), ip + 2))
                    j0, j1 = (max(0, jp - 3), min(len(psi), jp + 4))
                    ids = labels[ir, branch, i0:i1, j0:j1]
                    for component in np.unique(ids[ids > 0]):
                        ii, jj = np.where(ids == component)
                        poses = qbranches[branch][ii + i0, jj + j0]
                        k = np.argmin(np.linalg.norm(poses - seam_q[ip], axis=1))
                        try:
                            certify(np.array([[seam_q[ip], poses[k]]]), collision)
                        except RuntimeError:
                            continue
                        candidates.append(int(component))
                for a in candidates[1:]:
                    union.join(candidates[0], a)
                if len(candidates) > 1:
                    joins.append([float(r), float(np.rad2deg(phi[ip]))])
        roots = {v: union.root(v) for v in range(1, offset + 1)}
        compact = {v: i + 1 for i, v in enumerate(sorted(set(roots.values())))}
        lookup = np.array([0] + [compact[roots[v]] for v in range(1, offset + 1)])
        labels[ir] = lookup[labels[ir]]
        for family, anchor in enumerate((-30.0, 30.0)):
            ip = int(np.argmin(abs(np.rad2deg(phi) - anchor)))
            options = []
            for branch in range(2):
                q = qbranches[branch][ip]
                good = masks[branch][ip] & (abs(q[:, 0]) < FIRST_LIMIT)
                for jp in np.flatnonzero(good):
                    options.append(
                        (float(collision.clearance(q[jp])), int(labels[ir, branch, ip, jp]))
                    )
            selected[ir, family] = max(options)[1]
        components = []
        for component in range(1, int(labels[ir].max()) + 1):
            branch, ip, jp = np.where(labels[ir] == component)
            poses = np.concatenate(
                [qbranches[b][ip[branch == b], jp[branch == b]] for b in range(2)]
            )
            components.append(
                dict(
                    id=component,
                    samples=len(ip),
                    phi_range_degrees=np.rad2deg(phi[[ip.min(), ip.max()]]).tolist(),
                    last_link_range_degrees=np.rad2deg(psi[[jp.min(), jp.max()]]).tolist(),
                    elbow_signs=sorted(set((-1, 1)[b] for b in branch)),
                    first_link_range_degrees=np.rad2deg(
                        [poses[:, 0].min(), poses[:, 0].max()]
                    ).tolist(),
                    selected_family="connected anchors"
                    if component == selected[ir, 0] == selected[ir, 1]
                    else "down"
                    if component == selected[ir, 0]
                    else "up"
                    if component == selected[ir, 1]
                    else None,
                )
            )
        ranges = [
            next(c["phi_range_degrees"] for c in components if c["id"] == selected[ir, f])
            for f in range(2)
        ]
        reports.append(
            dict(
                radius=float(r),
                components=components,
                family_phi_ranges=ranges,
                anchors_connected=bool(selected[ir, 0] == selected[ir, 1]),
                overlap_degrees=[ranges[1][0], ranges[0][1]],
            )
        )
        print(
            f"fibre r={r}: selected down/up ranges={ranges}, components={len(components)}",
            flush=True,
        )
    common = [
        max(v["family_phi_ranges"][1][0] for v in reports),
        min(v["family_phi_ranges"][0][1] for v in reports),
    ]
    FAMILY_COVERAGE = (radii, np.array([v["family_phi_ranges"] for v in reports]))
    if common[0] >= common[1]:
        raise RuntimeError("No common overlap band across the radius samples")
    report = dict(
        links=list(LINKS),
        obstacles=OBSTACLES.tolist(),
        radial_interval=[RMIN, RMAX],
        radius_reports=reports,
        common_overlap_degrees=common,
        grid=dict(
            phi_step_degrees=0.1,
            last_link_step_degrees=0.25,
            radii=radii.tolist(),
            elbow_signs=[-1, 1],
        ),
        safe_elbow_seam_joins=len(joins),
        collision_queries=collision.queries,
        seconds=perf_counter() - start,
        connectivity_statement="Endpoint-anchored families are separate in the sampled labelled fibre graph at each reported radius, including certified safe elbow-sign seam joins. This sampling does not prove absence of a connection at all intermediate radii or arbitrarily narrow unsampled passages. Small extra grid components may be discretization fragments; outer first-link corridors are also reported.",
    )
    return (dict(phi=phi, psi=psi, radii=radii, labels=labels, selected=selected), report)


def fibre_seed(x, family, fibre, collision, previous=None):
    """Re-seed only from the selected component, allowing either elbow sign."""
    poses, clearance, members = safe_fibre(x, collision)
    q = poses[(members == family) & (clearance >= MARGIN)]
    if not len(q):
        phi = np.rad2deg(np.arctan2(x[1], x[0]))
        raise RuntimeError(f"No safe seed in family {family} at phi={phi:.4f}")
    if previous is None:
        primary = q[(1 if family == 0 else -1) * q[:, 2] >= 0.3]
        if len(primary):
            q = primary
        return q[np.argmax(collision.clearance(q))]
    return q[np.argmin(np.linalg.norm(q - previous, axis=1))]


def patch_st(path, phi_range):
    st = np.asarray(path).copy()
    lo, hi = (np.asarray(phi_range) + 30) / 60
    st[:, 1] = (st[:, 1] - lo) / (hi - lo)
    return st


def stretch_collocation(st, phi_range=None, spacing=0.004):
    """Original waypoint indices plus patch-face splits and short target steps."""
    points, originals = ([st[0]], [0])
    for a, b in zip(st[:-1], st[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(workspace(b) - workspace(a)) / spacing)))
        fractions = list(np.linspace(0, 1, n + 1)[1:])
        if phi_range is not None:
            aa, bb = patch_st(np.array([a, b]), phi_range)
            for vector, levels in (
                (np.array([NR, 0]), np.arange(NR + 1)),
                (np.array([0, NT]), np.arange(NT + 1)),
                (np.array([-NR, NT]), np.arange(-NR, NT + 1)),
            ):
                z0, z1 = (aa @ vector, bb @ vector)
                if abs(z1 - z0) > 1e-14:
                    f = (levels - z0) / (z1 - z0)
                    fractions.extend(f[(f > 1e-12) & (f < 1 - 1e-12)])
        fractions = np.unique(np.round(fractions, 12))
        points.extend(a + fractions[:, None] * (b - a))
        originals.append(len(points) - 1)
    return (np.asarray(points), np.asarray(originals))


def initial_family(requested):
    """Family forced at the first exclusive sample, scanning only a short prefix."""
    queries = 0
    for i in range(0, len(requested), 8):
        _, forced, _, used = availability(requested[i : i + 8])
        queries += used
        if (forced >= 0).any():
            return (int(forced[forced >= 0][0]), queries)
    raise RuntimeError("No exclusive sample on the path")


def geometry_plan(order, density, fibre, diagnostics=True):
    """Plan by warm-started IK. plan_seconds = start-family scan, IK, resets and
    certification; the full availability scan that gives the discrete minimum is a
    separate diagnostic (availability_seconds) and never steers the greedy switches."""
    start = perf_counter()
    original = request(order, density)
    requested, original_ids = stretch_collocation(original)
    collision = Collision(OBSTACLES)
    family, start_queries = initial_family(requested)
    seed = fibre_seed(workspace(requested[0]), family, fibre, collision)
    q, path, families, breaks, mapping, events = ([], [], [], [], [], [])
    for st in requested:
        x = workspace(st)
        reason = None
        try:
            candidate = solve_ik(x, seed, collision, margin=MARGIN)
            if component_of(candidate, x) != family:
                raise RuntimeError("IK left current component")
            if q:
                certify(np.array([[seed, candidate]]), collision)
        except RuntimeError as error:
            reason = str(error)
            if reason is not None:
                other = 1 - family
                switch_x = workspace(path[-1])
                reset = fibre_seed(switch_x, other, fibre, collision)
                breaks.append(True)
                q.append(reset)
                path.append(path[-1].copy())
                families.append(other)
                events.append(
                    dict(
                        edge=len(q) - 2,
                        from_family=family,
                        to_family=other,
                        cause=reason,
                        phi_degrees=float(np.rad2deg(np.arctan2(switch_x[1], switch_x[0]))),
                    )
                )
                family, seed = (other, reset)
                candidate = solve_ik(x, seed, collision, margin=MARGIN)
                certify(np.array([[seed, candidate]]), collision)
        if q:
            breaks.append(False)
        q.append(candidate)
        path.append(st)
        families.append(family)
        mapping.append(len(q) - 1)
        seed = candidate
    q, path, breaks = (np.asarray(q), np.asarray(path), np.asarray(breaks))
    continuous = np.stack((q[:-1][~breaks], q[1:][~breaks]), 1)
    lower = certify(continuous, collision)
    plan_seconds = perf_counter() - start
    data = dict(
        q=q,
        collocation_st=path,
        waypoint_indices=np.asarray(mapping)[original_ids],
        requested_st=original,
        family=np.asarray(families),
        break_edges=breaks,
    )
    info = dict(
        switches=len(events),
        switch_events=events,
        guaranteed_clearance_lower_bound=lower,
        plan_seconds=plan_seconds,
        plan_collision_queries=collision.queries + start_queries,
        plan_ik_solves=collision.ik_solves,
    )
    if diagnostics:
        diag_start = perf_counter()
        available, forced, minimum, analysis_queries = availability(requested)
        info.update(
            theoretical_minimum=minimum,
            availability_seconds=perf_counter() - diag_start,
            availability_analysis_collision_queries=analysis_queries,
        )
        data.update(availability=available, forced_family=forced, availability_st=requested)
        if len(events) != minimum:
            raise RuntimeError(
                f"Greedy switches {len(events)} differ from forced-region minimum {minimum}, m{order} {density}"
            )
    return (data, info)


def kinematics_plan(order, density, patches, ranges, split_phi):
    start = perf_counter()
    raw = hilbert_2d(order)
    tcut = (split_phi + 30) / 60
    sourcecut = tcut if density == "uniform" else 0.5 * tcut + 0.5 * tcut * tcut
    n = 2**order
    rows = int(np.clip(np.rint(sourcecut * n), 1, n - 1))
    sourcegridcut = rows / n
    originals = []
    for family in (0, 1):
        mask = raw[:, 1] < sourcegridcut if family == 0 else raw[:, 1] >= sourcegridcut
        st = raw[mask].copy()
        if family == 0:
            st[:, 1] *= sourcecut / sourcegridcut
        else:
            st[:, 1] = sourcecut + (st[:, 1] - sourcegridcut) * (1 - sourcecut) / (
                1 - sourcegridcut
            )
        if density == "gradient":
            st[:, 1] = -0.5 + np.sqrt(0.25 + 2 * st[:, 1])
        originals.append(st)
    handoff = np.array([0.5, tcut])
    low, low_ids = stretch_collocation(np.vstack((originals[0], handoff)), ranges[0])
    high, high_ids = stretch_collocation(np.vstack((handoff, originals[1])), ranges[1])
    q = np.concatenate(
        [patch_lookup(patch_st(st, ranges[f]), patches[f]) for f, st in enumerate((low, high))]
    )
    path = np.vstack((low, high))
    families = np.r_[np.zeros(len(low), dtype=int), np.ones(len(high), dtype=int)]
    breaks = np.zeros(len(q) - 1, dtype=bool)
    breaks[len(low) - 1] = True
    ids = np.r_[low_ids[:-1], len(low) + high_ids[1:]]
    collision = Collision(OBSTACLES)
    lower = certify(np.stack((q[:-1][~breaks], q[1:][~breaks]), 1), collision)
    return (
        dict(
            q=q,
            collocation_st=path,
            waypoint_indices=ids,
            requested_st=np.vstack(originals),
            family=families,
            break_edges=breaks,
        ),
        dict(
            switches=1,
            theoretical_minimum=1,
            switch_events=[
                dict(
                    edge=len(low) - 1,
                    from_family=0,
                    to_family=1,
                    cause="Single scheduled overlap handoff",
                    phi_degrees=split_phi,
                )
            ],
            guaranteed_clearance_lower_bound=lower,
            plan_seconds=perf_counter() - start,
            plan_collision_queries=collision.queries,
            plan_ik_solves=0,
            source_split_probability=sourcecut,
            lower_waypoint_count=len(originals[0]),
            availability_analysis_collision_queries=0,
        ),
    )


def evaluate(data, density):
    """Same evaluator for both pipelines; reset edges are never interpolated."""
    q, breaks = (data["q"], data["break_edges"])
    dense = [q[0]]
    for i, (a, b) in enumerate(zip(q[:-1], q[1:])):
        if breaks[i]:
            dense.append(b)
            continue
        n = max(1, int(np.ceil(abs(b - a) @ WEIGHTS / 0.0005)))
        dense.extend(a + np.linspace(0, 1, n + 1)[1:, None] * (b - a))
    dense = np.asarray(dense)
    x = forward(dense)
    s, t = np.meshgrid(np.linspace(0, 1, 33), np.linspace(0, 1, 129))
    grid = workspace(np.stack((s.ravel(), t.ravel()), -1))
    coverage = float(cKDTree(x).query(grid)[0].max())
    waypoints = forward(q[data["waypoint_indices"]])
    coords = np.stack(
        (
            (np.sum(waypoints * waypoints, -1) - RMIN**2) / (RMAX**2 - RMIN**2),
            (np.arctan2(waypoints[:, 1], waypoints[:, 0]) + ANGLE) / (2 * ANGLE),
        ),
        -1,
    )
    es, et = (np.linspace(0, 1, 5), np.linspace(0, 1, 9))
    masses = np.diff(et) if density == "uniform" else np.diff(0.5 * et + 0.5 * et * et)
    expected = np.broadcast_to(masses / 4, (4, 8))
    es[[0, -1]], et[[0, -1]] = ([-np.inf, np.inf], [-np.inf, np.inf])
    histogram = np.histogram2d(*coords.T, bins=(es, et))[0] / len(coords)
    collision = Collision(OBSTACLES)
    desired = workspace(data["collocation_st"])
    return (
        dict(
            minimum_clearance=float(collision.clearance(dense).min()),
            coverage_error=coverage,
            density_error=float(0.5 * abs(histogram - expected).sum()),
            maximum_collocation_tracking_error=float(
                np.linalg.norm(forward(q) - desired, axis=1).max()
            ),
            sampled_target_excursion=float(
                np.maximum.reduce(
                    (
                        np.zeros(len(x)),
                        RMIN - np.linalg.norm(x, axis=1),
                        np.linalg.norm(x, axis=1) - RMAX,
                        (abs(np.arctan2(x[:, 1], x[:, 0])) - ANGLE) * RMAX,
                    )
                ).max()
            ),
            evaluation_samples=len(x),
            evaluation_collision_queries=collision.queries,
            maximum_reset_tool_displacement=float(
                np.linalg.norm(np.diff(forward(q), axis=0)[breaks], axis=1).max()
            ),
        ),
        x,
    )


GRID_PHI, GRID_R = (241, 13)


def availability_grid():
    """Availability samples for the figure on a (phi, r) grid, same test as the planner.

    A sample is available for a family when some fibre pose of that family has clearance >= MARGIN
    (801 first-joint samples per elbow branch); absence of a sample is not an impossibility proof."""
    phi = np.linspace(-30, 30, GRID_PHI)
    radius = np.linspace(RMIN, RMAX, GRID_R)
    p, r = np.meshgrid(np.deg2rad(phi), radius)
    x = (r[..., None] * np.stack((np.cos(p), np.sin(p)), -1)).reshape(-1, 2)
    collision = Collision(OBSTACLES)
    found = []
    for chunk in np.array_split(x, int(np.ceil(len(x) / 128))):
        q, clearance, family = safe_fibre(chunk, collision)
        found.append(
            np.stack([np.any((family == f) & (clearance >= MARGIN), axis=1) for f in (0, 1)], -1)
        )
    available = np.concatenate(found).reshape(GRID_R, GRID_PHI, 2)
    return dict(phi=phi, radius=radius, available=available)


def run(orders=(2, 3, 4), fibre_only=False):
    if any(order not in (2, 3, 4) for order in orders):
        raise ValueError("This sweep is limited to Hilbert orders 2, 3 and 4")
    fibre, report = fibre_analysis()
    store.save_npz(OUT / "fibre.npz", **fibre)
    store.save_json(OUT / "fibre.json", report)
    store.save_npz(OUT / "availability_grid.npz", **availability_grid())
    if fibre_only:
        return
    lo, hi = report["common_overlap_degrees"]
    split = (lo + hi) / 2
    ranges = [(-30.0, hi - 0.4), (lo + 0.4, 30.0)]
    patches, preprocess = ([], [])
    for family in (0, 1):
        start = perf_counter()
        q, info = build_patch(
            ranges[family],
            OBSTACLES,
            seed_function=lambda x, c, family=family, fibre=fibre: fibre_seed(x, family, fibre, c),
            margin=0.004,
            radial_range=(RMIN, RMAX),
            reverse_theta=family == 1,
        )
        info.update(
            seconds=perf_counter() - start, family=family, phi_range_degrees=list(ranges[family])
        )
        patches.append(q)
        preprocess.append(info)
        store.save_npz(OUT / f"patch_{family}.npz", q=q, phi_range=ranges[family])
        print(f"Certified patch {family}, phi={ranges[family]}", flush=True)
    rows, events = ([], {})
    for order in sorted(set(orders)):
        for density in ("uniform", "gradient"):
            for method in ("geometry", "kinematics"):
                if method == "geometry":
                    data, info = geometry_plan(order, density, fibre)
                else:
                    data, info = kinematics_plan(order, density, patches, ranges, split)
                evaluation_start = perf_counter()
                metrics, evaluated_x = evaluate(data, density)
                info["evaluation_seconds"] = perf_counter() - evaluation_start
                events[f"{method}_m{order}_{density}"] = info.pop("switch_events")
                row = dict(
                    method=method,
                    order=order,
                    density=density,
                    waypoints=4**order,
                    collocation_points=len(data["q"]),
                    preprocessing_seconds=sum(p["seconds"] for p in preprocess)
                    if method == "kinematics"
                    else 0,
                    **info,
                    **metrics,
                )
                rows.append(row)
                store.save_npz(
                    OUT / f"{method}_m{order}_{density}.npz", **data, evaluation_x=evaluated_x
                )
                print(
                    f"{method} m{order} {density}: switches={info['switches']}, minimum={info['theoretical_minimum']}",
                    flush=True,
                )
    result = dict(
        requests=rows,
        switch_events=events,
        preprocessing=preprocess,
        fibre=report,
        split_phi_degrees=split,
        orders=sorted(set(orders)),
        minimum_statement="Alternations after deleting overlap samples from the selected two-family availability sequence. This is the exact combinatorial minimum conditional on the sampled safe-fibre availability model, not a continuous-space impossibility proof. Geometry-first checks its greedy count against it.",
        discontinuity_statement="Switch edges are recorded discrete posture resets. They are excluded from interpolation, collision certification and tool-path coverage. No collision-free transition between disconnected sheets is claimed.",
        kinematics_ordering="Each rectangular subgrid uses the existing Hilbert ordering; the two restricted traversals are concatenated at one common overlap point. Radial columns times allocated angular rows preserve 4**m original waypoints. Allocation is rounded to the nearest angular row of the desired density mass.",
        evaluation_statement="Both methods: 33x129 sampled target grid, <=.0005 tool-motion sampling on continuous stretches, 4x8 area-coordinate density bins evaluated at original waypoints only. Clearance lower bounds certify every continuous stretch; coverage, density and tracking are sampled. Timings are recorded in JSON as secondary diagnostics, not plotted.",
    )
    store.save_json(OUT / "results.json", result)


def kinematics_main(argv=None):
    """Plan the planar 3R comparison: fibre analysis, certified patches and both methods."""
    parser = argparse.ArgumentParser(description=kinematics_main.__doc__)
    parser.add_argument("--orders", nargs="+", type=int, choices=(2, 3, 4), default=[2, 3, 4])
    parser.add_argument("--fibre-only", action="store_true")
    store.add_check(parser)
    args = parser.parse_args(argv)
    store.configure(args)
    run(args.orders, fibre_only=args.fibre_only)
    store.finish()
