"""Posture sheets and the two construction orders: geometry-first and kinematics-first planning."""

from __future__ import annotations

import argparse
import itertools
import os
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter

import numpy as np
from scipy.ndimage import distance_transform_edt, label, maximum_filter1d
from scipy.spatial import cKDTree

from mapchain import store
from mapchain.kinematics.arm import (
    ANGLE,
    DEG,
    LINKS,
    MARGIN,
    NR,
    NT,
    OBSTACLES,
    RMAX,
    RMIN,
    WEIGHTS,
    Collision,
    band_limits,
    build_patch,
    certify,
    forward,
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


L1, L2, L3 = (float(v) for v in LINKS)


FIBRE = None


# Sampled radii for the labelled fibre.
RADII = 7


# Label search radius (grid cells) around a pose whose own cell is unsafe due to discretization.
SNAP = 3


def load_fibre(fibre=None):
    """Install the labelled fibre used by memberships (from fibre.npz unless given)."""
    global FIBRE
    if fibre is None:
        fibre = dict(np.load(OUT / "fibre.npz"))
    labels = fibre["labels"]
    if FIBRE is not None and FIBRE["labels"] is labels:
        FIBRE = dict(fibre, snapped=FIBRE["snapped"], tables=family_tables(fibre))
        return FIBRE
    snapped = labels.copy()
    for index in np.ndindex(labels.shape[:2]):
        empty = labels[index] == 0
        if empty.all():
            continue
        distance, (ii, jj) = distance_transform_edt(empty, return_indices=True)
        near = empty & (distance <= SNAP)
        snapped[index][near] = labels[index][ii[near], jj[near]]
    FIBRE = dict(fibre, snapped=snapped, tables=family_tables(fibre))
    return FIBRE


def family_tables(fibre):
    """Label -> family membership per radius, (radii, labels, families) booleans.

    "chain": the patch families, one selected component per radius (several
    families may share one). "sheets": every path component of the sampled safe
    fibre space, i.e. components linked across radii (fibre["sheet_of"])."""
    labels = fibre["labels"]
    nr, nl = (len(labels), int(labels.max()) + 1)
    selected = np.asarray(fibre["selected"])
    chain = np.zeros((nr, nl, selected.shape[1]), dtype=bool)
    for ir in range(nr):
        chain[ir, selected[ir], np.arange(selected.shape[1])] = True
    tables = dict(chain=chain)
    if "sheet_of" in fibre:
        sheet_of = np.asarray(fibre["sheet_of"])
        sheets = np.zeros((nr, nl, int(sheet_of.max()) + 1), dtype=bool)
        ir, label = np.nonzero(sheet_of >= 0)
        sheets[ir, label, sheet_of[ir, label]] = True
        tables["sheets"] = sheets
    for table in tables.values():
        table[:, 0] = False
    return tables


def memberships(q, x, kind="chain"):
    """Families containing each pose, as booleans (..., families); kind "chain" or "sheets".

    A pose is located on the wrist grid by its target angle, tool angle and
    second-joint sign, at the nearest sampled radius (then the next nearest).
    Chain families can share a component where their sheets merge, so a pose may
    belong to several. Classification is sampled evidence, like the labelling.
    """
    fibre = FIBRE if FIBRE is not None else load_fibre()
    q = np.asarray(q)
    x = np.broadcast_to(np.asarray(x), (*q.shape[:-1], 2))
    phi, psi, radii = (fibre["phi"], fibre["psi"], fibre["radii"])
    wrapped = (q + np.pi) % (2 * np.pi) - np.pi
    branch = (wrapped[..., 1] > 0).astype(int)
    tool = (q.sum(-1) + np.pi) % (2 * np.pi) - np.pi
    ip = np.clip(
        np.rint((np.arctan2(x[..., 1], x[..., 0]) - phi[0]) / (phi[1] - phi[0])), 0, len(phi) - 1
    )
    jp = np.clip(np.rint((tool - psi[0]) / (psi[1] - psi[0])), 0, len(psi) - 1)
    ip, jp = (ip.astype(int), jp.astype(int))
    order = np.argsort(abs(np.linalg.norm(x, axis=-1)[..., None] - radii), axis=-1)
    table = fibre["tables"][kind]
    member = np.zeros((*q.shape[:-1], table.shape[-1]), dtype=bool)
    for rank in range(min(2, len(radii))):
        ir = order[..., rank]
        here = table[ir, fibre["snapped"][ir, branch, ip, jp]]
        member = np.where(member.any(-1, keepdims=True), member, here)
    return member


def safe_fibre(x, collision, samples=801, kind="chain"):
    """Dense first-joint fibre, equivalent to the labelled (phi,psi) sheets.

    Use exact reachable q1 limits, rather than missing narrow wrist-grid cusp
    tips. Returned family memberships include both wrist elbow signs. Sampling can
    establish feasibility; absence of a sample is not a universal impossibility proof.
    """
    x = np.asarray(x)
    phi = np.arctan2(x[..., 1], x[..., 0])
    r = np.linalg.norm(x, axis=-1)
    reach = np.arccos(np.clip((r * r + L1 * L1 - (L2 + L3) ** 2) / (2 * L1 * r), -1, 1))
    lo, hi = (phi - reach, phi + reach)
    a = lo[..., None] + np.linspace(0, 1, samples) * (hi - lo)[..., None]
    w = x[..., None, :] - L1 * np.stack((np.cos(a), np.sin(a)), -1)
    c = (np.sum(w * w, -1) - L2 * L2 - L3 * L3) / (2 * L2 * L3)
    parts = []
    for sign in (-1, 1):
        b = sign * np.arccos(np.clip(c, -1, 1))
        d = np.arctan2(w[..., 1], w[..., 0]) - a - np.arctan2(L3 * np.sin(b), L2 + L3 * np.cos(b))
        parts.append(np.stack((a, d, b), -1))
    q = np.concatenate(parts, axis=-2)
    clearance = collision.clearance(q)
    return (q, clearance, memberships(q, x[..., None, :], kind))


def chain_length(kind="chain"):
    """Number of families of a kind: chain patch families or sheets."""
    fibre = FIBRE if FIBRE is not None else load_fibre()
    return fibre["tables"][kind].shape[-1]


def fibre_available(x, collision, kind="chain"):
    """Per target, whether each family has a fibre pose with clearance >= MARGIN."""
    found = []
    for chunk in np.array_split(x, max(1, int(np.ceil(len(x) / 128)))):
        q, clearance, member = safe_fibre(chunk, collision, kind=kind)
        found.append(np.any(member & (clearance >= MARGIN)[..., None], axis=1))
    return np.concatenate(found)


def fewest_switches(available):
    """Exact minimum family changes along a sequence of availability rows (dynamic program)."""
    cost = np.where(available[0], 0, np.inf)
    for row in available[1:]:
        cost = np.where(row, np.minimum(cost, cost.min() + 1), np.inf)
    return int(cost.min())


def availability(path, kind="sheets"):
    """Sheet (or chain family) availability along a path, forced families, fewest switches."""
    collision = Collision(OBSTACLES)
    available = fibre_available(workspace(path), collision, kind)
    if not available.any(axis=1).all():
        raise RuntimeError("No family covers a path sample")
    forced = np.where(available.sum(axis=1) == 1, np.argmax(available, axis=1), -1)
    return (available, forced, fewest_switches(available), collision.queries)


def wrist_ik(x, psi, sign):
    """Analytic closure of the first two links for a given tool angle, both elbow signs."""
    psi = np.asarray(psi)
    w = np.asarray(x) - L3 * np.stack((np.cos(psi), np.sin(psi)), -1)
    c = (np.sum(w * w, -1) - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    q2 = sign * np.arccos(np.clip(c, -1, 1))
    q1 = np.arctan2(w[..., 1], w[..., 0]) - np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2))
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
    """Label all sampled safe components and chain them across the sector.

    Per-radius eight-neighbour grid connectivity is sampled evidence. Safe
    branch joins use exact wrist singular poses and certified short joint edges.
    Components at adjacent radii sharing grid cells are tracked into families,
    which may share a component where sheets merge. The chain is the fewest
    families whose common angular intervals run from -DEG to DEG with overlaps
    wide enough for a handoff.
    """
    start = perf_counter()
    phi = np.deg2rad(np.linspace(-DEG, DEG, 601))
    psi = np.deg2rad(np.linspace(-180, 180, 1441))
    radii = np.linspace(RMIN, RMAX, RADII)
    labels = np.zeros((len(radii), 2, len(phi), len(psi)), dtype=np.int32)
    components, joins = ([], [])
    collision = Collision(OBSTACLES)
    for ir, r in enumerate(radii):
        qbranches, offset = ([], 0)
        x = r * np.stack((np.cos(phi), np.sin(phi)), -1)[:, None, :]
        for sign in (-1, 1):
            q, reachable = wrist_ik(x, psi[None, :], sign)
            safe = reachable & (collision.clearance(q) > 1e-09)
            lab, n = label(safe, np.ones((3, 3)))
            lab[safe] += offset
            labels[ir, len(qbranches)] = lab
            qbranches.append(q)
            offset += n
        union = Union(offset + 1)
        beta = np.arccos(np.clip((r * r + L3 * L3 - (L1 + L2) ** 2) / (2 * r * L3), -1, 1))
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
        for component in range(1, int(labels[ir].max()) + 1):
            branch, ip, jp = np.where(labels[ir] == component)
            poses = np.concatenate(
                [qbranches[b][ip[branch == b], jp[branch == b]] for b in range(2)]
            )
            components.append(
                dict(
                    radius_index=ir,
                    id=component,
                    samples=len(ip),
                    phi_range_degrees=np.rad2deg(phi[[ip.min(), ip.max()]]).tolist(),
                    last_link_range_degrees=np.rad2deg(psi[[jp.min(), jp.max()]]).tolist(),
                    elbow_signs=sorted(set((-1, 1)[b] for b in branch)),
                    first_link_range_degrees=np.rad2deg(
                        [poses[:, 0].min(), poses[:, 0].max()]
                    ).tolist(),
                )
            )
        print(f"fibre r={r}: components={int(labels[ir].max())}", flush=True)
    tracks, spans = track_spans(labels, components, phi)
    sheet_of = sheet_labels(labels)
    base = dict(phi=phi, psi=psi, radii=radii, labels=labels, sheet_of=sheet_of)
    handoff_start = perf_counter()
    chain, selected, curves, room, rejected = choose_chain(tracks, spans, base)
    handoff_seconds = perf_counter() - handoff_start
    reports = []
    for ir, r in enumerate(radii):
        mine = [dict(c) for c in components if c["radius_index"] == ir]
        for c in mine:
            del c["radius_index"]
            c["chain_families"] = [k for k, f in enumerate(chain) if tracks[f][ir] == c["id"]]
        ranges = [spans[f][ir].tolist() for f in chain]
        reports.append(dict(radius=float(r), components=mine, family_phi_ranges=ranges))
        print(f"fibre r={r:.3f}: chain usable spans={np.round(ranges, 1).tolist()}", flush=True)
    report = dict(
        links=list(LINKS),
        obstacles=OBSTACLES.tolist(),
        radial_interval=[RMIN, RMAX],
        radius_reports=reports,
        chain_length=len(chain),
        sheets=int(sheet_of.max()) + 1,
        sheet_summary=sheet_summary(sheet_of, components, len(radii)),
        chain_sheets=[int(sheet_of[0, tracks[f][0]]) for f in chain],
        tracks=[[int(c) for c in t] for t in tracks],
        track_spans=[span.tolist() for span in spans],
        chain=[int(f) for f in chain],
        rejected_chains=rejected,
        common_overlap_degrees=[
            np.stack((spans[g][:, 0], spans[f][:, 1]), -1).tolist()
            for f, g in zip(chain[:-1], chain[1:])
        ],
        handoff_s=radial_chart(radii).tolist(),
        handoff_curves=curves.tolist(),
        handoff_clearance=room,
        handoff_seconds=handoff_seconds,
        grid=dict(
            phi_step_degrees=float(np.rad2deg(phi[1] - phi[0])),
            last_link_step_degrees=float(np.rad2deg(psi[1] - psi[0])),
            radii=radii.tolist(),
            elbow_signs=[-1, 1],
        ),
        safe_elbow_seam_joins=len(joins),
        collision_queries=collision.queries,
        seconds=perf_counter() - start,
        connectivity_statement="Chained families are separate in the sampled labelled fibre graph at each reported radius, including certified safe elbow-sign seam joins. Families are tracked across radii by shared grid cells and may share a component where sheets merge. A family's usable span at a radius keeps only angle columns where its component shares cells with the adjacent radii; consecutive spans overlap at every sampled radius. Handoff curves follow those overlaps. This is sampled evidence: it neither proves a continuous section exists nor rules out unsampled connections.",
    )
    fibre = dict(base, selected=selected)
    load_fibre(fibre)
    return (fibre, report)


# Minimum common overlap (degrees) between consecutive chain families: 2 x 0.4 patch inset plus slack.
HANDOFF = 1.0


# Shared wrist-grid cells needed to link components at adjacent radii.
LINK_CELLS = 20


def sheet_labels(labels):
    """Path components of the sampled safe fibre space: components at adjacent radii
    sharing at least LINK_CELLS wrist-grid cells are joined (transitively).
    Returns sheet index per (radius, component label), -1 for unused labels."""
    nr = len(labels)
    sizes = [int(labels[ir].max()) for ir in range(nr)]
    offsets = np.cumsum([0] + sizes)
    union = Union(int(offsets[-1]) + 1)
    for ir in range(nr - 1):
        a, b = (labels[ir].ravel(), labels[ir + 1].ravel())
        both = (a > 0) & (b > 0)
        pairs, counts = np.unique(np.stack((a[both], b[both])), axis=1, return_counts=True)
        for (ca, cb), n in zip(pairs.T, counts):
            if n >= LINK_CELLS:
                union.join(int(offsets[ir] + ca), int(offsets[ir + 1] + cb))
    roots = {}
    sheet_of = np.full((nr, max(sizes) + 1), -1, dtype=int)
    for ir in range(nr):
        for c in range(1, sizes[ir] + 1):
            sheet_of[ir, c] = roots.setdefault(union.root(int(offsets[ir] + c)), len(roots))
    return sheet_of


def sheet_summary(sheet_of, components, nr):
    """Per sheet: samples, radii where present, and angular extent at the inner and outer
    radius. A sheet present at every sampled radius is a main sheet, otherwise a fragment."""
    summary = []
    for sheet in range(int(sheet_of.max()) + 1):
        mine = [c for c in components if sheet_of[c["radius_index"], c["id"]] == sheet]
        present = sorted({c["radius_index"] for c in mine})

        def extent(ir, mine=mine):
            ranges = [c["phi_range_degrees"] for c in mine if c["radius_index"] == ir]
            return [min(r[0] for r in ranges), max(r[1] for r in ranges)] if ranges else None

        summary.append(
            dict(
                sheet=sheet,
                samples=int(sum(c["samples"] for c in mine)),
                radii_present=len(present),
                main=len(present) == nr,
                inner_extent_degrees=extent(0),
                outer_extent_degrees=extent(nr - 1),
            )
        )
    return summary


def cover_number(available):
    """c: fewest families whose (sampled) projections cover every covered target."""
    available = np.asarray(available, dtype=bool).reshape(-1, np.shape(available)[-1])
    available = available[available.any(axis=1)]
    for size in range(1, available.shape[1] + 1):
        for subset in itertools.combinations(range(available.shape[1]), size):
            if available[:, subset].any(axis=1).all():
                return size, list(subset)
    return (available.shape[1], list(range(available.shape[1])))


def longest_run(mask):
    """(first, last) index of the longest run of True, or None."""
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    if not len(edges):
        return None
    starts, ends = (edges[::2], edges[1::2])
    i = int(np.argmax(ends - starts))
    return (int(starts[i]), int(ends[i] - 1))


def track_spans(labels, components, phi):
    """Tracks across radii and, per track and radius, its usable angular span (degrees).

    A track is one component per sampled radius, consecutive ones sharing at
    least LINK_CELLS wrist-grid cells; tracks may share components where sheets
    split or merge. A phi column is usable at a radius only where the component
    shares cells in that column with the track's components at the adjacent
    radii, i.e. the sheet continues across radii there. Projected coverage
    alone would accept tips that pinch off. The span is the longest usable run.
    """
    nr = len(labels)
    links = []
    for ir in range(nr - 1):
        a, b = (labels[ir].ravel(), labels[ir + 1].ravel())
        both = (a > 0) & (b > 0)
        pairs, counts = np.unique(np.stack((a[both], b[both])), axis=1, return_counts=True)
        step = {}
        for (ca, cb), n in zip(pairs.T, counts):
            if n >= LINK_CELLS:
                step.setdefault(int(ca), []).append(int(cb))
        links.append(step)
    tracks = [(c["id"],) for c in components if c["radius_index"] == 0]
    for ir in range(nr - 1):
        tracks = [t + (b,) for t in tracks for b in links[ir].get(t[-1], [])]
    degrees = np.rad2deg(phi)
    kept, spans = ([], [])
    for t in tracks:
        masks = [labels[ir] == c for ir, c in enumerate(t)]
        usable = [m.any(axis=(0, 2)) for m in masks]
        for ir in range(nr - 1):
            shared = (masks[ir] & masks[ir + 1]).any(axis=(0, 2))
            usable[ir] = usable[ir] & shared
            usable[ir + 1] = usable[ir + 1] & shared
        runs = [longest_run(u) for u in usable]
        if all(run is not None for run in runs):
            kept.append(t)
            spans.append(np.array([[degrees[a], degrees[b]] for a, b in runs]))
    return (kept, spans)


# Longest chain considered, and chain candidates tried before giving up.
MAX_CHAIN, MAX_TRIES = (6, 12)


def candidate_chains(spans):
    """Valid chains of tracks over the sector, fewest families first, widest overlap first.

    Consecutive spans must overlap by HANDOFF at every radius and their overlap
    midpoints must advance; the first span starts at -DEG, the last ends at DEG.
    """
    tol = 1e-6

    def narrowest(p):
        return min(
            [float((spans[a][:, 1] - spans[b][:, 0]).min()) for a, b in zip(p[:-1], p[1:])]
            or [np.inf]
        )

    level = [[f] for f, span in enumerate(spans) if (span[:, 0] <= -DEG + tol).all()]
    for _ in range(MAX_CHAIN):
        done = [p for p in level if (spans[p[-1]][:, 1] >= DEG - tol).all()]
        yield from sorted(done, key=narrowest, reverse=True)
        grown = []
        for p in level:
            if p in done:
                continue
            hi = spans[p[-1]][:, 1]
            floor = (spans[p[-1]][:, 0] + spans[p[-2]][:, 1]) / 2 if len(p) > 1 else -DEG
            for f, span in enumerate(spans):
                lo_f, hi_f = (span[:, 0], span[:, 1])
                if f in p or not (hi_f > hi).all() or not (hi - lo_f >= HANDOFF).all():
                    continue
                if ((lo_f + hi) / 2 > floor + HANDOFF).all():
                    grown.append(p + [f])
        level = grown
        if not level:
            return


# Least handoff clearance accepted for a chain (worse family, along the whole curve).
HANDOFF_ROOM = 0.03


def radial_chart(radii):
    """Radial chart coordinate s of radii (workspace() maps s linearly in r squared)."""
    return (np.asarray(radii) ** 2 - RMIN**2) / (RMAX**2 - RMIN**2)


def pair_room(x, k, collision):
    """Best fibre clearance of chain families k and k+1 at each target; the worse of the two."""
    best = []
    for chunk in np.array_split(x, max(1, int(np.ceil(len(x) / 128)))):
        q, clearance, member = safe_fibre(chunk, collision)
        best.append(np.where(member[..., k : k + 2], clearance[..., None], -np.inf).max(1))
    return np.concatenate(best).min(-1)


def handoff_curve(k, lo, hi, floor, radii, collision, step=0.1, slope=3.0):
    """Handoff angle at each sampled radius between chain families k and k+1.

    Inside each radius' overlap [lo, hi] (less PATCH_PAD) the curve maximises the
    worse family's clearance, minimised along the curve, moving at most `slope`
    degrees between adjacent radii (bottleneck dynamic program). The curve is
    then checked midway between radii. Returns (curve, room) or None."""
    grid = np.arange(lo.min() + PATCH_PAD, hi.max() - PATCH_PAD + 1e-9, step)
    room = np.full((len(radii), len(grid)), -np.inf)
    for i, r in enumerate(radii):
        inside = (grid >= lo[i] + PATCH_PAD) & (grid <= hi[i] - PATCH_PAD)
        inside &= grid > floor[i] + HANDOFF
        if not inside.any():
            return None
        p = np.deg2rad(grid[inside])
        room[i, inside] = pair_room(r * np.stack((np.cos(p), np.sin(p)), -1), k, collision)
    window = max(1, int(round(slope / step)))
    best = [room[0]]
    for i in range(1, len(radii)):
        reach = maximum_filter1d(best[-1], 2 * window + 1, mode="constant", cval=-np.inf)
        best.append(np.minimum(room[i], reach))
    j = int(np.argmax(best[-1]))
    if not np.isfinite(best[-1][j]):
        return None
    path = [j]
    for i in range(len(radii) - 2, -1, -1):
        w0 = max(0, j - window)
        near = best[i][w0 : j + window + 1]
        top = np.flatnonzero(near >= near.max())
        j = w0 + int(top[np.argmin(abs(w0 + top - j))])
        path.append(j)
    curve = grid[path[::-1]]
    mid_r = (radii[:-1] + radii[1:]) / 2
    mid_phi = np.deg2rad(np.interp(radial_chart(mid_r), radial_chart(radii), curve))
    mid = pair_room(mid_r[:, None] * np.stack((np.cos(mid_phi), np.sin(mid_phi)), -1), k, collision)
    return (curve, float(min(best[-1].max(), mid.min())))


class NoChain(RuntimeError):
    """No usable chain; carries the posture spans so a scene check can show the gap."""

    def __init__(self, message, spans, radii):
        super().__init__(message)
        self.spans, self.radii = (spans, radii)


def choose_chain(tracks, spans, base):
    """First candidate chain whose handoff curves all keep HANDOFF_ROOM clearance.

    Rejects chains before any patch is built; returns the chain, its per-radius
    component selection, handoff curves (degrees at the sampled radii), their
    clearances and the rejected candidates with reasons."""
    collision = Collision(OBSTACLES)
    nr = len(base["radii"])
    rejected = []
    for tries, chain in enumerate(candidate_chains(spans)):
        if tries >= MAX_TRIES:
            break
        selected = np.array([[tracks[f][ir] for f in chain] for ir in range(nr)], dtype=int)
        load_fibre(dict(base, selected=selected))
        curves, rooms, floor, reason = ([], [], np.full(nr, -DEG), None)
        for k, (f, g) in enumerate(zip(chain[:-1], chain[1:])):
            found = handoff_curve(
                k, spans[g][:, 0], spans[f][:, 1], floor, base["radii"], collision
            )
            if found is None or found[1] < HANDOFF_ROOM:
                reason = f"handoff {k}->{k + 1} clearance " + (
                    "none" if found is None else f"{found[1]:.4f} < {HANDOFF_ROOM}"
                )
                break
            curves.append(found[0])
            rooms.append(found[1])
            floor = found[0]
        if reason is None:
            print(f"chain of {len(chain)} families, handoff clearance {np.round(rooms, 3)}")
            return (chain, selected, np.array(curves).reshape(-1, nr), rooms, rejected)
        rejected.append(dict(families=len(chain), tracks=list(chain), reason=reason))
        print(f"chain of {len(chain)} families rejected: {reason}", flush=True)
    common = sorted(
        (round(float(span[:, 0].max()), 1), round(float(span[:, 1].min()), 1)) for span in spans
    )
    raise NoChain(
        f"No chain of families with handoff clearance >= {HANDOFF_ROOM}; tried {len(rejected)}"
        + "".join(f"; rejected {r['families']} families: {r['reason']}" for r in rejected)
        + f"; usable span of each posture at every radius (degrees): {common}",
        spans,
        base["radii"],
    )


def fibre_seed(x, family, fibre, collision, previous=None, kind="chain"):
    """Re-seed only from the given family, allowing either elbow sign."""
    poses, clearance, members = safe_fibre(x, collision, kind=kind)
    q = poses[members[..., family] & (clearance >= MARGIN)]
    if not len(q):
        phi = np.rad2deg(np.arctan2(x[1], x[0]))
        raise RuntimeError(f"No safe seed in family {family} at phi={phi:.4f}")
    if previous is None:
        n = chain_length()
        primary = q[(1 if family == 0 else -1) * q[:, 2] >= 0.3]
        if kind == "chain" and family in (0, n - 1) and n > 1 and len(primary):
            q = primary
        return q[np.argmax(collision.clearance(q))]
    return q[np.argmin(np.linalg.norm(q - previous, axis=1))]


def patch_st(path, band):
    """Global (s, t) -> chart coordinates of a patch over band, clipped to the chart."""
    st = np.asarray(path, dtype=float).copy()
    lo, hi = ((np.asarray(v) + DEG) / (2 * DEG) for v in band_limits(band, st[:, 0]))
    st[:, 1] = (st[:, 1] - lo) / (hi - lo)
    return np.clip(st, 0, 1)


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


class AvailabilityRows:
    """Lazily computed availability rows of one request, filled in chunks and reused."""

    CHUNK = 8

    def __init__(self, requested, kind="sheets"):
        self.requested, self.kind = (requested, kind)
        self.rows = np.zeros((len(requested), chain_length(kind)), dtype=bool)
        self.done = np.zeros(len(requested), dtype=bool)
        self.collision = Collision(OBSTACLES)

    def row(self, i):
        if not self.done[i]:
            j = min(len(self.requested), i + self.CHUNK)
            todo = np.arange(i, j)[~self.done[i:j]]
            self.rows[todo] = fibre_available(
                workspace(self.requested[todo]), self.collision, self.kind
            )
            self.done[todo] = True
        return self.rows[i]

    def fill(self):
        """Complete every row; returns the collision queries this call spent."""
        before = self.collision.queries
        todo = np.flatnonzero(~self.done)
        if len(todo):
            self.rows[todo] = fibre_available(
                workspace(self.requested[todo]), self.collision, self.kind
            )
            self.done[todo] = True
        return self.collision.queries - before


def furthest_family(rows, start, candidates):
    """Candidate family that stays available furthest from row start onwards.

    Furthest-reaching choice is the optimal greedy for covering a sequence by
    availability intervals, so it attains the fewest_switches minimum."""
    alive = np.asarray(candidates, dtype=bool)
    if not alive.any():
        raise RuntimeError("No sheet available at the switch point")
    for i in range(start, len(rows.requested)):
        row = rows.row(i)
        if not (alive & row).any():
            break
        alive = alive & row
        if alive.sum() == 1:
            break
    return int(np.argmax(alive))


# Largest tool deviation from the chord allowed for a continuous joint step; a larger one is
# a reconfiguration with the flow off, so it is recorded as a reset.
TRACK = 0.002


def tracks(a, b, x_a, x_b):
    """Whether the joint step a -> b keeps the tool on the chord x_a -> x_b."""
    t = np.linspace(0, 1, 9)[1:-1, None]
    return bool(
        np.linalg.norm(forward(a + t * (b - a)) - (x_a + t * (x_b - x_a)), axis=1).max() <= TRACK
    )


def geometry_plan(order, density, fibre, diagnostics=True):
    """Plan by warm-started IK over the sheets (any path component of the safe fibre space),
    changing sheet only when the current one fails. plan_seconds = start and switch lookahead, IK, resets
    and certification. The lookahead reads availability rows only as far as it needs; the
    diagnostic (availability_seconds) completes the remaining rows to give the discrete
    minimum and never steers the greedy switches."""
    start = perf_counter()
    original = request(order, density)
    requested, original_ids = stretch_collocation(original)
    collision = Collision(OBSTACLES)
    rows = AvailabilityRows(requested)
    family = furthest_family(rows, 0, np.ones(chain_length("sheets"), dtype=bool))
    seed = fibre_seed(workspace(requested[0]), family, fibre, collision, kind="sheets")
    q, path, families, breaks, mapping, events = ([], [], [], [], [], [])
    in_sheet = []
    for st in requested:
        x = workspace(st)
        branch = None
        try:
            try:
                candidate = solve_ik(x, seed, collision, margin=MARGIN)
                if not memberships(candidate, x, "sheets")[family]:
                    raise RuntimeError("IK left current sheet")
                if q:
                    if not tracks(seed, candidate, workspace(path[-1]), x):
                        raise RuntimeError("IK step left the tool path")
                    certify(np.array([[seed, candidate]]), collision)
            except RuntimeError as first:
                # Same-sheet retry from the nearest fibre pose; a certified step that tracks
                # the path is no reset.
                retry = fibre_seed(x, family, fibre, collision, previous=seed, kind="sheets")
                candidate = solve_ik(x, retry, collision, margin=MARGIN)
                if not memberships(candidate, x, "sheets")[family]:
                    raise RuntimeError("IK left current sheet") from first
                if q and not tracks(seed, candidate, workspace(path[-1]), x):
                    branch = candidate
                    raise RuntimeError(f"Other branch of the sheet only: {first}") from first
                if q:
                    certify(np.array([[seed, candidate]]), collision)
        except RuntimeError as error:
            reason = str(error)
            here = rows.row(max(len(mapping) - 1, 0)) & rows.row(len(mapping))
            # The current sheet's other branch is a candidate too, when a retry found it.
            here[family] = branch is not None
            switch_x = workspace(path[-1])
            other = furthest_family(rows, len(mapping), here)
            if other == family:
                # Reach the current sheet's other branch by an in-sheet reset at the
                # previous point (flow off): a reset, but not a sheet switch.
                reset = solve_ik(switch_x, branch, collision, margin=MARGIN)
                if not memberships(reset, switch_x, "sheets")[family] or not tracks(
                    reset, branch, switch_x, x
                ):
                    raise RuntimeError(f"No reset reaches the path at {x}") from error
                in_sheet.append(
                    dict(
                        edge=len(q) - 1,
                        family=family,
                        cause=reason,
                        phi_degrees=float(np.rad2deg(np.arctan2(switch_x[1], switch_x[0]))),
                    )
                )
            else:
                reset = fibre_seed(switch_x, other, fibre, collision, kind="sheets")
                events.append(
                    dict(
                        edge=len(q) - 1,
                        from_family=family,
                        to_family=other,
                        cause=reason,
                        phi_degrees=float(np.rad2deg(np.arctan2(switch_x[1], switch_x[0]))),
                    )
                )
            breaks.append(True)
            q.append(reset)
            path.append(path[-1].copy())
            families.append(other)
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
        in_sheet_resets=len(in_sheet),
        in_sheet_reset_events=in_sheet,
        guaranteed_clearance_lower_bound=lower,
        plan_seconds=plan_seconds,
        plan_collision_queries=collision.queries + rows.collision.queries,
        plan_ik_solves=collision.ik_solves,
    )
    if diagnostics:
        diag_start = perf_counter()
        analysis_queries = rows.fill()
        available = rows.rows
        if not available.any(axis=1).all():
            raise RuntimeError("No chain family covers a path sample")
        forced = np.where(available.sum(axis=1) == 1, np.argmax(available, axis=1), -1)
        minimum = fewest_switches(available)
        info.update(
            theoretical_minimum=minimum,
            availability_seconds=perf_counter() - diag_start,
            availability_analysis_collision_queries=analysis_queries,
        )
        data.update(availability=available, forced_family=forced, availability_st=requested)
        # Warm-started IK can lose a narrow sheet before its sampled availability ends,
        # so greedy may exceed the minimum; it can never beat it.
        info["switch_excess"] = len(events) - minimum
        if len(events) < minimum:
            raise RuntimeError(
                f"Greedy switches {len(events)} below forced-region minimum {minimum}, m{order} {density}"
            )
    return (data, info)


def gilbert(width, height):
    """Generalised Hilbert order of a width x height grid, from (0, 0) to (width - 1, 0)
    (after J. Cervany's gilbert2d); steps are to grid neighbours except, for some odd
    sizes, one diagonal step."""

    def sign(v):
        return int(v > 0) - int(v < 0)

    def walk(x, y, ax, ay, bx, by):
        w, h = (abs(ax + ay), abs(bx + by))
        dax, day, dbx, dby = (sign(ax), sign(ay), sign(bx), sign(by))
        if h == 1:
            for i in range(w):
                yield (x + i * dax, y + i * day)
            return
        if w == 1:
            for i in range(h):
                yield (x + i * dbx, y + i * dby)
            return
        ax2, ay2, bx2, by2 = (ax // 2, ay // 2, bx // 2, by // 2)
        w2, h2 = (abs(ax2 + ay2), abs(bx2 + by2))
        if 2 * w > 3 * h:
            if w2 % 2 and w > 2:
                ax2, ay2 = (ax2 + dax, ay2 + day)
            yield from walk(x, y, ax2, ay2, bx, by)
            yield from walk(x + ax2, y + ay2, ax - ax2, ay - ay2, bx, by)
        else:
            if h2 % 2 and h > 2:
                bx2, by2 = (bx2 + dbx, by2 + dby)
            yield from walk(x, y, bx2, by2, ax2, ay2)
            yield from walk(x + bx2, y + by2, ax, ay, bx - bx2, by - by2)
            yield from walk(
                x + (ax - dax) + (bx2 - dbx),
                y + (ay - day) + (by2 - dby),
                -bx2,
                -by2,
                -(ax - ax2),
                -(ay - ay2),
            )

    return np.array(list(walk(0, 0, int(width), 0, 0, int(height))), dtype=int).reshape(-1, 2)


def mass_cdf(density):
    """Cumulative target mass in the angular chart coordinate v (area-preserving chart)
    and its inverse; the gradient density rises linearly from 0.5 to 1.5 in v."""
    if density == "uniform":
        return (lambda v: np.asarray(v, dtype=float), lambda g: np.asarray(g, dtype=float))
    return (
        lambda v: 0.5 * np.asarray(v) + 0.5 * np.asarray(v) ** 2,
        lambda g: -0.5 + np.sqrt(0.25 + 2 * np.asarray(g)),
    )


def band_v(band, s):
    """Angular chart limits (v in [0, 1]) of a band at radial chart coordinates s."""
    return tuple((np.asarray(v) + DEG) / (2 * DEG) for v in band_limits(band, s))


def patch_mass(band, density):
    """Target mass w_i of the band's own region (between its handoff curves, unpadded)."""
    cdf, _ = mass_cdf(density)
    s_fine = np.linspace(0, 1, 1025)
    lo, hi = band_v(band, s_fine)
    return float(np.trapezoid(cdf(hi) - cdf(lo), s_fine))


def source_piece(region, density, rows, columns):
    """Own source piece of one patch: a rows x columns cell grid of the unit square,
    in generalised Hilbert order from its low-angle side to its high-angle side, mapped
    into the patch region by the Knothe-Rosenblatt map of the conditional target density
    (radial marginal, then angle given radius). Returns chart points (s, v)."""
    if not columns:
        return np.empty((0, 2))
    cdf, inverse = mass_cdf(density)
    s_fine = np.linspace(0, 1, 2049)
    lo, hi = band_v(region, s_fine)
    marginal = cdf(hi) - cdf(lo)
    cumulative = np.r_[0, np.cumsum((marginal[1:] + marginal[:-1]) / 2 * np.diff(s_fine))]
    order = gilbert(columns, rows)
    u = (order + 0.5) / [columns, rows]
    s = np.interp(u[:, 1] * cumulative[-1], cumulative, s_fine)
    lo, hi = band_v(region, s)
    v = inverse(cdf(lo) + u[:, 0] * (cdf(hi) - cdf(lo)))
    return np.stack((s, v), -1)


def allocate(shares, total):
    """Largest-remainder integer split of total in proportion to shares."""
    raw = np.asarray(shares, dtype=float) / np.sum(shares) * total
    counts = np.floor(raw).astype(int)
    counts[np.argsort(counts - raw)[: total - counts.sum()]] += 1
    return counts


def kinematics_plan(order, density, patches, bands, curves, s_nodes):
    """Proposition construction: patches P_i between handoff curves, target mass split
    w_i = mu(P_i), each patch traversed once by its own source piece carrying its
    conditional density, consecutive pieces joined through one handoff point per curve."""
    start = perf_counter()
    k, n = (len(patches), 2**order)
    curves = np.asarray(curves, dtype=float).reshape(-1, len(s_nodes))
    edges = [np.full(len(s_nodes), -DEG), *curves, np.full(len(s_nodes), DEG)]
    regions = [np.stack((s_nodes, edges[f], edges[f + 1])) for f in range(k)]
    shares = [patch_mass(region, density) for region in regions]
    columns = allocate(shares, n)
    originals = [source_piece(regions[f], density, n, columns[f]) for f in range(k)]
    inner = [o[0, 0] if len(o) else 0.5 / n for o in originals]
    handoffs = []
    for f, c in enumerate(curves):
        s_h = (inner[f] + inner[f + 1]) / 2
        handoffs.append(np.array([s_h, (np.interp(s_h, s_nodes, c) + DEG) / (2 * DEG)]))
    q, path, families, ids, ends = ([], [], [], [], [])
    for family in range(k):
        stops = [originals[family]]
        if family:
            stops.insert(0, handoffs[family - 1][None])
        if family < k - 1:
            stops.append(handoffs[family][None])
        st, st_ids = stretch_collocation(np.vstack(stops), bands[family])
        st_ids = st_ids[(1 if family else 0) : (-1 if family < k - 1 else None)]
        ids.append(sum(len(p) for p in path) + st_ids)
        q.append(patch_lookup(patch_st(st, bands[family]), patches[family]))
        path.append(st)
        families.append(np.full(len(st), family))
        ends.append(sum(len(p) for p in path) - 1)
    q, path, families = (np.concatenate(q), np.vstack(path), np.concatenate(families))
    breaks = np.zeros(len(q) - 1, dtype=bool)
    breaks[ends[:-1]] = True
    collision = Collision(OBSTACLES)
    lower = certify(np.stack((q[:-1][~breaks], q[1:][~breaks]), 1), collision)
    return (
        dict(
            q=q,
            collocation_st=path,
            waypoint_indices=np.concatenate(ids),
            requested_st=np.vstack(originals),
            family=families,
            break_edges=breaks,
            handoff_edges=np.asarray(ends[:-1], dtype=int),
        ),
        dict(
            switches=k - 1,
            in_sheet_resets=0,
            theoretical_minimum=k - 1,
            switch_events=[
                dict(
                    edge=int(ends[f]),
                    from_family=f,
                    to_family=f + 1,
                    cause="Scheduled overlap handoff",
                    phi_degrees=float(-DEG + 2 * DEG * handoffs[f][1]),
                )
                for f in range(k - 1)
            ],
            guaranteed_clearance_lower_bound=lower,
            plan_seconds=perf_counter() - start,
            plan_collision_queries=collision.queries,
            plan_ik_solves=0,
            mass_shares=[float(w) for w in shares],
            waypoint_shares=[float(c / n) for c in columns],
            family_waypoint_counts=[len(o) for o in originals],
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
            **tool_travel(q, breaks, data["waypoint_indices"], data.get("handoff_edges", ())),
        ),
        x,
    )


def tool_travel(q, breaks, ids, handoff_edges=()):
    """Tool path length over continuous stretches, and the part of it spent travelling to
    and from handoff points (kinematics-first resets off the requested waypoints; a
    geometry-first reset lies on its own path, so it has none)."""
    step = np.linalg.norm(np.diff(forward(q), axis=0), axis=1)
    step[breaks] = 0
    ids = np.asarray(ids)
    handoff = 0.0
    for e in np.asarray(handoff_edges, dtype=int):
        before, after = (ids[ids <= e], ids[ids > e])
        if len(before):
            handoff += step[before.max() : e].sum()
        if len(after):
            handoff += step[e + 1 : after.min()].sum()
    return dict(tool_path_length=float(step.sum()), handoff_travel_length=float(handoff))


GRID_PHI, GRID_R = (241, 13)


# Angular padding (degrees) of each patch beyond its handoff points.
PATCH_PAD = 0.3


def row_curves(report):
    """Handoff curves resampled at the patch rows (s = i/NR), so their kinks fall on rows
    where the piecewise-linear chart is exact; returns (curves, s_nodes)."""
    s_rows = np.linspace(0, 1, NR + 1)
    curves = np.array(report["handoff_curves"], dtype=float).reshape(-1, len(report["handoff_s"]))
    return (np.array([np.interp(s_rows, report["handoff_s"], c) for c in curves]), s_rows)


def patch_bands(curves, s_nodes):
    """Each chain family's patch band: between its handoff curves, padded by PATCH_PAD."""
    curves = np.asarray(curves, dtype=float).reshape(-1, len(s_nodes))
    k = len(curves) + 1
    edge = np.full(len(s_nodes), DEG)
    bands = []
    for f in range(k):
        lo = -edge if f == 0 else np.maximum(-DEG, curves[f - 1] - PATCH_PAD)
        hi = edge if f == k - 1 else np.minimum(DEG, curves[f] + PATCH_PAD)
        bands.append(np.stack((np.asarray(s_nodes, dtype=float), lo, hi)))
    return bands


def availability_grid():
    """Availability samples for the figure on a (phi, r) grid, same test as the planner.

    A sample is available for a family when some fibre pose of that family has clearance >= MARGIN
    (801 first-joint samples per elbow branch); absence of a sample is not an impossibility proof."""
    phi = np.linspace(-DEG, DEG, GRID_PHI)
    radius = np.linspace(RMIN, RMAX, GRID_R)
    p, r = np.meshgrid(np.deg2rad(phi), radius)
    x = (r[..., None] * np.stack((np.cos(p), np.sin(p)), -1)).reshape(-1, 2)
    available = fibre_available(x, Collision(OBSTACLES), "sheets")
    return dict(phi=phi, radius=radius, available=available.reshape(GRID_R, GRID_PHI, -1))


ORDERS = (2, 3, 4, 5)


def family_patch(family, band, k):
    """Build and time one family's certified patch (module level so workers can run it)."""
    start = perf_counter()
    q, info = build_patch(
        band,
        OBSTACLES,
        seed_function=lambda x, c: fibre_seed(x, family, None, c),
        margin=0.004,
        radial_range=(RMIN, RMAX),
        reverse_theta=family == k - 1 and k > 1,
    )
    info.update(seconds=perf_counter() - start, family=family, band=np.asarray(band).tolist())
    return (q, info)


def plan_request(method, order, density, patches, bands, curves, s_nodes):
    """Plan and evaluate one request; plan and evaluation are timed inside the worker."""
    if method == "geometry":
        data, info = geometry_plan(order, density, None)
    else:
        data, info = kinematics_plan(order, density, patches, bands, curves, s_nodes)
    evaluation_start = perf_counter()
    metrics, evaluated_x = evaluate(data, density)
    info["evaluation_seconds"] = perf_counter() - evaluation_start
    return (data, info, metrics, evaluated_x)


def parallel(function, args, jobs, fibre):
    """Map function over argument tuples; workers get this run's fibre, not the file (--check)."""
    jobs = min(len(args), jobs or os.cpu_count() or 1)
    if jobs <= 1:
        return [function(*a) for a in args]
    with ProcessPoolExecutor(jobs, initializer=load_fibre, initargs=(fibre,)) as pool:
        return list(pool.map(function, *zip(*args)))


def run(orders=(3, 4, 5), fibre_only=False, jobs=None):
    """Fibre, handoffs, patches, then every request. Patches and requests are independent,
    so they run in worker processes; each is timed inside its worker (CPU contention
    makes those timings indicative; the benchmark gives the reported ones)."""
    if any(order not in ORDERS for order in orders):
        raise ValueError(f"This sweep is limited to Hilbert orders {ORDERS}")
    fibre, report = fibre_analysis()
    store.save_npz(OUT / "fibre.npz", **fibre)
    store.save_json(OUT / "fibre.json", report)
    grid = availability_grid()
    store.save_npz(OUT / "availability_grid.npz", **grid)
    c, cover = cover_number(grid["available"])
    k = report["chain_length"]
    print(
        f"sheets {report['sheets']}, c = {c} (sheets {cover}); kinematics-first uses {k} patches "
        f"in sheets {report['chain_sheets']}, so {k - 1} resets against the bound c - 1 = {c - 1}",
        flush=True,
    )
    if fibre_only:
        return
    curves, s_nodes = row_curves(report)
    bands = patch_bands(curves, s_nodes)
    k = len(bands)
    built = parallel(family_patch, [(f, bands[f], k) for f in range(k)], jobs, fibre)
    patches, preprocess = ([q for q, _ in built], [info for _, info in built])
    for family, q in enumerate(patches):
        store.save_npz(OUT / f"patch_{family}.npz", q=q, band=bands[family])
        lo, hi = (np.round(v, 1).tolist() for v in bands[family][1:])
        print(f"Certified patch {family}, phi from {lo} to {hi}", flush=True)
    keys = [
        (method, order, density)
        for order in sorted(set(orders))
        for density in ("uniform", "gradient")
        for method in ("geometry", "kinematics")
    ]
    args = [(*key, patches, bands, curves, s_nodes) for key in keys]
    done = parallel(plan_request, args, jobs, fibre)
    rows, events = ([], {})
    for (method, order, density), (data, info, metrics, evaluated_x) in zip(keys, done):
        events[f"{method}_m{order}_{density}"] = info.pop("switch_events")
        row = dict(
            method=method,
            order=order,
            density=density,
            waypoints=4**order,
            collocation_points=len(data["q"]),
            preprocessing_seconds=report["handoff_seconds"] + sum(p["seconds"] for p in preprocess)
            if method == "kinematics"
            else 0,
            **info,
            **metrics,
        )
        # Proposition: every plan needs >= c - 1 switches; the patch construction attains it.
        row.update(
            cover_number=c,
            bound_c_minus_1=c - 1,
            meets_bound=bool(
                row["switches"] == c - 1
                if method == "kinematics"
                else row["theoretical_minimum"] >= c - 1
            ),
        )
        rows.append(row)
        store.save_npz(OUT / f"{method}_m{order}_{density}.npz", **data, evaluation_x=evaluated_x)
        print(
            f"{method} m{order} {density}: switches={info['switches']}, minimum={info['theoretical_minimum']}",
            flush=True,
        )
    result = dict(
        requests=rows,
        switch_events=events,
        preprocessing=preprocess,
        fibre=report,
        handoff_s=s_nodes.tolist(),
        handoff_curves=curves.tolist(),
        split_phi_degrees=[float(np.interp(0.5, s_nodes, c)) for c in curves],
        family_bands=[b.tolist() for b in bands],
        orders=sorted(set(orders)),
        sheets=report["sheets"],
        cover_number=c,
        cover_sheets=cover,
        chain_sheets=report["chain_sheets"],
        minimum_statement="Fewest sheet changes along the sampled sheet availability sequence (dynamic program). This is the exact combinatorial minimum conditional on the sampled safe-fibre availability model, not a continuous-space impossibility proof. Geometry-first never goes below it; any excess is recorded as switch_excess.",
        discontinuity_statement="Switch edges are recorded discrete posture resets. They are excluded from interpolation, collision certification and tool-path coverage. No collision-free transition between disconnected sheets is claimed.",
        kinematics_ordering="Patches lie between handoff curves that follow the sheet overlaps. The target mass is split by patch, w_i = mu(P_i), into 2**m radial rows times a largest-remainder share of 2**m angular columns, so 4**m waypoints in all. Each patch is traversed once by its own source piece: a generalised Hilbert order of its cell grid, from its low-angle side to its high-angle side, mapped into the patch by the Knothe-Rosenblatt map of its conditional density. Consecutive pieces meet at one handoff point on each curve near the inner radius.",
        evaluation_statement="Both methods: 33x129 sampled target grid, <=.0005 tool-motion sampling on continuous stretches, 4x8 area-coordinate density bins evaluated at original waypoints only. Clearance lower bounds certify every continuous stretch; coverage, density and tracking are sampled. Timings are recorded in JSON as secondary diagnostics, not plotted.",
    )
    store.save_json(OUT / "results.json", result)


def scout_patch(family, band, k):
    """Patch build outcome for scout: guaranteed clearance, or the failure."""
    try:
        return f"ok, guaranteed clearance {family_patch(family, band, k)[1]['guaranteed_clearance_lower_bound']:.4f}"
    except RuntimeError as error:
        return f"FAILED: {error}"


def scout_minimum(order, density):
    """Fewest resets any planner needs on this request, or None if a target is uncovered."""
    requested = stretch_collocation(request(order, density))[0]
    available = fibre_available(workspace(requested), Collision(OBSTACLES), "sheets")
    return fewest_switches(available) if available.any(axis=1).all() else None


def scout_plot(spans, radii, verdict, chain=(), curves=None, s_nodes=None):
    """Scene layout and posture spans: scout.png in the results folder (overwritten)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Wedge

    fig, (ax_w, ax_s) = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    ax_w.add_patch(Wedge((0, 0), RMAX, -DEG, DEG, width=RMAX - RMIN, fc="0.88", ec="0.6"))
    for i, (cx, cy, radius) in enumerate(OBSTACLES):
        ax_w.add_patch(Circle((cx, cy), radius, fc="0.45", ec="0.2"))
        ax_w.annotate(f"obstacle {i}", (cx, cy + radius + 0.05), ha="center", fontsize=8)
    ax_w.plot(0, 0, "ks")
    ax_w.set(aspect="equal", xlim=(-0.2, RMAX + 0.2), ylim=(-RMAX * 0.8, RMAX * 0.8))
    ax_w.set_title("Scene")
    from mapchain.figures.palette import categorical

    step = (radii[1] - radii[0]) / (len(spans) + 2) if len(radii) > 1 else 0.01
    for i, span in enumerate(spans):
        used = i in chain
        label = f"posture {list(chain).index(i) + 1}" if used else None
        for j, (lo, hi) in enumerate(span):
            ax_s.plot(
                [lo, hi],
                [radii[j] + (i - len(spans) / 2) * step] * 2,
                color=categorical(i),
                lw=4 if used else 1.2,
                alpha=1 if used else 0.6,
                solid_capstyle="butt",
                label=label if j == 0 else None,
            )
    grid = np.linspace(-DEG, DEG, 901)
    for j, r in enumerate(radii):
        covered = np.zeros(len(grid), dtype=bool)
        for span in spans:
            covered |= (grid >= span[j, 0]) & (grid <= span[j, 1])
        gaps = np.flatnonzero(np.diff(np.r_[0, (~covered).astype(int), 0]))
        half = (len(spans) / 2 + 1) * step
        for a, b in zip(gaps[::2], gaps[1::2]):
            ax_s.fill_between(
                [grid[a], grid[b - 1]], r - half, r + half, color="red", alpha=0.35, lw=0
            )
    if curves is not None:
        rr = np.sqrt(RMIN**2 + np.asarray(s_nodes) * (RMAX**2 - RMIN**2))
        for c in curves:
            ax_s.plot(c, rr, "k--", lw=1.2, label="handoff" if c is curves[0] else None)
    ax_s.set(
        xlim=(-DEG, DEG),
        xlabel="phi (deg)",
        ylabel="r",
        title="Usable span of each posture per radius (red = no posture)",
    )
    if ax_s.get_legend_handles_labels()[0]:
        ax_s.legend(fontsize=7, loc="upper right")
    fig.suptitle(verdict, fontsize=9)
    path = OUT / "scout.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  plot: {path}")


def scout(orders=(3, 4, 5), jobs=None):
    """Quick scene check, nothing saved: kinematics-first resets (chain length - 1) and
    whether its patches build, against the fewest resets geometry-first could achieve
    on each request using any tracked posture (its sampled lower bound)."""
    start = perf_counter()
    try:
        fibre, report = fibre_analysis()
    except NoChain as error:
        print()
        print(f"SCOUT: scene {OBSTACLES.tolist()}, r {RMIN}-{RMAX}")
        print(f"  verdict: NOT USABLE by kinematics-first: {error}")
        scout_plot(error.spans, error.radii, "NOT USABLE: no chain of postures across the sector")
        return
    k = report["chain_length"]
    bands = patch_bands(*row_curves(report))
    built = parallel(scout_patch, [(f, bands[f], k) for f in range(k)], jobs, fibre)
    keys = [(order, density) for order in orders for density in ("uniform", "gradient")]
    minima = parallel(scout_minimum, keys, jobs, fibre)
    print()
    print(f"SCOUT ({perf_counter() - start:.0f} s): scene {OBSTACLES.tolist()}, r {RMIN}-{RMAX}")
    print(
        f"  kinematics-first: {k} postures -> {k - 1} resets at every order; "
        f"handoff clearance {np.round(report['handoff_clearance'], 3).tolist()}"
    )
    for f, outcome in enumerate(built):
        print(f"    patch {f}: {outcome}")
    ready = all(outcome.startswith("ok") for outcome in built)
    for (order, density), minimum in zip(keys, minima):
        gap = "uncovered target" if minimum is None else f"advantage {minimum - (k - 1):+d}"
        print(f"  m{order} {density:8s}: geometry-first needs >= {minimum} resets ({gap})")
    verdict = "kinematics-first patches build" if ready else "PATCHES FAIL, not usable"
    print("  verdict: " + verdict)
    worst = [m - (k - 1) for m in minima if m is not None]
    scout_plot(
        [np.array(span) for span in report["track_spans"]],
        fibre["radii"],
        f"{verdict}; kinematics-first {k - 1} resets, advantage {min(worst, default=0):+d} "
        f"to {max(worst, default=0):+d}",
        report["chain"],
        *row_curves(report),
    )


def kinematics_main(argv=None):
    """Plan the planar 3R comparison: fibre analysis, certified patches and both methods."""
    parser = argparse.ArgumentParser(description=kinematics_main.__doc__)
    parser.add_argument("--orders", nargs="+", type=int, choices=ORDERS, default=[3, 4, 5])
    parser.add_argument("--fibre-only", action="store_true")
    parser.add_argument(
        "--scout", action="store_true", help="quick scene check: reset counts only, nothing saved"
    )
    parser.add_argument(
        "--jobs", type=int, default=None, help="worker processes (default: CPU count; 1 = serial)"
    )
    store.add_check(parser)
    args = parser.parse_args(argv)
    store.configure(args)
    if args.scout:
        scout(args.orders, args.jobs)
        return
    run(args.orders, fibre_only=args.fibre_only, jobs=args.jobs)
    store.finish()
