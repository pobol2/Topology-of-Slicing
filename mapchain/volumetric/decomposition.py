"""Repair by decomposition: near-convex source pieces, each with its own transported path."""

from __future__ import annotations

import argparse
import os
import platform
import time
import traceback
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull

from mapchain import store
from mapchain.evaluation.whole_part import versions
from mapchain.store import ROOT
from mapchain.volumetric.hilbert import hilbert_3d
from mapchain.volumetric.parts import geometry, load_or_create_mesh, normalization_params
from mapchain.volumetric.transport import (
    NESTED_METHOD,
    acceptance_report,
    nested_solve,
    tetrahedralize,
)

SPACING = 0.001


TAU = 1e-09


EPS = 1e-09


CONVEX_RTOL = 1e-09


def decompose(mesh, threshold, seed):
    """CoACD convex hulls (vertex arrays) of a mesh."""
    import coacd

    coacd.set_log_level("error")
    return [
        np.asarray(v, float)
        for v, _ in coacd.run_coacd(
            coacd.Mesh(mesh.vertices, mesh.faces), threshold=threshold, seed=seed
        )
    ]


class Membership:
    """Inside test and exact distance for one closed, consistently oriented surface (libigl)."""

    def __init__(self, surface):
        import igl

        self.igl = igl
        self.V = np.asarray(surface.vertices, float)
        self.F = np.asarray(surface.faces, np.int64)

    def winding(self, points, batch=500000):
        return (
            np.concatenate(
                [
                    self.igl.winding_number(self.V, self.F, points[a : a + batch])
                    for a in range(0, len(points), batch)
                ]
            )
            if len(points)
            else np.empty(0)
        )

    def distance(self, points):
        return (
            np.sqrt(self.igl.point_mesh_squared_distance(points, self.V, self.F)[0])
            if len(points)
            else np.empty(0)
        )

    def outside(self, points):
        """(outside flags, distance of each point flagged outside)."""
        out = self.winding(points) < 0.5
        d = np.zeros(len(points))
        d[out] = self.distance(points[out])
        out &= d > TAU
        return (out, d)


def segment_samples(points):
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    counts = np.maximum(1, np.ceil(lengths / SPACING).astype(np.int64)) + 1
    seg = np.repeat(np.arange(len(lengths)), counts)
    t = (np.arange(counts.sum()) - np.repeat(np.r_[0, np.cumsum(counts)[:-1]], counts)) / np.repeat(
        counts - 1, counts
    )
    return (points[:-1][seg] + t[:, None] * (points[1:] - points[:-1])[seg], seg, lengths)


def path_measurements(member, points):
    """Sampled measurements of one independent path against its assigned piece."""
    wp_out, _ = member.outside(points)
    q, seg, lengths = segment_samples(points)
    out, d = member.outside(q)
    seg_out = np.bincount(seg, weights=out, minlength=len(lengths)) > 0
    depth = np.zeros(len(lengths))
    np.maximum.at(depth, seg[out], d[out])
    return dict(
        waypoints=len(points),
        waypoints_inside=int(len(points) - wp_out.sum()),
        segments=len(lengths),
        exterior_segments=int(seg_out.sum()),
        exterior_segment_fraction=float(seg_out.mean()) if len(lengths) else 0.0,
        max_sampled_excursion=float(depth.max()) if len(depth) else 0.0,
        samples=int(len(q)),
        max_sample_spacing=float((lengths / np.maximum(1, np.ceil(lengths / SPACING))).max()),
        exterior_segment_indices=np.flatnonzero(seg_out),
    )


def surface_report(surface):
    """closed: every edge has at least two faces (pieces touching themselves along an edge give
    non-manifold edges, which is not a leak); winding_consistent: faces consistently oriented."""
    _, counts = np.unique(surface.edges_sorted, axis=0, return_counts=True)
    return dict(
        closed=bool(np.all(counts >= 2)),
        winding_consistent=bool(surface.is_winding_consistent),
        nonmanifold_edges=int(np.sum(counts > 2)),
        boundary_edges=int(np.sum(counts == 1)),
        body_count=int(surface.body_count),
        faces=len(surface.faces),
        surface_volume=float(surface.volume),
    )


def hull_gap_bounds(surface, tol, max_cubes=2000000, max_level=20):
    """Verified convexity, or a certified upper bound U_j on
        delta_j = sup_{z in conv(Omega_j)} dist(z, Omega_j)      (dist = 0 inside Omega_j).
    Branch and bound on cubes of half-size s (r = s sqrt 3) covering conv(Omega_j): a cube meets the
    hull only if its centre is within r of every facet plane; dist(., Omega_j) is 1-Lipschitz, so a
    cube's supremum is at most max(0, sd(c) + r) + EPS, with sd(c) = -d(c) only if c is robustly inside
    (winding within 1/4 of 1 and d > TAU) and +d(c) otherwise. The lower bound L is the largest d at a
    robustly outside centre inside the hull. Cubes with bound <= L are pruned; the rest are split until
    U - L <= tol. Assumption: the generalised winding number is correct to within 1/4 at points
    farther than TAU from the surface."""
    vertices = np.asarray(surface.vertices, float)
    hull = ConvexHull(vertices)
    volume = float(surface.volume)
    base = dict(
        volume=volume, hull_volume=float(hull.volume), volume_over_hull=volume / float(hull.volume)
    )
    if float(hull.volume) - volume <= CONVEX_RTOL * float(hull.volume):
        return dict(
            base,
            status="convex (verified)",
            upper_bound=0.0,
            lower_bound=0.0,
            method=f"piece volume equals its convex-hull volume within relative {CONVEX_RTOL:g}; a closed piece inside its hull with equal volume is the hull, so delta_j = 0",
        )
    import igl

    V = np.asarray(surface.vertices, float)
    F = np.asarray(surface.faces, np.int64)
    eq = hull.equations
    lo, hi = (vertices.min(axis=0), vertices.max(axis=0))
    s = float((hi - lo).max()) / 16
    n = np.ceil((hi - lo) / (2 * s)).astype(int)
    centres = np.stack(
        np.meshgrid(*[lo[k] + s + 2 * s * np.arange(n[k]) for k in range(3)], indexing="ij"), -1
    ).reshape(-1, 3)
    L = 0.0
    level = 0
    evaluated = 0
    ambiguous = 0
    while True:
        r = s * np.sqrt(3)
        facet = centres @ eq[:, :3].T + eq[:, 3]
        centres = centres[(facet <= r + EPS).all(axis=1)]
        in_hull = (centres @ eq[:, :3].T + eq[:, 3] <= -EPS).all(axis=1)
        d = np.sqrt(igl.point_mesh_squared_distance(centres, V, F)[0])
        w = igl.winding_number(V, F, centres)
        inside = (np.abs(w - 1) < 0.25) & (d > TAU)
        outside = (np.abs(w) < 0.25) & (d > TAU)
        ambiguous += int(np.sum(~inside & ~outside & (d > TAU)))
        evaluated += len(centres)
        lower = np.where(in_hull & outside, d, 0.0)
        L = max(L, float(lower.max()) if len(lower) else 0.0)
        bound = np.maximum(0.0, np.where(inside, -d, d) + r) + EPS
        keep = bound > L
        U = max(L, float(bound[keep].max()) if keep.any() else L)
        budget = 8 * int(keep.sum()) > max_cubes
        if U - L <= tol or level >= max_level or (not keep.any()) or budget:
            stopped = (
                "tolerance"
                if U - L <= tol
                else "cube budget"
                if budget
                else "max level"
                if keep.any()
                else "all pruned"
            )
            return dict(
                base,
                status="upper bound (certified)",
                upper_bound=U,
                lower_bound=L,
                gap=U - L,
                tolerance=tol,
                stopped_by=stopped,
                levels=level,
                final_half_size=s,
                evaluated_centres=evaluated,
                ambiguous_centres=ambiguous,
                method="branch and bound on cubes; exact point-triangle distance (libigl AABB); sign from the generalised winding number",
                assumption="winding number correct to within 1/4 farther than TAU from the surface; ambiguous centres use +distance (conservative)",
            )
        s /= 2
        level += 1
        offsets = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]) * s
        centres = (centres[keep][:, None, :] + offsets[None]).reshape(-1, 3)


def colours():
    """tab10 without red, which marks excursions."""
    import matplotlib.pyplot as plt

    return [c for i, c in enumerate(plt.get_cmap("tab10").colors) if i != 3]


def surfaces_from(npz):
    data = np.load(npz)
    return [
        trimesh.Trimesh(data[f"v{j}"], data[f"f{j}"], process=False)
        for j in range(int(data["count"]))
    ]


OUT = store.OUT / "volumetric" / "decomposition"
CACHE = store.ROOT / ".cache" / "coacd"


FACES = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])


L_BOX_BOXES = (
    ("A", [-1.5, -1.5, -0.5], [1.5, -0.5, 0.5]),
    ("B", [-1.5, -0.5, -0.5], [-0.5, 1.5, 0.5]),
)


def coacd_hulls(obj, mesh, threshold, seed):
    """CoACD hulls, cached in .cache/coacd (CoACD is deterministic for a seed)."""
    path = CACHE / f"coacd_{obj}_t{threshold:g}_s{seed}.npz"
    if path.exists():
        data = np.load(path)
        return ([data[f"h{i}"] for i in range(int(data["count"]))], float(data["seconds"]), True)
    start = time.perf_counter()
    hulls = decompose(mesh, threshold, seed)
    seconds = time.perf_counter() - start
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, count=len(hulls), seconds=seconds, **{f"h{i}": h for i, h in enumerate(hulls)})
    return (hulls, seconds, False)


def hull_labels(centres, hulls):
    """Owner = hull with the largest depth min_f -(n_f.x+d_f) at the tetrahedron centre."""
    depth = np.stack(
        [
            -(centres @ e[:, :3].T + e[:, 3]).max(axis=1)
            for e in (ConvexHull(h).equations for h in hulls)
        ],
        axis=1,
    )
    return (depth.argmax(axis=1), int(np.sum(depth.max(axis=1) < 0)))


def box_labels(centres):
    centre, scale = normalization_params(load_or_create_mesh("l_box", ROOT / "models"))
    inside = np.stack(
        [
            np.all(
                (centres >= (np.asarray(lo) - centre) * scale + 0.5)
                & (centres <= (np.asarray(hi) - centre) * scale + 0.5),
                axis=1,
            )
            for _, lo, hi in L_BOX_BOXES
        ],
        axis=1,
    )
    if not np.all(inside.sum(axis=1) == 1):
        raise RuntimeError("L-bracket tetrahedra do not conform to the two boxes")
    return inside.argmax(axis=1)


def face_pairs(tets):
    faces = np.sort(tets[:, FACES].reshape(-1, 3), axis=1)
    _, inverse, counts = np.unique(faces, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    order = np.argsort(inverse, kind="stable")
    shared = counts[inverse[order]] == 2
    return np.column_stack([order[shared][0::2] // 4, order[shared][1::2] // 4])


def components(labels, pairs, n):
    same = labels[pairs[:, 0]] == labels[pairs[:, 1]]
    p = pairs[same]
    return connected_components(
        coo_matrix((np.ones(len(p)), (p[:, 0], p[:, 1])), shape=(n, n)), directed=False
    )[1]


def merge_small(labels, pairs, volumes, min_volume):
    """Face-connected components per label; components below min_volume merge into the neighbour
    they share most faces with, one at a time. Returns (piece of each tetrahedron, merges)."""
    merges = 0
    while True:
        comp = components(labels, pairs, len(labels))
        sizes = np.bincount(comp, weights=volumes)
        cross = pairs[labels[pairs[:, 0]] != labels[pairs[:, 1]]]
        a, b = (comp[cross[:, 0]], comp[cross[:, 1]])
        for s in np.flatnonzero(sizes < min_volume)[np.argsort(sizes[sizes < min_volume])]:
            touching = np.concatenate([b[a == s], a[b == s]])
            if len(touching):
                labels[comp == s] = labels[
                    np.flatnonzero(comp == np.bincount(touching).argmax())[0]
                ]
                merges += 1
                break
        else:
            return (np.unique(comp, return_inverse=True)[1].ravel(), merges)


def piece_surfaces(vertices, tets, piece_of):
    """Outward boundary of each piece: faces whose neighbour is in another piece or outside."""
    faces = np.sort(tets[:, FACES].reshape(-1, 3), axis=1)
    inverse = np.unique(faces, axis=0, return_inverse=True)[1].ravel()
    key = inverse * (piece_of.max() + 1) + np.repeat(piece_of, 4)
    kinv, kcount = np.unique(key, return_inverse=True, return_counts=True)[1:]
    boundary = np.flatnonzero(kcount[kinv.ravel()] == 1)
    t, f = (boundary // 4, boundary % 4)
    surfaces = []
    for j in range(piece_of.max() + 1):
        keep = piece_of[t] == j
        tri = tets[t[keep]][np.arange(keep.sum())[:, None], FACES[f[keep]]]
        a, b, c = (vertices[tri[:, i]] for i in range(3))
        flip = (
            np.einsum("ij,ij->i", np.cross(b - a, c - a), vertices[tets[t[keep], f[keep]]] - a) > 0
        )
        tri[flip] = tri[flip][:, [0, 2, 1]]
        s = trimesh.Trimesh(vertices, tri, process=False)
        s.remove_unreferenced_vertices()
        surfaces.append(s)
    return surfaces


def partition_checks(mesh, samples, tets, volumes, piece_of, pairs, surfaces):
    k = len(surfaces)
    piece_volume = np.bincount(piece_of, weights=volumes, minlength=k)
    reports = [surface_report(s) for s in surfaces]
    in_solid = Membership(mesh).winding(samples) > 0.5
    in_piece = np.stack([Membership(s).winding(samples) > 0.5 for s in surfaces], axis=1)
    count = in_piece.sum(axis=1)
    comp = components(piece_of, pairs, len(tets))
    per_piece = np.array([len(np.unique(comp[piece_of == j])) for j in range(k)])
    covered = abs(volumes.sum() - mesh.volume) / mesh.volume
    surface_residual = max(
        (abs(r["surface_volume"] - v) / v for r, v in zip(reports, piece_volume))
    )
    checks = dict(
        coverage_volume=dict(
            method="sum of tetrahedron volumes vs polygonal-solid volume",
            residual=covered,
            tolerance=1e-05,
            passed=bool(covered <= 1e-05),
        ),
        coverage_samples=dict(
            method="evaluation samples inside the polygonal solid (winding > 1/2) that lie in no piece",
            samples_in_solid=int(in_solid.sum()),
            samples_outside_polygonal_solid=int((~in_solid).sum()),
            count=int(((count == 0) & in_solid).sum()),
            tolerance=0,
            passed=bool(((count == 0) & in_solid).sum() == 0),
        ),
        ownership=dict(
            method="tetrahedron-to-piece ownership array: every tetrahedron has exactly one owner",
            tetrahedra=len(tets),
            unowned=int(np.sum(piece_of < 0)),
            passed=bool(np.all(piece_of >= 0)),
        ),
        overlap_samples=dict(
            method="evaluation samples that lie in two or more pieces",
            count=int((count > 1).sum()),
            tolerance=0,
            passed=bool((count > 1).sum() == 0),
        ),
        connectivity=dict(
            method="face-connected components of each piece",
            components_per_piece=per_piece,
            passed=bool(np.all(per_piece == 1)),
        ),
        volume_conservation=dict(
            method="sum of piece volumes vs object tetrahedral volume; each piece surface volume vs its tetrahedral volume",
            sum_residual=float(abs(piece_volume.sum() - volumes.sum()) / volumes.sum()),
            max_surface_residual=float(surface_residual),
            tolerance=1e-06,
            passed=bool(
                abs(piece_volume.sum() - volumes.sum()) <= 1e-12 * volumes.sum()
                and surface_residual <= 1e-06
            ),
        ),
        piece_surfaces=dict(
            method="every piece surface closed and consistently oriented (needed for winding numbers)",
            passed=bool(all(r["closed"] and r["winding_consistent"] for r in reports)),
        ),
    )
    checks["passed"] = all(c["passed"] for c in checks.values())
    return (checks, piece_volume, reports)


def solve_piece(j, vertices, tets, surface, m):
    """Nested solve on one piece in its own process. Unit nodal density on the piece's tetrahedra:
    the solver normalises it to total mass one, i.e. the conditional density rho_j = rho / w_j."""
    start = time.perf_counter()
    try:
        result, initialization = nested_solve(surface, vertices, tets, m, np.ones(len(vertices)))
        acceptance = acceptance_report(result, hilbert_3d(m))
        passed = bool(result["converged"] and acceptance["passed"])
        return dict(
            piece=j,
            passed=passed,
            failure=None if passed else "acceptance failed",
            converged=bool(result["converged"]),
            acceptance=acceptance,
            iterations=result["iterations"],
            max_relative_mass_error=float(result["max_relative_mass_error"]),
            max_cell_diameter=float(np.max(result["diameters"])),
            nested_levels=[
                dict(
                    order=level["order"],
                    iterations=level["iterations"],
                    runtime_seconds=level["runtime_seconds"],
                )
                for level in initialization["levels"]
            ],
            centroids=np.asarray(result["centroids"]),
            solve_seconds=time.perf_counter() - start,
        )
    except Exception:
        return dict(
            piece=j,
            passed=False,
            failure=traceback.format_exc(),
            centroids=None,
            solve_seconds=time.perf_counter() - start,
        )


def local_mesh(vertices, tets):
    used, local = np.unique(tets, return_inverse=True)
    return (vertices[used], local.reshape(-1, 4))


def summary(pieces):
    solved = [p for p in pieces if p["solve"]["passed"]]
    m = [p["measurements"] for p in solved]
    bounds = [p["certificate"].get("upper_bound") for p in pieces]
    all_bounds = all(b is not None for b in bounds)
    return dict(
        pieces=len(pieces),
        all_solves_passed=len(solved) == len(pieces),
        failed_pieces=[p["piece_id"] for p in pieces if not p["solve"]["passed"]],
        total_waypoints=sum(x["waypoints"] for x in m),
        waypoints_inside=sum(x["waypoints_inside"] for x in m),
        waypoint_containment=sum(x["waypoints_inside"] for x in m)
        / max(1, sum(x["waypoints"] for x in m)),
        exterior_segments=sum(x["exterior_segments"] for x in m),
        segments=sum(x["segments"] for x in m),
        exterior_segment_fraction=sum(x["exterior_segments"] for x in m)
        / max(1, sum(x["segments"] for x in m)),
        max_sampled_excursion=max((x["max_sampled_excursion"] for x in m), default=None),
        verified_convex_pieces=sum(
            p["certificate"]["status"] == "convex (verified)" for p in pieces
        ),
        max_upper_bound=max(bounds) if all_bounds else None,
        max_upper_bound_status="every piece certified"
        if all_bounds
        else "unavailable: not every piece has a valid bound",
        partial_upper_bounds={p["piece_id"]: b for p, b in zip(pieces, bounds) if b is not None},
    )


def decompose_main(argv=None):
    """Decompose one object into near-convex pieces and transport a Hilbert path into each."""
    parser = argparse.ArgumentParser(description=decompose_main.__doc__)
    parser.add_argument("--object", required=True, choices=("l_box", "torus", "bunny"))
    parser.add_argument(
        "--threshold", type=float, default=0.05, help="CoACD concavity threshold (torus, bunny)"
    )
    parser.add_argument("--seed", type=int, default=0, help="CoACD seed")
    parser.add_argument(
        "--min-fraction",
        type=float,
        default=0.01,
        help="Merge components below this volume fraction",
    )
    parser.add_argument("--m", type=int, default=4, help="Hilbert order of every piece")
    parser.add_argument(
        "--tol", type=float, default=0.002, help="Certificate: stop when U_j - L_j <= tol"
    )
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--out",
        help="Folder name under results/volumetric/decomposition (default: the object); keeps other runs intact",
    )
    store.add_check(parser)
    args = parser.parse_args(argv)
    store.configure(args)
    obj = args.object
    folder = OUT / (args.out or obj)
    started = time.perf_counter()
    mesh, b, definition = geometry(obj)
    raw = load_or_create_mesh(obj, ROOT / "models")
    centre, scale = normalization_params(raw)
    start = time.perf_counter()
    vertices, tets, meshing = tetrahedralize(mesh, max_volume=None)
    vertices = np.asarray(vertices, float)
    tets = np.asarray(tets, np.int64)
    corners = vertices[tets]
    signed = (
        np.einsum(
            "ij,ij->i",
            np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]),
            corners[:, 3] - corners[:, 0],
        )
        / 6
    )
    tets[signed < 0] = tets[signed < 0][:, [0, 2, 1, 3]]
    volumes = np.abs(signed)
    tet_seconds = time.perf_counter() - start
    start = time.perf_counter()
    pairs = face_pairs(tets)
    if obj == "l_box":
        method = "explicit two boxes (exact convex control): each tetrahedron owned by the box containing its centre"
        piece_of = box_labels(corners.mean(axis=1))
        merges = 0
        hull_seconds = 0.0
        cached = None
        labels = piece_of
        outside_hulls = 0
        names = [n for n, _, _ in L_BOX_BOXES]
        params = dict(
            boxes={
                n: dict(lower=lo, upper=hi, frame="models/l_box.stl") for n, lo, hi in L_BOX_BOXES
            }
        )
    else:
        method = "CoACD-guided: each tetrahedron owned by the hull containing its centre most deeply (closest if none); face-connected components; components below min_fraction merged into the neighbour sharing most faces"
        hulls, hull_seconds, cached = coacd_hulls(obj, mesh, args.threshold, args.seed)
        labels, outside_hulls = hull_labels(corners.mean(axis=1), hulls)
        piece_of, merges = merge_small(
            labels.copy(), pairs, volumes, args.min_fraction * volumes.sum()
        )
        names = [f"p{j}" for j in range(piece_of.max() + 1)]
        params = dict(
            coacd_threshold=args.threshold,
            coacd_seed=args.seed,
            coacd_hulls=len(hulls),
            min_fraction=args.min_fraction,
            tetrahedra_outside_every_hull=outside_hulls,
            components_merged=merges,
        )
    surfaces = piece_surfaces(vertices, tets, piece_of)
    partition_seconds = time.perf_counter() - start
    start = time.perf_counter()
    checks, piece_volume, surface_reports = partition_checks(
        mesh, np.asarray(b.target_eval), tets, volumes, piece_of, pairs, surfaces
    )
    check_seconds = time.perf_counter() - start
    k = len(surfaces)
    shares = piece_volume / volumes.sum()
    print(f"{obj}: {k} pieces, partition checks passed={checks['passed']}", flush=True)
    start = time.perf_counter()
    workers = args.workers or min(k, os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        solved = list(
            pool.map(
                solve_piece,
                range(k),
                *zip(*[local_mesh(vertices, tets[piece_of == j]) for j in range(k)]),
                surfaces,
                [args.m] * k,
            )
        )
    solve_wall = time.perf_counter() - start
    pieces = []
    paths = {}
    for j, (surface, s) in enumerate(zip(surfaces, solved)):
        t0 = time.perf_counter()
        certificate = hull_gap_bounds(surface, args.tol)
        certificate["certificate_seconds"] = time.perf_counter() - t0
        piece = dict(
            piece_id=f"{obj}-{names[j]}",
            label=names[j],
            tetrahedra=int((piece_of == j).sum()),
            volume=float(piece_volume[j]),
            mass_share=float(shares[j]),
            order=args.m,
            waypoints=8**args.m,
            time_share_per_waypoint=float(shares[j]) / 8**args.m,
            density=dict(
                target="uniform: rho = 1/|Omega| on the represented solid",
                conditional=f"rho_j = rho / w_j = 1/|Omega_j| on piece {names[j]}",
                solver_input="unit nodal density on the piece tetrahedra, normalised by the solver to total mass 1",
            ),
            surface=surface_reports[j],
            certificate=certificate,
            solve={key: v for key, v in s.items() if key != "centroids"},
        )
        if obj != "l_box":
            piece["coacd_hulls"] = sorted(set(labels[piece_of == j].tolist()))
        if s["passed"]:
            t0 = time.perf_counter()
            mm = path_measurements(Membership(surface), s["centroids"])
            paths[f"p{j}"] = s["centroids"]
            paths[f"e{j}"] = mm.pop("exterior_segment_indices")
            mm["measurement_seconds"] = time.perf_counter() - t0
            piece["measurements"] = mm
        pieces.append(piece)
        mm = piece.get("measurements", {})
        print(
            f"  {piece['piece_id']}: w={shares[j]:.4f} solve={('ok' if s['passed'] else 'FAILED')} {certificate['status']} U={certificate.get('upper_bound')} inside={mm.get('waypoints_inside')}/{mm.get('waypoints')} ext={mm.get('exterior_segments')}/{mm.get('segments')} depth={mm.get('max_sampled_excursion')}",
            flush=True,
        )
    record = dict(
        object=obj,
        run_id=f"{obj}-{('boxes' if obj == 'l_box' else 'coacd')}-m{args.m}",
        files=dict(
            record=f"results/volumetric/decomposition/{folder.name}/record.json",
            geometry=f"results/volumetric/decomposition/{folder.name}/geometry.npz (mesh_vertices, mesh_faces, vertices, tetrahedra, piece_of, piece surfaces v<j>/f<j>)",
            paths=f"results/volumetric/decomposition/{folder.name}/paths.npz (p<j>: path of piece j; e<j>: indices of its exterior segments)",
        ),
        geometry=dict(
            model=f"models/{obj}.stl",
            construction=definition,
            coordinates="normalised: x_norm = (x_model - centre) * scale + 0.5, which fits the model bounding box in [0.05, 0.95]^3; lengths are model units times scale",
            centre=centre,
            scale=scale,
            tetrahedra=len(tets),
            tetgen=meshing,
            polygonal_solid_volume=float(mesh.volume),
            tetrahedral_volume=float(volumes.sum()),
        ),
        partition=dict(method=method, parameters=params, pieces=k, checks=checks),
        transport=dict(
            method=NESTED_METHOD,
            order=args.m,
            acceptance="mapchain.volumetric.transport.acceptance_report: per-cell relative mass error < 1e-6, no empty or non-positive cells, finite centroids, centroid i power-nearest to Hilbert site i",
            paths="independent: one Hilbert-ordered centroid polyline per piece; never connected",
        ),
        measurement_protocol=dict(
            domain="the assigned piece Omega_j used by its solve",
            membership="generalised winding number of the closed piece surface (libigl); inside when >= 1/2",
            distance="exact point-to-triangle distance to the piece surface (libigl AABB)",
            boundary_tolerance=f"a sample counts as outside only if winding < 1/2 and distance > {TAU:g}",
            sampling=f"uniform samples along each segment, ceil(length/{SPACING:g})+1 per segment, endpoints included; spacing <= {SPACING:g}",
            note="sampled measurements: zero exterior segments is not a containment certificate and the sampled depth is not a bound",
        ),
        certificate_protocol=dict(
            quantity="delta_j = sup over conv(Omega_j) of dist(z, Omega_j), normalised units",
            convexity=f"verified convex when |conv(Omega_j)| - |Omega_j| <= {CONVEX_RTOL:g} |conv(Omega_j)| (then delta_j = 0)",
            bound=hull_gap_bounds.__doc__.split("Branch and bound", 1)[1].strip(),
            tolerance=args.tol,
            eps=EPS,
            tau=TAU,
        ),
        pieces=pieces,
        summary=summary(pieces),
        timing=dict(
            boundaries="coacd: CoACD hull computation (cached runs report the original time); tetrahedralization: TetGen; partition: ownership, components, piece surfaces; checks: partition checks; solve_wall: all piece solves in parallel (each solve_seconds: nested solve + acceptance in its worker); certificate and measurement per piece; total: whole script",
            coacd_seconds=hull_seconds,
            coacd_cached=cached,
            tetrahedralization_seconds=tet_seconds,
            partition_seconds=partition_seconds,
            checks_seconds=check_seconds,
            solve_wall_seconds=solve_wall,
            sum_piece_solve_seconds=sum(s["solve_seconds"] for s in solved),
            total_seconds=time.perf_counter() - started,
            concurrency=f"{workers} worker processes, one piece per task",
            hardware=dict(
                platform=platform.platform(),
                processor=platform.processor(),
                logical_cpus=os.cpu_count(),
                memory_bytes=__import__("psutil").virtual_memory().total,
            ),
        ),
        versions=versions() | {"coacd": __import__("importlib.metadata").metadata.version("coacd")},
    )
    store.save_json(folder / "record.json", record)
    store.save_npz(
        folder / "geometry.npz",
        mesh_vertices=np.asarray(mesh.vertices),
        mesh_faces=np.asarray(mesh.faces),
        vertices=vertices,
        tetrahedra=tets,
        piece_of=piece_of,
        count=k,
        **{f"v{j}": np.asarray(s.vertices) for j, s in enumerate(surfaces)},
        **{f"f{j}": np.asarray(s.faces) for j, s in enumerate(surfaces)},
    )
    store.save_npz(folder / "paths.npz", count=k, **paths)
    s = record["summary"]
    print(
        f"{obj}: solves passed={s['all_solves_passed']} convex={s['verified_convex_pieces']}/{k} max U={s['max_upper_bound']} inside={s['waypoints_inside']}/{s['total_waypoints']} ext={s['exterior_segments']}/{s['segments']} depth={s['max_sampled_excursion']} total={record['timing']['total_seconds']:.1f}s",
        flush=True,
    )
    store.finish()
