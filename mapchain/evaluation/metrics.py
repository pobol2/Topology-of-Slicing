"""Sampled containment, deposition, coverage and locality metrics of ordered paths."""

from __future__ import annotations

import time
from functools import partial
from types import SimpleNamespace

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from mapchain.store import ROOT
from mapchain.volumetric.parts import (
    TORUS_MAJOR_RADIUS,
    TORUS_MINOR_RADIUS,
    load_or_create_mesh,
    normalization_params,
)

BINS = 20


BOUNDS = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])


SPACING = 0.001


ANGLES = (45, 30)


CORNER_ANGLES = (1, 15, 45, 90)


def surface_distances(model, exterior_points):
    """Exact exterior-to-surface distances for the analytic inside_fn solids.

    Use the original STL only to recover the identical center and scale. The
    torus uses its circle-of-revolution distance; the L-prism is the union of
    two closed boxes, so outside distance is the minimum box distance.
    """
    mesh = load_or_create_mesh(model, ROOT / "models")
    center, scale = normalization_params(mesh, padding=0.05)
    original = (exterior_points - 0.5) / scale + center
    if model == "torus":
        radial = np.linalg.norm(original[:, :2], axis=1)
        distances = np.abs(
            np.hypot(radial - TORUS_MAJOR_RADIUS, original[:, 2]) - TORUS_MINOR_RADIUS
        )
    elif model == "sphere":
        distances = np.abs(np.linalg.norm(original, axis=1) - 1.0)
    elif model == "cylinder":
        radial = np.linalg.norm(original[:, :2], axis=1)
        distances = np.hypot(
            np.maximum(radial - 1.0, 0.0), np.maximum(np.abs(original[:, 2]) - 1.0, 0.0)
        )
    elif model == "l_box":
        lower = np.array([-1.5, -1.5, -0.5])
        upper_a = np.array([1.5, -0.5, 0.5])
        upper_b = np.array([-0.5, 1.5, 0.5])

        def box_distance(upper):
            offsets = np.maximum(np.maximum(lower - original, original - upper), 0.0)
            return np.linalg.norm(offsets, axis=1)

        distances = np.minimum(box_distance(upper_a), box_distance(upper_b))
    else:
        raise ValueError(model)
    return distances * scale


def ordered_cone_metrics(points, w):
    """Exhaustive earlier-waypoint comparisons, in memory-bounded row blocks.

    Every earlier deposit counts, including those classified unsupported or
    exterior. Strict height/order inequalities are applied without tolerances.
    """
    n = len(points)
    z = points[:, 2]
    plate = z <= z.min() + w / 2.0
    supported = {a: plate.copy() for a in ANGLES}
    clearance = {a: np.zeros(n, dtype=bool) for a in ANGLES}
    indices = np.arange(n)
    for start in range(0, n, 128):
        stop = min(start + 128, n)
        current = points[start:stop]
        dx = current[:, None, 0] - points[None, :, 0]
        dy = current[:, None, 1] - points[None, :, 1]
        lateral = np.hypot(dx, dy)
        dz = current[:, None, 2] - z[None, :]
        earlier = indices[None, :] < indices[start:stop, None]
        below = earlier & (dz > 0.0) & (dz <= w)
        above = earlier & (-dz > w / 2.0)
        for angle in ANGLES:
            tangent = 1.0 if angle == 45 else float(np.tan(np.deg2rad(angle)))
            supported[angle][start:stop] |= np.any(
                below & (lateral <= w / 2.0 + dz * tangent), axis=1
            )
            clearance[angle][start:stop] = np.any(above & (lateral <= -dz * tangent), axis=1)
    return (
        {
            str(angle): {
                "unsupported_waypoint_count": int(np.count_nonzero(~supported[angle])),
                "unsupported_waypoint_fraction": float(np.mean(~supported[angle])),
                "clearance_violating_waypoint_count": int(np.count_nonzero(clearance[angle])),
                "clearance_violating_waypoint_fraction": float(np.mean(clearance[angle])),
            }
            for angle in ANGLES
        },
        int(np.count_nonzero(plate)),
    )


def part_metrics(model, b, points, sanity, *, cone_fn=None, distance_fn=None):
    cone_fn = cone_fn or ordered_cone_metrics
    distance_fn = distance_fn or surface_distances
    started = time.perf_counter()
    n = len(points)
    vectors = np.diff(points, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    tree = cKDTree(points)
    nn, _ = tree.query(points, k=2)
    w = float(np.mean(nn[:, 1]))
    intervals = np.maximum(1, np.ceil(lengths / SPACING).astype(np.int64))
    sample_counts = intervals + 1
    offsets = np.concatenate(([0], np.cumsum(sample_counts)))
    samples = np.concatenate(
        [
            (1.0 - t[:, None]) * points[i] + t[:, None] * points[i + 1]
            for i, count in enumerate(sample_counts)
            for t in [np.linspace(0.0, 1.0, int(count))]
        ]
    )
    outside = ~b.inside_fn(samples)
    outside_counts = np.add.reduceat(outside.astype(np.int64), offsets[:-1])
    exterior = outside_counts > 0
    endpoint_inside = b.inside_fn(points)
    arc_length = float(np.sum(lengths * (outside_counts / sample_counts)))
    path_length = float(np.sum(lengths))
    max_depth = float(np.max(distance_fn(model, samples[outside]))) if np.any(outside) else 0.0
    valid_corners = (lengths[:-1] > 0.0) & (lengths[1:] > 0.0)
    cosine = np.sum(vectors[:-1][valid_corners] * vectors[1:][valid_corners], axis=1)
    cosine /= lengths[:-1][valid_corners] * lengths[1:][valid_corners]
    turns = np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0)))
    cones, plate_count = cone_fn(points, w)
    target_distances, _ = tree.query(b.target_eval)
    return {
        "model_name": model,
        "waypoint_count": n,
        "segment_count": len(lengths),
        "sanity_check": sanity,
        "parameters": {
            "p_ot": 3,
            "p_path": 4,
            "project": False,
            "k": 8,
            "normalization_padding": 0.05,
            "w": w,
            "w_definition": "mean k=2 self-query second-neighbour distance",
            "maximum_sampling_spacing": SPACING,
            "actual_maximum_sampling_spacing": float(np.max(lengths / intervals)),
            "segment_sampling": "ceil(length/spacing)+1 uniform samples; endpoints included",
            "shared_endpoints_counted_once_per_incident_segment": True,
            "corner_threshold_degrees": list(CORNER_ANGLES),
            "support_and_clearance_half_angles_degrees": list(ANGLES),
            "tan_45_degrees": 1.0,
            "tan_30_degrees": float(np.tan(np.deg2rad(30))),
            "z_min": float(points[:, 2].min()),
            "build_plate_maximum_height": float(points[:, 2].min() + w / 2.0),
            "distance_method": "exact analytic surface distance in inside_fn normalization frame",
        },
        "w": w,
        "total_segment_samples": int(len(samples)),
        "exterior_sample_count": int(np.count_nonzero(outside)),
        "exterior_segment_count": int(np.count_nonzero(exterior)),
        "exterior_segment_fraction": float(np.mean(exterior)),
        "exterior_arc_length": arc_length,
        "exterior_arc_length_fraction": arc_length / path_length,
        "maximum_excursion_depth": max_depth,
        "exterior_segments_with_both_endpoints_inside": int(
            np.count_nonzero(exterior & endpoint_inside[:-1] & endpoint_inside[1:])
        ),
        "path_length": path_length,
        "corners_above_degrees": {
            str(angle): int(np.count_nonzero(turns > angle)) for angle in CORNER_ANGLES
        },
        "zero_length_segment_count": int(np.count_nonzero(lengths == 0.0)),
        "interior_waypoints_with_undefined_turn": int(np.count_nonzero(~valid_corners)),
        "build_plate_waypoint_count": plate_count,
        "ordered_deposition": cones,
        "covering_radius_target_eval": float(np.max(target_distances)),
        "covering_radius_target_eval_count": len(b.target_eval),
        "metric_runtime_seconds": time.perf_counter() - started,
    }


def fast_cones(points, w, *, batch_size=2048):
    """Exact existential cone tests using spatial boxes plus earlier-index bounds.

    Boxes are conservatively pruned. At leaves use the original strict height,
    order, and lateral inequalities. Whole-box witnesses also satisfy all of
    them; no approximate nearest-neighbour query or changed tolerance is used.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    n = len(points)
    tree = cKDTree(points, leafsize=16)
    lower, upper, imin, imax, left, right, leaf = ([], [], [], [], [], [], [])

    def build(node):
        index = len(lower)
        ids = node.indices
        pts = points[ids]
        lower.append(pts.min(axis=0))
        upper.append(pts.max(axis=0))
        imin.append(int(ids.min()))
        imax.append(int(ids.max()))
        left.append(-1)
        right.append(-1)
        leaf.append(None)
        if node.split_dim < 0:
            leaf[index] = ids.copy()
        else:
            left[index] = build(node.lesser)
            right[index] = build(node.greater)
        return index

    build(tree.tree)
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    imin = np.asarray(imin)
    imax = np.asarray(imax)
    left = np.asarray(left)
    right = np.asarray(right)
    leaf_count = np.array([len(ids) if ids is not None else 0 for ids in leaf])
    leaf_start = np.r_[0, np.cumsum(leaf_count[:-1])]
    leaf_ids = np.concatenate([ids for ids in leaf if ids is not None])
    plate = points[:, 2] <= points[:, 2].min() + w / 2

    def query(kind, tangent, initial):
        done = initial.copy()
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            q = np.flatnonzero(~done[start:stop]) + start
            nodes = np.zeros(len(q), dtype=np.int64)
            while len(q):
                use = ~done[q] & (imin[nodes] < q)
                q, nodes = (q[use], nodes[use])
                if not len(q):
                    break
                p, lo, hi = (points[q], lower[nodes], upper[nodes])
                dmin = np.maximum(np.maximum(lo[:, :2] - p[:, :2], p[:, :2] - hi[:, :2]), 0)
                lateral_min = np.hypot(dmin[:, 0], dmin[:, 1])
                dmax = np.maximum(np.abs(lo[:, :2] - p[:, :2]), np.abs(hi[:, :2] - p[:, :2]))
                lateral_max = np.hypot(dmax[:, 0], dmax[:, 1])
                if kind == "support":
                    min_dz = p[:, 2] - hi[:, 2]
                    max_dz = p[:, 2] - lo[:, 2]
                    possible = (
                        (max_dz > 0)
                        & (min_dz <= w)
                        & (lateral_min <= w / 2 + np.minimum(w, max_dz) * tangent)
                    )
                    witness = (
                        (imax[nodes] < q)
                        & (min_dz > 0)
                        & (max_dz <= w)
                        & (lateral_max <= w / 2 + min_dz * tangent)
                    )
                else:
                    min_dz = lo[:, 2] - p[:, 2]
                    max_dz = hi[:, 2] - p[:, 2]
                    possible = (max_dz > w / 2) & (lateral_min <= max_dz * tangent)
                    witness = (
                        (imax[nodes] < q) & (min_dz > w / 2) & (lateral_max <= min_dz * tangent)
                    )
                if kind == "support":
                    possible |= (
                        (max_dz > 0)
                        & (min_dz <= w)
                        & (
                            lateral_min
                            <= np.nextafter(w / 2 + np.minimum(w, max_dz) * tangent, np.inf)
                        )
                    )
                else:
                    possible |= (max_dz > w / 2) & (
                        lateral_min <= np.nextafter(max_dz * tangent, np.inf)
                    )
                done[q[witness]] = True
                use = possible & ~done[q]
                q, nodes = (q[use], nodes[use])
                is_leaf = left[nodes] < 0
                if np.any(is_leaf):
                    ql, nl = (q[is_leaf], nodes[is_leaf])
                    sizes = leaf_count[nl]
                    offsets = np.r_[0, np.cumsum(sizes)]
                    flat_q = np.repeat(ql, sizes)
                    positions = np.repeat(leaf_start[nl] - offsets[:-1], sizes) + np.arange(
                        offsets[-1]
                    )
                    ids = leaf_ids[positions]
                    earlier = ids < flat_q
                    candidates = points[ids]
                    lateral = np.hypot(
                        points[flat_q, 0] - candidates[:, 0], points[flat_q, 1] - candidates[:, 1]
                    )
                    dz = points[flat_q, 2] - candidates[:, 2]
                    if kind == "support":
                        hit = earlier & (dz > 0) & (dz <= w) & (lateral <= w / 2 + dz * tangent)
                    else:
                        hit = earlier & (-dz > w / 2) & (lateral <= -dz * tangent)
                    done[ql[np.logical_or.reduceat(hit, offsets[:-1])]] = True
                q, nodes = (q[~is_leaf], nodes[~is_leaf])
                q, nodes = (np.repeat(q, 2), np.stack((left[nodes], right[nodes]), axis=1).ravel())
        return done

    output = {}
    for angle in (45, 30):
        tangent = 1.0 if angle == 45 else float(np.tan(np.deg2rad(30)))
        supported = query("support", tangent, plate)
        violation = query("clearance", tangent, np.zeros(n, dtype=bool))
        output[str(angle)] = {
            "unsupported_waypoint_count": int(np.count_nonzero(~supported)),
            "unsupported_waypoint_fraction": float(np.mean(~supported)),
            "clearance_violating_waypoint_count": int(np.count_nonzero(violation)),
            "clearance_violating_waypoint_fraction": float(np.mean(violation)),
        }
    return (output, int(plate.sum()))


def run_stats(exterior, lengths):
    starts = np.flatnonzero(exterior & ~np.r_[False, exterior[:-1]])
    ends = np.flatnonzero(exterior & ~np.r_[exterior[1:], False]) + 1
    return {
        "count": len(starts),
        "total_run_length": float(lengths[exterior].sum()),
        "longest_run_segments": int(np.max(ends - starts)) if len(starts) else 0,
        "longest_run_length": float(
            max((lengths[a:b].sum() for a, b in zip(starts, ends)), default=0.0)
        ),
    }


def segment_engine(
    model,
    b,
    points,
    *,
    order=4,
    project=False,
    support_order=3,
    depositing=None,
    cone_batch_size=2048,
    distance_fn=None,
):
    """Call existing engine, capture its exact sample classifications, mask travels.

    Ordered-deposition tests use every visited cell centre (every centre deposits).
    Travel segments are excluded from arc/segment/turn metrics; turns across a
    travel move are not deposition corners. w and covering use all deposited tips.
    """
    started = time.perf_counter()
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    counts = np.maximum(1, np.ceil(lengths / SPACING).astype(np.int64)) + 1
    offsets = np.r_[0, np.cumsum(counts)]
    captured = {}

    def inside_capture(samples):
        mask = b.inside_fn(samples)
        if not captured:
            captured["outside"] = ~mask
            captured["samples"] = samples
        return mask

    adapter = SimpleNamespace(inside_fn=inside_capture, target_eval=b.target_eval)
    cones = (
        partial(fast_cones, batch_size=cone_batch_size)
        if len(points) > 8192
        else ordered_cone_metrics
    )
    distance_fn = distance_fn or surface_distances
    m = part_metrics(model, adapter, points, {}, cone_fn=cones, distance_fn=distance_fn)
    outside_counts = np.add.reduceat(captured["outside"].astype(np.int64), offsets[:-1])
    exterior = outside_counts > 0
    m["parameters"].update(p_ot=support_order, p_path=order, project=project)
    m["parameters"]["cone_algorithm"] = (
        "exact spatial box pruning" if len(points) > 8192 else "imported exhaustive comparisons"
    )
    if depositing is not None:
        keep = np.asarray(depositing, dtype=bool)
        inside = b.inside_fn(points)
        selected_samples = np.repeat(keep, counts)
        out = captured["outside"] & selected_samples
        m.update(
            segment_count=int(keep.sum()),
            total_segment_samples=int(counts[keep].sum()),
            exterior_sample_count=int(out.sum()),
            exterior_segment_count=int((exterior & keep).sum()),
            exterior_segment_fraction=float(exterior[keep].mean()) if keep.any() else 0.0,
            exterior_arc_length=float(np.sum(lengths[keep] * outside_counts[keep] / counts[keep])),
            path_length=float(lengths[keep].sum()),
            maximum_excursion_depth=float(distance_fn(model, captured["samples"][out]).max())
            if out.any()
            else 0.0,
            exterior_segments_with_both_endpoints_inside=int(
                np.count_nonzero(exterior & keep & inside[:-1] & inside[1:])
            ),
            zero_length_segment_count=int(np.count_nonzero(keep & (lengths == 0))),
        )
        m["exterior_arc_length_fraction"] = (
            m["exterior_arc_length"] / m["path_length"] if m["path_length"] else 0.0
        )
        vec = np.diff(points, axis=0)
        adjacent = keep[:-1] & keep[1:]
        valid = adjacent & (lengths[:-1] > 0) & (lengths[1:] > 0)
        turns = np.rad2deg(
            np.arccos(
                np.clip(
                    np.sum(vec[:-1][valid] * vec[1:][valid], axis=1)
                    / (lengths[:-1][valid] * lengths[1:][valid]),
                    -1,
                    1,
                )
            )
        )
        m["corners_above_degrees"] = {
            str(a): int(np.count_nonzero(turns > a)) for a in CORNER_ANGLES
        }
        m["interior_waypoints_with_undefined_turn"] = int(np.count_nonzero(adjacent & ~valid))
        m["parameters"]["segment_scope"] = (
            "depositing segments only; all deposited waypoint centres for w/support/clearance/covering"
        )
        m["parameters"]["actual_maximum_sampling_spacing"] = (
            float(np.max(lengths[keep] / (counts[keep] - 1))) if keep.any() else 0.0
        )
    m["metric_runtime_seconds"] = time.perf_counter() - started
    return (m, exterior, lengths)


def table4(points, b):
    return compute_all(points, b.target_eval, b.inside_fn, bounds=BOUNDS, bins=BINS)


def row_metrics(model, b, pts, **kwargs):
    started = time.perf_counter()
    segment, exterior, lengths = segment_engine(model, b, pts, **kwargs)
    result = {
        "waypoint_count": len(pts),
        "table4": table4(pts, b),
        "segment_metrics": segment,
        "detected_jumps": jump_count(pts),
        "exterior_runs": run_stats(exterior, lengths),
    }
    result["runtime_seconds"] = time.perf_counter() - started
    return result


def weighted_cloud(points, weights, target):
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    counts, _ = np.histogramdd(points, bins=(BINS,) * 3, range=BOUNDS.T, weights=weights)
    target_counts, _ = np.histogramdd(target, bins=(BINS,) * 3, range=BOUNDS.T)
    occupied = target_counts > 0
    values = counts[occupied]
    forward = cKDTree(target).query(points)[0]
    reverse = cKDTree(points).query(target)[0]
    return {
        "coverage": float(np.count_nonzero((counts > 0) & occupied) / occupied.sum()),
        "coverage_cv": float(values.std() / values.mean()),
        "chamfer": float(0.5 * (np.dot(weights, forward) + reverse.mean())),
        "total_time_weight": float(weights.sum()),
        "target_occupied_voxel_count": int(occupied.sum()),
    }


def exterior_segments(points, inside):
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    counts = np.maximum(1, np.ceil(lengths / SPACING).astype(int)) + 1
    out = np.zeros(len(lengths), bool)
    for start in range(0, len(lengths), 1024):
        end = min(start + 1024, len(lengths))
        batches = []
        offsets = [0]
        for i in range(start, end):
            t = np.linspace(0, 1, counts[i])
            batches.append((1 - t[:, None]) * points[i] + t[:, None] * points[i + 1])
            offsets.append(offsets[-1] + counts[i])
        outside = ~inside(np.concatenate(batches))
        out[start:end] = np.logical_or.reduceat(outside, np.asarray(offsets[:-1]))
    return out


def _as_points(points: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3).")
    if len(array) == 0:
        raise ValueError(f"{name} cannot be empty.")
    return array


def inside_fraction(
    waypoints: np.ndarray,
    mesh: trimesh.Trimesh,
) -> float:
    """Fraction of waypointsstructed points lying inside the target volume."""
    points = _as_points(waypoints, "waypoints")
    return float(np.mean(mesh.contains(points)))


def chamfer_distance(
    waypoints: np.ndarray,
    target_points: np.ndarray,
) -> float:
    """Symmetric mean nearest-neighbour distance between two point clouds."""
    waypointsstructed = _as_points(waypoints, "waypoints")
    target = _as_points(target_points, "target_points")
    waypointsstructed_tree = cKDTree(waypointsstructed)
    target_tree = cKDTree(target)
    target_to_waypointsstructed, _ = waypointsstructed_tree.query(target)
    waypointsstructed_to_target, _ = target_tree.query(waypointsstructed)
    return float(
        0.5 * (np.mean(target_to_waypointsstructed) + np.mean(waypointsstructed_to_target))
    )


def _voxel_counts(points: np.ndarray, bounds: np.ndarray, bins: int) -> np.ndarray:
    if bins < 2:
        raise ValueError("bins must be at least 2.")
    lower = np.asarray(bounds[0], dtype=float)
    upper = np.asarray(bounds[1], dtype=float)
    ranges = [(float(lower[a]), float(upper[a])) for a in range(3)]
    counts, _ = np.histogramdd(_as_points(points, "points"), bins=(bins, bins, bins), range=ranges)
    return counts


def voxel_coverage_fraction(
    waypoints: np.ndarray,
    target_points: np.ndarray,
    *,
    bounds: np.ndarray,
    bins: int = 16,
) -> float:
    """Fraction of target-occupied voxels also reached by the waypointsstruction."""
    waypointsstructed_counts = _voxel_counts(waypoints, bounds, bins)
    target_counts = _voxel_counts(target_points, bounds, bins)
    target_occupied = target_counts > 0
    n_target = int(np.count_nonzero(target_occupied))
    if n_target == 0:
        raise ValueError("The target point cloud occupies no voxels.")
    covered = target_occupied & (waypointsstructed_counts > 0)
    return float(np.count_nonzero(covered) / n_target)


def coverage_cv(
    waypoints: np.ndarray,
    target_points: np.ndarray,
    *,
    bounds: np.ndarray,
    bins: int = 16,
) -> float:
    """Coefficient of variation of interior point density (3D voxel grid).

    Restricted to voxels the target occupies, so empty exterior voxels do not
    dilate the statistic. Zero is perfectly uniform; higher is more clumped.
    """
    waypointsstructed_counts = _voxel_counts(waypoints, bounds, bins)
    target_counts = _voxel_counts(target_points, bounds, bins)
    occupied = target_counts > 0
    values = waypointsstructed_counts[occupied]
    mean = float(np.mean(values))
    if mean == 0.0:
        return float("inf")
    return float(np.std(values) / mean)


def step_lengths(ordered_points: np.ndarray) -> np.ndarray:
    pts = _as_points(ordered_points, "ordered_points")
    return np.linalg.norm(np.diff(pts, axis=0), axis=1)


def locality_ratio(ordered_points: np.ndarray) -> float:
    """Mean consecutive step divided by mean spatial nearest-neighbour distance.

    Equals 1 for a perfectly locality-preserving traversal; larger means the
    ordering leaps around relative to the intrinsic point spacing.
    """
    pts = _as_points(ordered_points, "ordered_points")
    steps = step_lengths(pts)
    nn, _ = cKDTree(pts).query(pts, k=2)
    mean_nn = float(np.mean(nn[:, 1]))
    if mean_nn == 0.0:
        return float("inf")
    return float(np.mean(steps) / mean_nn)


def jump_count(ordered_points: np.ndarray, *, factor: float = 5.0) -> int:
    """Number of consecutive steps longer than ``factor`` times the mean step."""
    steps = step_lengths(ordered_points)
    return int(np.count_nonzero(steps > factor * float(np.mean(steps))))


def compute_all(
    waypointsstructed_ordered: np.ndarray,
    target_points: np.ndarray,
    inside_fn,
    *,
    bounds: np.ndarray,
    bins: int = 16,
) -> dict[str, float]:
    """Compute the full metric set used in the study tables.

    ``inside_fn`` maps points in [0,1]^3 to a boolean interior mask.
    """
    steps = step_lengths(waypointsstructed_ordered)
    return {
        "inside_fraction": float(np.mean(inside_fn(waypointsstructed_ordered))),
        "chamfer": chamfer_distance(waypointsstructed_ordered, target_points),
        "coverage": voxel_coverage_fraction(
            waypointsstructed_ordered, target_points, bounds=bounds, bins=bins
        ),
        "coverage_cv": coverage_cv(
            waypointsstructed_ordered, target_points, bounds=bounds, bins=bins
        ),
        "locality": locality_ratio(waypointsstructed_ordered),
        "jumps": float(jump_count(waypointsstructed_ordered)),
        "mean_step": float(np.mean(steps)),
        "max_step": float(np.max(steps)),
    }
