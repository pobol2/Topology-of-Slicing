"""Decomposed versus whole-part paths: the decomposition summary and the matched-budget torus comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json

import numpy as np
from scipy.spatial import cKDTree

from mapchain import store
from mapchain.evaluation.metrics import surface_distances, weighted_cloud
from mapchain.evaluation.whole_part import uniform_path, waypoints
from mapchain.store import ROOT
from mapchain.volumetric.decomposition import (
    SPACING,
    Membership,
    segment_samples,
    surfaces_from,
)
from mapchain.volumetric.parts import geometry

OBJECTS = (
    ("l_box", "L-bracket", "two boxes"),
    ("torus", "Torus", "CoACD-guided"),
    ("bunny", "Bunny", "CoACD-guided"),
)


DIR = store.OUT / "volumetric" / "decomposition"
OUT = store.OUT / "evaluation" / "pieces"
SUMMARY = OUT / "summary.json"


def records():
    return [
        (obj, label, kind, json.loads((DIR / obj / "record.json").read_text()))
        for obj, label, kind in OBJECTS
    ]


def consistency(record):
    """Sampled depth never exceeds the piece's numerical bound; every bound met its tolerance."""
    rows = []
    for p in record["pieces"]:
        c, mm = (p["certificate"], p.get("measurements"))
        rows.append(
            dict(
                piece_id=p["piece_id"],
                bound=c.get("upper_bound"),
                depth=mm and mm["max_sampled_excursion"],
                depth_within_bound=None
                if not mm or c.get("upper_bound") is None
                else mm["max_sampled_excursion"] <= c["upper_bound"],
                certificate_met_tolerance=c["status"] == "convex (verified)"
                or c.get("stopped_by") in ("tolerance", "all pruned"),
            )
        )
    return dict(
        all_depths_within_bounds=all(r["depth_within_bound"] for r in rows),
        all_certificates_met_tolerance=all(r["certificate_met_tolerance"] for r in rows),
        pieces=rows,
    )


def caption_claims(rows):
    """The table captions state these facts in fixed text; list every one the records contradict."""
    problems = []
    for obj, _label, _kind, r in rows:
        s, c = (r["summary"], r["partition"]["checks"])
        if not c["passed"]:
            problems.append(f'{obj}: partition checks failed ("All checks pass")')
        if not s["all_solves_passed"]:
            problems.append(f"{obj}: failed pieces {s['failed_pieces']}")
        if s["max_upper_bound"] is None:
            problems.append(f"{obj}: not every piece has a valid bound")
        if not consistency(r)["all_certificates_met_tolerance"]:
            problems.append(f"{obj}: a certificate stopped before its tolerance")
        if r["certificate_protocol"]["tolerance"] != 0.002:
            problems.append(f"{obj}: certificate tolerance is not 2e-3")
        if r["certificate_protocol"]["tau"] != 1e-09:
            problems.append(f"{obj}: boundary tolerance is not 1e-9")
        if any(p["order"] != 4 or p["waypoints"] != 4096 for p in r["pieces"]):
            problems.append(f"{obj}: not every piece has order 4 (4096 waypoints)")
        if f"{SPACING:g}" not in r["measurement_protocol"]["sampling"] or SPACING != 0.001:
            problems.append(f"{obj}: sample spacing is not 1e-3")
        samples = c["coverage_samples"]
        if samples["samples_in_solid"] + samples["samples_outside_polygonal_solid"] != 10000:
            problems.append(f"{obj}: not 10^4 evaluation samples")
    return problems


def summarise():
    """Summarise the decomposition records (the data of the decomposition table) into summary.json."""
    rows = records()
    problems = caption_claims(rows)
    if problems:
        raise SystemExit(
            "STOP: the table captions would be wrong; nothing written:\n  " + "\n  ".join(problems)
        )
    summary = {
        obj: dict(
            label=label,
            partition=kind,
            run_id=r["run_id"],
            record=r["files"]["record"],
            **r["summary"],
            partition_checks_passed=r["partition"]["checks"]["passed"],
            consistency=consistency(r),
        )
        for obj, label, kind, r in rows
    }
    store.save_json(SUMMARY, summary)
    for obj, s in summary.items():
        print(
            f"{obj}: checks={s['partition_checks_passed']} solves={s['all_solves_passed']} convex={s['verified_convex_pieces']}/{s['pieces']} maxU={s['max_upper_bound']} depth<=U all={s['consistency']['all_depths_within_bounds']} certs at tol={s['consistency']['all_certificates_met_tolerance']}"
        )
    return summary


RESULT = OUT / "torus_whole_vs_pieces.json"


WHOLE = uniform_path("torus", 5)


DEC = DIR / "torus"


def polygonal(surface):
    """(outside flags, distance of each flagged point) for a closed polygonal surface."""
    return Membership(surface).outside


def analytic(b):

    def outside(points):
        out = ~b.inside_fn(points)
        d = np.zeros(len(points))
        if out.any():
            d[out] = surface_distances("torus", points[out])
        return (out, d)

    return outside


def measure(points, outside):
    """Sampled exterior measurements of one polyline; arc length counts the outside share of each segment's samples."""
    waypoint_out, _ = outside(points)
    q, seg, lengths = segment_samples(points)
    out, d = outside(q)
    n = np.bincount(seg, minlength=len(lengths))
    k = np.bincount(seg, weights=out, minlength=len(lengths))
    exterior = k > 0
    return (
        dict(
            waypoints=len(points),
            waypoints_inside=int(len(points) - waypoint_out.sum()),
            segments=len(lengths),
            exterior_segments=int(exterior.sum()),
            exterior_arc_length=float(np.sum(lengths * k / n)),
            path_length=float(lengths.sum()),
            max_excursion=float(d[out].max()) if out.any() else 0.0,
            samples=len(q),
        ),
        exterior,
        waypoint_out,
    )


def combine(parts, weights):
    """Totals over independent piece paths; containment also weighted by piece mass."""

    def total(key):
        return sum(p[key] for p in parts)

    return dict(
        waypoints=total("waypoints"),
        waypoints_inside=total("waypoints_inside"),
        segments=total("segments"),
        exterior_segments=total("exterior_segments"),
        exterior_arc_length=total("exterior_arc_length"),
        path_length=total("path_length"),
        max_excursion=max(p["max_excursion"] for p in parts),
        samples=total("samples"),
        mass_weighted_containment=float(
            sum((w * p["waypoints_inside"] / p["waypoints"] for p, w in zip(parts, weights)))
        ),
    )


def finish_comparison(m):
    m["containment"] = m["waypoints_inside"] / m["waypoints"]
    m.setdefault("mass_weighted_containment", m["containment"])
    m["exterior_segment_fraction"] = m["exterior_segments"] / m["segments"]
    m["exterior_length_fraction"] = m["exterior_arc_length"] / m["path_length"]
    return m


def hull_gap(certificate, protocol):
    """The record's bound on delta_j with the conditions it holds under (copied, not recomputed)."""
    return dict(
        quantity=protocol["quantity"],
        status=certificate["status"],
        upper_bound=certificate.get("upper_bound"),
        lower_bound=certificate.get("lower_bound"),
        tolerance=certificate.get("tolerance"),
        stopped_by=certificate.get("stopped_by"),
        method=certificate.get("method"),
        conditional_on=[
            certificate.get("assumption") or protocol["bound"].split("Assumption:", 1)[1].strip(),
            f"boundary tolerance TAU = {protocol['tau']:g} and rounding allowance EPS = {protocol['eps']:g}",
            "the closed polygonal piece surface stored in geometry.npz",
            "a bound on the piece path excursion only via the paper corollary (path inside conv(Omega_j)), up to the solver tolerances of the accepted solve",
        ],
    )


def coverage(points, weights, b):
    cloud = weighted_cloud(points, weights, b.target_eval)
    cloud["covering_radius"] = float(cKDTree(points).query(b.target_eval)[0].max())
    return cloud


def compare():
    whole = json.loads(WHOLE.read_text())
    record = json.loads((DEC / "record.json").read_text())
    partition = np.load(DEC / "geometry.npz")
    paths = np.load(DEC / "paths.npz")
    mesh, b, _ = geometry("torus")
    k = int(partition["count"])
    surfaces = surfaces_from(DEC / "geometry.npz")
    W = np.asarray(waypoints("torus", 5), float)
    P = [np.asarray(paths[f"p{j}"], float) for j in range(k)]
    weights = np.array([p["mass_share"] for p in record["pieces"]])
    corners = partition["vertices"][partition["tetrahedra"]]
    volumes = (
        np.abs(
            np.einsum(
                "ij,ij->i",
                np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]),
                corners[:, 3] - corners[:, 0],
            )
        )
        / 6
    )
    recomputed = np.bincount(partition["piece_of"], weights=volumes, minlength=k) / volumes.sum()
    model = hashlib.sha256((ROOT / "models/torus.stl").read_bytes()).hexdigest()
    tetgen_whole, tetgen_pieces = (whole["meshing"], record["geometry"]["tetgen"])
    inputs = dict(
        model_sha256=model,
        whole_run_model_hash_matches=whole["construction"]["model_sha256"] == model,
        piece_run_model_hash_matches=record["geometry"]["construction"]["model_sha256"] == model,
        piece_run_surface_equals_current_normalised_model=bool(
            np.array_equal(partition["mesh_vertices"], mesh.vertices)
            and np.array_equal(partition["mesh_faces"], mesh.faces)
        ),
        tetgen_counts_match=tetgen_whole["vertex_count"]
        == tetgen_pieces["vertex_count"]
        == len(partition["vertices"])
        and tetgen_whole["tetrahedron_count"]
        == tetgen_pieces["tetrahedron_count"]
        == len(partition["tetrahedra"]),
        tetgen_settings_match=tetgen_whole["switches"] == tetgen_pieces["switches"]
        and tetgen_whole["max_tetrahedron_volume"] == tetgen_pieces["max_tetrahedron_volume"],
        tetrahedral_mesh_identity="unverified: the whole-part run saved only counts and settings; the piece run saved its arrays (geometry.npz)",
        evaluation_samples=dict(
            count=len(b.target_eval),
            sha256=hashlib.sha256(np.ascontiguousarray(b.target_eval).tobytes()).hexdigest(),
            use="every coverage metric of both paths in this comparison is computed on this array",
        ),
        same_waypoint_count=len(W) == sum(len(p) for p in P) == 8**5,
        piece_weights_match_tetrahedral_volumes=float(np.max(np.abs(recomputed - weights))),
        piece_weights_sum=float(weights.sum()),
        all_piece_solves_accepted=all(
            p["solve"]["passed"] and p["solve"]["acceptance"]["passed"] for p in record["pieces"]
        ),
        whole_solve_accepted=bool(whole["passed"] and whole["acceptance"]["passed"]),
    )
    torus, exact = (polygonal(mesh), analytic(b))
    whole_poly, whole_poly_ext, _ = measure(W, torus)
    whole_exact, whole_exact_ext, _ = measure(W, exact)
    assigned, pieces_poly, pieces_exact, piece_rows = ([], [], [], [])
    exterior_whole_object = []
    subset_segments = subset_waypoints = True
    for j, (p, s) in enumerate(zip(P, surfaces)):
        a, a_ext, a_out = measure(p, polygonal(s))
        o, o_ext, o_out = measure(p, torus)
        e, _, _ = measure(p, exact)
        assigned.append(a)
        pieces_poly.append(o)
        pieces_exact.append(e)
        exterior_whole_object.append(o_ext)
        subset_segments &= bool(np.all(a_ext[o_ext]))
        subset_waypoints &= bool(np.all(a_out[o_out]))
        saved = record["pieces"][j]["measurements"]
        piece_rows.append(
            dict(
                piece=record["pieces"][j]["label"],
                mass_share=float(weights[j]),
                reproduces_saved_assigned_measurement=bool(
                    a["waypoints_inside"] == saved["waypoints_inside"]
                    and a["exterior_segments"] == saved["exterior_segments"]
                    and (abs(a["max_excursion"] - saved["max_sampled_excursion"]) <= 1e-12)
                    and np.array_equal(np.flatnonzero(a_ext), paths[f"e{j}"])
                ),
                assigned_piece=finish_comparison(dict(a)),
                whole_polygonal_torus=finish_comparison(dict(o)),
                analytic_torus=finish_comparison(dict(e)),
                hull_gap=hull_gap(
                    record["pieces"][j]["certificate"], record["certificate_protocol"]
                ),
            )
        )
    rows = dict(
        whole_m5=dict(
            polygonal_torus=finish_comparison(whole_poly),
            analytic_torus=finish_comparison(whole_exact),
        ),
        pieces_m4=dict(
            polygonal_torus=finish_comparison(combine(pieces_poly, weights)),
            analytic_torus=finish_comparison(combine(pieces_exact, weights)),
            assigned_piece=finish_comparison(combine(assigned, weights)),
        ),
    )
    uniform = np.full(len(W), 1 / len(W))
    piece_weights = np.concatenate([np.full(len(p), w / len(p)) for p, w in zip(P, weights)])
    pieces_points = np.concatenate(P)
    rows["whole_m5"]["coverage"] = coverage(W, uniform, b)
    rows["pieces_m4"]["coverage"] = coverage(pieces_points, piece_weights, b)
    rows["pieces_m4"]["coverage_ignoring_piece_weights"] = coverage(
        pieces_points, np.full(len(pieces_points), 1 / len(pieces_points)), b
    )
    saved_seg, saved_t4 = (whole["metrics"]["segment_metrics"], whole["metrics"]["table4"])
    w, c = (rows["whole_m5"]["analytic_torus"], rows["whole_m5"]["coverage"])

    def close(x, y):
        return abs(x - y) <= 1e-12 * max(1, abs(y))

    validation = dict(
        inputs=inputs,
        whole_reproduces_saved_metrics=dict(
            exterior_segments=w["exterior_segments"] == saved_seg["exterior_segment_count"],
            exterior_length_fraction=close(
                w["exterior_length_fraction"], saved_seg["exterior_arc_length_fraction"]
            ),
            max_excursion=close(w["max_excursion"], saved_seg["maximum_excursion_depth"]),
            samples=w["samples"] == saved_seg["total_segment_samples"],
            containment=close(w["containment"], saved_t4["inside_fraction"]),
            coverage=close(c["coverage"], saved_t4["coverage"]),
            coverage_cv=close(c["coverage_cv"], saved_t4["coverage_cv"]),
            chamfer=close(c["chamfer"], saved_t4["chamfer"]),
            covering_radius=close(c["covering_radius"], saved_seg["covering_radius_target_eval"]),
        ),
        pieces_reproduce_saved_assigned_measurements=all(
            r["reproduces_saved_assigned_measurement"] for r in piece_rows
        ),
        torus_exterior_implies_assigned_exterior=dict(
            segments=subset_segments, waypoints=subset_waypoints
        ),
    )
    flat = (
        [v for v in validation["inputs"].values() if isinstance(v, bool)]
        + list(validation["whole_reproduces_saved_metrics"].values())
        + [validation["pieces_reproduce_saved_assigned_measurements"]]
        + list(validation["torus_exterior_implies_assigned_exterior"].values())
    )
    validation["passed"] = bool(
        all(flat)
        and inputs["piece_weights_match_tetrahedral_volumes"] <= 1e-12
        and (abs(inputs["piece_weights_sum"] - 1) <= 1e-12)
    )
    t = record["timing"]
    runtime = dict(
        whole_m5=dict(
            tetrahedralization=whole["meshing"]["runtime_seconds"],
            solve=whole["solving"]["runtime_seconds"],
            to_waypoints=whole["meshing"]["runtime_seconds"] + whole["solving"]["runtime_seconds"],
            concurrency="one nested solve",
        ),
        pieces_m4=dict(
            tetrahedralization=t["tetrahedralization_seconds"],
            coacd=t["coacd_seconds"],
            partition_excluding_coacd=t["partition_seconds"] - t["coacd_seconds"],
            checks=t["checks_seconds"],
            solve=t["solve_wall_seconds"],
            sum_of_piece_elapsed_solve_times=t["sum_piece_solve_seconds"],
            to_waypoints=t["tetrahedralization_seconds"]
            + t["partition_seconds"]
            + t["checks_seconds"]
            + t["solve_wall_seconds"],
            to_waypoints_excluding_coacd=t["tetrahedralization_seconds"]
            + t["partition_seconds"]
            - t["coacd_seconds"]
            + t["checks_seconds"]
            + t["solve_wall_seconds"],
            concurrency=t["concurrency"],
        ),
        source="Saved stage timings of the two original runs (same machine), not re-timed here",
        definitions=dict(
            solve="wall-clock time of the solve stage",
            sum_of_piece_elapsed_solve_times="sum of the eight per-piece elapsed (wall-clock) times of nested solve + acceptance, each measured inside its own concurrently running worker process; not CPU time and not a serial runtime",
            to_waypoints="whole: tetrahedralization + solve; pieces: tetrahedralization + partition (including CoACD, except in to_waypoints_excluding_coacd) + checks + solve wall time",
        ),
    )
    data = dict(
        sources=dict(
            whole=WHOLE.relative_to(ROOT).as_posix(), pieces=DEC.relative_to(ROOT).as_posix()
        ),
        definitions=dict(
            sampler=f"mapchain.volumetric.decomposition.segment_samples: ceil(length/{SPACING:g})+1 samples per segment, endpoints included",
            polygonal_torus="normalised models/torus.stl, the domain of both solves; outside if winding < 1/2 and distance > 1e-9",
            analytic_torus="exact torus of the saved whole-part metrics (mapchain.evaluation.metrics.surface_distances)",
            assigned_piece="closed surface of the piece Omega_j used by its own solve",
            containment="waypoint fraction inside; mass_weighted_containment weights piece j by w_j",
            exterior_length_fraction="sum over segments of length x outside share of its samples, over total path length",
            max_excursion="largest sampled distance outside (not a bound)",
            sampled_scope="exterior segments, exterior length and excursion are evaluated at the samples only: a value of zero means no sample was classified outside, not that the polyline is proven to lie in the solid; waypoint containment classifies the waypoints themselves with the same membership test",
            scope="this comparison only: the saved whole-part order-5 torus run and the saved eight-piece CoACD order-4 torus run; no conclusion is drawn for other parts, partitions, orders or settings",
            hull_gap="per piece, copied from the decomposition record with its conditions (pieces[].hull_gap.conditional_on)",
            assigned_piece_crossings="a piece path segment with samples outside its assigned piece Omega_j; it lies in a region whose mass is assigned to another piece, so that piece and its path are no longer the only deposit there. Its effect on local deposition density, overlap and sequencing is not measured here",
            coverage="weighted_cloud on the 10^4 evaluation samples, 20^3 voxels: occupancy, mass-weighted CV, weighted chamfer; covering radius = max sample-to-waypoint distance",
            weights="each waypoint carries its cell mass: 1/32768 (whole), w_j/4096 (piece j)",
            paths="piece paths are independent; no segment joins two pieces",
        ),
        validation=validation,
        comparison=rows,
        pieces=piece_rows,
        runtime=runtime,
    )
    return (
        data,
        dict(W=W, P=P, whole_ext=whole_poly_ext, piece_ext=exterior_whole_object, mesh=mesh),
    )


SAMPLED = "sampled (spacing <= 1e-3); 0 = no sample outside, not proven containment"


TABLE = (
    ("waypoints", "Waypoints", "{:.0f}", ""),
    ("segments", "Segments", "{:.0f}", "piece paths are independent"),
    (
        "containment",
        "Containment (waypoint fraction)",
        "{:.5f}",
        "waypoints classified with the same membership test",
    ),
    (
        "mass_weighted_containment",
        "Containment (mass-weighted)",
        "{:.5f}",
        "piece j weighted by w_j",
    ),
    ("exterior_segment_fraction", "Exterior segment fraction (sampled)", "{:.5f}", SAMPLED),
    ("exterior_length_fraction", "Exterior path-length fraction (sampled)", "{:.5f}", SAMPLED),
    ("max_excursion", "Max sampled excursion", "{:.5f}", SAMPLED + "; not a bound"),
)


def comparison_csv(data):
    """Comparison table of this torus comparison only (see comparison.json definitions.scope)."""
    c, r = (data["comparison"], data["runtime"])
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    w.writerow(
        [
            "metric",
            "reference",
            "whole_order5",
            "pieces_order4",
            "pieces_order4_ignoring_piece_weights",
            "note",
        ]
    )
    for ref, label in (
        ("polygonal_torus", "whole polygonal torus"),
        ("analytic_torus", "analytic torus"),
        ("assigned_piece", "assigned piece"),
    ):
        for key, name, fmt, note in TABLE:
            whole = c["whole_m5"][ref][key] if ref in c["whole_m5"] else None
            if ref == "assigned_piece" and key in (
                "exterior_segment_fraction",
                "exterior_length_fraction",
                "max_excursion",
            ):
                note += "; crossings enter regions assigned to other pieces (definitions.assigned_piece_crossings)"
            w.writerow(
                [
                    name,
                    label,
                    "" if whole is None else fmt.format(whole),
                    fmt.format(c["pieces_m4"][ref][key]),
                    "",
                    note,
                ]
            )
    bounds = [p["hull_gap"]["upper_bound"] for p in data["pieces"]]
    w.writerow(
        [
            "Max hull-gap upper bound U_j",
            "assigned piece",
            "",
            f"{max(bounds):.5f}" if None not in bounds else "unavailable",
            "",
            "conditional: winding number correct within 1/4 farther than TAU=1e-9 from the piece surface, EPS=1e-9 (pieces[].hull_gap.conditional_on)",
        ]
    )
    for key, name in (
        ("coverage", "Coverage (occupied target voxels)"),
        ("coverage_cv", "Coverage CV"),
        ("chamfer", "Chamfer"),
        ("covering_radius", "Covering radius"),
    ):
        w.writerow(
            [
                name,
                "evaluation samples",
                f"{c['whole_m5']['coverage'][key]:.5f}",
                f"{c['pieces_m4']['coverage'][key]:.5f}",
                f"{c['pieces_m4']['coverage_ignoring_piece_weights'][key]:.5f}",
                "10^4 evaluation samples; waypoints weighted by cell mass",
            ]
        )
    w.writerow(
        [
            "Runtime to waypoints (s)",
            "saved timings",
            f"{r['whole_m5']['to_waypoints']:.1f}",
            f"{r['pieces_m4']['to_waypoints']:.1f}",
            "",
            "pieces include CoACD; saved timings, not re-timed",
        ]
    )
    w.writerow(
        [
            "Solve wall time (s)",
            "saved timings",
            f"{r['whole_m5']['solve']:.1f}",
            f"{r['pieces_m4']['solve']:.1f}",
            "",
            "pieces: 8 concurrent worker processes",
        ]
    )
    w.writerow(
        [
            "Runtime to waypoints excluding CoACD (s)",
            "saved timings",
            "",
            f"{r['pieces_m4']['to_waypoints_excluding_coacd']:.1f}",
            "",
            "",
        ]
    )
    w.writerow(
        [
            "Sum of per-piece elapsed solve times (s)",
            "saved timings",
            "",
            f"{r['pieces_m4']['sum_of_piece_elapsed_solve_times']:.1f}",
            "",
            "wall-clock times of concurrent workers summed; not CPU time",
        ]
    )
    return out.getvalue()


def compare_main(argv=None):
    """Compare the whole torus path with the decomposed one at a matched waypoint budget."""
    parser = argparse.ArgumentParser(description=compare_main.__doc__)
    store.add_check(parser)
    store.configure(parser.parse_args(argv))
    data, _ = compare()
    v = data["validation"]
    print("validation passed:", v["passed"], flush=True)
    if not v["passed"]:
        print(json.dumps(v, indent=1, default=store.serial), flush=True)
        raise SystemExit("STOP: validation failed; nothing written")
    store.save_json(RESULT, data)
    print(comparison_csv(data), flush=True)
    store.finish()
