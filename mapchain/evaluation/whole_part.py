"""Whole-part runs: solver validation and uniform transported paths."""

from __future__ import annotations

import argparse
import importlib.metadata
import itertools
import json
import time
import traceback

import numpy as np
import trimesh

from mapchain import store
from mapchain.evaluation.metrics import row_metrics
from mapchain.store import ROOT
from mapchain.volumetric.hilbert import hilbert_3d
from mapchain.volumetric.parts import (
    AVAILABLE_PARTS,
    MESH_PARTS,
    PARTS,
    geometry,
    load_or_create_mesh,
    normalize_mesh_to_unit_cube,
)
from mapchain.volumetric.transport import (
    NESTED_LBFGS_HANDOFF,
    NESTED_METHOD,
    StageMemory,
    adapter_solve,
    map_part,
    nested_initial_weights,
)

OUT = store.OUT / "evaluation" / "whole_part"
# Runs whose waypoints are read back: by the figures and by the torus comparison.
SAVED_WAYPOINTS = {("box", 2), ("l_box", 4), ("torus", 4), ("torus", 5), ("bunny", 5)}


def uniform_path(part, m):
    return OUT / f"{part}_m{m}.json"


def waypoints(part, m):
    """Saved waypoints of a run in SAVED_WAYPOINTS."""
    with np.load(uniform_path(part, m).with_suffix(".npz")) as data:
        return data["waypoints"]


def versions():
    return {
        p: importlib.metadata.version(p)
        for p in ("numpy", "scipy", "trimesh", "tetgen", "manifold3d", "psutil")
    } | {"geogram": "1.10.1"}


def validation_path(
    *, regularized=False, direct=False, warmstart=False, multilevel=False, nested=False
):
    flags = [
        n
        for n, v in (
            ("nested", nested),
            ("regularized", regularized),
            ("direct", direct),
            ("warmstart", warmstart),
            ("multilevel", multilevel),
        )
        if v
    ]
    return store.OUT / "volumetric" / f"validation_{'_'.join(flags) or 'default'}.json"


def validated(**flags):
    """Return the passing validation record for this solver configuration."""
    path = validation_path(**flags)
    if not path.exists():
        raise RuntimeError(f"Run python -m mapchain validate first ({path.name})")
    if not json.loads(path.read_text())["passed"]:
        raise RuntimeError(f"{path.name} failed; fix the solver first")
    return path.relative_to(ROOT).as_posix()


def distances(mesh, points):
    return (
        np.concatenate(
            [
                trimesh.proximity.closest_point(mesh, points[a : a + 512])[1]
                for a in range(0, len(points), 512)
            ]
        )
        if len(points)
        else np.empty(0)
    )


def metrics(part, mesh, b, points, m):
    distance_fn = (
        (lambda model, p: distances(mesh, p)) if part == "complex" or part in MESH_PARTS else None
    )
    with StageMemory() as stage:
        result = row_metrics(
            part,
            b,
            points,
            order=m,
            support_order=None,
            project=False,
            cone_batch_size=16,
            distance_fn=distance_fn,
        )
    result["stage"] = stage.report()
    result["p99_step"] = float(np.percentile(np.linalg.norm(np.diff(points, axis=0), axis=1), 99))
    params = result["segment_metrics"]["parameters"]
    params.pop("k", None)
    params["map"] = "Semi-discrete OT; no interpolation or projection"
    params["cone_working_batch_size"] = 16 if len(points) > 8192 else 128
    if part == "complex":
        params["distance_method"] = "Exact distance to the polygonal CSG surface"
    if part in MESH_PARTS:
        params["distance_method"] = f"Exact distance to the polygonal {part} surface"
    return result


def run_uniform(part, m, *, timeout=None, solver_only=False, max_volume="auto"):
    path = uniform_path(part, m)
    start = time.perf_counter()
    data = dict(
        part=part,
        m=m,
        waypoint_count=8**m,
        density="uniform",
        validation=validated(
            regularized=False, direct=False, warmstart=False, multilevel=False, nested=True
        ),
        versions=versions(),
        weight_regularization=0.0,
    )
    data["linear_solver"] = "Geogram OT_PRECG"
    data["geogram_initialization"] = NESTED_METHOD
    print(f"START {part} m={m}", flush=True)
    try:
        mesh, b, definition = geometry(part)
        data["construction"] = definition
        result = map_part(
            mesh,
            m,
            timeout_seconds=timeout,
            regularized=False,
            direct=False,
            warmstart=False,
            multilevel=False,
            nested=True,
            max_volume=max_volume,
        )
        data["meshing"] = result["meshing"]
        data["solving"] = result["stage"]
        data["adapter_stage"] = result["adapter_stage"]
        data["containment_stage"] = result["containment_stage"]
        data["initialization"] = result["initialization"]
        data["initialization_reports"] = result["initialization_reports"]
        data["solver_report"] = {
            k: result[k]
            for k in (
                "iterations",
                "empty_cells",
                "upstream_gradient_norm",
                "upstream_report",
                "max_relative_mass_error",
                "max_relative_adapter_mass_disagreement",
                "converged",
                "weight_gauge",
                "raw_weight_mean",
            )
        }
        data["cell_metrics"] = dict(
            max_diameter=float(result["diameters"].max()),
            p99_diameter=float(np.percentile(result["diameters"], 99)),
            fraction_centroids_outside=result["fraction_centroids_outside"],
        )
        data["acceptance"] = result["acceptance"]
        if not result["converged"]:
            raise RuntimeError("Geogram did not meet required per-cell mass tolerance")
        if not result["acceptance"]["passed"]:
            raise RuntimeError(f"Acceptance failed: {result['acceptance']}")
        if (part, m) in SAVED_WAYPOINTS:
            store.save_npz(path.with_suffix(".npz"), waypoints=result["centroids"])
        if not solver_only:
            data["metrics"] = metrics(part, mesh, b, result["centroids"], m)
        else:
            data["evaluation_scope"] = (
                "Solver acceptance and containment only; study metrics not requested"
            )
        data["passed"] = True
        print(
            f"DONE {part} m={m} solve={data['solving']['runtime_seconds']:.2f}s diameter={data['cell_metrics']['max_diameter']:.6g}",
            flush=True,
        )
    except Exception as error:
        data.update(getattr(error, "semidiscrete_stages", {}))
        data["passed"] = False
        data["failure"] = traceback.format_exc()
        print(data["failure"], flush=True)
    data["runtime_seconds"] = time.perf_counter() - start
    store.save_json(path, data)
    return data


def uniform_main(argv=None):
    """Transport uniform Hilbert paths into the target parts."""
    p = argparse.ArgumentParser(description=uniform_main.__doc__)
    p.add_argument("--parts", nargs="+", choices=AVAILABLE_PARTS, default=PARTS)
    p.add_argument("--orders", nargs="+", type=int, default=[4, 5, 6])
    p.add_argument("--timeout", type=float)
    p.add_argument("--solver-only", action="store_true")
    p.add_argument("--max-volume", type=float)
    store.add_check(p)
    args = p.parse_args(argv)
    store.configure(args)
    if any(m < 2 for m in args.orders):
        p.error("orders must be at least 2")
    for part in args.parts:
        for m in args.orders:
            run_uniform(
                part,
                m,
                timeout=args.timeout,
                solver_only=args.solver_only,
                max_volume="auto" if args.max_volume is None else args.max_volume,
            )
    store.finish()


def validate_main(argv=None):
    """Validate the semi-discrete solver on the unit cube and the order-4 torus."""
    parser = argparse.ArgumentParser(description=validate_main.__doc__)
    parser.add_argument("--regularized", action="store_true")
    parser.add_argument("--direct", action="store_true")
    parser.add_argument("--warmstart", action="store_true")
    parser.add_argument("--multilevel", action="store_true")
    parser.add_argument(
        "--nested",
        action="store_true",
        help="Nested m-1 start + HLBFGS handoff + Newton (default unregularised CG builds)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare with the saved validation instead of overwriting it",
    )
    parser.set_defaults(nested=True)
    args = parser.parse_args(argv)
    if args.nested and (args.regularized or args.direct or args.warmstart or args.multilevel):
        raise SystemExit("--nested uses only the default unregularised CG builds")
    start = time.perf_counter()
    report = dict(
        solver="Geogram/Exploragram",
        version="1.10.1",
        tests={},
        weight_regularization=1e-06 / 12 if args.regularized else 0.0,
    )
    report["linear_solver"] = "Geogram OT_SUPERLU" if args.direct else "Geogram OT_PRECG"
    report["geogram_initialization"] = (
        NESTED_METHOD
        if args.nested
        else "HLBFGS"
        if args.warmstart
        else "multilevel Newton"
        if args.multilevel
        else None
    )
    report["build_report"] = ".geogram/semidiscrete_geogram_build_v1.10.1*/build_report.json"
    try:
        vertices = np.asarray(list(itertools.product((0.0, 1.0), repeat=3)))
        lookup = {tuple(p): i for i, p in enumerate(vertices)}
        tetrahedra = []
        for order in itertools.permutations(range(3)):
            p = np.zeros(3)
            tet = [lookup[tuple(p)]]
            for axis in order:
                p[axis] = 1
                tet.append(lookup[tuple(p)])
            tetrahedra.append(tet)
        sites = hilbert_3d(3)
        nested_report = None
        if args.nested:
            coarse_sites = hilbert_3d(2)
            coarse_start = 0.5 * np.sum((coarse_sites - 0.5) ** 2, axis=1)
            coarse_start -= coarse_start.mean()
            coarse = adapter_solve(
                vertices,
                tetrahedra,
                coarse_sites,
                initial_weights=coarse_start,
                relative_tolerance=1e-12,
            )
            initial = nested_initial_weights(coarse["weights"], 3)
            nested_report = dict(
                coarse_order=2,
                coarse_converged=coarse["converged"],
                coarse_iterations=coarse["iterations"],
                coarse_max_relative_mass_error=coarse["max_relative_mass_error"],
                interpolated_start_max_absolute_weight=float(np.max(np.abs(initial))),
            )
            cube = adapter_solve(
                vertices,
                tetrahedra,
                sites,
                initial_weights=initial,
                keep_vertices=True,
                warmstart=True,
                lbfgs_handoff=NESTED_LBFGS_HANDOFF,
                relative_tolerance=1e-12,
            )
        else:
            cube = adapter_solve(
                vertices,
                tetrahedra,
                sites,
                keep_vertices=True,
                regularized=args.regularized,
                direct=args.direct,
                warmstart=args.warmstart,
                multilevel=args.multilevel,
                relative_tolerance=1e-12,
            )
        error = 0.0
        empty_geometry = []
        corners = np.asarray(list(itertools.product((-1 / 16, 1 / 16), repeat=3)))
        for i, site in enumerate(sites):
            lo, hi = cube["vertex_offsets"][i : i + 2]
            cell = cube["vertices"][int(lo) : int(hi)]
            if not len(cell):
                empty_geometry.append(i)
                continue
            expected = site + corners
            error = max(
                error,
                float(np.max(np.abs(cell.min(0) - (site - 1 / 16)))),
                float(np.max(np.abs(cell.max(0) - (site + 1 / 16)))),
                float(
                    np.max(
                        np.min(np.linalg.norm(expected[:, None] - cell[None, :], axis=2), axis=1)
                    )
                ),
            )
        cube_report = dict(
            stage=cube["stage"],
            site_count=512,
            tetrahedron_count=6,
            max_absolute_weight=float(np.max(np.abs(cube["weights"]))),
            raw_weight_mean=cube["raw_weight_mean"],
            weight_gauge=cube["weight_gauge"],
            max_centroid_coordinate_error=float(np.max(np.abs(cube["centroids"] - sites)))
            if np.isfinite(cube["centroids"]).all()
            else None,
            max_grid_cell_geometry_error=error if not empty_geometry else None,
            empty_geometry_cell_ids=empty_geometry,
            empty_cells=cube["empty_cells"],
            max_relative_mass_error=cube["max_relative_mass_error"],
            max_relative_volume_disagreement=cube["max_relative_adapter_mass_disagreement"],
            iterations=cube["iterations"],
            upstream_report=cube["upstream_report"],
            upstream_stdout=cube["stdout"],
            upstream_stderr=cube["stderr"],
            initialization_reports=cube["initialization_reports"],
        )
        cube_report["passed"] = (
            cube["converged"]
            and (not empty_geometry)
            and (error < 1e-12)
            and (cube_report["max_absolute_weight"] < 1e-10)
            and (cube_report["max_centroid_coordinate_error"] is not None)
            and (cube_report["max_centroid_coordinate_error"] < 1e-12)
        )
        if args.nested:
            nearest = (
                np.argmin(
                    np.linalg.norm(cube["centroids"][:, None] - sites[None, :], axis=2), axis=1
                )
                if np.isfinite(cube["centroids"]).all()
                else None
            )
            nested_report["hilbert_order_preserved"] = bool(
                nearest is not None and np.array_equal(nearest, np.arange(len(sites)))
            )
            nested_report["coarse_passed"] = bool(coarse["converged"])
            cube_report["nested"] = nested_report
            cube_report["passed"] = bool(
                cube_report["passed"]
                and nested_report["hilbert_order_preserved"]
                and nested_report["coarse_passed"]
            )
        report["tests"]["cube"] = cube_report
        print("CUBE", json.dumps(cube_report), flush=True)
        if not cube_report["passed"]:
            raise RuntimeError("Cube gate failed; torus not run")
        mesh = normalize_mesh_to_unit_cube(load_or_create_mesh("torus"))
        torus = map_part(
            mesh,
            4,
            timeout_seconds=1800,
            regularized=args.regularized,
            direct=args.direct,
            warmstart=args.warmstart,
            multilevel=args.multilevel,
            nested=args.nested,
        )
        torus_report = {
            key: torus[key]
            for key in (
                "meshing",
                "stage",
                "initialization",
                "max_relative_mass_error",
                "max_relative_adapter_mass_disagreement",
                "fraction_centroids_outside",
                "iterations",
                "upstream_report",
                "converged",
            )
        }
        torus_report["empty_cells"] = torus["empty_cells"]
        torus_report["site_count"] = 4096
        torus_report["initialization_reports"] = torus["initialization_reports"]
        torus_report["acceptance"] = torus["acceptance"]
        torus_report["passed"] = (
            torus["converged"]
            and torus["max_relative_adapter_mass_disagreement"] < 1e-06
            and torus["acceptance"]["passed"]
        )
        report["tests"]["torus_m4"] = torus_report
        print("TORUS", json.dumps(torus_report), flush=True)
    except Exception:
        report["failure"] = traceback.format_exc()
        print(report["failure"], flush=True)
    report["passed"] = len(report["tests"]) == 2 and all(
        r["passed"] for r in report["tests"].values()
    )
    report["runtime_seconds"] = time.perf_counter() - start
    path = validation_path(
        regularized=args.regularized,
        direct=args.direct,
        warmstart=args.warmstart,
        multilevel=args.multilevel,
        nested=args.nested,
    )
    store.configure(args)
    store.save_json(path, report)
    print(f"VALIDATION passed={report['passed']}: {path}", flush=True)
    store.finish()
    return 0 if report["passed"] else 1
