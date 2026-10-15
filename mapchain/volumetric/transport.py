"""Semi-discrete optimal transport with the Geogram adapter: Laguerre-cell centroids in Hilbert order."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from mapchain.store import NATIVE
from mapchain.volumetric.hilbert import hilbert_3d

BUILD = NATIVE / "semidiscrete_geogram_build_v1.10.1/build"


def contains_chunked(mesh, points, batch_size=512):
    """Identical mesh membership queries with a bounded ray-cast working set."""
    points = np.asarray(points)
    return np.concatenate(
        [mesh.contains(points[a : a + batch_size]) for a in range(0, len(points), batch_size)]
    )


class StageMemory:
    """Sample total process-tree RSS every 20 ms within each measured stage."""

    def __enter__(self):
        import psutil

        self.process = psutil.Process()
        self.peak = 0
        self.start_rss = self.process.memory_info().rss
        self.start = time.perf_counter()
        self.stop = threading.Event()

        def sample():
            while not self.stop.is_set():
                total = 0
                for process in [self.process, *self.process.children(recursive=True)]:
                    try:
                        info = process.memory_info()
                        total += info.rss
                    except psutil.Error:
                        pass
                self.peak = max(self.peak, total)
                self.stop.wait(0.02)

        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.runtime = time.perf_counter() - self.start

    def report(self):
        return dict(
            runtime_seconds=self.runtime,
            peak_process_tree_bytes=self.peak,
            initial_process_rss_bytes=self.start_rss,
            sampling_interval_seconds=0.02,
            memory_definition="Peak sampled sum of process-tree RSS during this stage (20 ms sampling)",
        )


def adapter_solve(
    vertices,
    tetrahedra,
    sites,
    *,
    density=None,
    initial_weights=None,
    relative_tolerance=1e-6,
    max_iterations=500,
    keep_vertices=False,
    timeout_seconds=None,
    slice_z=None,
    regularized=False,
    direct=False,
    warmstart=False,
    multilevel=False,
    lbfgs_handoff=None,
):
    """Call the upstream solver once, reject any failed convergence afterwards."""
    vertices = np.ascontiguousarray(vertices, dtype="<f8")
    tetrahedra = np.ascontiguousarray(tetrahedra, dtype="<u4")
    sites = np.ascontiguousarray(sites, dtype="<f8")
    n = len(sites)
    density = np.ones(len(vertices)) if density is None else np.asarray(density, dtype="<f8")
    initial_weights = (
        np.zeros(n) if initial_weights is None else np.asarray(initial_weights, dtype="<f8")
    )
    if vertices.shape[1:] != (3,) or sites.shape[1:] != (3,) or tetrahedra.shape[1:] != (4,):
        raise ValueError("Expected vertex/site xyz and four tetrahedron vertex indices")
    if len(density) != len(vertices) or not np.all(density > 0):
        raise ValueError("Expected strictly positive vertex density")
    build = (
        NATIVE / "semidiscrete_geogram_build_v1.10.1_regularized/build" if regularized else BUILD
    )
    if direct:
        if not regularized:
            raise ValueError("Direct solve requires the tiny regularisation")
        build = NATIVE / "semidiscrete_geogram_build_v1.10.1_regularized_superlu/build"
    if warmstart and multilevel:
        raise ValueError("Choose one upstream initialization method")
    if lbfgs_handoff is not None and not warmstart:
        raise ValueError("lbfgs_handoff requires warmstart")
    if warmstart or multilevel:
        variant = build.parent.name + ("_warmstart" if warmstart else "_multilevel")
        if lbfgs_handoff is not None:
            variant += f"_handoff{lbfgs_handoff:g}"
        build = build.parent.with_name(variant) / "build"
    executable = build / "bin/semidiscrete_adapter.exe"
    if not executable.is_file():
        raise RuntimeError("Build first: python -m mapchain build")
    scratch_root = NATIVE / "semidiscrete_scratch"
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch_root) as directory, StageMemory() as stage:
        directory = Path(directory)
        source, destination = directory / "input.bin", directory / "output.bin"
        with source.open("xb") as handle:
            np.asarray(
                [len(vertices), len(tetrahedra), n, max_iterations, keep_vertices], dtype="<u8"
            ).tofile(handle)

            np.asarray(
                [relative_tolerance / (4 if regularized else 1) / np.sqrt(n)], dtype="<f8"
            ).tofile(handle)
            np.asarray([np.nan if slice_z is None else slice_z], dtype="<f8").tofile(handle)
            for array in (vertices, tetrahedra, density, sites, initial_weights):
                array.tofile(handle)
        environment = os.environ.copy()
        environment["PATH"] = str(build / "bin") + os.pathsep + environment.get("PATH", "")
        completed = subprocess.run(
            [str(executable), str(source), str(destination)],
            cwd=directory,
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        if completed.returncode:
            raise RuntimeError(
                f"Geogram exit {completed.returncode}: {completed.stdout}\n{completed.stderr}"
            )
        with destination.open("rb") as handle:
            count, iterations, empty, text_length, vertex_count, piece_count, piece_vertex_count = (
                np.fromfile(handle, "<u8", 7).astype(int)
            )
            gradient_norm = float(np.fromfile(handle, "<f8", 1)[0])
            stats = handle.read(text_length).decode()
            result = dict(
                weights=np.fromfile(handle, "<f8", count),
                geogram_masses=np.fromfile(handle, "<f8", count),
                volumes=np.fromfile(handle, "<f8", count),
                measured_masses=np.fromfile(handle, "<f8", count),
                geogram_centroids=np.fromfile(handle, "<f8", 3 * count).reshape(-1, 3),
                diameters=np.fromfile(handle, "<f8", count),
                centroids=np.fromfile(handle, "<f8", 3 * count).reshape(-1, 3),
                vertex_offsets=np.fromfile(handle, "<u8", count + 1),
                vertices=np.fromfile(handle, "<f8", 3 * vertex_count).reshape(-1, 3),
                iterations=int(iterations),
                empty_cells=int(empty),
                upstream_gradient_norm=gradient_norm,
                upstream_report=stats,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )
            result["slice_piece_sites"] = np.fromfile(handle, "<u4", piece_count)
            result["slice_piece_offsets"] = np.fromfile(handle, "<u8", piece_count + 1)
            result["slice_piece_vertices"] = np.fromfile(
                handle, "<f8", 3 * piece_vertex_count
            ).reshape(-1, 3)
            if handle.read(1):
                raise RuntimeError("Unexpected trailing adapter output")
    result["stage"] = stage.report()
    result["weight_regularization"] = 1e-6 / 12 if regularized else 0.0
    result["linear_solver"] = "Geogram OT_SUPERLU" if direct else "Geogram OT_PRECG"
    result["geogram_initialization"] = (
        "HLBFGS" if warmstart else "multilevel Newton" if multilevel else None
    )
    if warmstart:
        result["lbfgs_handoff_factor"] = 0.05 if lbfgs_handoff is None else lbfgs_handoff
    result["initialization_reports"] = [
        json.loads(line.split(" ", 1)[1])
        for line in completed.stderr.splitlines()
        if line.startswith(("GEOGRAM_WARMSTART ", "GEOGRAM_MULTILEVEL "))
    ]
    result["accepted_Newton_steps"] = max(0, result["iterations"] - 1)
    result["raw_weight_mean"] = float(result["weights"].mean())
    result["weights"] -= result["raw_weight_mean"]
    result["weight_gauge"] = "Mean zero; subtracting a common constant leaves all cells unchanged"
    total_mass = result["geogram_masses"].sum()
    result["masses"] = result["geogram_masses"] / total_mass
    result["max_relative_mass_error"] = float(np.max(np.abs(result["masses"] * n - 1)))
    result["max_relative_adapter_mass_disagreement"] = float(
        np.max(
            np.abs(result["measured_masses"] - result["geogram_masses"])
            / np.maximum(result["geogram_masses"], np.finfo(float).tiny)
        )
    )
    result["converged"] = bool(
        result["max_relative_mass_error"] < relative_tolerance
        and np.isfinite(result["centroids"]).all()
    )
    return result


def tetrahedralize(mesh, *, max_volume=2e-5):
    """Conforming TetGen mesh of the existing polygonal watertight part.

    max_volume=None drops the interior volume limit; the surface and the
    radius-edge quality bound are unchanged. Only valid for uniform density,
    because a P1 density is represented on the mesh vertices.
    """
    import tetgen

    if not mesh.is_watertight:
        raise ValueError("A watertight mesh is required")
    switches = "pq1.4Q" if max_volume is None else f"pq1.4a{max_volume:.17g}Q"
    with StageMemory() as stage:
        generator = tetgen.TetGen(np.asarray(mesh.vertices), np.asarray(mesh.faces))
        result = generator.tetrahedralize(switches=switches)
        vertices, tetrahedra = result[:2]
    return (
        vertices,
        tetrahedra,
        dict(
            **stage.report(),
            vertex_count=len(vertices),
            tetrahedron_count=len(tetrahedra),
            max_tetrahedron_volume=max_volume,
            switches=switches,
        ),
    )


def positive_cell_initial_weights(mesh, vertices, tetrahedra, sites):
    """Exact affine contraction giving every initial Laguerre cell positive mass.

    If B(q,r) is inside Omega, choose s so the contracted unit cube fits inside
    the ball. Weights (1-s)|y-q|^2 make the power diagram the affine contraction
    of the zero-weight grid diagram. No site or output waypoint is moved.
    """
    import trimesh
    from scipy.spatial import cKDTree

    centres = np.asarray(vertices)[np.asarray(tetrahedra)].mean(axis=1)

    triangles = np.asarray(mesh.triangles)
    edge_midpoints = ((triangles + np.roll(triangles, -1, axis=1)) / 2).reshape(-1, 3)
    on_surface = np.concatenate([np.asarray(mesh.vertices), triangles.mean(axis=1), edge_midpoints])
    bounds = cKDTree(on_surface).query(centres)[0]
    order = np.argsort(-bounds, kind="stable")
    count = 512
    while True:
        candidates = order[:count]
        exact = np.concatenate(
            [
                trimesh.proximity.closest_point(mesh, centres[candidates[a : a + 512]])[1]
                for a in range(0, len(candidates), 512)
            ]
        )
        best = float(exact.max())
        if count >= len(centres) or best > bounds[order[count]]:
            break
        count = min(2 * count, len(centres))
    ties = candidates[exact == best]
    index = int(ties.min())
    q, radius = centres[index], best
    corner_radius = float(np.linalg.norm(np.maximum(q, 1 - q)))
    scale = radius / corner_radius
    if not 0 < scale < 1:
        raise ValueError("Could not obtain an interior ball for positive cells")
    weights = (1 - scale) * np.sum((sites - q) ** 2, axis=1)
    weights -= weights.mean()
    return weights, dict(
        method="Exact contracted grid inside an interior ball",
        centre=q.tolist(),
        radius=radius,
        contraction=scale,
        exact_distance_queries=int(count),
        tetrahedron_centres=len(centres),
    )


NESTED_START_ORDER = 2


NESTED_LBFGS_HANDOFF = 0.5


NESTED_METHOD = f"nested m-1 interpolation + HLBFGS handoff {NESTED_LBFGS_HANDOFF:g}"


def nested_initial_weights(coarse_weights, m):
    """Order-m start from converged order-(m-1) weights on the nested Hilbert cell-centre grids.

    Trilinear interpolation with linear extrapolation in the outer half-cells; affine
    weights (including the all-zero cube solution) are reproduced exactly. The start may
    contain empty cells, so it is only used with the upstream HLBFGS stage before Newton.
    """
    from scipy.interpolate import RegularGridInterpolator

    k = 2 ** (m - 1)
    coarse_sites = hilbert_3d(m - 1)
    index = np.floor(coarse_sites * k).astype(int)
    grid = np.empty((k, k, k))
    grid[index[:, 0], index[:, 1], index[:, 2]] = coarse_weights
    axis = (np.arange(k) + 0.5) / k
    interpolate = RegularGridInterpolator(
        (axis, axis, axis), grid, method="linear", bounds_error=False, fill_value=None
    )
    weights = interpolate(hilbert_3d(m))
    return weights - weights.mean()


def nested_solve(
    mesh,
    vertices,
    tetrahedra,
    m,
    rho,
    *,
    keep_vertices=False,
    max_iterations=500,
    timeout_seconds=None,
    slice_z=None,
):
    """Contraction-start Newton at NESTED_START_ORDER, then per order: interpolated start,
    upstream HLBFGS (handoff NESTED_LBFGS_HANDOFF), upstream Newton. Every level must converge."""
    if m < NESTED_START_ORDER:
        raise ValueError(f"Nested solve needs m >= {NESTED_START_ORDER}")
    levels = []
    result = {}
    for order in range(NESTED_START_ORDER, m + 1):
        sites = hilbert_3d(order)
        start = time.perf_counter()
        if order == NESTED_START_ORDER:
            initial, initialization = positive_cell_initial_weights(
                mesh, vertices, tetrahedra, sites
            )
            options = {}
        else:
            initial = nested_initial_weights(result["weights"], order)
            initialization = dict(
                method="Nested order-(m-1) trilinear interpolation", source_order=order - 1
            )
            options = dict(warmstart=True, lbfgs_handoff=NESTED_LBFGS_HANDOFF)
        final = order == m
        result = adapter_solve(
            vertices,
            tetrahedra,
            sites,
            density=rho,
            initial_weights=initial,
            keep_vertices=keep_vertices and final,
            max_iterations=max_iterations,
            timeout_seconds=timeout_seconds,
            slice_z=slice_z if final else None,
            **options,
        )
        levels.append(
            dict(
                order=order,
                initialization=initialization,
                runtime_seconds=time.perf_counter() - start,
                adapter_stage=result["stage"],
                iterations=result["iterations"],
                empty_cells=result["empty_cells"],
                initialization_reports=result["initialization_reports"],
                max_relative_mass_error=result["max_relative_mass_error"],
                converged=result["converged"],
                upstream_report=result["upstream_report"],
            )
        )
        if not result["converged"]:
            raise RuntimeError(f"Nested level m={order} did not converge: {levels[-1]}")
    return result, dict(
        method=NESTED_METHOD,
        start_order=NESTED_START_ORDER,
        lbfgs_handoff_factor=NESTED_LBFGS_HANDOFF,
        levels=levels,
    )


def acceptance_report(result, sites, *, relative_tolerance=1e-6):
    """Acceptance criteria for a final result: per-cell mass error, no empty cell,
    finite centroids, and Hilbert output order.

    Order check: a Laguerre cell is convex, so the centroid of its (possibly
    non-convex) clipped part lies inside it. Hence centroid i must be power-nearest
    to site i. Power distance |c-y|^2-w equals squared 4D distance to the lifted
    site (y, sqrt(W-w)) minus W, so a 4D KD-tree finds the power-nearest site.
    """
    from scipy.spatial import cKDTree

    weights, centroids = np.asarray(result["weights"]), np.asarray(result["centroids"])
    finite = bool(np.isfinite(centroids).all())
    empty = int(result["empty_cells"])
    nonpositive = int(
        np.sum(~(result["geogram_masses"] > 0)) + np.sum(~(result["measured_masses"] > 0))
    )
    misordered = None
    if finite:
        lift = np.sqrt(weights.max() - weights)
        _, nearest = cKDTree(np.column_stack([sites, lift])).query(
            np.column_stack([centroids, np.zeros(len(sites))]), workers=-1
        )
        wrong = np.flatnonzero(nearest != np.arange(len(sites)))
        if len(wrong):
            own = np.sum((centroids[wrong] - sites[wrong]) ** 2, axis=1) - weights[wrong]
            other = (
                np.sum((centroids[wrong] - sites[nearest[wrong]]) ** 2, axis=1)
                - weights[nearest[wrong]]
            )
            wrong = wrong[own - other > 1e-12 * (1 + np.abs(other))]
        misordered = int(len(wrong))
    report = dict(
        max_relative_mass_error=result["max_relative_mass_error"],
        relative_tolerance=relative_tolerance,
        empty_cells=empty,
        nonpositive_cell_masses=nonpositive,
        finite_centroids=finite,
        hilbert_misordered_cells=misordered,
        hilbert_order_check="centroid i power-nearest to Hilbert site i (convex Laguerre cell)",
    )
    report["passed"] = bool(
        result["max_relative_mass_error"] < relative_tolerance
        and empty == 0
        and nonpositive == 0
        and finite
        and misordered == 0
    )
    return report


def map_part(
    mesh,
    m,
    *,
    density=None,
    max_volume="auto",
    keep_vertices=False,
    max_iterations=500,
    timeout_seconds=None,
    slice_z=None,
    regularized=False,
    direct=False,
    warmstart=False,
    multilevel=False,
    nested=False,
):
    """The input mesh must already use the pipeline's 0.05-padded frame.

    max_volume='auto': no interior volume limit for the nested method with uniform
    density (the mesh only describes the shape); otherwise the original 2e-5.
    """
    if nested and (regularized or direct or warmstart or multilevel):
        raise ValueError(
            "The nested method is validated only with the default unregularised CG build"
        )
    if max_volume == "auto":
        max_volume = None if nested and density is None else 2e-5
    vertices, tetrahedra, meshing = tetrahedralize(mesh, max_volume=max_volume)
    sites = hilbert_3d(m)
    try:
        with StageMemory() as solve_stage:
            rho = (
                np.ones(len(vertices))
                if density is None
                else np.asarray(density(vertices), dtype=float)
            )
            if nested:
                result, initialization = nested_solve(
                    mesh,
                    vertices,
                    tetrahedra,
                    m,
                    rho,
                    keep_vertices=keep_vertices,
                    max_iterations=max_iterations,
                    timeout_seconds=timeout_seconds,
                    slice_z=slice_z,
                )
            else:
                initial, initialization = positive_cell_initial_weights(
                    mesh, vertices, tetrahedra, sites
                )
                result = adapter_solve(
                    vertices,
                    tetrahedra,
                    sites,
                    density=rho,
                    initial_weights=initial,
                    keep_vertices=keep_vertices,
                    max_iterations=max_iterations,
                    timeout_seconds=timeout_seconds,
                    slice_z=slice_z,
                    regularized=regularized,
                    direct=direct,
                    warmstart=warmstart,
                    multilevel=multilevel,
                )
    except Exception as error:
        error.semidiscrete_stages = {"meshing": meshing, "solving": solve_stage.report()}
        raise
    result["adapter_stage"] = result["stage"]
    result["stage"] = solve_stage.report()
    with StageMemory() as containment_stage:
        finite = np.isfinite(result["centroids"]).all(axis=1)
        result["inside"] = np.zeros(len(sites), bool)
        if finite.any():
            result["inside"][finite] = contains_chunked(mesh, result["centroids"][finite])
    result["containment_stage"] = containment_stage.report()
    result["fraction_centroids_outside"] = float(np.mean(~result["inside"]))
    result["meshing"] = meshing
    result["initialization"] = initialization
    result["density_representation"] = "P1 nodal interpolation on tetrahedra"
    result["m"] = m
    result["sites"] = sites
    result["acceptance"] = acceptance_report(result, sites)
    return result
