"""Timings of both planners against the saved plans."""

from __future__ import annotations

import argparse
import json
import os
import platform
from time import perf_counter

import numpy as np

from mapchain.kinematics.arm import OBSTACLES, RMAX, RMIN, build_patch
from mapchain.kinematics.sheets import (
    OUT,
    availability,
    evaluate,
    fibre_analysis,
    fibre_seed,
    geometry_plan,
    kinematics_plan,
    request,
    stretch_collocation,
)

PATH = OUT / "benchmark.json"


DENSITIES = ("uniform", "gradient")


def load_benchmark():
    if PATH.exists():
        return json.loads(PATH.read_text())
    return dict(
        protocol=dict(
            timer="time.perf_counter, wall clock, single process",
            warm_up="one untimed order-2 request per invocation (imports, first SLSQP/NumPy calls)",
            statistic="median over repeats; variability = interquartile range and min-max",
            request_plan="planning + certification of every continuous stretch; excludes evaluation and (geometry) availability scan",
            retained=dict(
                shared="fibre analysis (fibre.json)",
                kinematics="two certified patches (patch_*.npz) and the overlap split",
                geometry="nothing beyond the shared fibre analysis",
            ),
            equality_check="every timed repeat reproduces the saved q arrays and switch counts",
        ),
        prep={},
        requests={},
    )


def environment():
    import scipy

    return dict(
        platform=platform.platform(),
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        cpu_count=os.cpu_count(),
    )


def summary(values):
    v = np.asarray(values, float)
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return dict(
        values=v.tolist(),
        median=float(med),
        q1=float(q1),
        q3=float(q3),
        min=float(v.min()),
        max=float(v.max()),
        repeats=len(v),
    )


def prepare(repeats):
    """Time the shared fibre analysis and the kinematics-first patch builds; compare with saved patches."""
    result = json.loads((OUT / "results.json").read_text())
    ranges = [p["phi_range_degrees"] for p in result["preprocessing"]]
    shared, patch = ([], [[], []])
    for rep in range(repeats):
        start = perf_counter()
        fibre, report = fibre_analysis()
        shared.append(perf_counter() - start)
        lo, hi = report["common_overlap_degrees"]
        assert np.allclose([(-30.0, hi - 0.4), (lo + 0.4, 30.0)], ranges), (
            "fresh fibre analysis gives different patch ranges"
        )
        for family in (0, 1):
            start = perf_counter()
            q, info = build_patch(
                ranges[family],
                OBSTACLES,
                seed_function=lambda x, c, family=family, fibre=fibre: fibre_seed(
                    x, family, fibre, c
                ),
                margin=0.004,
                radial_range=(RMIN, RMAX),
                reverse_theta=family == 1,
            )
            patch[family].append(perf_counter() - start)
            saved = np.load(OUT / f"patch_{family}.npz")["q"]
            assert np.allclose(q, saved, atol=1e-06), f"patch {family} differs from saved"
        print(
            f"prep repeat {rep + 1}/{repeats}: shared {shared[-1]:.2f}s, patches {patch[0][-1]:.2f}s+{patch[1][-1]:.2f}s",
            flush=True,
        )
    data = load_benchmark()
    data["prep"] = dict(
        shared_fibre_analysis=summary(shared),
        patch_0=summary(patch[0]),
        patch_1=summary(patch[1]),
        kinematics_preparation=summary(np.array(patch[0]) + np.array(patch[1])),
        fibre_collision_queries=report["collision_queries"],
        patch_collision_queries=sum(
            p["preprocessing_collision_queries"] for p in result["preprocessing"]
        ),
        patch_ik_solves=sum(p["preprocessing_ik_solves"] for p in result["preprocessing"]),
    )
    data["environment"] = environment()
    PATH.write_text(json.dumps(data, indent=2) + "\n")


def requests(method, orders, repeats, densities=DENSITIES, eval_repeats=3, diag_repeats=3):
    result = json.loads((OUT / "results.json").read_text())
    ranges = [p["phi_range_degrees"] for p in result["preprocessing"]]
    split = result["split_phi_degrees"]
    patches = [np.load(OUT / f"patch_{f}.npz")["q"] for f in (0, 1)]
    saved_rows = {(r["method"], r["order"], r["density"]): r for r in result["requests"]}
    fibre = None
    plan = (
        (lambda o, d, diag=False: geometry_plan(o, d, fibre, diagnostics=diag))
        if method == "geometry"
        else lambda o, d, diag=False: kinematics_plan(o, d, patches, ranges, split)
    )
    plan(2, "uniform")
    data = load_benchmark()
    for order in orders:
        for density in densities:
            saved = dict(np.load(OUT / f"{method}_m{order}_{density}.npz"))
            times, evals, diags = ([], [], [])
            for _rep in range(repeats):
                start = perf_counter()
                out, info = plan(order, density)
                times.append(perf_counter() - start)
                assert (
                    np.allclose(out["q"], saved["q"], atol=1e-09)
                    and info["switches"] == saved_rows[method, order, density]["switches"]
                )
            entry = dict(
                method=method,
                order=order,
                density=density,
                plan_seconds=summary(times),
                switches=info["switches"],
                plan_collision_queries=info["plan_collision_queries"],
                plan_ik_solves=info["plan_ik_solves"],
            )
            for _rep in range(eval_repeats):
                start = perf_counter()
                evaluate(saved, density)
                evals.append(perf_counter() - start)
            entry["evaluation_seconds"] = summary(evals)
            if method == "geometry":
                requested = stretch_collocation(request(order, density))[0]
                for _rep in range(diag_repeats):
                    start = perf_counter()
                    availability(requested)
                    diags.append(perf_counter() - start)
                entry["availability_seconds"] = summary(diags)
            data["requests"][f"{method}_m{order}_{density}"] = entry
            data["environment"] = environment()
            PATH.write_text(json.dumps(data, indent=2) + "\n")
            print(
                f"{method} m{order} {density}: plan median {entry['plan_seconds']['median']:.4f}s [{entry['plan_seconds']['min']:.4f}, {entry['plan_seconds']['max']:.4f}] over {repeats}",
                flush=True,
            )


def benchmark_main(argv=None):
    """Time the 3R planners and check that every timed repeat reproduces the saved plans."""
    parser = argparse.ArgumentParser(
        description=benchmark_main.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--prep", action="store_true")
    parser.add_argument("--method", choices=("geometry", "kinematics"))
    parser.add_argument("--orders", nargs="+", type=int, default=[2, 3, 4])
    parser.add_argument("--densities", nargs="+", default=list(DENSITIES), choices=DENSITIES)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args(argv)
    if any(o not in (2, 3, 4) for o in args.orders):
        parser.error("orders 2-4 only")
    if args.prep:
        prepare(args.repeats)
    if args.method:
        requests(args.method, args.orders, args.repeats, args.densities)
