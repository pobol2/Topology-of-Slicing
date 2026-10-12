"""Checks of the saved 3R results (results/kinematics): plans, certificates, benchmark; no replanning.

Skipped when the results have not been generated (python -m mapchain kinematics / benchmark).
"""

import json
import unittest

import numpy as np

from mapchain import store
from mapchain.figures import three_link_posture_comparison as figure
from mapchain.kinematics.arm import (
    LINKS,
    NR,
    NT,
    OBSTACLES,
    RMAX,
    RMIN,
    Collision,
    certify,
    forward,
    joints,
    workspace,
)
from mapchain.kinematics.sheets import OUT, availability, component_of, evaluate, wrist_ik


def check_plans():
    """Saved plans: geometry, fibre labels, certified stretches, switch counts and metrics."""
    result = json.loads((OUT / "results.json").read_text())
    fibre = dict(np.load(OUT / "fibre.npz"))
    assert result["orders"] == [2, 3, 4]
    assert len(OBSTACLES) == 3
    assert LINKS == (1.0, 0.5, 0.5)
    assert np.allclose(
        OBSTACLES, [[0.9, np.sqrt(0.19), 0.2], [0.9, -np.sqrt(0.19), 0.2], [1.35, 0.1, 0.1]]
    )
    assert RMIN == 1.6 and RMAX == 1.85
    assert np.isclose(np.linalg.norm(workspace([0.0, 0.5])), 1.6)
    assert np.isclose(np.linalg.norm(workspace([1.0, 0.5])), 1.85)
    assert np.allclose(fibre["radii"], [1.6, 1.75, 1.8, 1.85])
    q_test = np.array([0.1, -0.5, 0.6])
    distances = Collision(OBSTACLES).distances(q_test).reshape(3, 3)
    points = joints(q_test)
    for io, (cx, cy, radius) in enumerate(OBSTACLES):
        for link in range(3):
            a, b = points[link : link + 2]
            d = b - a
            t = np.clip((np.array([cx, cy]) - a) @ d / (d @ d), 0, 1)
            assert np.isclose(distances[io, link], np.linalg.norm(a + t * d - [cx, cy]) - radius)
    assert np.all(fibre["selected"][:, 0] != fibre["selected"][:, 1])
    assert result["fibre"]["safe_elbow_seam_joins"] > 0
    lo, hi = result["fibre"]["common_overlap_degrees"]
    assert lo < result["split_phi_degrees"] < hi
    for ir, r in enumerate(fibre["radii"]):
        for branch, sign in enumerate((-1, 1)):
            for family in (0, 1):
                i, j = np.where(fibre["labels"][ir, branch] == fibre["selected"][ir, family])
                indices = np.linspace(0, len(i) - 1, min(300, len(i))).astype(int)
                x = r * np.stack(
                    (np.cos(fibre["phi"][i[indices]]), np.sin(fibre["phi"][i[indices]])), -1
                )
                q, _ = wrist_ik(x, fibre["psi"][j[indices]], sign)
                assert np.all(component_of(q, x) == family)
                assert np.all(Collision(OBSTACLES).clearance(q) > 0)
    for family in (0, 1):
        data = np.load(OUT / f"patch_{family}.npz")
        q = data["q"]
        faces = []
        for i in range(NR):
            for j in range(NT):
                faces.extend(
                    (
                        (q[i, j], q[i + 1, j], q[i + 1, j + 1]),
                        (q[i, j], q[i + 1, j + 1], q[i, j + 1]),
                    )
                )
        assert certify(faces, Collision(OBSTACLES)) > 0
    assert len(result["requests"]) == 12
    for row in result["requests"]:
        method, order, density = (row["method"], row["order"], row["density"])
        data = dict(np.load(OUT / f"{method}_m{order}_{density}.npz"))
        q, breaks, family, ids = (
            data["q"],
            data["break_edges"],
            data["family"],
            data["waypoint_indices"],
        )
        assert len(ids) == 4**order
        assert len(breaks) == len(q) - 1
        assert np.array_equal(np.diff(family) != 0, breaks)
        assert int(breaks.sum()) == row["switches"]
        assert np.max(abs(q)) <= np.pi
        assert np.all(component_of(q, forward(q)) == family)
        cuts = np.r_[0, np.flatnonzero(breaks) + 1, len(q)]
        for a, b in zip(cuts[:-1], cuts[1:]):
            assert b - a >= 2
            assert certify(np.stack((q[a : b - 1], q[a + 1 : b]), 1), Collision(OBSTACLES)) > 0
        metrics, x = evaluate(data, density)
        assert np.allclose(x, data["evaluation_x"])
        for name, value in metrics.items():
            assert np.isclose(value, row[name], atol=1e-10, rtol=0), name
        assert row["minimum_clearance"] > 0
        assert row["maximum_collocation_tracking_error"] < 0.002
        assert row["maximum_reset_tool_displacement"] < 0.002
        assert "plan_seconds" in row and row["plan_seconds"] >= 0
        if method == "kinematics":
            assert row["switches"] == 1
            assert np.all(family[ids[: row["lower_waypoint_count"]]] == 0)
            assert np.all(family[ids[row["lower_waypoint_count"] :]] == 1)
        else:
            available = data["availability"]
            forced = np.where(available.sum(axis=1) == 1, available.argmax(axis=1), -1)
            assert np.array_equal(forced, data["forced_family"])
            sequence = forced[forced >= 0]
            minimum = int(np.count_nonzero(np.diff(sequence)))
            costs = np.where(available[0], 0, 10**6)
            for allowed in available[1:]:
                costs = np.where(allowed, np.minimum(costs, costs[::-1] + 1), 10**6)
            assert int(costs.min()) == minimum == row["theoretical_minimum"] == row["switches"]


def check_benchmark():
    """Availability grid and benchmark agree with the saved results; no replanning."""
    result = json.loads((OUT / "results.json").read_text())
    bench = json.loads((OUT / "benchmark.json").read_text())
    grid = dict(np.load(OUT / "availability_grid.npz"))
    path = np.array(
        [
            [
                (RMIN**2 + s * (RMAX**2 - RMIN**2) - RMIN**2) / (RMAX**2 - RMIN**2),
                t,
            ]
            for s, t in [(0.0, 0.5), (0.5, 0.1), (0.5, 0.5), (0.5, 0.9), (1.0, 0.3), (1.0, 0.95)]
        ]
    )
    available, _, _, _ = availability(path)
    x = workspace(path)
    phi, r = figure.chart(x)
    for a, p_, rr in zip(available, phi, r):
        i, j = (np.argmin(abs(grid["radius"] - rr)), np.argmin(abs(grid["phi"] - p_)))
        if abs(grid["radius"][i] - rr) < 1e-09 and abs(grid["phi"][j] - p_) < 1e-09:
            assert np.array_equal(grid["available"][i, j], a)
    assert grid["available"].any(axis=-1).all(), "unsampled gap in the target"
    assert len(result["requests"]) == len(bench["requests"]) == 12
    for src in result["requests"]:
        t = bench["requests"][f"{src['method']}_m{src['order']}_{src['density']}"]
        assert t["switches"] == src["switches"] and t["plan_seconds"]["repeats"] >= 3
        assert np.isclose(t["plan_seconds"]["median"], np.median(t["plan_seconds"]["values"]))
        assert t["plan_seconds"]["min"] <= t["plan_seconds"]["median"] <= t["plan_seconds"]["max"]
    assert (
        bench["prep"]["shared_fibre_analysis"]["median"] > 0
        and bench["prep"]["kinematics_preparation"]["median"] > 0
    )
    for name in ("three_link_posture_comparison.pdf",):
        assert (store.FIGURES / name).stat().st_size > 0


@unittest.skipUnless((OUT / "benchmark.json").exists(), "3R results not generated")
class SavedResults(unittest.TestCase):
    def test_plans(self):
        check_plans()

    def test_benchmark(self):
        check_benchmark()


if __name__ == "__main__":
    unittest.main()
