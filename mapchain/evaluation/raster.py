"""Layered raster baseline on the same parts."""

from __future__ import annotations

import argparse
import time

import numpy as np

from mapchain import store
from mapchain.evaluation.metrics import segment_engine
from mapchain.evaluation.whole_part import distances
from mapchain.volumetric.parts import PARTS, geometry

OUT = store.OUT / "evaluation" / "raster"


def layered(model, b, order, *, engine=segment_engine):
    started = time.perf_counter()
    side = 2**order
    h = 1 / side
    rows = np.arange(side)
    columns = np.tile(rows, (side, 1))
    columns[1::2] = columns[1::2, ::-1]
    yy = np.repeat(rows, side)
    xx = columns.ravel()
    point_layers, cell_layers = ([], [])
    active = []
    for k in range(side):
        cells = np.column_stack((xx, yy, np.full(len(xx), k)))
        pts = (cells + 0.5) * h
        inside = b.inside_fn(pts)
        if inside.any():
            point_layers.append(pts[inside])
            cell_layers.append(cells[inside])
            active.append(k)
    pts = np.concatenate(point_layers)
    cells = np.concatenate(cell_layers)
    between = np.diff(cells[:, 2]) != 0
    adjacent = (np.abs(np.diff(cells, axis=0)).sum(axis=1) == 1) & ~between
    metrics, exterior, lengths = engine(model, b, pts, order=order, depositing=adjacent)
    depositing = adjacent & ~exterior
    if not np.array_equal(depositing, adjacent):
        metrics, _, _ = engine(model, b, pts, order=order, depositing=depositing)
    within = ~between & ~depositing
    dlengths = lengths[depositing]
    return {
        "p_path": order,
        "h": h,
        "nominal_layer_count": side,
        "active_layer_count": len(active),
        "active_layer_indices": active,
        "waypoint_count": len(pts),
        "inter_layer_interruptions": int(between.sum()),
        "within_layer_interruptions": int(within.sum()),
        "total_interruptions": int((~depositing).sum()),
        "all_moves_path_length": float(lengths.sum()),
        "travel_length": float(lengths[~depositing].sum()),
        "depositing_path_length": float(dlengths.sum()),
        "depositing_segment_count": int(depositing.sum()),
        "detected_jumps_depositing_segments": int(np.count_nonzero(dlengths > 5 * dlengths.mean())),
        "segment_metrics": metrics,
        "runtime_seconds": time.perf_counter() - started,
    }


def raster(part, order):
    mesh, bundle, _ = geometry(part)
    distance_fn = (lambda model, p: distances(mesh, p)) if part == "complex" else None

    def engine(model, bundle, points, **kwargs):
        return segment_engine(
            model, bundle, points, cone_batch_size=16, distance_fn=distance_fn, **kwargs
        )

    data = layered(part, bundle, order, engine=engine)
    data["cell_count"] = data["waypoint_count"]
    data["travel_move_count"] = data["total_interruptions"]
    data["between_layer_stops"] = data["inter_layer_interruptions"]
    data["travel_scope"] = (
        "All within-layer gaps/nondepositing moves plus transitions between successive occupied layers"
    )
    return data


def raster_main(argv=None):
    """Layered raster baseline on the same parts."""
    parser = argparse.ArgumentParser(description=raster_main.__doc__)
    parser.add_argument("--parts", nargs="+", choices=PARTS, default=PARTS)
    parser.add_argument("--orders", nargs="+", type=int, choices=(4, 5, 6), default=(4, 5, 6))
    store.add_check(parser)
    args = parser.parse_args(argv)
    store.configure(args)
    for part in args.parts:
        for order in args.orders:
            store.save_json(OUT / f"{part}_m{order}.json", raster(part, order))
    store.finish()
