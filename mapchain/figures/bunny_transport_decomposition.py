"""Teaser: cube-to-part morphs of the whole part and of its decomposition."""

from __future__ import annotations

import argparse
import itertools
import json
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pymupdf as fitz
import pyvista as pv
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.optimize import linear_sum_assignment

from mapchain import store
from mapchain.evaluation.metrics import exterior_segments
from mapchain.evaluation.whole_part import uniform_path, waypoints
from mapchain.figures.style import MORPH_CAMERA_FOCUS, MORPH_CAMERA_POSITION
from mapchain.volumetric.decomposition import OUT as DECOMPOSITION
from mapchain.volumetric.hilbert import hilbert_3d
from mapchain.volumetric.parts import (
    load_or_create_mesh,
    normalize_mesh_to_unit_cube,
    sample_bundle,
)


def add_outline(plotter, kind, bounds=None):
    """Draw endpoint geometry without filling or covering the transported path."""
    curves = []
    if kind == "cube":
        for axis in range(3):
            other = [i for i in range(3) if i != axis]
            for a in (0, 1):
                for b in (0, 1):
                    edge = np.zeros((2, 3))
                    edge[:, other] = [a, b]
                    edge[:, axis] = [0, 1]
                    if bounds is not None:
                        low, high = np.asarray(bounds)
                        edge = low + edge * (high - low)
                    curves.append(edge)
    elif kind:
        mesh = normalize_mesh_to_unit_cube(load_or_create_mesh(kind, store.ROOT / "models"))
        faces = np.column_stack((np.full(len(mesh.faces), 3), mesh.faces)).ravel()
        surface = pv.PolyData(np.asarray(mesh.vertices), faces)
        plotter.add_silhouette(surface, color="#707780", opacity=0.4, line_width=1.5)
    for points in curves:
        line = pv.PolyData(points)
        line.lines = np.r_[len(points), np.arange(len(points))]
        plotter.add_mesh(line, color="#707780", opacity=0.4, line_width=1.5, lighting=False)


def render_tube(
    points,
    time,
    outline=None,
    order=5,
    keep=None,
    elevation=None,
    path_lengths=None,
    cube_bounds=None,
    radius=None,
):
    path = pv.PolyData(points)
    if path_lengths is not None:
        if keep is not None:
            raise ValueError("Independent paths cannot use a single segment mask")
        lengths = np.asarray(path_lengths, dtype=int)
        if np.any(lengths < 2) or lengths.sum() != len(points):
            raise ValueError("Path lengths must partition all points into polylines")
        starts = np.r_[0, np.cumsum(lengths)[:-1]]
        path.lines = np.concatenate(
            [np.r_[n, np.arange(start, start + n)] for start, n in zip(starts, lengths)]
        )
    elif keep is None:
        path.lines = np.r_[len(points), np.arange(len(points))]
    else:
        indices = np.flatnonzero(keep)
        runs = np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1)
        path.lines = np.concatenate(
            [np.r_[len(run) + 1, run, run[-1] + 1] for run in runs if len(run)]
        )
    path.point_data["t"] = time
    tube = path.tube(
        radius=radius if radius is not None else 0.0014 * 2 ** (5 - order), n_sides=8, capping=True
    )
    plotter = pv.Plotter(off_screen=True, window_size=(1600, 1600))
    plotter.set_background("white")
    plotter.add_mesh(
        tube,
        scalars="t",
        cmap="viridis",
        clim=(0, 1),
        show_scalar_bar=False,
        smooth_shading=True,
        ambient=0.55,
        diffuse=0.45,
        specular=0.08,
    )
    if outline:
        add_outline(plotter, outline)
    if cube_bounds is not None:
        for bounds in cube_bounds:
            add_outline(plotter, "cube", bounds=bounds)
    camera = MORPH_CAMERA_POSITION.copy()
    if elevation is not None:
        azimuth = np.arctan2(-3.5, 2.1)
        angle = np.deg2rad(elevation)
        camera = np.full(3, 0.5) + 5 * np.array(
            [np.cos(angle) * np.cos(azimuth), np.cos(angle) * np.sin(azimuth), np.sin(angle)]
        )
    plotter.camera_position = [camera, MORPH_CAMERA_FOCUS, (0, 0, 1)]
    plotter.enable_parallel_projection()
    plotter.camera.parallel_scale = 0.83
    plotter.enable_anti_aliasing("ssaa")
    image = plotter.screenshot(return_img=True)
    plotter.close()
    return image


def whole_main(argv=None):
    """Render the cube-to-part morph of one whole-part path."""
    parser = argparse.ArgumentParser(description=whole_main.__doc__)
    parser.add_argument("--part", choices=("torus", "bunny"), default="torus")
    parser.add_argument("--order", type=int, choices=(4, 5, 6), default=5)
    display = parser.add_mutually_exclusive_group()
    display.add_argument(
        "--show-exterior", action="store_true", help="Draw the complete original path"
    )
    display.add_argument(
        "--remove-exterior",
        action="store_true",
        help="Omit exterior target segments solely for display",
    )
    parser.add_argument("--output", type=Path, help="Destination PDF")
    parser.add_argument(
        "--elevation", type=float, help="Camera elevation in degrees above the XY plane"
    )
    args = parser.parse_args(argv)
    if args.elevation is not None and (not 0 < args.elevation < 90):
        parser.error("--elevation must be between 0 and 90 degrees")
    order = args.order
    count = 8**order
    part = args.part
    filename = "teaser_morph.pdf" if part == "torus" else f"teaser_morph_{part}.pdf"
    if args.remove_exterior:
        filename = Path(filename).stem + "_filtered.pdf"
    output = args.output or Path(tempfile.gettempdir()) / filename
    output.parent.mkdir(exist_ok=True)
    result_path = uniform_path(part, order)
    result = json.loads(result_path.read_text())
    assert result.get("passed", True) and result["solver_report"]["converged"]
    assert result["acceptance"]["passed"]
    assert result["acceptance"]["hilbert_misordered_cells"] == 0
    assert result["part"] == part and result["m"] == order
    assert result["density"] == "uniform"
    source = hilbert_3d(order)
    target = np.asarray(waypoints(part, order), dtype=float)
    print(f"Loaded validated semi-discrete centroids from {result_path.name}.", flush=True)
    assert source.shape == target.shape == (count, 3)
    assert np.isfinite(target).all()
    if part == "bunny":
        mesh = normalize_mesh_to_unit_cube(load_or_create_mesh(part, store.ROOT / "models"))
        faces = np.column_stack((np.full(len(mesh.faces), 3), mesh.faces)).ravel()
        surface = pv.PolyData(np.asarray(mesh.vertices), faces)

        def inside(points):
            return (
                pv.PolyData(points)
                .select_enclosed_points(surface, tolerance=1e-09, check_surface=True)[
                    "SelectedPoints"
                ]
                .astype(bool)
            )
    else:
        inside = sample_bundle(part).inside_fn
    unfiltered = args.show_exterior or (part == "bunny" and (not args.remove_exterior))
    keep = None if unfiltered else ~exterior_segments(target, inside)
    removed = 0 if keep is None else int(np.count_nonzero(~keep))
    print(f"Omitting {removed} exterior target segments for display only.", flush=True)
    time = np.linspace(0, 1, len(source))
    values = [0, 0.25, 0.5, 0.75, 1]
    labels = ["0", "\\frac{1}{4}", "\\frac{1}{2}", "\\frac{3}{4}", "1"]
    fig = plt.figure(figsize=(12, 3.4))
    for i, (s, label) in enumerate(zip(values, labels)):
        points = (1 - s) * source + s * target
        print(f"Rendering s={s:g}: {len(points)} waypoints", flush=True)
        outline = "cube" if s == 0 else part if s == 1 else None
        image = render_tube(
            points, time, outline=outline, order=order, keep=keep, elevation=args.elevation
        )
        ax = fig.add_axes([0.005 + i * 0.2, 0.28, 0.19, 0.6])
        ax.imshow(image)
        ax.set_axis_off()
        ax.set_title(f"$s={label}$", fontsize=18, pad=0, loc="center")
    cax = fig.add_axes([0.005, 0.16, 0.99, 0.055])
    bar = fig.colorbar(
        ScalarMappable(norm=Normalize(0, 1), cmap="viridis"),
        cax=cax,
        orientation="horizontal",
        ticks=[0, 0.25, 0.5, 0.75, 1],
    )
    bar.outline.set_linewidth(0.4)
    bar.ax.tick_params(labelsize=14, length=3, pad=3)
    bar.ax.get_xticklabels()[0].set_ha("left")
    bar.ax.get_xticklabels()[-1].set_ha("right")
    bar.set_label("Traversal time $t$", fontsize=16, labelpad=4)
    fig.savefig(
        output,
        dpi=400,
        metadata={
            "Title": f"Cube-to-{part} morph using semi-discrete optimal transport",
            "Subject": f"Validated Hilbert-ordered cell centroids from {result_path.name}",
        },
    )
    plt.close(fig)
    print(f"Saved {output}; original result unchanged; {removed} segments hidden.", flush=True)


def load_paths(folder, part="torus"):
    record = json.loads((folder / "record.json").read_text())
    assert record["object"] == part
    assert record["partition"]["checks"]["passed"]
    assert record["summary"]["all_solves_passed"]
    pieces = record["pieces"]
    count = len(pieces)
    assert count == {"torus": 8, "bunny": 9}[part]
    assert record["summary"]["total_waypoints"] == count * 4096
    targets, times = ([], [])
    shares = np.asarray([p["mass_share"] for p in pieces])
    assert np.all(shares > 0) and np.isclose(shares.sum(), 1)
    boundaries = np.r_[0, np.cumsum(shares)]
    with np.load(folder / "paths.npz") as archive:
        assert int(archive["count"]) == len(pieces)
        for j, piece in enumerate(pieces):
            solve = piece["solve"]
            assert piece["order"] == 4 and solve["converged"] and solve["passed"]
            assert solve["acceptance"]["passed"]
            assert solve["acceptance"]["hilbert_misordered_cells"] == 0
            target = np.asarray(archive[f"p{j}"], dtype=float)
            assert target.shape == (4096, 3) and np.isfinite(target).all()
            targets.append(target)
            times.append(np.linspace(boundaries[j], boundaries[j + 1], len(target)))
    side = 0.47 if count == 8 else 0.3
    lows = np.asarray(list(itertools.product((0.0, 1.0 - side), repeat=3)))
    if count == 9:
        lows = np.vstack((lows, np.full(3, (1.0 - side) / 2)))
    centres = lows + side / 2
    target_centres = np.asarray([p.mean(axis=0) for p in targets])
    rows, slots = linear_sum_assignment(
        np.linalg.norm(target_centres[:, None, :] - centres[None, :, :], axis=2)
    )
    placement = np.empty(count, dtype=int)
    placement[rows] = slots
    lows = lows[placement]
    source = hilbert_3d(4)
    sources = [low + side * source for low in lows]
    return (sources, targets, times, np.stack((lows, lows + side), axis=1))


def combine_rows(whole, decomposition, output, part="torus", piece_count=8):
    """Stack aligned morph images with one set of stage labels and one bottom bar."""
    result = fitz.open()
    with fitz.open(whole) as top, fitz.open(decomposition) as bottom:
        width = top[0].rect.width
        frames = [
            (item[0], bottom[0].get_image_rects(item[0])[0])
            for item in bottom[0].get_images()
            if item[2] == item[3]
        ]
        assert len(frames) == 5
        first_y = min((rect.y0 for _, rect in frames))
        last_y = max((rect.y1 for _, rect in frames))
        top_frames = [
            rect
            for item in top[0].get_images()
            if item[2] == item[3]
            for rect in top[0].get_image_rects(item[0])
        ]
        top_height = max(rect.y1 for rect in top_frames) + 2
        gap = 4.0
        scale = width / bottom[0].rect.width
        image_height = (last_y - first_y) * scale
        bar_start = last_y + 2
        bar_height = (bottom[0].rect.height - bar_start) * scale
        page = result.new_page(width=width, height=top_height + gap + image_height + bar_height)
        top[0].add_redact_annot(fitz.Rect(0, top_height, width, top[0].rect.height))
        top[0].apply_redactions()
        page.show_pdf_page(
            fitz.Rect(0, 0, width, top_height), top, 0, clip=fitz.Rect(0, 0, width, top_height)
        )
        lower_y = top_height + gap
        for xref, rect in frames:
            destination = fitz.Rect(
                rect.x0 * scale,
                lower_y + (rect.y0 - first_y) * scale,
                rect.x1 * scale,
                lower_y + (rect.y1 - first_y) * scale,
            )
            page.insert_image(destination, stream=bottom.extract_image(xref)["image"])
        bottom[0].add_redact_annot(fitz.Rect(0, 0, bottom[0].rect.width, bar_start))
        bottom[0].apply_redactions()
        page.show_pdf_page(
            fitz.Rect(0, lower_y + image_height, width, page.rect.height),
            bottom,
            0,
            clip=fitz.Rect(0, bar_start, bottom[0].rect.width, bottom[0].rect.height),
        )
    result.set_metadata(
        {
            "title": f"Whole-part and decomposed {part} morphs",
            "subject": f"Whole-part order 5; {piece_count} independent order-4 lower-row paths; no joins",
        }
    )
    result.save(output)
    result.close()


def pieces_main(argv=None):
    """Render the morph of the decomposed part and stack it under the whole-part morph."""
    parser = argparse.ArgumentParser(description=pieces_main.__doc__)
    parser.add_argument("--part", choices=("torus", "bunny"), default="torus")
    parser.add_argument("--folder", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--whole", type=Path, help="Original teaser PDF to place above the decomposition"
    )
    args = parser.parse_args(argv)
    suffix = "" if args.part == "torus" else f"_{args.part}"
    args.folder = args.folder or DECOMPOSITION / args.part
    args.output = (
        args.output or Path(tempfile.gettempdir()) / f"teaser_morph_decomposition{suffix}.pdf"
    )
    args.whole = args.whole or Path(tempfile.gettempdir()) / f"teaser_morph{suffix}.pdf"
    sources, targets, times, bounds = load_paths(args.folder, args.part)
    lengths = [len(p) for p in targets]
    source, target, time = map(np.concatenate, (sources, targets, times))
    segments = sum(n - 1 for n in lengths)
    fig = plt.figure(figsize=(12, 3.4))
    values = [0, 0.25, 0.5, 0.75, 1]
    labels = ["0", "\\frac{1}{4}", "\\frac{1}{2}", "\\frac{3}{4}", "1"]
    for i, (s, label) in enumerate(zip(values, labels)):
        print(
            f"Rendering {args.part} decomposition s={s:g}: {len(lengths)} independent paths",
            flush=True,
        )
        image = render_tube(
            (1 - s) * source + s * target,
            time,
            outline=args.part if s == 1 else None,
            path_lengths=lengths,
            cube_bounds=bounds if s == 0 else None,
            radius=0.0014,
        )
        ax = fig.add_axes([0.005 + i * 0.2, 0.28, 0.19, 0.6])
        ax.imshow(image)
        ax.set_axis_off()
        ax.set_title(f"$s={label}$", fontsize=18, pad=0, loc="center")
    cax = fig.add_axes([0.005, 0.16, 0.99, 0.055])
    bar = fig.colorbar(
        ScalarMappable(norm=Normalize(0, 1), cmap="viridis"),
        cax=cax,
        orientation="horizontal",
        ticks=values,
    )
    bar.outline.set_linewidth(0.4)
    bar.ax.tick_params(labelsize=14, length=3, pad=3)
    bar.ax.get_xticklabels()[0].set_ha("left")
    bar.ax.get_xticklabels()[-1].set_ha("right")
    bar.set_label("Traversal time $t$", fontsize=16, labelpad=4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        args.output,
        dpi=400,
        metadata={
            "Title": f"{len(lengths)} independent source cubes transported into a {args.part} partition",
            "Subject": "Validated order-4 piece centroids; affine source display placement; no joins or filtering",
        },
    )
    plt.close(fig)
    row = args.output.with_name(args.output.stem + "_row.pdf")
    row.write_bytes(args.output.read_bytes())
    combine_rows(args.whole, row, args.output, args.part, len(lengths))
    row.unlink()
    print(
        f"Saved combined figure {args.output}; all {segments} internal segments retained; no joins.",
        flush=True,
    )
