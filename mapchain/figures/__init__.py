"""The five figures used by the paper, written to paper/figures/ at the paper's text width."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pymupdf as fitz

from mapchain import store
from mapchain.figures import (
    bunny_transport_decomposition,
    cube_layer_interruptions,
    hilbert_transport,
    nonconvex_containment_decomposition,
    three_link_posture_comparison,
)
from mapchain.figures.layout import layout

NAMES = (
    "bunny_transport_decomposition",
    "cube_layer_interruptions",
    "nonconvex_containment_decomposition",
    "three_link_posture_comparison",
    "hilbert_transport",
)


def build(stage, names):
    """Draw the selected figures into the stage directory."""
    for name, draw in (
        ("cube_layer_interruptions", cube_layer_interruptions.draw),
        ("nonconvex_containment_decomposition", nonconvex_containment_decomposition.draw),
        ("hilbert_transport", hilbert_transport.draw),
    ):
        if name in names:
            fig = draw()
            fig.savefig(stage / (name + ".pdf"))
            plt.close(fig)
    if "three_link_posture_comparison" in names:
        with plt.rc_context():
            three_link_posture_comparison.draw(stage / "three_link_posture_comparison.pdf")
    if "bunny_transport_decomposition" in names:
        whole = stage / "bunny_transport_whole.pdf"
        bunny_transport_decomposition.whole_main(
            ["--part", "bunny", "--show-exterior", "--output", str(whole)]
        )
        bunny_transport_decomposition.pieces_main(
            [
                "--part",
                "bunny",
                "--whole",
                str(whole),
                "--output",
                str(stage / "bunny_transport_decomposition.pdf"),
            ]
        )


def render(path):
    """First page as an RGB array, for pixel comparison."""
    with fitz.open(path) as pdf:
        pixmap = pdf[0].get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    return np.frombuffer(pixmap.samples, np.uint8).reshape(pixmap.h, pixmap.w, 3)


def figures_main(argv=None):
    """Write the paper figures to paper/figures/, or with --check compare them pixel by pixel."""
    parser = argparse.ArgumentParser(description=figures_main.__doc__)
    parser.add_argument("--only", nargs="+", choices=NAMES)
    store.add_check(parser)
    args = parser.parse_args(argv)
    names = args.only or NAMES
    with tempfile.TemporaryDirectory(prefix="mapchain-figures-") as directory:
        stage = Path(directory)
        with layout():
            build(stage, names)
        changed = 0
        for name in sorted(names):
            new, old = (stage / (name + ".pdf"), store.FIGURES / (name + ".pdf"))
            assert new.exists(), name
            if args.check:
                same = old.exists() and np.array_equal(render(new), render(old))
                changed += not same
                state = (
                    "same" if same else "no saved figure" if not old.exists() else "pixels differ"
                )
                print(f"CHECK {name}: {state}")
            else:
                store.FIGURES.mkdir(exist_ok=True)
                shutil.copy2(new, old)
        if args.check:
            print(
                "CHECK: all figures unchanged"
                if not changed
                else f"CHECK: {changed} figures changed"
            )
            sys.exit(1 if changed else 0)
