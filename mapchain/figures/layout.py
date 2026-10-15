"""Paper layout: text width, LaTeX fonts and padding for every saved PDF."""

from __future__ import annotations

import contextlib
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pymupdf as fitz
from matplotlib.figure import Figure
from matplotlib.text import Text
from matplotlib.transforms import Bbox

from mapchain.figures.style import set_morph_camera

WIDTH = 452.9679 * 72 / 72.27  # \textwidth in PDF points
FONT = 10.95  # body font size in TeX points
PAD = 3.0
NATIVE_SAVE = Figure.savefig
RC = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": FONT,
    "axes.titlesize": FONT,
    "axes.labelsize": FONT,
    "xtick.labelsize": FONT,
    "ytick.labelsize": FONT,
    "legend.fontsize": FONT,
    "text.latex.preamble": "\\usepackage[T1]{fontenc}\\usepackage{lmodern}\\usepackage{bm}\\usepackage{upgreek}",
}


def latex_text(text):
    return re.sub("(?<!\\\\)%", "\\\\%", text)


def prepare(fig, name):
    """Scale the figure to the text width, set every text in the body font and return the crop box."""
    original_width, original_height = fig.get_size_inches()
    fig.set_size_inches(WIDTH / 72, original_height * (WIDTH / 72) / original_width)
    for ax in fig.axes:
        ax.tick_params(axis="both", pad=3)
        if hasattr(ax, "zaxis"):
            if not getattr(ax, "keep_view", False):
                set_morph_camera(ax)
            ax.set(xlabel="", ylabel="", zlabel="")
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.set_ticklabels([])
            ax.tick_params(axis="z", pad=3)
            ax.xaxis.labelpad = 2
            ax.yaxis.labelpad = 2
            ax.zaxis.labelpad = 2
    for text in fig.findobj(match=Text):
        text.set_fontsize(FONT)
        text.set_fontfamily("serif")
        text.set_usetex(True)
        text.set_text(latex_text(text.get_text()))
    if name in ("bunny_transport_whole", "bunny_transport_decomposition"):
        fig.set_size_inches(WIDTH / 72, 2.15)
        for ax in fig.axes[:-1]:
            pos = ax.get_position()
            ax.set_position([pos.x0, 0.35, pos.width, 0.5])
        # Colour bar from the first to the last column centre: each tick under its image.
        fig.axes[-1].set_position([0.1, 0.18, 0.8, 0.055])
    fig.canvas.draw()
    for text in fig.findobj(match=Text):
        text.set_fontsize(FONT)
        text.set_usetex(True)
    content_width = (WIDTH - 2 * PAD) / 72
    for _ in range(12):
        bbox = fig.get_tightbbox(fig.canvas.get_renderer())
        if abs(bbox.width - content_width) < 0.0005:
            break
        fig.set_size_inches(fig.get_figwidth() * content_width / bbox.width, fig.get_figheight())
    fig.canvas.draw()
    bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    centre = (bbox.x0 + bbox.x1) / 2
    return Bbox.from_extents(
        centre - WIDTH / 144, bbox.y0 - PAD / 72, centre + WIDTH / 144, bbox.y1 + PAD / 72
    )


def savefig(fig, filename, *args, **kwargs):
    """Figure.savefig inside layout(): PDFs get the paper layout; other formats are skipped."""
    path = Path(filename)
    if path.suffix != ".pdf":
        return
    kwargs.update(bbox_inches=prepare(fig, path.stem), pad_inches=0, dpi=300)
    NATIVE_SAVE(fig, path, *args, **kwargs)
    with fitz.open(path) as pdf:
        assert abs(pdf[0].rect.width - WIDTH) < 0.001
        fonts = [font[3] for font in pdf[0].get_fonts()]
    assert all(font.startswith(("LM", "CM", "EU")) for font in fonts), (path.stem, fonts)
    print("LATEX", path.stem, flush=True)


@contextlib.contextmanager
def layout():
    """Inside the block every Figure.savefig(*.pdf) gets the paper width, body font and padding."""
    old = dict(plt.rcParams)
    plt.rcParams.update(RC)
    Figure.savefig = savefig
    try:
        yield
    finally:
        Figure.savefig = NATIVE_SAVE
        plt.rcParams.update(old)
