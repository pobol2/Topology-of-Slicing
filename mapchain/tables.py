"""The paper's data tables, written to paper/tables/ from the saved results."""

from __future__ import annotations

import argparse
import json
import math

from mapchain import store
from mapchain.evaluation import pieces
from mapchain.evaluation.raster import OUT as RASTER
from mapchain.evaluation.whole_part import OUT as WHOLE_PART
from mapchain.kinematics.sheets import OUT as KINEMATICS

PARTS = (
    ("cylinder", "Cylinder"),
    ("sphere", "Sphere"),
    ("torus", "Torus"),
    ("l_box", "L-bracket"),
    ("complex", "Complex part"),
)
ORDERS = (4, 5, 6)
DENSITIES = (("uniform", "Uniform"), ("gradient", "Gradient"))
KINEMATIC_ORDERS = (2, 3, 4)


def load(path):
    return json.loads(path.read_text())


def count(n):
    """Integer as in the paper: thin-space thousands separator from five digits on."""
    return f"{n}" if n < 10000 else "$" + f"{n:,}".replace(",", r"\,") + "$"


def percent(x):
    return f"{100 * x:.2f}" + r"\%"


def length(x):
    return f"${x:.4f}$"


def bound(x):
    """Upper bounds are rounded up, so the printed value is still a bound."""
    return "$0$" if x == 0 else f"${math.ceil(x * 1e4) / 1e4:.4f}$"


def small(x):
    """Three decimals; exact zeros as 0 and positive values below the last digit as <0.001."""
    return "$0$" if x == 0 else "$<0.001$" if x < 0.0005 else f"${x:.3f}$"


def latex_row(cells):
    return "    " + " & ".join(cells) + r" \\"


def fill(template, **values):
    for key, value in values.items():
        template = template.replace(key, value)
    return template


MAIN_STOPS = r"""% Written by python -m mapchain tables from results/evaluation/raster and results/evaluation/whole_part.
\begin{papertable}
  \begin{papertabular}{lrrr}
    \toprule
    & \multicolumn{2}{c}{Layered raster} & Transported path \\
    \cmidrule(lr){2-3}\cmidrule(lr){4-4}
    Part & Between layers & All travel & Exterior runs \\
    \midrule
ROWS
    \bottomrule
  \end{papertabular}
  \caption{Stops at Hilbert orders 4 / 5 / 6. Most raster travel lies within layers, where a better in-layer planner could remove some of it; only the stops between layers are forced.}
  \label{tab:main-stops}
\end{papertable}
"""


def main_stops():
    """Section 4: stops of the layered raster against exterior runs of the transported path."""
    rows = []
    for part, label in PARTS:
        raster = [load(RASTER / f"{part}_m{m}.json") for m in ORDERS]
        whole = [load(WHOLE_PART / f"{part}_m{m}.json") for m in ORDERS]
        rows.append(
            latex_row(
                [
                    label,
                    " / ".join(str(r["between_layer_stops"]) for r in raster),
                    " / ".join(str(r["total_interruptions"]) for r in raster),
                    " / ".join(str(w["metrics"]["exterior_runs"]["count"]) for w in whole),
                ]
            )
        )
    return fill(MAIN_STOPS, ROWS="\n".join(rows))


EXPERIMENTS_GEOMETRIES = r"""% Written by python -m mapchain tables from results/evaluation/whole_part.
\begin{papertable}
  \begin{papertabular}{lrrrrrrrrrr}
    \toprule
    Part & $m$ & Inside & \papertablehead{Ext.\\seg.} & \papertablehead{Ext.\\length} & Depth & \papertablehead{Ext.\\runs} & Jumps & $D_m$ & \papertablehead{Cover.\\radius} & \papertablehead{Max.\\step} \\
    \midrule
ROWS
    \bottomrule
  \end{papertabular}
  \caption{All fifteen uniform runs, at $8^m$ waypoints (BUDGETS). ``Inside'' is the fraction of waypoints inside $\Omega$; ``Ext.\ seg.'' and ``Ext.\ length'' are the fractions of exterior segments and exterior path length; ``Depth'' is the deepest sampled excursion. Distances ($D_m$, depth, covering radius, and maximum step) are in normalised units (Appendix~\ref{sec:implementation}); exterior runs and jumps are counts.}
  \label{tab:experiments-geometries}
\end{papertable}
"""


def experiments_geometries():
    """Appendix C: every whole-part uniform run."""
    rows = []
    for part, label in PARTS:
        for i, m in enumerate(ORDERS):
            run = load(WHOLE_PART / f"{part}_m{m}.json")
            metrics, segments = (run["metrics"], run["metrics"]["segment_metrics"])
            rows.append(
                latex_row(
                    [
                        label if i == 0 else "",
                        str(m),
                        f"${metrics['table4']['inside_fraction']:.3f}$",
                        small(segments["exterior_segment_fraction"]),
                        small(segments["exterior_arc_length_fraction"]),
                        small(segments["maximum_excursion_depth"]),
                        f"${metrics['exterior_runs']['count']}$",
                        f"${metrics['detected_jumps']}$",
                        f"${run['cell_metrics']['max_diameter']:.3f}$",
                        f"${segments['covering_radius_target_eval']:.3f}$",
                        f"${metrics['table4']['max_step']:.3f}$",
                    ]
                )
            )
    budgets = [count(8**m) for m in ORDERS]
    return fill(
        EXPERIMENTS_GEOMETRIES,
        ROWS="\n".join(rows).replace("    " + " & ", "    & "),
        BUDGETS=", ".join(budgets[:-1]) + ", and " + budgets[-1],
    )


SHEETS_3R_RESULTS = r"""% Written by python -m mapchain tables from results/kinematics/results.json and benchmark.json.
\begin{papertable}
  \begin{papertabular}{llrrrrrrrr}
    \toprule
    & & \multicolumn{2}{c}{Resets} & \multicolumn{2}{c}{Coverage error} & \multicolumn{2}{c}{Density error} & \multicolumn{2}{c}{Planning (s)} \\
    \cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}
    Density & $m$ & G & K & G & K & G & K & G & K \\
    \midrule
ROWS
    \bottomrule
  \end{papertabular}
  \caption{Three-link comparison of geometry-first (G) and kinematics-first (K) at $4^m$ waypoints. Resets are observed discrete posture changes. Coverage error is in target units; density error is total variation of original-waypoint shares on $4\times8$ area bins. Planning time is the median time to plan and certify one request. Kinematics-first also has a one-time preparation of $PREPARATION$~s; the shared fibre analysis ($FIBRE$~s) is excluded for both.}
  \label{tab:sheets-3r-results}
\end{papertable}
"""


def sheets_3r_results():
    """Section 5: geometry-first against kinematics-first on the planar 3R arm."""
    result, bench = (load(KINEMATICS / "results.json"), load(KINEMATICS / "benchmark.json"))
    plans = {(q["method"], q["order"], q["density"]): q for q in result["requests"]}
    blocks = []
    for density, label in DENSITIES:
        rows = []
        for i, m in enumerate(KINEMATIC_ORDERS):
            g, k = (plans["geometry", m, density], plans["kinematics", m, density])
            timed = {
                method: bench["requests"][f"{method}_m{m}_{density}"]
                for method in ("geometry", "kinematics")
            }
            if any(timed[q["method"]]["switches"] != q["switches"] for q in (g, k)):
                raise SystemExit(f"STOP: benchmark and results disagree on resets at m={m}")
            rows.append(
                latex_row(
                    [
                        label if i == 0 else "",
                        str(m),
                        str(g["switches"]),
                        str(k["switches"]),
                        f"{g['coverage_error']:.3f}",
                        f"{k['coverage_error']:.3f}",
                        f"{g['density_error']:.3f}",
                        f"{k['density_error']:.3f}",
                        f"{timed['geometry']['plan_seconds']['median']:.2f}",
                        f"{timed['kinematics']['plan_seconds']['median']:.3f}",
                    ]
                ).replace("    " + " & ", "    & ")
            )
        blocks.append("\n".join(rows))
    return fill(
        SHEETS_3R_RESULTS,
        ROWS="\n    \\addlinespace\n".join(blocks),
        PREPARATION=f"{bench['prep']['kinematics_preparation']['median']:.1f}",
        FIBRE=f"{bench['prep']['shared_fibre_analysis']['median']:.1f}",
    )


DECOMPOSITION_MAIN = r"""% Written by python -m mapchain tables from results/evaluation/pieces/summary.json.
\begin{papertable}
  \begin{papertabular}{lrrrrrr}
    \toprule
    Object & Pieces & Waypoints & $\max_j U_j$ & In piece & \papertablehead{Exterior\\segments} & \papertablehead{Sampled\\depth} \\
    \midrule
ROWS
    \bottomrule
  \end{papertabular}
  \caption{Independent piece paths, uniform density, 4096 waypoints per piece. ``In piece'' is the percentage of waypoints inside their assigned piece; exterior segments are measured against that same piece. Depth is the largest sampled excursion; $U_j$ is the numerical hull-gap upper bound. Distances are in normalised units (Appendix~\ref{sec:implementation}).}
  \label{tab:decomposition}
\end{papertable}
"""


def decomposition_main(summary):
    """Appendix C: the L-bracket and torus rows of the decomposition summary."""
    rows = [
        latex_row(
            [
                label,
                str(s["pieces"]),
                count(s["total_waypoints"]),
                bound(s["max_upper_bound"]),
                percent(s["waypoint_containment"]),
                percent(s["exterior_segment_fraction"]),
                length(s["max_sampled_excursion"]),
            ]
        )
        for obj, label in (("l_box", "L-bracket"), ("torus", "Torus"))
        for s in [summary[obj]]
    ]
    return fill(DECOMPOSITION_MAIN, ROWS="\n".join(rows))


TORUS_WHOLE_VS_PIECES = r"""% Written by python -m mapchain tables from results/evaluation/pieces/torus_whole_vs_pieces.json.
\begin{papertable}
  \begin{papertabular}{lrrrrrrr}
    \toprule
    Path / domain & Inside & Ext.\ seg. & Ext.\ length & Depth & Coverage & CV & Radius \\
    \midrule
ROWS
    \bottomrule
  \end{papertabular}
  \caption{Torus at the same budget: one whole-part path and eight piece paths, BUDGET waypoints each. \emph{Torus} measures against the whole object, \emph{assigned} against each path's own piece. ``Inside'' is the percentage of waypoints inside the evaluation domain; ``Ext.\ seg.'' and ``Ext.\ length'' are the percentages of exterior segments and exterior path length. Depth is the deepest sampled excursion; radius is the sampled covering radius. Distances are in normalised units; coverage and the coefficient of variation (CV) are defined in Appendix~\ref{sec:experiments-setup}. See Appendix~\ref{sec:experiments-torus-comparison} for the comparison protocol.}
  \label{tab:torus-whole-vs-pieces}
\end{papertable}
"""


def torus_whole_vs_pieces():
    """Section 4: whole torus path against piece paths at the same budget."""
    comparison = load(pieces.RESULT)["comparison"]
    whole, parts = (comparison["whole_m5"], comparison["pieces_m4"])

    def row(label, m, cover=None):
        return latex_row(
            [
                label,
                percent(m["containment"]),
                percent(m["exterior_segment_fraction"]),
                percent(m["exterior_length_fraction"]),
                length(m["max_excursion"]),
                *(
                    ["--"] * 3
                    if cover is None
                    else [
                        f"${cover['coverage']:.3f}$",
                        f"${cover['coverage_cv']:.3f}$",
                        length(cover["covering_radius"]),
                    ]
                ),
            ]
        )

    rows = [
        row("Whole / torus", whole["polygonal_torus"], whole["coverage"]),
        row("Pieces / torus", parts["polygonal_torus"], parts["coverage"]),
        row("Pieces / assigned", parts["assigned_piece"]),
    ]
    return fill(
        TORUS_WHOLE_VS_PIECES,
        ROWS="\n".join(rows),
        BUDGET=count(whole["polygonal_torus"]["waypoints"]),
    )


def tables_main(argv=None):
    """Write the paper's data tables to paper/tables/ from the saved results."""
    parser = argparse.ArgumentParser(description=tables_main.__doc__)
    store.add_check(parser)
    store.configure(parser.parse_args(argv))
    summary = pieces.summarise()
    for name, text in (
        ("main_stops", main_stops()),
        ("torus_whole_vs_pieces", torus_whole_vs_pieces()),
        ("sheets_3r_results", sheets_3r_results()),
        ("experiments_geometries", experiments_geometries()),
        ("decomposition_main", decomposition_main(summary)),
    ):
        store.save_text(store.TABLES / f"{name}.tex", text)
    store.finish()
