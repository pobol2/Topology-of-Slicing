# Saved results

These are the numerical results used by the [paper](../paper/main.pdf), figures
and tables. Run commands below from the repository root after following the
[setup instructions](../README.md#setup). Commands overwrite their own outputs;
higher Hilbert orders can take substantial time and memory.

## Formats

JSON files contain readable run settings, solver reports, acceptance checks,
measurement protocols, metrics and timings. NPZ files are compressed NumPy arrays
containing the geometry and paths needed to reproduce figures and comparisons.
Not every run retains its waypoints: a JSON file without a matching NPZ is expected.

| Files | Contents | Generator (`python -m mapchain ...`) |
| --- | --- | --- |
| [`volumetric/validation_nested.json`](volumetric/validation_nested.json) | Cube and torus solver validation reports | `validate` |
| [`evaluation/whole_part/`](evaluation/whole_part/) | `<part>_m<order>.json`: transport settings, solver acceptance and path metrics; matching NPZ files store an `(8**m, 3)` `waypoints` array | `uniform --orders 4 5 6`; also `uniform --parts box --orders 2` and `uniform --parts bunny --orders 5` |
| [`evaluation/raster/`](evaluation/raster/) | Per-part, per-order JSON metrics for the layered raster baseline | `raster` |
| [`volumetric/decomposition/`](volumetric/decomposition/) | Per-object `record.json`: partition checks, piece certificates, solves and measurements; `geometry.npz`: mesh vertices/faces, tetrahedral vertices and connectivity, `piece_of` assignments, piece count and piece surfaces (`v0`, `f0`, ...); `paths.npz`: piece count, waypoint arrays (`p0`, ...) and exterior-segment indices (`e0`, ...) | See decomposition commands below |
| [`evaluation/pieces/summary.json`](evaluation/pieces/summary.json) | Summary of the L-bracket, torus and bunny decomposition records | `tables` |
| [`evaluation/pieces/torus_whole_vs_pieces.json`](evaluation/pieces/torus_whole_vs_pieces.json) | Whole-part versus decomposed torus measurements at a matched waypoint budget | `compare` |
| [`kinematics/`](kinematics/) | `results.json`: both planners' metrics, switch events and preprocessing; `fibre.json`: fibre analysis; `fibre.npz`: angle/radius samples, sheet labels and selections; `availability_grid.npz`: sampled sheet availability; `patch_0.npz` and `patch_1.npz`: joint configurations `q` and angular ranges; `<method>_m<order>_<density>.npz`: joint paths, requested/collocation coordinates, sheet families, break edges, availability and evaluated Cartesian positions | `kinematics`; `kinematics --fibre-only` regenerates only the fibre and availability files |
| [`kinematics/benchmark.json`](kinematics/benchmark.json) | Repeated preprocessing, planning and evaluation timings, checked against saved plans | `benchmark --prep`, then `benchmark --method geometry` and `benchmark --method kinematics` |

Regenerate the decomposition folders with:

```powershell
python -m mapchain decompose --object l_box
python -m mapchain decompose --object torus --threshold 0.05
python -m mapchain decompose --object torus --threshold 0.05 --m 3 --out torus_m3
python -m mapchain decompose --object bunny --threshold 0.1
```

The matched-budget torus comparison needs the saved whole-part paths and the
`torus_m3` decomposition. Regenerate those inputs before running `compare`.
`tables` reads saved results and writes `paper/tables/*.tex`; `figures` reads them
and writes `paper/figures/*.pdf`.

## Inspecting and checking

For example, inspect a saved path without running the solver:

```python
import numpy as np

with np.load("results/evaluation/whole_part/bunny_m5.npz", allow_pickle=False) as data:
    print(data.files)
    print(data["waypoints"].shape)
```

Append `--check` to result commands that support it to recompute and compare
without overwriting files (for example, `python -m mapchain uniform --parts box
--orders 2 --check`). Comparisons use numerical tolerances and ignore volatile
timing, environment and selected solver diagnostics; see
[`mapchain/store.py`](../mapchain/store.py). Benchmark commands write timings
and do not offer `--check`.

Saved 3R consistency and generated tables are checked by
`python -m unittest discover -s tests -t .`.
