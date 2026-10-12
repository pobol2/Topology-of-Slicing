"""Save results to results/ and, with --check, compare with the saved ones instead."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
PAPER = ROOT / "paper"
FIGURES = PAPER / "figures"
TABLES = PAPER / "tables"
NATIVE = ROOT / ".geogram"
CHECK = False
RTOL, ATOL = (1e-08, 1e-10)
IGNORED = (
    "runtime",
    "seconds",
    "stage",
    "memory",
    "peak",
    "rss",
    "versions",
    "executable",
    "platform",
    "stdout",
    "stderr",
    "upstream_report",
    "mass_error",
    "mass_disagreement",
    "gradient_norm",
)
CHANGED = []


def add_check(parser):
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare with the saved results instead of overwriting them",
    )


def configure(args):
    global CHECK
    CHECK = bool(getattr(args, "check", False))


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value))


def differences(old, new, where=""):
    if isinstance(old, dict) and isinstance(new, dict):
        out = []
        for key in sorted(set(old) | set(new)):
            if any(word in key.lower() for word in IGNORED):
                continue
            if key not in old or key not in new:
                out.append(f"{where}/{key}: only in {('new' if key in new else 'saved')}")
            else:
                out += differences(old[key], new[key], f"{where}/{key}")
        return out
    if isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            return [f"{where}: length {len(old)} -> {len(new)}"]
        try:
            a, b = (np.asarray(old, dtype=float), np.asarray(new, dtype=float))
            if a.shape == b.shape:
                bad = ~np.isclose(a, b, rtol=RTOL, atol=ATOL, equal_nan=True)
                return (
                    [
                        f"{where}: {int(bad.sum())} of {bad.size} values differ (max {np.abs(a - b)[bad].max():.3g})"
                    ]
                    if bad.any()
                    else []
                )
        except (TypeError, ValueError):
            pass
        return [
            d for i, (x, y) in enumerate(zip(old, new)) for d in differences(x, y, f"{where}[{i}]")
        ]
    if (
        isinstance(old, (int, float))
        and isinstance(new, (int, float))
        and (not isinstance(old, bool))
    ):
        return (
            []
            if math.isclose(old, new, rel_tol=RTOL, abs_tol=ATOL) or (old != old and new != new)
            else [f"{where}: {old} -> {new}"]
        )
    return [] if old == new else [f"{where}: {str(old)[:60]} -> {str(new)[:60]}"]


def _check(path, new):
    if not path.exists():
        CHANGED.append(path)
        print(f"CHECK {path.relative_to(ROOT).as_posix()}: no saved result", flush=True)
        return
    diffs = differences(json.loads(path.read_text()), new)
    if diffs:
        CHANGED.append(path)
    print(
        f"CHECK {path.relative_to(ROOT).as_posix()}: "
        + ("same" if not diffs else f"{len(diffs)} differences"),
        flush=True,
    )
    for d in diffs[:20]:
        print("   ", d, flush=True)


def save_json(path, data):
    path = Path(path)
    if CHECK:
        return _check(path, json.loads(json.dumps(data, default=serial)))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, default=serial, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def save_text(path, text):
    """Derived text (CSV, LaTeX); --check compares it with the saved file."""
    path = Path(path)
    if CHECK:
        same = path.exists() and path.read_text(encoding="utf-8") == text
        if not same:
            CHANGED.append(path)
        state = "same" if same else "differs" if path.exists() else "no saved result"
        print(f"CHECK {path.relative_to(ROOT).as_posix()}: {state}", flush=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_npz(path, **arrays):
    path = Path(path)
    if CHECK:
        if not path.exists():
            return _check(path, None)
        return _report_arrays(path, dict(np.load(path)), arrays)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _report_arrays(path, saved, arrays):
    diffs = differences(
        {k: v.tolist() for k, v in saved.items()},
        {k: np.asarray(v).tolist() for k, v in arrays.items()},
    )
    if diffs:
        CHANGED.append(path)
    print(
        f"CHECK {path.relative_to(ROOT).as_posix()}: "
        + ("same" if not diffs else f"{len(diffs)} differences"),
        flush=True,
    )
    for d in diffs[:20]:
        print("   ", d, flush=True)


def finish():
    """In --check mode: summary, and exit status 1 if anything changed."""
    if not CHECK:
        return
    print(
        "CHECK: all results unchanged" if not CHANGED else f"CHECK: {len(CHANGED)} results changed",
        flush=True,
    )
    sys.exit(1 if CHANGED else 0)
