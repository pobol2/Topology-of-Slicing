"""Command line: python -m mapchain COMMAND [--help]."""

import argparse
import importlib
import sys

COMMANDS = {
    "build": ("mapchain.volumetric.native.build", "main"),
    "validate": ("mapchain.evaluation.whole_part", "validate_main"),
    "uniform": ("mapchain.evaluation.whole_part", "uniform_main"),
    "raster": ("mapchain.evaluation.raster", "raster_main"),
    "decompose": ("mapchain.volumetric.decomposition", "decompose_main"),
    "compare": ("mapchain.evaluation.pieces", "compare_main"),
    "tables": ("mapchain.tables", "tables_main"),
    "kinematics": ("mapchain.kinematics.sheets", "kinematics_main"),
    "benchmark": ("mapchain.kinematics.benchmark", "benchmark_main"),
    "figures": ("mapchain.figures", "figures_main"),
}


def main():
    """Dispatch to a command; modules are imported only when their command runs."""
    parser = argparse.ArgumentParser(prog="python -m mapchain", description=__doc__)
    parser.add_argument("command", nargs="?", choices=COMMANDS)
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return
    module, function = COMMANDS[args.command]
    sys.argv[0] = f"python -m mapchain {args.command}"
    result = getattr(importlib.import_module(module), function)(args.args)
    if isinstance(result, int):
        raise SystemExit(result)


if __name__ == "__main__":
    main()
