"""Build the checksum-pinned Geogram adapter using MSVC, CMake, and Ninja."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import time
import urllib.request
import zipfile
from pathlib import Path

from mapchain.store import NATIVE

LOCAL = NATIVE / "semidiscrete_geogram_build_v1.10.1"


SOURCE = NATIVE / "semidiscrete_geogram_src_v1.10.1" / "geogram_1.10.1"


URL = "https://github.com/BrunoLevy/geogram/releases/download/v1.10.1/geogram_1.10.1.zip"


SHA256 = "49c7bef2a8da9b44eb1e621ade96f2bb9e743e211cbc13b97b9b403b4cb7b6fe"


def run(args, env, report):
    start = time.perf_counter()
    result = subprocess.run(args, env=env, cwd=LOCAL, capture_output=True, text=True)
    report["commands"].append(
        dict(
            argv=args,
            runtime_seconds=time.perf_counter() - start,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    )
    print(result.stdout[-5000:], result.stderr[-3000:], flush=True)
    if result.returncode:
        raise RuntimeError(f"Build command failed: {args}")
    return result


def main(argv=None):
    """Build the checksum-pinned Geogram adapter using MSVC, CMake, and Ninja."""
    global LOCAL
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "--regularized",
        action="store_true",
        help="Tiny built-in weight penalty, 1e-6/(4*3)",
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Geogram built-in SuperLU linear solver (requires --regularized)",
    )
    parser.add_argument(
        "--warmstart",
        action="store_true",
        help="Existing Geogram HLBFGS initialization before its Newton solve",
    )
    parser.add_argument(
        "--multilevel",
        action="store_true",
        help="Geogram multilevel Newton with its internal 20-neighbour weight fit",
    )
    parser.add_argument(
        "--lbfgs-handoff",
        type=float,
        help="HLBFGS stop factor before Newton (requires --warmstart; adapter default 0.05)",
    )
    args = parser.parse_args(argv)
    if args.lbfgs_handoff is not None and not args.warmstart:
        raise SystemExit("--lbfgs-handoff requires --warmstart")
    if args.regularized:
        LOCAL = NATIVE / "semidiscrete_geogram_build_v1.10.1_regularized"
    if args.direct:
        if not args.regularized:
            raise SystemExit("--direct requires --regularized to remove the singular weight gauge")
        LOCAL = NATIVE / "semidiscrete_geogram_build_v1.10.1_regularized_superlu"
    if args.warmstart:
        LOCAL = LOCAL.with_name(LOCAL.name + "_warmstart")
        if args.lbfgs_handoff is not None:
            LOCAL = LOCAL.with_name(LOCAL.name + f"_handoff{args.lbfgs_handoff:g}")
    if args.multilevel:
        if args.warmstart:
            raise SystemExit("Choose --warmstart or --multilevel")
        LOCAL = LOCAL.with_name(LOCAL.name + "_multilevel")
    vcvars_candidates = list(
        Path("C:/Program Files/Microsoft Visual Studio").glob(
            "*/Community/VC/Auxiliary/Build/vcvars64.bat"
        )
    )
    for edition in ("BuildTools", "Professional", "Enterprise"):
        vcvars_candidates += list(
            Path("C:/Program Files/Microsoft Visual Studio").glob(
                f"*/{edition}/VC/Auxiliary/Build/vcvars64.bat"
            )
        )
    if not vcvars_candidates:
        raise SystemExit("STOP: no installed MSVC toolchain. Nothing installed.")
    vcvars = sorted(vcvars_candidates)[-1]
    setup = subprocess.run(
        f'cmd.exe /d /c call "{vcvars}" >nul && set', capture_output=True, text=True
    )
    if setup.returncode:
        raise SystemExit("STOP: existing MSVC/SDK environment failed: " + setup.stderr)
    env = os.environ.copy()
    env.update(
        line.split("=", 1)
        for line in setup.stdout.splitlines()
        if "=" in line and not line.startswith("=")
    )
    compiler = shutil.which("cl.exe", path=env.get("Path", env.get("PATH")))
    cmake = shutil.which("cmake.exe")
    ninja = shutil.which("ninja.exe")
    if not all((compiler, cmake, ninja)):
        raise SystemExit(f"STOP: missing build tool: cl={compiler}, cmake={cmake}, ninja={ninja}")
    LOCAL.mkdir(parents=True, exist_ok=True)
    report = dict(
        geogram_version="1.10.1",
        source_url=URL,
        archive_sha256=SHA256,
        compiler=compiler,
        vcvars=str(vcvars),
        commands=[],
        licence=dict(
            spdx="BSD-3-Clause",
            url="https://github.com/BrunoLevy/geogram/blob/v1.10.1/LICENSE",
            exploragram="BSD-3-Clause (bundled release)",
        ),
        adapter_policy="Calls upstream solver and restricted power diagram; no OT/Newton implementation",
    )
    report["weight_regularization"] = 1e-6 / 12 if args.regularized else 0.0
    report["linear_solver"] = "Geogram OT_SUPERLU" if args.direct else "Geogram OT_PRECG"
    report["geogram_LBFGS_initialization"] = args.warmstart
    if args.warmstart:
        report["lbfgs_handoff_factor"] = 0.05 if args.lbfgs_handoff is None else args.lbfgs_handoff
    report["geogram_multilevel_initialization"] = args.multilevel
    if args.multilevel:
        report["initialization_settings"] = dict(
            BRIO_seed=0,
            BRIO_threshold=64,
            BRIO_ratio=0.125,
            weight_fit_degree=2,
            weight_fit_neighbours=20,
            note="Internal permutation of the fixed Hilbert sites only; no target samples. Output indices restored to Hilbert order.",
        )
    report["adapter_source_sha256"] = hashlib.sha256(
        (Path(__file__).with_name("adapter.cpp")).read_bytes()
    ).hexdigest()
    report["build_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if args.regularized:
        report["regularization_derivation"] = (
            "eta = mass tolerance/(4*max cost) = 1e-6/(4*3); unit-cube squared cost <=3. Actual per-cell masses are independently checked."
        )
    start = time.perf_counter()
    try:
        raw = urllib.request.urlopen(URL, timeout=60).read()
        if hashlib.sha256(raw).hexdigest() != SHA256:
            raise RuntimeError("Upstream release checksum mismatch")
        archive = zipfile.ZipFile(io.BytesIO(raw))
        source_hashes = {}
        for entry in archive.infolist():
            if entry.is_dir():
                continue
            relative = Path(entry.filename).relative_to("geogram_1.10.1")
            payload = archive.read(entry)
            path = SOURCE / relative
            digest = hashlib.sha256(payload).hexdigest()
            source_hashes[str(relative)] = digest
            if path.exists():
                if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    raise RuntimeError(f"Existing upstream file differs: {path}")
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as handle:
                    handle.write(payload)
        report["download_verify_seconds"] = time.perf_counter() - start
        report["licence"]["text"] = (SOURCE / "LICENSE").read_text()
        compiler_result = subprocess.run([compiler], env=env, capture_output=True, text=True)
        report["compiler_version_banner"] = compiler_result.stderr
    except Exception as error:
        report["failure"] = repr(error)
    if "failure" not in report:
        try:
            parent = LOCAL / "project"
            parent.mkdir(exist_ok=True)
            cmake_text = f'''cmake_minimum_required(VERSION 3.20)
project(SemidiscreteAdapter LANGUAGES C CXX)
add_subdirectory("{SOURCE.as_posix()}" upstream)
add_executable(semidiscrete_adapter "{(Path(__file__).with_name("adapter.cpp")).as_posix()}")
target_include_directories(semidiscrete_adapter PRIVATE "{(SOURCE / "src/lib").as_posix()}" "${{CMAKE_BINARY_DIR}}/upstream/src/lib")
target_compile_features(semidiscrete_adapter PRIVATE cxx_std_17)
target_compile_options(semidiscrete_adapter PRIVATE /O2 /fp:precise /EHsc)
target_compile_definitions(semidiscrete_adapter PRIVATE GEO_DYNAMIC_LIBS NOMINMAX)
target_link_libraries(semidiscrete_adapter PRIVATE exploragram geogram)
set_target_properties(semidiscrete_adapter PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${{CMAKE_BINARY_DIR}}/bin")
'''
            if args.regularized:
                cmake_text += "target_compile_definitions(semidiscrete_adapter PRIVATE SEMIDISCRETE_WEIGHT_REGULARIZATION=8.333333333333333e-8)\n"
            if args.direct:
                cmake_text += "target_compile_definitions(semidiscrete_adapter PRIVATE SEMIDISCRETE_DIRECT_SUPERLU)\n"
            if args.warmstart:
                cmake_text += "target_compile_definitions(semidiscrete_adapter PRIVATE SEMIDISCRETE_GEOGRAM_WARMSTART)\n"
                if args.lbfgs_handoff is not None:
                    cmake_text += f"target_compile_definitions(semidiscrete_adapter PRIVATE SEMIDISCRETE_LBFGS_HANDOFF={args.lbfgs_handoff!r})\n"
            if args.multilevel:
                cmake_text += "target_compile_definitions(semidiscrete_adapter PRIVATE SEMIDISCRETE_GEOGRAM_MULTILEVEL)\n"
            cmake_file = parent / "CMakeLists.txt"
            cmake_file.write_text(cmake_text)
            flags = [
                "-DCMAKE_BUILD_TYPE=Release",
                "-DVORPALINE_BUILD_DYNAMIC=ON",
                "-DGEOGRAM_WITH_GRAPHICS=OFF",
                "-DGEOGRAM_LIB_ONLY=ON",
                "-DGEOGRAM_WITH_EXPLORAGRAM=ON",
                "-DGEOGRAM_WITH_LEGACY_NUMERICS=ON",
                "-DGEOGRAM_WITH_TETGEN=OFF",
                "-DGEOGRAM_WITH_TRIANGLE=OFF",
                "-DGEOGRAM_WITH_LUA=OFF",
                "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
            ]
            report["configure_flags"] = flags
            run(
                [
                    cmake,
                    "-S",
                    str(parent),
                    "-B",
                    str(LOCAL / "build"),
                    "-G",
                    "Ninja",
                    f"-DCMAKE_MAKE_PROGRAM={ninja}",
                    f"-DCMAKE_C_COMPILER={compiler}",
                    f"-DCMAKE_CXX_COMPILER={compiler}",
                    *flags,
                ],
                env,
                report,
            )
            targets = (
                ["semidiscrete_adapter", "geogram_num_3rdparty"]
                if args.direct
                else ["semidiscrete_adapter"]
            )
            run(
                [cmake, "--build", str(LOCAL / "build"), "--target", *targets, "--parallel", "8"],
                env,
                report,
            )
            report["compile_commands"] = json.loads(
                (LOCAL / "build/compile_commands.json").read_text()
            )
            report["upstream_files_unchanged"] = all(
                hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() == digest
                for name, digest in source_hashes.items()
            )
            report["executable"] = str(LOCAL / "build/bin/semidiscrete_adapter.exe")
            report["passed"] = (
                report["upstream_files_unchanged"] and Path(report["executable"]).is_file()
            )
        except Exception as error:
            report["failure"] = repr(error)
            report["passed"] = False
    report["runtime_seconds"] = time.perf_counter() - start
    filename = LOCAL / "build_report.json"
    filename.parent.mkdir(parents=True, exist_ok=True)
    with filename.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"BUILD REPORT: {filename}", flush=True)
    return 0 if report.get("passed") else 1
