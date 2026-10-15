#!/usr/bin/env python3
"""P0024 throwaway router production mutations; completed numeric mismatches alone count.

Temporary trees copy the production public/private headers and sources plus the
P0024 executor. The worktree is guarded by hashes and is never edited.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DRIVER = ROOT / "tests/core/test_router_parity.py"
EXECUTOR = ROOT / "tests/core/core_exec.cpp"

MUTATIONS = (
    (
        "nonlinear-stage",
        "text",
        (("softplus[row] = detail::softplus_fp32(temp[row], certification_bits);",
          "softplus[row] = detail::sqrt_fp32(detail::softplus_fp32(temp[row], certification_bits));"),),
    ),
    (
        "omit-epsilon",
        "epsilon-dominant",
        (("df::quant::detail::add_fp32(pending_trace.sum_bits, kRoutingEpsilon);",
          "df::quant::detail::add_fp32(pending_trace.sum_bits, kRoutingEpsilon & 0u);"),),
    ),
    (
        "projection-order",
        "projection-order",
        (("for (std::size_t column = 0; column < gate_snapshot.k; ++column) {",
          "for (std::size_t column = gate_snapshot.k; column-- > 0;) {"),),
    ),
    (
        "bias-in-weight",
        "negative-selection",
        (("pending_trace.selected_bits[slot] = score[result.indices[slot]];",
          "pending_trace.selected_bits[slot] = selection[result.indices[slot]];"),),
    ),
    (
        "reverse-tie",
        "tie-9",
        (("return order > 0 || (order == 0 && candidate < incumbent);",
          "return order > 0 || (order == 0 && candidate > incumbent);"),),
    ),
    (
        "wrong-scale",
        "tiny-weight",
        (("constexpr std::uint32_t kRouteScale = 0x3FC00000u;",
          "constexpr std::uint32_t kRouteScale = 0x3F800000u;"),),
    ),
    (
        "ignore-image-bias",
        "image",
        (("image_token && bias_vl_snapshot.has_value() ? *bias_vl_snapshot : bias;",
          "false && image_token && bias_vl_snapshot.has_value() ? *bias_vl_snapshot : bias;"),),
    ),
    (
        "mismatched-weight",
        "matrix-0",
        (("const std::size_t offset = row * gate_snapshot.k;",
          "const std::size_t offset = ((row + 1) % gate_snapshot.n) * gate_snapshot.k;"),),
    ),
)


def checked_run(command, **kwargs):
    proc = subprocess.run(command, capture_output=True, text=True, timeout=300, **kwargs)
    if proc.returncode:
        raise RuntimeError(f"command failed {command!r}\n{proc.stdout}\n{proc.stderr}")
    if "warning:" in proc.stdout or "warning:" in proc.stderr:
        raise RuntimeError("compilation warning is not acceptance")
    return proc


def build(directory, cxx):
    executable = directory / "core_exec"
    checked_run([
        cxx, "-std=c++20", "-O1", "-g", "-Wall", "-Wextra", "-Wpedantic", "-Werror",
        "-ffp-contract=off", "-fno-fast-math", "-I" + str(directory / "include"),
        "-I" + str(directory / "src/core"), "-I" + str(directory / "src/quant"),
        str(directory / "core_exec.cpp"),
        *[str(directory / "src/quant" / name)
          for name in ("codec.cpp", "dequant.cpp", "compute.cpp")],
        str(directory / "src/core/router.cpp"),
        str(directory / "src/core/router_math.cpp"),
        "-o", str(executable),
    ])
    return executable


def report(executable, selector=None):
    command = [sys.executable, "-B", str(DRIVER), "--exe", str(executable),
               "--suite", "parity", "--numeric-report"]
    if selector:
        command += ["--case", selector]
    proc = subprocess.run(command, capture_output=True, text=True, timeout=900,
                          env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    lines = proc.stdout.splitlines()
    if len(lines) != 1 or not lines[0].startswith("P0024_NUMERIC "):
        raise RuntimeError(f"missing, duplicate or malformed numeric report\n{proc.stdout}\n{proc.stderr}")
    payload = json.loads(lines[0][len("P0024_NUMERIC "):])
    if set(payload) != {"cases", "errors", "skipped"} or payload["skipped"] != 0:
        raise RuntimeError("numeric report schema or skip failure")
    if not isinstance(payload["errors"], list) or not isinstance(payload["cases"], list):
        raise RuntimeError("numeric report field types")
    for case in payload["cases"]:
        if (set(case) != {"case", "mismatches"} or not isinstance(case["case"], str)
                or not isinstance(case["mismatches"], list)
                or any(not isinstance(stage, str) for stage in case["mismatches"])):
            raise RuntimeError("malformed numeric case")
    failed = bool(payload["errors"] or any(case["mismatches"] for case in payload["cases"]))
    if proc.returncode != int(failed):
        raise RuntimeError("driver exit disagrees with numeric report")
    if proc.stderr:
        raise RuntimeError(f"numeric driver emitted diagnostics: {proc.stderr[-2000:]}")
    return payload


def require_pristine(payload, label):
    if payload["errors"] or payload["skipped"] or not payload["cases"]:
        raise RuntimeError(f"{label}: pristine numeric selection did not complete")
    if any(case["mismatches"] for case in payload["cases"]):
        raise RuntimeError(f"{label}: pristine production has numeric mismatches")


def require_numeric(payload, selector):
    if payload["errors"] or payload["skipped"]:
        raise RuntimeError("exceptions or protocol failures cannot count as sensitivity")
    expected = {selector + ".ordinary", selector + ".trace"}
    if {case["case"] for case in payload["cases"]} != expected:
        raise RuntimeError(f"{selector}: ordinary and trace detectors did not both complete")
    if any(not case["mismatches"] for case in payload["cases"]):
        raise RuntimeError(f"{selector}: each completed entry must expose a numeric/stage mismatch")
    return sum(len(case["mismatches"]) for case in payload["cases"])


def copy_sources(destination):
    (destination / "src").mkdir(parents=True)
    shutil.copytree(ROOT / "include", destination / "include")
    shutil.copytree(ROOT / "src/core", destination / "src/core")
    shutil.copytree(ROOT / "src/quant", destination / "src/quant")
    shutil.copy(EXECUTOR, destination / "core_exec.cpp")


def guarded_hashes():
    paths = [EXECUTOR, DRIVER,
             *list((ROOT / "include/df/core").glob("*")),
             *list((ROOT / "src/core").glob("*")),
             *list((ROOT / "src/quant").glob("*")),
             *list((ROOT / "include/df/quant").glob("*")),
             *list((ROOT / "ref").glob("*.py"))]
    return {path: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths if path.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default=os.environ.get("CXX", "c++"))
    args = parser.parse_args()
    hashes = guarded_hashes()
    with tempfile.TemporaryDirectory(prefix="p0024-router-sensitivity-") as temporary:
        base = Path(temporary) / "pristine"
        copy_sources(base)
        pristine = build(base, args.cxx)
        baseline = report(pristine)
        require_pristine(baseline, "full baseline")
        print(f'P0024 pristine: {len(baseline["cases"])} numeric cases passed', flush=True)

        for slug, selector, edits in MUTATIONS:
            pristine_selection = report(pristine, selector)
            require_pristine(pristine_selection, slug + " same-case baseline")
            expected = {selector + ".ordinary", selector + ".trace"}
            if {case["case"] for case in pristine_selection["cases"]} != expected:
                raise RuntimeError(f"{slug}: selector did not name exactly ordinary and trace")

            directory = Path(temporary) / slug
            shutil.copytree(base, directory, ignore=shutil.ignore_patterns("core_exec"))
            source = directory / "src/core/router.cpp"
            original = source.read_text()
            mutated = original
            for anchor, replacement in edits:
                if mutated.count(anchor) != 1:
                    raise RuntimeError(f"{slug}: mutation anchor is not unique")
                mutated = mutated.replace(anchor, replacement)
            source.write_text(mutated)
            payload = report(build(directory, args.cxx), selector)
            count = require_numeric(payload, selector)
            print(f"P0024 mutation {slug}: {selector}; {count} completed numeric/stage "
                  "mismatches; errors=0 skips=0", flush=True)

            source.write_text(original)
            restored = report(build(directory, args.cxx), selector)
            require_pristine(restored, slug + " restoration")

        require_pristine(report(pristine), "final baseline")

    for path, digest in hashes.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError("guarded worktree source changed: " + str(path))
    print(f"P0024 sensitivity: {len(MUTATIONS)}/{len(MUTATIONS)} production variants "
          "detected by completed ordinary+trace numeric comparisons; restorations, final "
          "baseline and worktree hashes passed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
