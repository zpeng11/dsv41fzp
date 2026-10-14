#!/usr/bin/env python3
"""P0018: bounded checkpoint expert rows consumed by existing CPU quantization ops.

This explicit validator checks all headers, but reads only the selected weight and
scale rows. Artificial BF16 inputs validate CPU contracts, not model quality or
GPU equivalence. Exit: 0 pass, 1 validation failure, 2 argument or I/O error.
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "quant"))
from tools import ckpt_inventory as inv
from ref.act_quant import act_quant
from ref.compute import trace_expert_grouped_dot
from test_act_quant_parity import ActCase
from test_codec_parity import FP32, HarnessError, is_nan, run_executor
from test_compute_parity import ComputeCase, trace_words

PREFIX = "layers.0.ffn.experts.0"
INPUT_RECIPE = {
    "mixed": "i=0..K-1; g=i//FP4_BLOCK; sign=(i%3==0); exponent=121+g%11; fraction=(37*i+11)%128",
    "group_ramp": "i=0..K-1; g=i//FP4_BLOCK; sign=((i+g)%2==0); exponent=119+(3*g)%13; fraction=(53*i+7*g+29)%128",
    "encoding": "BF16 uint16=(sign<<15)|(exponent<<7)|fraction; SHA256 of little-endian uint16 sequence",
    "block": inv.FP4_BLOCK,
}


class ConsumptionError(Exception):
    """Stable failure code and contextual diagnostic; rejection is never parity."""

    def __init__(self, code, detail):
        super().__init__(f"{code}: {detail}")
        self.code, self.detail = code, str(detail)


@dataclass(frozen=True)
class Snapshot:
    root: Path
    revision: str
    expected: dict
    tensors: dict


@dataclass(frozen=True)
class Matrix:
    name: str
    n: int
    k: int
    weight: tuple
    scale: tuple


@dataclass(frozen=True)
class Row:
    matrix: str
    index: int
    k: int
    weight: bytes
    scale: bytes
    reads: list


def _count(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _checked(errors):
    if errors:
        error = errors[0]
        raise ConsumptionError(error.code, error.detail)


def _basename(name):
    if (not isinstance(name, str) or not name or name in (".", "..")
            or "/" in name or "\\" in name or "\0" in name or ":" in name):
        raise ConsumptionError("index_path", f"not a shard basename: {name!r}")
    return name


def inspect_checkpoint(path, reference=inv.DEFAULT_REFERENCE):
    """Validate P0016 structure and independent expectations without payload scans."""
    try:
        root = Path(path).absolute()
        if not root.is_dir():
            raise ConsumptionError("argument", f"not a checkpoint directory: {root}")
        _checked(inv.check_configs(root, reference))
        expected = inv.expected_tensors(inv.load_reference_config(reference))
        index_path = root / _basename(inv.INDEX_NAME)
        with index_path.open(encoding="utf-8") as fh:
            index = json.load(fh, object_pairs_hook=inv._reject_duplicates,
                              parse_constant=inv._reject_constant)
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict):
            raise ConsumptionError("index_json", "weight_map must be an object")
        names = {_basename(name) for name in weight_map.values()}
        names.update(_basename(p.name) for p in root.iterdir() if p.name.endswith(".safetensors"))
        shards = {name: inv.read_header(root / name) for name in sorted(names)}
        _checked(inv.check_index(index_path, shards))
        tensors = {entry.name: (shard, entry) for shard in shards.values() for entry in shard.entries}
        _checked(inv.compare(expected, tensors)[0])
        return Snapshot(root, root.resolve().name, expected, tensors)
    except inv.CheckError as exc:
        raise ConsumptionError(exc.code, exc.detail) from None
    except inv.UsageError as exc:
        raise ConsumptionError("argument", str(exc)) from None
    except OSError as exc:
        raise ConsumptionError("io", str(exc)) from None
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise ConsumptionError("structure", str(exc)) from None


def select_matrix(snapshot, matrix):
    """Select only the fixed expert, validating dimensions against independent Expected."""
    if matrix not in ("w1", "w2", "w3"):
        raise ConsumptionError("matrix", f"unsupported matrix {matrix!r}")
    name = f"{PREFIX}.{matrix}"
    try:
        wn, sn = name + ".weight", name + ".scale"
        ew, es = snapshot.expected[wn], snapshot.expected[sn]
        weight, scale = snapshot.tensors[wn], snapshot.tensors[sn]
        if (ew.name != wn or es.name != sn or ew.runtime != "float4_e2m1fn_x2"
                or es.runtime != "float8_e8m0fnu" or len(ew.shape) != 2):
            raise ConsumptionError("matrix", f"{name}: unsupported independent expectation")
        _checked(inv.compare({wn: ew, sn: es}, {wn: weight, sn: scale})[0])
        n, packed_k = ew.shape
        selected = Matrix(name, n, 2 * packed_k, weight, scale)
        _matrix_layout(selected)
        return selected
    except (KeyError, TypeError, ValueError) as exc:
        raise ConsumptionError("matrix", f"{name}: {exc}") from None


def _matrix_layout(matrix):
    """Revalidate descriptors so replaced/mismatched Matrix objects cannot read rows."""
    if (matrix.name not in tuple(f"{PREFIX}.{m}" for m in ("w1", "w2", "w3"))
            or not _count(matrix.n) or not _count(matrix.k) or matrix.k % inv.FP4_BLOCK):
        raise ConsumptionError("row_layout", f"invalid matrix {matrix.name} dimensions")
    for pair, suffix, dtype, width in ((matrix.weight, ".weight", "I8", matrix.k // 2),
                                       (matrix.scale, ".scale", "F8_E8M0", matrix.k // inv.FP4_BLOCK)):
        try:
            shard, entry = pair
            valid = (entry.name == matrix.name + suffix and entry.dtype == dtype
                     and entry.shape == (matrix.n, width)
                     and isinstance(entry.begin, int) and entry.begin >= 0
                     and entry.end - entry.begin == matrix.n * width
                     and isinstance(shard.data_start, int) and shard.data_start >= 8)
        except (TypeError, AttributeError, ValueError):
            valid = False
        if not valid:
            raise ConsumptionError("row_layout", f"{matrix.name}: invalid {suffix} descriptor or pairing")


def read_row(matrix, row):
    """Read exactly two row intervals using pread, retaining their absolute fingerprints."""
    _matrix_layout(matrix)
    if not isinstance(row, int) or isinstance(row, bool) or not 0 <= row < matrix.n:
        raise ConsumptionError("row_range", f"{matrix.name}: row {row!r} outside [0,{matrix.n})")
    buffers, reads = [], []
    try:
        for shard, entry in (matrix.weight, matrix.scale):
            length = entry.shape[1]
            offset = shard.data_start + entry.begin + row * length
            fd = os.open(shard.path, os.O_RDONLY)
            try:
                buf = os.pread(fd, length, offset)
            finally:
                os.close(fd)
            if len(buf) != length:
                raise ConsumptionError("short_read", f"{entry.name} row={row} offset={offset}: {len(buf)} != {length}")
            buffers.append(buf)
            reads.append({"tensor": entry.name, "shard": str(Path(shard.path).absolute()),
                          "offset": offset, "length": length, "sha256": hashlib.sha256(buf).hexdigest()})
    except OSError as exc:
        raise ConsumptionError("io", f"{matrix.name} row={row}: {exc}") from None
    return Row(matrix.name, row, matrix.k, *buffers, reads)


def input_cases(k):
    """Construct two nondegenerate finite BF16 patterns entirely with integer operations."""
    if not _count(k) or k % inv.FP4_BLOCK:
        raise ConsumptionError("input", "K must be positive and divisible by FP4_BLOCK")
    mixed, ramp = [], []
    for i in range(k):
        g = i // inv.FP4_BLOCK
        mixed.append((int(i % 3 == 0) << 15) | ((121 + g % 11) << 7) | ((37 * i + 11) % 128))
        ramp.append((int((i + g) % 2 == 0) << 15) | ((119 + (3 * g) % 13) << 7)
                    | ((53 * i + 7 * g + 29) % 128))
    return {"mixed": tuple(mixed), "group_ramp": tuple(ramp)}


def _execute(exe, case, context, labels=None):
    """Reuse strict protocol and existing contracts: NaN class, otherwise raw bits."""
    try:
        responses = run_executor(exe, [case])
        response = responses[0]
    except (HarnessError, subprocess.TimeoutExpired, OSError, UnicodeError) as exc:
        raise ConsumptionError("executor", f"{context}: {exc}") from None
    if response[0] != "ok":
        raise ConsumptionError("executor", f"{context}: valid input rejected: {response[1]}")
    want, got = case.expect[1], response[1]
    if len(want) != len(got):
        raise ConsumptionError("protocol", f"{context}: response length {len(got)} != {len(want)}")
    for i, (expected, actual) in enumerate(zip(want, got)):
        if expected != actual and not (is_nan(expected, case.kind) and is_nan(actual, case.kind)):
            stage = labels[i] if labels is not None else case.op
            raise ConsumptionError("mismatch", f"{context} stage={stage}[{i}]: expected {expected:#x}, got {actual:#x}")
    return got


def verify_row(row, inputs, exe):
    """Validate activation, point stages, and both direct production-buffer compositions."""
    context = f"matrix={row.matrix} row={row.index}"
    if (not _count(row.k) or row.k % inv.FP4_BLOCK or not isinstance(row.weight, bytes)
            or not isinstance(row.scale, bytes) or len(row.weight) != row.k // 2
            or len(row.scale) != row.k // inv.FP4_BLOCK):
        raise ConsumptionError("row_layout", context + ": invalid row lengths")
    if not isinstance(inputs, dict) or not inputs:
        raise ConsumptionError("input", context + ": empty or invalid input cases")
    records = []
    for name, bits in inputs.items():
        ctx = context + f" input={name}"
        try:
            if not isinstance(name, str) or not name or len(bits) != row.k:
                raise ConsumptionError("input", ctx + ": invalid name or K mismatch")
            a, sa = act_quant(bits)
            activations = [ActCase(name, bits, traced=traced) for traced in (False, True)]
        except (ValueError, TypeError) as exc:
            raise ConsumptionError("input", ctx + f" stage=activation: {exc}") from None
        activation_count = 0
        for case in activations:
            _execute(exe, case, ctx + " stage=activation", case.labels)
            activation_count += len(case.expect[1])
        try:
            trace = trace_expert_grouped_dot(a, row.weight, sa, row.scale)
        except (ValueError, TypeError) as exc:
            raise ConsumptionError("oracle", ctx + f" stage=compute: {exc}") from None
        words = trace_words(trace)
        labels = (["products"] * row.k + ["partial_sums"] * row.k
                  + [stage for stage in ("dot", "after_a", "block", "acc") for _ in trace.blocks]
                  + ["fp32_bits"])
        case = ComputeCase(name, "expert_grouped_dot_trace", (a, row.weight, sa, row.scale),
                           len(words), 4, ("ok", words), FP32)
        _execute(exe, case, ctx + " stage=compute", labels)
        outputs = {}
        for kind, value, width in (("fp32", trace.fp32_bits, 4), ("bf16", trace.bf16_bits, 2)):
            case = ComputeCase(name, "act_quant_dot_" + kind, (bits, row.weight, row.scale),
                               1, width, ("ok", [value]), kind)
            outputs[kind + "_bits"] = _execute(exe, case, ctx + " stage=combination." + kind)[0]
        records.append({"case": name, **outputs, "expected_fp32_bits": trace.fp32_bits,
                        "expected_bf16_bits": trace.bf16_bits, "activation_comparisons": activation_count,
                        "compute_comparisons": len(words), "output_comparisons": 2, "pass": True})
    return records


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_identity():
    try:
        def git(*args):
            return subprocess.run(["git", "-C", str(ROOT), *args], check=True, capture_output=True,
                                  text=True, timeout=30).stdout.strip()
        return {"git_sha": git("rev-parse", "HEAD"), "version": (ROOT / "VERSION").read_text().strip(),
                "dirty": bool(git("status", "--porcelain", "--untracked-files=normal"))}
    except (OSError, subprocess.SubprocessError) as exc:
        raise ConsumptionError("io", f"cannot record source identity: {exc}") from None


def run(checkpoint, exe, reference=inv.DEFAULT_REFERENCE):
    """Return a serializable report only after every fixed row and input comparison passes."""
    snapshot = inspect_checkpoint(checkpoint, reference)
    executable = Path(exe).absolute()
    try:
        exe_sha = _file_sha256(executable)
    except OSError as exc:
        raise ConsumptionError("io", f"executor {executable}: {exc}") from None
    identity = _source_identity()
    report = {"plan": "P0018", "pass": False, "environment_contract": "ENV-A",
              "environment": {"platform": platform.platform(), "machine": platform.machine(),
                              "python": platform.python_version(), "byteorder": sys.byteorder},
              "checkpoint": str(snapshot.root), "revision": snapshot.revision, **identity,
              "exe": {"path": str(executable), "sha256": exe_sha}, "inputs": INPUT_RECIPE,
              "comparison_policy": "finite/Inf/signed-zero bit-exact; NaN by existing COMPUTE classification",
              "reference": str(Path(reference).absolute()), "matrices": [],
              "command": [sys.executable, str(Path(__file__).resolve()), str(snapshot.root),
                          "--exe", str(executable), "--reference", str(Path(reference).absolute())]}
    comparisons = {"activation": 0, "compute": 0, "outputs": 0}
    for short in ("w1", "w3", "w2"):
        matrix = select_matrix(snapshot, short)
        inputs = input_cases(matrix.k)
        entry = {"name": matrix.name, "n": matrix.n, "k": matrix.k, "inputs": [], "rows": []}
        for name, bits in inputs.items():
            raw = b"".join(code.to_bytes(2, "little") for code in bits)
            entry["inputs"].append({"case": name, "k": matrix.k, "sha256": hashlib.sha256(raw).hexdigest(),
                                    "encoding": "BF16 little-endian uint16"})
        for index in sorted({0, matrix.n // 2, matrix.n - 1}):
            row = read_row(matrix, index)
            cases = verify_row(row, inputs, executable)
            entry["rows"].append({"index": index, "k": row.k, "reads": row.reads, "cases": cases})
            for case in cases:
                for key, field in (("activation", "activation_comparisons"), ("compute", "compute_comparisons"),
                                   ("outputs", "output_comparisons")):
                    comparisons[key] += case[field]
        report["matrices"].append(entry)
    report.update({"pass": True, "comparisons": {**comparisons, "total": sum(comparisons.values())}})
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument("--reference", default=inv.DEFAULT_REFERENCE, type=Path)
    parser.add_argument("--json", dest="json_path", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.json_path:
            target = args.json_path.absolute()
            if os.path.lexists(target):
                raise ConsumptionError("argument", f"report target already exists: {target}")
            if target.resolve().is_relative_to(args.checkpoint.resolve()):
                raise ConsumptionError("argument", f"report target is inside checkpoint: {target}")
        report = run(args.checkpoint, args.exe, args.reference)
        if args.json_path:
            payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
            with args.json_path.open("x", encoding="utf-8") as fh:
                fh.write(payload)
    except ConsumptionError as exc:
        print(f"RESULT FAIL {exc}", file=sys.stderr)
        return 2 if exc.code in ("argument", "io") else 1
    except OSError as exc:
        print(f"RESULT FAIL I/O error: {exc}", file=sys.stderr)
        return 2
    rows = sum(len(m["rows"]) for m in report["matrices"])
    print(f"P0018 checkpoint {report['revision']}: {rows} rows; "
          f"{report['comparisons']['outputs']} FP32/BF16 outputs; "
          f"{report['comparisons']['total']} value comparisons (finite bit-exact, NaN class); RESULT PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
