#!/usr/bin/env python3
"""P0019: offline full-w1 validation through the production CPU linear entry.

Read only every row of layers.0.ffn.experts.0.w1, at full K. Artificial BF16
inputs validate the CPU contract, not model quality or GPU equivalence.
Exit: 0 pass, 1 validation failure, 2 argument or I/O error; never skip.
"""

import argparse
import hashlib
import json
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import ckpt_expert as expert
from ref.act_quant import act_quant
from ref.compute import trace_expert_grouped_dot

ConsumptionError = expert.ConsumptionError


@dataclass(frozen=True)
class Payload:
    name: str
    n: int
    k: int
    weight: bytes
    scale: bytes
    rows: list


@dataclass(frozen=True)
class LinearCase:
    """P0019 executor adapter: one production matrix call with three payloads."""

    name: str
    kind: str
    n: int
    bits: tuple
    weight: bytes
    scale: bytes
    expected: list

    @property
    def op(self):
        return "expert_linear_" + self.kind

    @property
    def cap(self):
        return self.n

    @property
    def out_width(self):
        return 4 if self.kind == "fp32" else 2

    @property
    def expect(self):
        return "ok", self.expected

    def line(self):
        sections = " ; ".join(" ".join(f"{v:x}" for v in values)
                              for values in (self.bits, self.weight, self.scale))
        return f"{self.op} shape={self.n} cap={self.n} : {sections}"


def read_matrix(matrix):
    """Assemble exactly one row-major matrix using P0018's bounded row reader."""
    expert._matrix_layout(matrix)
    if matrix.name != expert.PREFIX + ".w1":
        raise ConsumptionError("matrix", "the offline linear validator only accepts fixed expert w1")
    weights, scales, rows = bytearray(), bytearray(), []
    for index in range(matrix.n):
        row = expert.read_row(matrix, index)
        weights.extend(row.weight)
        scales.extend(row.scale)
        rows.append({"index": index, "reads": row.reads})
    return Payload(matrix.name, matrix.n, matrix.k, bytes(weights), bytes(scales), rows)


def verify_matrix(payload, inputs, exe):
    """Quantize once per oracle input, discard each row trace, compare full outputs."""
    context = f"matrix={payload.name}"
    if (payload.name != expert.PREFIX + ".w1" or not expert._count(payload.n)
            or not expert._count(payload.k) or payload.k % expert.inv.FP4_BLOCK
            or not isinstance(payload.weight, bytes) or not isinstance(payload.scale, bytes)
            or len(payload.weight) != payload.n * (payload.k // 2)
            or len(payload.scale) != payload.n * (payload.k // expert.inv.FP4_BLOCK)):
        raise ConsumptionError("matrix", context + ": invalid full matrix payload")
    if not isinstance(inputs, dict) or not inputs:
        raise ConsumptionError("input", context + ": empty or invalid input cases")
    records = []
    for name, bits in inputs.items():
        ctx = context + f" input={name}"
        try:
            if not isinstance(name, str) or not name or len(bits) != payload.k:
                raise ConsumptionError("input", ctx + ": invalid name or K mismatch")
            bits = tuple(bits)
            a, sa = act_quant(bits)
        except (TypeError, ValueError) as exc:
            raise ConsumptionError("input", ctx + f" stage=activation: {exc}") from None
        expected = {"fp32": [], "bf16": []}
        for index in range(payload.n):
            w, s = payload.k // 2, payload.k // expert.inv.FP4_BLOCK
            try:
                trace = trace_expert_grouped_dot(a, payload.weight[index * w:(index + 1) * w],
                                                sa, payload.scale[index * s:(index + 1) * s])
            except (ValueError, TypeError) as exc:
                raise ConsumptionError("oracle", ctx + f" row={index} stage=compute: {exc}") from None
            expected["fp32"].append(trace.fp32_bits)
            expected["bf16"].append(trace.bf16_bits)
            del trace
        outputs = {}
        for kind in ("fp32", "bf16"):
            case = LinearCase(name, kind, payload.n, bits, payload.weight, payload.scale, expected[kind])
            outputs[kind + "_bits"] = expert._execute(
                exe, case, ctx + f" stage={case.op}", [f"row={i}" for i in range(payload.n)])
        records.append({"case": name, **outputs, "expected_fp32_bits": expected["fp32"],
                        "expected_bf16_bits": expected["bf16"], "output_comparisons": 2 * payload.n,
                        "oracle_activation_calls": 1, "oracle_rows": payload.n,
                        "production_matrix_calls": 2, "pass": True})
    return records


def _fingerprint(data):
    return {"length": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def run(checkpoint, exe, reference=expert.inv.DEFAULT_REFERENCE):
    """Validate the fixed full matrix, returning evidence only if every output passes."""
    snapshot = expert.inspect_checkpoint(checkpoint, reference)
    executable = Path(exe).absolute()
    try:
        exe_sha = expert._file_sha256(executable)
    except OSError as exc:
        raise ConsumptionError("io", f"executor {executable}: {exc}") from None
    identity = expert._source_identity()
    matrix = expert.select_matrix(snapshot, "w1")
    payload = read_matrix(matrix)
    inputs = expert.input_cases(matrix.k)
    records = verify_matrix(payload, inputs, executable)
    comparisons = sum(record["output_comparisons"] for record in records)
    return {"plan": "P0019", "pass": True, "environment_contract": "ENV-A",
            "environment": {"platform": platform.platform(), "machine": platform.machine(),
                            "python": platform.python_version(), "byteorder": sys.byteorder},
            "checkpoint": str(snapshot.root), "revision": snapshot.revision, **identity,
            "exe": {"path": str(executable), "sha256": exe_sha}, "inputs": expert.INPUT_RECIPE,
            "reference": str(Path(reference).absolute()),
            "comparison_policy": "finite/Inf/signed-zero bit-exact; NaN by existing COMPUTE classification",
            "matrices": [{"name": payload.name, "n": payload.n, "k": payload.k,
                          "layout": "contiguous row-major packed weights and paired scale_b rows",
                          "payloads": {"weight": _fingerprint(payload.weight), "scale_b": _fingerprint(payload.scale)},
                          "inputs": [{"case": name, "k": matrix.k,
                                      **_fingerprint(b"".join(code.to_bytes(2, "little") for code in bits)),
                                      "encoding": "BF16 little-endian uint16"} for name, bits in inputs.items()],
                          "rows": payload.rows, "cases": records}],
            "comparisons": {"outputs": comparisons, "total": comparisons},
            "command": [sys.executable, str(Path(__file__).resolve()), str(snapshot.root),
                        "--exe", str(executable), "--reference", str(Path(reference).absolute())]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument("--reference", default=expert.inv.DEFAULT_REFERENCE, type=Path)
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
            report["command"] += ["--json", str(args.json_path.absolute())]
            with args.json_path.open("x", encoding="utf-8") as fh:
                fh.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    except ConsumptionError as exc:
        print(f"RESULT FAIL {exc}", file=sys.stderr)
        return 2 if exc.code in ("argument", "io") else 1
    except OSError as exc:
        print(f"RESULT FAIL I/O error: {exc}", file=sys.stderr)
        return 2
    matrix = report["matrices"][0]
    print(f"P0019 checkpoint {report['revision']}: {matrix['name']} N={matrix['n']} K={matrix['k']}; "
          f"{report['comparisons']['outputs']} FP32/BF16 full-vector comparisons "
          "(finite bit-exact, NaN class); RESULT PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
