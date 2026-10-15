#!/usr/bin/env python3
"""P0022: offline full-expert checkpoint parity through production CPU entries.

Read only the fixed layers.0.ffn.experts.0 W1/W3/W2 payloads at full shape.
Artificial finite BF16 inputs validate the CPU contract, not model quality or GPU
parity. Exit: 0 pass, 1 validation failure, 2 argument or I/O error; never skip.
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
sys.path.insert(0, str(ROOT / "tests" / "quant"))
from tools import ckpt_expert as bounded
from ref.expert import FP4Matrix, trace_expert

ConsumptionError = bounded.ConsumptionError
PREFIX = bounded.PREFIX
MATRIX_ORDER = ("w1", "w3", "w2")
PAYLOAD_ORDER = ("input", "w1packed", "w1scale", "w2packed", "w2scale",
                 "w3packed", "w3scale")
ROUTES = (("unweighted", None), ("route_half", 0x3F000000))
ORDINARY_FIELDS = (
    ("input_codes", "input_codes", "raw8"),
    ("input_scales", "input_scales", "raw8"),
    ("gate_bf16", "gate_bf16", "bf16"),
    ("up_bf16", "up_bf16", "bf16"),
    ("bridge.bf16_bits", "hidden_bf16", "bf16"),
    ("hidden_codes", "hidden_codes", "raw8"),
    ("hidden_scales", "hidden_scales", "raw8"),
    ("output_bf16", "output_bf16", "bf16"),
)
TRACE_FIELDS = ORDINARY_FIELDS[:4] + (
    ("bridge.gate_bits", "gate_bits", "fp32"),
    ("bridge.up_bits", "up_bits", "fp32"),
    ("bridge.silu_bits", "silu_bits", "fp32"),
    ("bridge.gated_bits", "gated_bits", "fp32"),
    ("bridge.weighted_bits", "weighted_bits", "fp32"),
) + ORDINARY_FIELDS[4:]
@dataclass(frozen=True)
class MatrixPayload:
    name: str
    n: int
    k: int
    packed: bytes
    scale: bytes
    rows: tuple


@dataclass(frozen=True)
class OracleCase:
    name: str
    input_bf16: tuple
    route_weight_bits: int | None
    trace: object
    output: tuple


def _fingerprint(data):
    return {"length": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _input_bytes(values):
    return b"".join(value.to_bytes(2, "little") for value in values)


def read_matrix(matrix):
    """Assemble one fixed full matrix solely through P0018 bounded row reads."""
    bounded._matrix_layout(matrix)
    if matrix.name not in tuple(f"{PREFIX}.{short}" for short in MATRIX_ORDER):
        raise ConsumptionError("matrix", f"not a fixed expert matrix: {matrix.name}")
    packed, scale, rows = bytearray(), bytearray(), []
    for index in range(matrix.n):
        row = bounded.read_row(matrix, index)
        packed.extend(row.weight)
        scale.extend(row.scale)
        rows.append({"index": index, "reads": row.reads})
    return MatrixPayload(matrix.name, matrix.n, matrix.k, bytes(packed), bytes(scale),
                         tuple(rows))


def read_expert(snapshot):
    """Read exactly all rows of W1, W3 and W2, checking the expert shape."""
    payloads = {short: read_matrix(bounded.select_matrix(snapshot, short))
                for short in MATRIX_ORDER}
    w1, w3, w2 = (payloads[name] for name in MATRIX_ORDER)
    if (w3.n, w3.k) != (w1.n, w1.k) or (w2.n, w2.k) != (w1.k, w1.n):
        raise ConsumptionError(
            "matrix", f"expected w1/w3 H-by-D and w2 D-by-H; got "
            f"w1={w1.n}x{w1.k}, w3={w3.n}x{w3.k}, w2={w2.n}x{w2.k}")
    return payloads


def _matrix(payload):
    return FP4Matrix(payload.n, payload.k, payload.packed, payload.scale)


def oracle_cases(payloads, inputs):
    """Compute one trace oracle for every input/route combination and reuse its output."""
    matrices = {name: _matrix(payloads[name]) for name in MATRIX_ORDER}
    cases = []
    for input_name, values in inputs.items():
        if input_name not in ("mixed", "group_ramp"):
            raise ConsumptionError("input", f"unsupported input case {input_name!r}")
        for route_name, route in ROUTES:
            name = f"{input_name}.{route_name}"
            try:
                trace = trace_expert(values, matrices["w1"], matrices["w2"], matrices["w3"],
                                     route_weight_bits=route)
            except (TypeError, ValueError, ArithmeticError) as exc:
                raise ConsumptionError("oracle", f"input={name} stage=full_expert: {exc}") from None
            cases.append(OracleCase(name, tuple(values), route, trace,
                                    tuple(trace.output_bf16)))
    return cases


def _expected_values(trace, traced):
    values, labels, kinds = [], [], []
    for path, stage, kind in TRACE_FIELDS if traced else ORDINARY_FIELDS:
        target = trace
        for component in path.split("."):
            target = getattr(target, component)
        values.extend(target)
        labels.extend(f"{stage}[{index}]" for index in range(len(target)))
        kinds.extend([kind] * len(target))
    return values, labels, kinds


def _load_protocol():
    """Load the production executor adapter supplied by the P0022 parity harness."""
    try:
        from test_expert_parity import ExpertCase, OMIT, flatten_trace, execute
    except (ImportError, AttributeError) as exc:
        raise ConsumptionError(
            "protocol", f"P0022 ExpertCase/flatten_trace adapter unavailable: {exc}") from None
    return ExpertCase, OMIT, flatten_trace, execute


def _construct_case(case_type, omit, *, name, traced, payloads, oracle):
    """Construct the shared adapter using its formal precomputed-trace interface."""
    w1, w2, w3 = payloads["w1"], payloads["w2"], payloads["w3"]
    d, h = w1.k, w1.n
    matrices = (_matrix(w1), _matrix(w2), _matrix(w3))
    expected, labels, kinds = _expected_values(oracle.trace, traced)
    try:
        case = case_type(
            name, oracle.input_bf16, *matrices,
            route=omit if oracle.route_weight_bits is None else oracle.route_weight_bits,
            traced=traced, cap=d,
            work_shape=(d, d // bounded.inv.FP4_BLOCK, h, h, h,
                        h, h // bounded.inv.FP4_BLOCK),
            trace_shape=(h, h, h, h, h) if traced else None,
            expected_trace=oracle.trace,
        )
    except (TypeError, ValueError, AttributeError) as exc:
        raise ConsumptionError("protocol", f"cannot construct ExpertCase: {exc}") from None
    return case, expected, labels, kinds


def _protocol_values(case, response):
    try:
        values = list(response[1])
    except (TypeError, ValueError, IndexError) as exc:
        raise ConsumptionError("protocol", f"cannot flatten ExpertCase response: {exc}") from None
    if len(values) != len(case.labels):
        raise ConsumptionError(
            "protocol", f"response length {len(values)} != adapter capacity {len(case.labels)}")
    return values


def _nan(bits, kind):
    if kind == "fp32":
        return (bits & 0x7FFFFFFF) > 0x7F800000
    if kind == "bf16":
        return (bits & 0x7FFF) > 0x7F80
    return False


def _compare(expected, actual, labels, kinds, context):
    if len(actual) != len(expected):
        raise ConsumptionError("protocol", f"{context}: response length {len(actual)} != {len(expected)}")
    limits = {"raw8": 0xFF, "bf16": 0xFFFF, "fp32": 0xFFFFFFFF}
    for index, (want, got, label, kind) in enumerate(zip(expected, actual, labels, kinds)):
        if not isinstance(got, int) or isinstance(got, bool) or not 0 <= got <= limits[kind]:
            raise ConsumptionError("protocol", f"{context} stage={label}: {kind} value out of range: {got!r}")
        if want != got and not (_nan(want, kind) and _nan(got, kind)):
            raise ConsumptionError("mismatch", f"{context} stage={label}: expected {want:#x}, got {got:#x}")
    return len(expected)


def _comparison_evidence(trace, traced, actual):
    """Fingerprint every compared stage without inflating reports with full vectors."""
    records, offset = [], 0
    widths = {"raw8": 1, "bf16": 2, "fp32": 4}
    for path, stage, kind in TRACE_FIELDS if traced else ORDINARY_FIELDS:
        target = trace
        for component in path.split("."):
            target = getattr(target, component)
        expected = list(target)
        observed = actual[offset:offset + len(expected)]
        width = widths[kind]
        encode = lambda values: b"".join(value.to_bytes(width, "little") for value in values)
        records.append({"stage": stage, "kind": kind, "comparisons": len(expected),
                        "expected": _fingerprint(encode(expected)),
                        "actual": _fingerprint(encode(observed)), "pass": True})
        offset += len(expected)
    if offset != len(actual):
        raise ConsumptionError("protocol", f"comparison evidence consumed {offset} of {len(actual)} values")
    return records


def verify(payloads, cases, executables):
    """Call ordinary and traced production expert entries and compare every stage."""
    ExpertCase, omit, flatten_trace, execute = _load_protocol()
    records, totals = [], {"ordinary": 0, "trace": 0}
    for executable in executables:
        exe_totals = {"ordinary": 0, "trace": 0}
        try:
            executable_sha256 = bounded._file_sha256(executable)
        except OSError as exc:
            raise ConsumptionError("executor", f"executor {executable}: {exc}") from None
        exe_record = {"path": str(executable), "sha256": executable_sha256, "cases": []}
        for oracle in cases:
            record = {"case": oracle.name, "route_weight_bits": oracle.route_weight_bits,
                      "entries": {}}
            for traced, entry in ((False, "expert_bf16"), (True, "expert_trace")):
                context = f"exe={executable} input={oracle.name} entry={entry}"
                case, expected, labels, kinds = _construct_case(
                    ExpertCase, omit, name=oracle.name, traced=traced,
                    payloads=payloads, oracle=oracle)
                if getattr(case, "op", entry) != entry:
                    raise ConsumptionError("protocol", f"{context}: adapter op is {getattr(case, 'op', None)!r}")
                if traced and expected != list(flatten_trace(oracle.trace)):
                    raise ConsumptionError("protocol", f"{context}: local trace order differs from shared flatten_trace")
                try:
                    response = execute(executable, [case])[0]
                except (bounded.HarnessError, OSError, UnicodeError) as exc:
                    raise ConsumptionError("executor", f"{context}: {exc}") from None
                if response[0] != "ok":
                    raise ConsumptionError("executor", f"{context}: valid request rejected: {response[1]}")
                actual = _protocol_values(case, response)
                count = _compare(expected, actual, labels, kinds, context)
                category = "trace" if traced else "ordinary"
                totals[category] += count
                exe_totals[category] += count
                record["entries"][entry] = {
                    "comparisons": count,
                    "stages": _comparison_evidence(oracle.trace, traced, actual),
                    "pass": True,
                }
            exe_record["cases"].append(record)
        exe_record["comparisons"] = {**exe_totals, "total": sum(exe_totals.values())}
        records.append(exe_record)
    return records, {**totals, "total": sum(totals.values())}


def _matrix_report(payload):
    return {"name": payload.name, "n": payload.n, "k": payload.k,
            "layout": "contiguous row-major; low nibble even K, high nibble odd K; paired scale_b rows",
            "payloads": {"packed": _fingerprint(payload.packed),
                         "scale_b": _fingerprint(payload.scale)},
            "rows": list(payload.rows)}


def run(checkpoint, executables, reference=bounded.inv.DEFAULT_REFERENCE):
    """Return evidence only after bounded reads and all production comparisons pass."""
    if isinstance(executables, (str, os.PathLike)):
        executables = [executables]
    executables = [Path(path).absolute() for path in executables]
    if not executables:
        raise ConsumptionError("argument", "at least one executor is required")
    for executable in executables:
        if not executable.is_file():
            raise ConsumptionError("io", f"executor is not a file: {executable}")
    snapshot = bounded.inspect_checkpoint(checkpoint, reference)
    identity = bounded._source_identity()
    payloads = read_expert(snapshot)
    inputs = bounded.input_cases(payloads["w1"].k)
    cases = oracle_cases(payloads, inputs)
    executor_records, comparisons = verify(payloads, cases, executables)
    command = [sys.executable, str(Path(__file__).resolve()), str(snapshot.root)]
    for executable in executables:
        command += ["--exe", str(executable)]
    command += ["--reference", str(Path(reference).absolute())]
    return {"plan": "P0022", "decision": "D0017", "pass": True,
            "environment_contract": "ENV-A",
            "environment": {"platform": platform.platform(), "machine": platform.machine(),
                            "python": platform.python_version(), "byteorder": sys.byteorder},
            "checkpoint": str(snapshot.root), "revision": snapshot.revision,
            "reference": str(Path(reference).absolute()), **identity,
            "payload_order": list(PAYLOAD_ORDER), "inputs_recipe": bounded.INPUT_RECIPE,
            "inputs": [{"case": case.name, "d": len(case.input_bf16),
                        "route_weight_bits": case.route_weight_bits,
                        **_fingerprint(_input_bytes(case.input_bf16)),
                        "encoding": "BF16 little-endian uint16"} for case in cases],
            "matrices": [_matrix_report(payloads[name]) for name in MATRIX_ORDER],
            "executors": executor_records,
            "comparison_policy": "finite/Inf/signed-zero bit-exact; NaN by stage numeric class",
            "ordinary_stages": [stage for _, stage, _ in ORDINARY_FIELDS],
            "trace_stages": [stage for _, stage, _ in TRACE_FIELDS],
            "comparisons": comparisons, "command": command}


def _validate_report_target(checkpoint, json_path):
    target = json_path.absolute()
    if os.path.lexists(target):
        raise ConsumptionError("argument", f"report target already exists: {target}")
    try:
        inside = target.resolve().is_relative_to(checkpoint.resolve())
    except OSError as exc:
        raise ConsumptionError("io", f"cannot resolve report target: {exc}") from None
    if inside:
        raise ConsumptionError("argument", f"report target is inside checkpoint: {target}")
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--exe", required=True, action="append", type=Path,
                        help="production executor; repeat for Debug/Release while sharing one in-memory oracle")
    parser.add_argument("--reference", default=bounded.inv.DEFAULT_REFERENCE, type=Path)
    parser.add_argument("--json", dest="json_path", type=Path)
    args = parser.parse_args(argv)
    try:
        target = _validate_report_target(args.checkpoint, args.json_path) if args.json_path else None
        report = run(args.checkpoint, args.exe, args.reference)
        if target:
            report["command"] += ["--json", str(target)]
            with target.open("x", encoding="utf-8") as output:
                output.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    except ConsumptionError as exc:
        print(f"RESULT FAIL {exc}", file=sys.stderr)
        return 2 if exc.code in ("argument", "io") else 1
    except OSError as exc:
        print(f"RESULT FAIL I/O error: {exc}", file=sys.stderr)
        return 2
    dims = ", ".join(f"{m['name'].rsplit('.', 1)[1]}={m['n']}x{m['k']}" for m in report["matrices"])
    print(f"P0022 checkpoint {report['revision']}: {dims}; {len(report['executors'])} executor(s); "
          f"{report['comparisons']['total']} comparisons (finite bit-exact, NaN class); RESULT PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
