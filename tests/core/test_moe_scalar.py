#!/usr/bin/env python3
"""P0026 production single-token routed-branch acceptance.

This module imports P0025 helpers as expectations and fixtures only.  It never
collects the older oracle suite.  Production ordinary and private trace calls
are compared with ref/moe.py through the independent P0025 expectations.
"""

import argparse
import dataclasses
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "quant"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_moe as oracle_tests
from ref import moe as oracle
from test_codec_parity import HarnessError

EXE = None
EXE_ARGS = []
FILL16 = 0xA5A5
FILL32 = 0xA5A5A5A5
SUCCESS_FIELDS = ("indices", "weights", "experts", "partial", "out", "expert_output")
_EXPECTATIONS = {}


def is_nan32(bits):
    return (bits & 0x7FFFFFFF) > 0x7F800000


def equal32(actual, expected):
    return actual == expected or (is_nan32(actual) and is_nan32(expected))


@dataclasses.dataclass
class MoeCase:
    name: str
    inputs: tuple
    gate: object
    bias: tuple
    experts: tuple
    vision: object = None
    image: bool = False
    traced: bool = False
    router_caps: object = None
    expert_caps: object = None
    expert_out_cap: object = None
    accumulator_cap: object = None
    out_cap: object = None
    trace_cap: object = None

    def __post_init__(self):
        d = self.gate.k
        h = self.experts[0].w1.n
        if self.router_caps is None:
            self.router_caps = (self.gate.n + 2,) * 5
        if self.expert_caps is None:
            self.expert_caps = (d + 3, d // 32 + 2, h + 3, h + 2,
                                h + 1, h + 4, h // 32 + 2)
        if self.expert_out_cap is None:
            self.expert_out_cap = d + 3
        if self.accumulator_cap is None:
            self.accumulator_cap = d + 3
        if self.out_cap is None:
            self.out_cap = d + 3
        if self.trace_cap is None:
            self.trace_cap = 6 * d + 3

    def expected(self):
        key = (self.image, self.inputs, self.gate, self.bias, self.experts, self.vision)
        if key not in _EXPECTATIONS:
            expected = oracle_tests.expected_combination(
                self.inputs, self.gate, self.bias, self.experts,
                image_token=self.image, bias_vl_bits=self.vision)
            route, ordered, traces, partials, output = expected
            actual = oracle.trace_routed(
                self.inputs, self.gate, self.bias, self.experts,
                image_token=self.image, bias_vl_bits=self.vision)
            for field in oracle_tests.independent_router.TRACE_FIELDS:
                if getattr(actual.route, field) != route[field]:
                    raise AssertionError(f"oracle route trace disagrees at {field}")
            if (actual.expert_indices, actual.expert_traces,
                    actual.partial_sums_bits, actual.output_bits) != (
                        ordered, traces, partials, output):
                raise AssertionError("oracle trace disagrees with independent expectation")
            _EXPECTATIONS[key] = expected
        return _EXPECTATIONS[key]

    def line(self):
        head = (
            f"moe n={self.gate.n} k={self.gate.k} image={int(self.image)} "
            f"vl={int(self.vision is not None)} vl_present={int(self.vision is not None)} "
            f"trace={int(self.traced)} caps={','.join(map(str, self.router_caps))} "
            f"expert_caps={','.join(map(str, self.expert_caps))} "
            f"expert_out_cap={self.expert_out_cap} "
            f"accumulator_cap={self.accumulator_cap} out_cap={self.out_cap} "
            f"trace_cap={self.trace_cap}"
        )
        sections = [self.inputs, self.gate.weight_bf16, self.bias,
                    self.vision if self.vision is not None else ()]
        for expert in self.experts:
            for matrix in (expert.w1, expert.w2, expert.w3):
                sections.extend(((matrix.n, matrix.k), matrix.w_packed, matrix.scale_b))
        return head + " : " + " ; ".join(
            " ".join(f"{value:x}" for value in section) for section in sections)


@dataclasses.dataclass
class RawRequest:
    name: str
    text: str
    fields: tuple = ("value",)

    def line(self):
        return self.text


def parse_response(request, line):
    if line.startswith("err "):
        match = re.fullmatch(
            r"err kind=(invalid_argument|overflow_error|runtime_error) "
            r"changed=([01]) inputchanged=([01]) guardchanged=([01]) "
            r"tracechanged=([01]) workchanged=([01]) outchanged=([01])", line)
        if not match:
            raise HarnessError(f"{request.name}: malformed error response {line!r}")
        names = ("changed", "inputchanged", "guardchanged", "tracechanged",
                 "workchanged", "outchanged")
        result = {name: bool(int(value)) for name, value in zip(names, match.groups()[1:])}
        result["error"] = match[1]
        return result
    segments = line.split(" ; ")
    if not segments or segments[0] != "ok":
        raise HarnessError(f"{request.name}: malformed response {line[:200]!r}")
    result = {}
    widths = {"indices": 16, "experts": 16, "weights": 8, "partial": 8,
              "out": 8, "expert_output": 4, "value": 8}
    for segment in segments[1:]:
        if "=" not in segment:
            raise HarnessError(f"{request.name}: response field lacks '='")
        key, value = segment.split("=", 1)
        if key in result or key not in widths:
            raise HarnessError(f"{request.name}: unknown or duplicate field {key!r}")
        tokens = value.split()
        width = widths[key]
        if any(not re.fullmatch(rf"[0-9a-f]{{{width}}}", token) for token in tokens):
            raise HarnessError(f"{request.name}: malformed {key} bit pattern")
        result[key] = tuple(int(token, 16) for token in tokens)
    expected_fields = set(request.fields) if isinstance(request, RawRequest) else set(SUCCESS_FIELDS)
    if set(result) != expected_fields:
        raise HarnessError(f"{request.name}: response fields differ: {set(result) ^ expected_fields}")
    return result


def execute(exe, requests, args=()):
    try:
        process = subprocess.run(
            [str(exe), *args], input="".join(item.line() + "\n" for item in requests),
            capture_output=True, text=True, timeout=900)
    except OSError as error:
        raise HarnessError(f"cannot launch executor {exe}: {error}") from error
    if process.returncode:
        raise HarnessError(f"executor exited {process.returncode}: {process.stderr[-2000:]}")
    if process.stderr:
        raise HarnessError(f"executor emitted diagnostics: {process.stderr[-2000:]}")
    lines = process.stdout.splitlines()
    if len(lines) != len(requests):
        raise HarnessError(f"expected {len(requests)} responses, got {len(lines)}")
    return [parse_response(request, line) for request, line in zip(requests, lines)]


def directed_cases():
    inputs, gate, bias, vision, experts = oracle_tests.fixture()
    return [
        MoeCase(f"real-{branch}.{mode}", inputs, gate, bias, experts, vision,
                image=image, traced=traced)
        for branch, image in (("text", False), ("image", True))
        for mode, traced in (("ordinary", False), ("trace", True))
    ]


def runtime_failure_case(traced=False):
    _, gate, bias, experts = oracle_tests.zero_fixture()
    inputs = (0x7F7F,) * gate.k
    bad = oracle.FP4Expert(
        oracle_tests.independent_expert.constant_matrix(32, 32, 0xE, 254),
        oracle_tests.independent_expert.constant_matrix(32, 32, 0x2, 127),
        oracle_tests.independent_expert.constant_matrix(32, 32, 0x2, 127),
    )
    return MoeCase(f"runtime-hidden-nonfinite.{traced}", inputs, gate, bias,
                   experts[:-1] + (bad,), traced=traced)


def accumulate_request(name, accumulator, output, *, traced=True, cap=None):
    cap = len(accumulator) if cap is None else cap
    text = (
        f"moe_accumulate k={len(output)} trace={int(traced)} out_cap={cap} "
        f"trace_cap={cap} : "
        + " ".join(f"{value:x}" for value in accumulator)
        + " ; " + " ".join(f"{value:x}" for value in output)
    )
    return RawRequest(name, text, ("partial", "out"))


def fp32_differences(actual, expected, label):
    differences = []
    if len(actual) != len(expected):
        return [f"{label}.length"]
    for index, (got, want) in enumerate(zip(actual, expected)):
        if not equal32(got, want):
            differences.append(f"{label}[{index}]")
    return differences


def mismatches(case, response):
    if "error" in response:
        raise HarnessError(f"{case.name}: production rejected with {response['error']}")
    route, ordered, traces, partials, output = case.expected()
    differences = []
    expected_out = output + (FILL32,) * (case.out_cap - case.gate.k)
    differences += fp32_differences(response["out"], expected_out, "out")
    last_output = traces[-1].output_bf16
    expected_expert_output = last_output + (FILL16,) * (case.expert_out_cap - case.gate.k)
    if response["expert_output"] != expected_expert_output:
        differences.append("expert_output")
    if case.traced:
        if response["indices"] != route["indices"]:
            differences.append("indices")
        if response["weights"] != route["weights_bits"]:
            differences.append("weights")
        if response["experts"] != ordered:
            differences.append("experts")
        expected_partial = tuple(value for partial in partials for value in partial)
        differences += fp32_differences(response["partial"], expected_partial, "partial")
    else:
        for field in ("indices", "weights", "experts", "partial"):
            if response[field]:
                differences.append(field + ".nonempty")
    return differences


class ProductionParity(unittest.TestCase):
    def test_real_text_image_ordinary_trace_against_oracle(self):
        cases = directed_cases()
        responses = execute(EXE, cases, EXE_ARGS)
        for case, response in zip(cases, responses):
            with self.subTest(case=case.name):
                route, ordered, traces, partials, output = case.expected()
                self.assertEqual(len(set(route["weights_bits"])), 6)
                self.assertNotEqual(route["indices"], ordered)
                self.assertEqual(ordered, tuple(sorted(route["indices"])))
                self.assertEqual(partials[-1], output)
                self.assertEqual(len(traces), 6)
                self.assertEqual(mismatches(case, response), [])

    def test_trace_and_ordinary_share_the_same_final_result(self):
        cases = directed_cases()
        responses = execute(EXE, cases, EXE_ARGS)
        by_branch = {case.name: response for case, response in zip(cases, responses)}
        for branch in ("text", "image"):
            ordinary = by_branch[f"real-{branch}.ordinary"]
            traced = by_branch[f"real-{branch}.trace"]
            self.assertEqual(ordinary["out"], traced["out"])
            self.assertEqual(ordinary["expert_output"], traced["expert_output"])


class AggregateSeamTests(unittest.TestCase):
    def test_real_accumulate_specials_rounding_order_and_stepwise_counterexamples(self):
        directed = (
            ((0,), (0x0001,), (0x00010000,), "subnormal"),
            ((0x80000000,), (0x8000,), (0x80000000,), "negative-zero"),
            ((0x3F800000,), (0x3380,), (0x3F800000,), "halfway-even"),
            ((0x3F800001,), (0x3380,), (0x3F800002,), "halfway-odd"),
            ((0x7F800000,), (0xFF80,), (0x7FC00000,), "opposite-infinities"),
            ((0,), (0x7FC1,), (0x7FC00000,), "nan"),
        )
        requests = [accumulate_request(label, acc, output, cap=3)
                    for acc, output, _, label in directed]
        responses = execute(EXE, requests, EXE_ARGS)
        for (accumulator, output, expected, label), response in zip(directed, responses):
            with self.subTest(label=label):
                self.assertEqual(oracle_tests.expected_accumulate(accumulator, output), expected)
                self.assertEqual(response["out"][1:], (FILL32, FILL32))
                self.assertEqual(response["partial"][1:], (FILL32, FILL32))
                self.assertTrue(equal32(response["out"][0], expected[0]))
                self.assertTrue(equal32(response["partial"][0], expected[0]))

        examples = {
            "order": ((0x60AD,), (0x3F80,), (0xE0AD,)),
            "early-bf16": ((0x3F80,), (0x3B80,), (0x3B80,)),
            "nonstep": ((0x4B80,), (0x3F80,), (0xCB80,)),
        }
        for label, outputs in examples.items():
            accumulator = (0,)
            observed = []
            for step, output in enumerate(outputs):
                response = execute(
                    EXE, [accumulate_request(f"{label}-{step}", accumulator, output)],
                    EXE_ARGS)[0]
                accumulator = response["out"]
                observed.append(accumulator)
            expected_partials, expected = oracle_tests.expected_partials(outputs, 1)
            self.assertEqual(tuple(observed), expected_partials, label)
            self.assertEqual(accumulator, expected, label)
        wrong_order = (examples["order"][0], examples["order"][2],
                       examples["order"][1])
        self.assertNotEqual(oracle_tests.expected_partials(wrong_order, 1)[1],
                            oracle_tests.expected_partials(examples["order"], 1)[1])
        self.assertNotEqual((0x3F800000,),
                            oracle_tests.expected_partials(examples["early-bf16"], 1)[1])
        self.assertNotEqual((0x3F800000,),
                            oracle_tests.expected_partials(examples["nonstep"], 1)[1])


class HarnessTests(unittest.TestCase):
    def test_private_contract_probe_reaches_success_and_runtime_commit_checks(self):
        response = execute(EXE, [RawRequest("moe-contract", "moe_contract :", ())],
                           EXE_ARGS)[0]
        self.assertEqual(response, {})

    def test_accumulate_protocol_rejects_wrong_lengths_without_result(self):
        short_input = accumulate_request("short-initial", (), (0x3F80,))
        short_output = accumulate_request("short-output", (0,), ())
        unrelated_field = RawRequest(
            "unrelated-field", "moe_accumulate k=1 trace=0 out_cap=1 trace_cap=0 "
            "vl_present=0 : 0 ; 3f80", ("partial", "out"))
        for request in (short_input, short_output, unrelated_field):
            with self.subTest(request=request.name), self.assertRaises(HarnessError):
                execute(EXE, [request], EXE_ARGS)

    def test_protocol_faults_and_completed_wrong_numeric_result_fail_closed(self):
        case = directed_cases()[1]
        for fault in ("crash", "exit", "malformed", "missing"):
            with self.subTest(fault=fault), self.assertRaises(HarnessError):
                execute(EXE, [case], [*EXE_ARGS, f"--fault={fault}"])
        wrong = execute(EXE, [case], [*EXE_ARGS, "--fault=wrong"])[0]
        self.assertTrue(mismatches(case, wrong))

    def test_missing_nonexecutable_and_unrelated_executor(self):
        case = directed_cases()[0]
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing"
            with self.assertRaises(HarnessError):
                execute(missing, [case])
            missing.write_text("not executable\n", encoding="utf-8")
            with self.assertRaises(HarnessError):
                execute(missing, [case])
        with self.assertRaises(HarnessError):
            execute(Path("/bin/true"), [case])


class EnvironmentTests(unittest.TestCase):
    def test_production_paths_preserve_requested_mxcsr_and_status(self):
        probes = (RawRequest("before", "mxcsr :"), RawRequest("after", "mxcsr :"))
        success = directed_cases()
        rejected = []
        runtime = []
        for traced in (False, True):
            rejected.append(dataclasses.replace(
                success[0], name=f"invalid-unused-bias.{traced}", traced=traced,
                bias=success[0].bias[:-1] + (0x7F800000,)))
            runtime.append(runtime_failure_case(traced))
            base = runtime_failure_case(traced)
            weights = list(base.gate.weight_bf16)
            weights[base.gate.k] = 0x7F7F
            runtime.append(dataclasses.replace(
                base, name=f"runtime-router-overflow.{traced}",
                gate=dataclasses.replace(base.gate, weight_bf16=tuple(weights))))
        environments = {
            "default": 0x1F80, "up": 0x5F80, "down": 0x3F80, "zero": 0x7F80,
            "ftz": 0x9F80, "daz": 0x1FC0, "ftz_daz": 0x9FC0,
            "overflow_unmasked": 0x1B80,
        }
        for name, control in environments.items():
            for status in (0, 0x3F):
                requests = [probes[0], *success, *rejected, *runtime, probes[1]]
                responses = execute(
                    EXE, requests,
                    [*EXE_ARGS, f"--mxcsr={name}", f"--status={status:x}"])
                self.assertEqual(responses[0], {"value": (control | status,)})
                self.assertEqual(responses[-1], {"value": (control | status,)})
                offset = 1 + len(success)
                for case, response in zip(success, responses[1:offset]):
                    self.assertEqual(mismatches(case, response), [], f"{name}/{status:x}")
                for case, response in zip(rejected, responses[offset:offset + len(rejected)]):
                    with self.subTest(environment=name, status=status, case=case.name):
                        self.assertEqual(response["error"], "invalid_argument")
                        self.assertFalse(response["changed"])
                offset += len(rejected)
                for case, response in zip(runtime, responses[offset:-1]):
                    with self.subTest(environment=name, status=status, case=case.name):
                        self.assertEqual(response["error"], "invalid_argument")
                        self.assertTrue(response["workchanged"])
                        for field in ("inputchanged", "guardchanged", "tracechanged", "outchanged"):
                            self.assertFalse(response[field], field)


SUITES = {
    "production": (ProductionParity, AggregateSeamTests),
    "harness": (HarnessTests,),
    "environment": (EnvironmentTests,),
    "all": (ProductionParity, AggregateSeamTests, HarnessTests, EnvironmentTests),
}


def iter_tests(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from iter_tests(item)
        else:
            yield item


def numeric_report(cases):
    payload = {"cases": [], "errors": [], "skipped": 0}
    try:
        responses = execute(EXE, cases, EXE_ARGS)
        for case, response in zip(cases, responses):
            payload["cases"].append({"case": case.name,
                                     "mismatches": mismatches(case, response)})
    except Exception as error:
        payload["errors"].append(f"{type(error).__name__}: {error}")
    print("P0026_NUMERIC " + json.dumps(payload, sort_keys=True))
    return int(bool(payload["errors"] or
                    any(item["mismatches"] for item in payload["cases"])))


def executable_error(path):
    if not path.exists():
        return f"executor does not exist: {path}"
    if not path.is_file():
        return f"executor is not a file: {path}"
    if not os.access(path, os.X_OK):
        return f"executor is not executable: {path}"
    return None


def main(argv=None):
    global EXE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument("--exe-arg", action="append", default=[])
    parser.add_argument("--suite", choices=tuple(SUITES), default="production")
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--numeric-report", action="store_true")
    args = parser.parse_args(argv)
    error = executable_error(args.exe)
    if error:
        parser.error(error)
    EXE = args.exe.resolve()
    EXE_ARGS.extend(args.exe_arg)
    cases = [case for case in directed_cases()
             if not args.case or any(token in case.name for token in args.case)]
    if args.numeric_report:
        if not cases:
            print("P0026_NUMERIC " + json.dumps(
                {"cases": [], "errors": ["no cases selected"], "skipped": 0}))
            return 1
        return numeric_report(cases)
    loader = unittest.TestLoader()
    suite = unittest.TestSuite(loader.loadTestsFromTestCase(cls)
                               for cls in SUITES[args.suite])
    if args.case:
        suite = unittest.TestSuite(
            test for test in iter_tests(suite)
            if any(token in test.id() for token in args.case))
    if not suite.countTestCases():
        print("no tests selected", file=sys.stderr)
        return 2
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() and result.testsRun and not result.skipped else 1


if __name__ == "__main__":
    sys.exit(main())
