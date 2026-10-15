#!/usr/bin/env python3
"""P0024 production router acceptance; imports never collect older suites."""

import argparse
import dataclasses
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "quant"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ref import router as oracle
import test_router as independent
import test_router_expert as integration
import test_expert_parity as expert_parity
import test_expert as independent_expert
from test_codec_parity import HarnessError
from test_dequant_parity import ENVIRONMENTS, MXCSR_STATUS

EXE = None
EXPERT_EXE = None
EXE_ARGS = []
FILL = 0xA5A5A5A5
STAGES = ("linear", "temp", "softplus", "score", "selection")
TRACE = ("selected", "sum", "denominator", "normalized")


@dataclasses.dataclass
class Request:
    name: str
    text: str

    def line(self):
        return self.text


@dataclasses.dataclass
class RouteCase:
    name: str
    inputs: tuple
    gate: object
    bias: tuple
    vl: object = None
    image: bool = False
    traced: bool = True
    caps: object = None
    options: str = ""

    def line(self):
        caps = self.caps if self.caps is not None else (self.gate.n + 2,) * 5
        head = (f"route n={self.gate.n} k={self.gate.k} image={int(self.image)} "
                f"vl={int(self.vl is not None)} trace={int(self.traced)} "
                f"caps={','.join(map(str, caps))} {self.options}")
        sections = (self.inputs, self.gate.weight_bf16, self.bias,
                    self.vl if self.vl is not None else ())
        return head + " : " + " ; ".join(" ".join(f"{value:x}" for value in part)
                                               for part in sections)


def parse_response(request, line):
    if line.startswith("err "):
        match = re.fullmatch(
            r"err kind=(invalid_argument|overflow_error|runtime_error) "
            r"changed=([01]) inputchanged=([01]) guardchanged=([01]) tracechanged=([01])", line)
        if not match:
            raise HarnessError(f"{request.name}: malformed error response {line!r}")
        return {"error": match[1], "changed": bool(int(match[2])),
                "inputchanged": bool(int(match[3])), "guardchanged": bool(int(match[4])),
                "tracechanged": bool(int(match[5]))}
    segments = line.split(" ; ")
    if segments[0] != "ok":
        raise HarnessError(f"{request.name}: malformed response {line[:160]!r}")
    result = {}
    for segment in segments[1:]:
        if "=" not in segment:
            raise HarnessError(f"{request.name}: missing response key")
        key, value = segment.split("=", 1)
        if key in result or not re.fullmatch(r"[a-z_]+", key):
            raise HarnessError(f"{request.name}: invalid/duplicate response key {key!r}")
        tokens = value.split()
        width = 16 if key == "indices" else 8
        if any(not re.fullmatch(rf"[0-9a-f]{{{width}}}", token) for token in tokens):
            raise HarnessError(f"{request.name}: malformed {key} bit pattern")
        result[key] = tuple(int(token, 16) for token in tokens)
    return result


def execute(exe, requests, args=()):
    try:
        proc = subprocess.run([str(exe), *args], input="".join(c.line() + "\n" for c in requests),
                              capture_output=True, text=True, timeout=900)
    except OSError as exc:
        raise HarnessError(f"cannot launch executor {exe}: {exc}") from exc
    if proc.returncode:
        raise HarnessError(f"executor exited {proc.returncode}: {proc.stderr[-2000:]}")
    if proc.stderr:
        raise HarnessError(f"executor emitted diagnostics: {proc.stderr[-2000:]}")
    lines = proc.stdout.splitlines()
    if len(lines) != len(requests):
        raise HarnessError(f"expected {len(requests)} responses, got {len(lines)}")
    return [parse_response(c, line) for c, line in zip(requests, lines)]


def scalar(name, op, *values, options=""):
    return Request(name, f"{op} {options} : " + " ; ".join(f"{value:x}" for value in values))


def oracle_trace(case):
    trace = oracle.trace_route(case.inputs, case.gate, case.bias,
                               image_token=case.image, bias_vl_bits=case.vl)
    return {field: getattr(trace, field) for field in independent.TRACE_FIELDS}


def mismatches(case, response, want=None):
    if "error" in response:
        raise HarnessError(f"{case.name}: production rejected with {response['error']}")
    want = oracle_trace(case) if want is None else want
    keys = {*STAGES, "indices", "weights", "input", "gate", "bias", "bias_vl"}
    if case.traced:
        keys.update(TRACE)
    if set(response) != keys:
        raise HarnessError(f"{case.name}: response fields {set(response) ^ keys}")
    differences = []
    caps = case.caps if case.caps is not None else (case.gate.n + 2,) * 5
    for stage, cap in zip(STAGES, caps):
        values = response[stage]
        if len(values) != cap:
            raise HarnessError(f"{case.name}: wrong {stage} capacity")
        if values[:case.gate.n] != want[stage + "_bits"]:
            differences.append(stage)
        for index in range(case.gate.n, cap):
            expected_tail = FILL
            overlap = re.search(r"overlap=([a-z_]+),([a-z_]+)", case.options)
            if overlap and stage in overlap.groups():
                pair = overlap.groups()
                first = re.search(r"in_offset=(\d+)", case.options)
                second = re.search(r"out_offset=(\d+)", case.options)
                offsets = (int(first[1]) if first else 0, int(second[1]) if second else 0)
                absolute = offsets[pair.index(stage)] + index
                for other, start in zip(pair, offsets):
                    if other in STAGES and start <= absolute < start + case.gate.n:
                        expected_tail = want[other + "_bits"][absolute - start]
            if values[index] != expected_tail:
                raise HarnessError(f"{case.name}: changed {stage} tail outside write union")
    for name, values in (("indices", want["indices"]), ("weights", want["weights_bits"])):
        if len(response[name]) != 6:
            raise HarnessError(f"{case.name}: wrong {name} length")
        if response[name] != values:
            differences.append(name)
    if case.traced:
        for name in TRACE:
            values = want[name + "_bits"]
            values = (values,) if isinstance(values, int) else values
            if len(response[name]) != len(values):
                raise HarnessError(f"{case.name}: wrong {name} length")
            if response[name] != values:
                differences.append(name)
    for key, original in (("input", case.inputs), ("gate", case.gate.weight_bf16),
                          ("bias", case.bias), ("bias_vl", case.vl or ())):
        if response[key] != tuple(original):
            raise HarnessError(f"{case.name}: changed input {key}")
    return differences


def directed_cases():
    cases = []
    inputs, gate, bias, vl = independent.basic_fixture()
    for image, vision, name in ((False, vl, "text"), (True, vl, "image"),
                                (True, None, "fallback")):
        cases.append(RouteCase(name, inputs, gate, bias, vision, image))
    rows = [[independent.bf16(independent.fmt.power2(24)), 0x3F80,
             independent.bf16(-independent.fmt.power2(24))]]
    rows += [[independent.bf16(Fraction(v)), 0, 0] for v in (-4, -2, -1, 0, 1, 2)]
    gate = independent.gate_from_rows(rows)
    cases.append(RouteCase("projection-order", (0x3F80,) * 3, gate, (0,) * gate.n))
    gate = independent.gate_from_rows([[0x3400, 0x3380]] * 6)
    cases.append(RouteCase("projection-subnormal", (0x0080,) * 2, gate, (0,) * 6))
    for n in (6, 9):
        gate = independent.gate_from_rows([[0]] * n)
        cases.append(RouteCase(f"tie-{n}", (0,), gate, (0, 0x80000000) + (0,) * (n - 2)))
    gate = independent.gate_from_rows([[independent.bf16(Fraction(-100))]] * 6)
    cases.append(RouteCase("epsilon-dominant", (0x3F80,), gate, (0,) * 6))
    gate = independent.gate_from_rows([[0]] * 9)
    bias = tuple(independent.fp32(Fraction(i) * independent.fmt.power2(-24)) for i in range(9))
    cases.append(RouteCase("cutoff-neighbors", (0,), gate, bias))
    negative = tuple(independent.fp32(Fraction(-i - 2)) for i in range(9))
    cases.append(RouteCase("negative-selection", (0,), gate, negative))
    for value in (-128, -104, -103, 20, 21):
        gate = independent.gate_from_rows([[independent.bf16(Fraction(value))]] * 6)
        cases.append(RouteCase(f"nonlinear-{value}", (0x3F80,), gate, (0,) * 6))
    inputs, gate, bias = independent.environment_fixture()
    cases.append(RouteCase("tiny-weight", inputs, gate, bias))
    rng = random.Random(24)
    for number, (n, k) in enumerate(((7, 4), (11, 17), (6, 33))):
        inputs = tuple(independent.bf16(Fraction(rng.randrange(-16, 17), 16)) for _ in range(k))
        rows = [[independent.bf16(Fraction(rng.randrange(-16, 17), 8)) for _ in range(k)]
                for _ in range(n)]
        gate = independent.gate_from_rows(rows)
        bias = tuple(independent.fp32(Fraction(rng.randrange(-8, 9), 4)) for _ in range(n))
        cases.append(RouteCase(f"matrix-{number}", inputs, gate, bias))
    return [dataclasses.replace(case, name=case.name + suffix, traced=traced)
            for case in cases for traced, suffix in ((False, ".ordinary"), (True, ".trace"))]


class Parity(unittest.TestCase):
    def test_directed_pipeline_and_independent_anchors(self):
        cases = directed_cases()
        responses = execute(EXE, cases, EXE_ARGS)
        for case, response in zip(cases, responses):
            with self.subTest(case=case.name):
                want = oracle_trace(case)
                self.assertEqual(want, independent.expected_route(
                    case.inputs, case.gate, case.bias, image_token=case.image,
                    bias_vl_bits=case.vl))
                self.assertEqual(mismatches(case, response, want), [])

    def test_sqrt_and_division_final_grid_rounding(self):
        values = (0, 0x80000000, 1, 2, 3, 0x007FFFFF, 0x00800000, 0x3F800000,
                  0x3F800001, 0x40000000, 0x40800000, 0x7F7FFFFF)
        rng = random.Random(2400)
        values += tuple(rng.randrange(0x7F800000) for _ in range(128))
        cases = [scalar(f"sqrt-{bits:x}", "sqrt", bits) for bits in values]
        expected = [independent.expected_sqrt(bits) for bits in values]
        pairs = [(a, b) for a in (0, 0x80000000, 1, 3, 0x007FFFFF, 0x00800000,
                                  0x3F800001, 0x7F7FFFFF, 0x80000001)
                 for b in (1, 0x007FFFFF, 0x00800000, 0x3F000000, 0x3F800000,
                           0x3FC00000, 0x40000000, 0x7F7FFFFF, 0xBF800000)]
        pairs += [(1, 0x40000000), (3, 0x40000000), (0x00FFFFFF, 0x40000000)]
        rng = random.Random(2401)
        for _ in range(96):
            a, b = rng.randrange(0xFF000000), rng.randrange(0xFF000000)
            if (a & 0x7FFFFFFF) >= 0x7F800000 or (b & 0x7FFFFFFF) >= 0x7F800000 or not b & 0x7FFFFFFF:
                continue
            pairs.append((a, b))
        cases += [scalar(f"div-{a:x}-{b:x}", "div", a, b) for a, b in pairs]
        expected += [independent._div(a, b) for a, b in pairs]
        responses = execute(EXE, cases, EXE_ARGS)
        for case, response, want in zip(cases, responses, expected):
            with self.subTest(case=case.name):
                self.assertEqual(response, {"value": (want,)})

    def test_softplus_certified_bounds_and_decimal_reference(self):
        values = (0, 0x80000000, 1, 0x80000001, 0x41A00000, 0x419FFFFF,
                  0x41A00001, 0xC3000000, 0xC2CFFFFF, 0xC2D00000,
                  0xC2CE0000, 0xC2A00000, 0x3F800000, 0xBF800000,
                  0x7F7FFFFF, 0xFF7FFFFF)
        cases = [scalar(f"softplus-{bits:x}", "softplus", bits, options="precision=12288")
                 for bits in values]
        responses = execute(EXE, cases, EXE_ARGS)
        for bits, response in zip(values, responses):
            want = oracle._softplus_fp32_bits(bits)
            self.assertEqual(response, {"value": (want,)}, hex(bits))
            self.assertEqual(want, independent.expected_softplus(bits))
        bounds = [scalar(f"bounds-{bits:x}", "bounds", bits, options="precision=192") for bits in values]
        for bits, response in zip(values, execute(EXE, bounds, EXE_ARGS)):
            self.assertEqual(set(response), {"lower", "upper"})
            want = oracle._softplus_fp32_bits(bits)
            self.assertLessEqual(response["lower"][0], want)
            self.assertGreaterEqual(response["upper"][0], want)
            self.assertEqual(response["lower"], response["upper"], hex(bits))
        for precision in (384, 768, 1536, 3072, 6144, 12288):
            request = scalar(f"capacity-{precision}", "bounds", 0xC2FE0000, options=f"precision={precision}")
            response = execute(EXE, [request], EXE_ARGS)[0]
            self.assertEqual(response, {"lower": (0,), "upper": (0,)})

    def test_environment_control_and_sticky_states(self):
        available = {case.name: case for case in directed_cases()}
        selected = [available[name] for name in ("tiny-weight.trace", "projection-order.ordinary",
                    "projection-subnormal.trace", "image.trace", "nonlinear--128.trace")]
        bad = dataclasses.replace(selected[0], name="unused-bad-bias", vl=(0,) * 5 + (0x7F800000,))
        overflow = independent.overflow_fixtures()[0]
        rejected = RouteCase("product-overflow", *overflow)
        budget = dataclasses.replace(selected[1], name="budget", traced=True, options="precision=0")
        cases = [Request("before", "mxcsr :"), *selected, bad, rejected, budget,
                 scalar("subnormal-div", "div", 3, 0x40000000), Request("after", "mxcsr :")]
        for env, control in ENVIRONMENTS.items():
            for status in (0, MXCSR_STATUS):
                args = [*EXE_ARGS, f"--mxcsr={env}", f"--status={status:x}"]
                responses = execute(EXE, cases, args)
                self.assertEqual(responses[0], {"value": (control | status,)})
                self.assertEqual(responses[-1], {"value": (control | status,)})
                for case, response in zip(selected, responses[1:1 + len(selected)]):
                    self.assertEqual(mismatches(case, response), [], f"{env}/{status}: {case.name}")
                self.assertEqual(responses[-5]["error"], "invalid_argument")
                self.assertFalse(responses[-5]["changed"])
                self.assertEqual(responses[-4]["error"], "invalid_argument")
                self.assertEqual(responses[-3]["error"], "runtime_error")
                self.assertEqual(responses[-2], {"value": (2,)})


class Contract(unittest.TestCase):
    def assert_rejected(self, cases, kind="invalid_argument", unchanged=True):
        responses = execute(EXE, cases, EXE_ARGS)
        for case, response in zip(cases, responses):
            with self.subTest(case=case.name):
                self.assertEqual(response.get("error"), kind, response)
                self.assertFalse(response["inputchanged"])
                self.assertFalse(response["guardchanged"])
                self.assertFalse(response["tracechanged"])
                if unchanged:
                    self.assertFalse(response["changed"])

    def test_dimensions_lengths_capacities_and_all_contents(self):
        base = directed_cases()[0]
        cases = []
        for traced in (False, True):
            c = dataclasses.replace(base, traced=traced)
            for n, k in ((5, 1), (6, 0)):
                cases.append(dataclasses.replace(c, name=f"shape-{n}-{k}", gate=oracle.GateMatrix(n, k, c.gate.weight_bf16)))
            for member in ("inputs", "bias", "vl"):
                sequence = getattr(c, member)
                for changed in (sequence[:-1], sequence + (0,)):
                    cases.append(dataclasses.replace(c, name=f"length-{member}", **{member: changed}))
            for weights in (c.gate.weight_bf16[:-1], c.gate.weight_bf16 + (0,)):
                cases.append(dataclasses.replace(c, name="gate-length", gate=dataclasses.replace(c.gate, weight_bf16=weights)))
            for index in range(5):
                caps = [c.gate.n + 2] * 5
                caps[index] = c.gate.n - 1
                cases.append(dataclasses.replace(c, name=f"capacity-{index}", caps=tuple(caps)))
            for member, bads in (("inputs", (0x7F80, 0xFF80, 0x7FC0)),
                                 ("bias", (0x7F800000, 0xFF800000, 0x7FC00000)),
                                 ("vl", (0x7F800000, 0xFF800000, 0x7FC00000))):
                for bad in bads:
                    values = getattr(c, member)
                    cases.append(dataclasses.replace(c, name=f"content-{member}", **{member: values[:-1] + (bad,)}))
            for bad in (0x7F80, 0xFF80, 0x7FC0):
                cases.append(dataclasses.replace(c, name="gate-content", gate=dataclasses.replace(
                    c.gate, weight_bf16=c.gate.weight_bf16[:-1] + (bad,))))
            cases.append(dataclasses.replace(c, name="present-empty-vision", vl=None,
                                             options="vl_present=1"))
        cases.extend(dataclasses.replace(base, name=f"invalid-certification-budget-{precision}",
                     traced=True, options=f"precision={precision}") for precision in (1, 191, 193, 24576))
        self.assert_rejected(cases)
        overflow = RouteCase("precheck-before-overflow", *independent.overflow_fixtures()[0],
                             vl=(0,) * 5 + (0x7F800000,))
        self.assert_rejected([overflow])

    def test_numeric_and_certification_failures_do_not_return_partial_results(self):
        cases = [RouteCase(f"overflow-{i}", *data) for i, data in enumerate(independent.overflow_fixtures())]
        self.assert_rejected(cases, unchanged=False)
        base = dataclasses.replace(directed_cases()[1], options="precision=0")
        self.assert_rejected([base], kind="runtime_error", unchanged=False)
        cases = [scalar("bad-softplus", "softplus", 0x7F800000, options="precision=12288"),
                 scalar("bad-sqrt", "sqrt", 0xBF800000), scalar("nan-sqrt", "sqrt", 0x7FC00000),
                 scalar("zero-denominator", "div", 0, 0), scalar("nonfinite-div", "div", 0x7F800000, 0x3F800000)]
        self.assert_rejected(cases)
        self.assert_rejected([scalar("budget-zero", "softplus", 0, options="precision=0")],
                             kind="runtime_error")

    def test_typed_overlaps_read_sharing_and_adjacency(self):
        gate = independent.gate_from_rows([[0]] * 8)
        base = RouteCase("overlap", (0,), gate, (0,) * 8, (0,) * 8)
        cases = []
        for traced in (False, True):
            for left in ("bias", "bias_vl", *STAGES):
                for right in STAGES:
                    if left == right:
                        continue
                    for a, b in ((0, 0), (0, 1), (1, 0), (0, 7)):
                        cases.append(dataclasses.replace(base, name=f"overlap-{left}-{right}-{a}-{b}",
                                     traced=traced, options=f"overlap={left},{right} in_offset={a} out_offset={b}"))
        self.assert_rejected(cases)
        good = []
        for left, right in (("bias", "bias_vl"), ("bias", "linear"), ("linear", "temp")):
            offsets = ((0, 0),) if right == "bias_vl" else ((0, 8), (8, 0))
            for a, b in offsets:
                good.append(dataclasses.replace(base, name=f"adjacent-{left}-{right}-{a}-{b}",
                            caps=(8,) * 5, options=f"overlap={left},{right} in_offset={a} out_offset={b}"))
        good.extend(dataclasses.replace(base, name=f"unused-tail-sharing-{a}-{b}",
                    options=f"overlap=linear,temp in_offset={a} out_offset={b}")
                    for a, b in ((0, 8), (8, 0)))
        for case, response in zip(good, execute(EXE, good, EXE_ARGS)):
            self.assertEqual(mismatches(case, response), [])

    def test_byte_storage_counts_and_sanitizer_preflight(self):
        maximum = (1 << 64) - 1
        cases = [Request("count-NK-overflow", f"counts n=6 k={maximum} :"),
                 Request("count-work-bytes", f"counts n={maximum // 4 + 1} k=1 :"),
                 Request("count-weight-bytes", f"counts n=6 k={maximum // 12 + 1} :")]
        self.assert_rejected(cases)
        good, bad = [], []
        for read_size, write_size in ((2, 4), (4, 4), (64, 4), (4, 64)):
            for offset in range(read_size + 1):
                request = Request(f"byte-{read_size}-{write_size}-{offset}",
                                  f"storage read_size={read_size} write_size={write_size} read_offset=0 write_offset={offset} :")
                (good if offset == read_size else bad).append(request)
        self.assert_rejected(bad)
        self.assertEqual(execute(EXE, good, EXE_ARGS), [{}] * len(good))
        asan = execute(EXE, [Request("asan", "asan :")], EXE_ARGS)[0]
        self.assertIn(asan["value"], ((0,), (1,)))
        if asan["value"] == (1,):
            base = directed_cases()[0]
            invalid = dataclasses.replace(base, name="poisoned-precheck", caps=(base.gate.n - 1,) * 5,
                                          options="poison")
            self.assert_rejected([invalid])
            proc = subprocess.run([str(EXE), *EXE_ARGS], input="poison_probe :\n",
                                  capture_output=True, text=True, timeout=30)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("AddressSanitizer", proc.stderr)


class Harness(unittest.TestCase):
    def test_process_protocol_and_wrong_results_are_not_accepted(self):
        case = directed_cases()[0]
        for fault in ("crash", "exit", "malformed", "missing"):
            with self.subTest(fault=fault), self.assertRaises(HarnessError):
                execute(EXE, [case], [*EXE_ARGS, f"--fault={fault}"])
        response = execute(EXE, [case], [*EXE_ARGS, "--fault=wrong"])[0]
        self.assertTrue(mismatches(case, response))
        for text in ("wat :", "sqrt : 100000000", "div : 1", "route n=6 k=1 image=2 vl=0 trace=0 caps=6,6,6,6,6 :"):
            with self.subTest(text=text), self.assertRaises(HarnessError):
                execute(EXE, [Request("bad-protocol", text)], EXE_ARGS)

    def test_missing_nonexecutable_and_fake_executor_fail_closed(self):
        case = directed_cases()[0]
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing"
            with self.assertRaises(HarnessError):
                execute(missing, [case])
            missing.write_text("not executable\n")
            with self.assertRaises(HarnessError):
                execute(missing, [case])
        with self.assertRaises(HarnessError):
            execute(Path("/bin/true"), [case])
        with self.assertRaises(HarnessError):
            parse_response(case, "ok ; value=100000000")


class Combination(unittest.TestCase):
    def test_actual_production_indices_and_weights_drive_experts(self):
        self.assertIsNotNone(EXPERT_EXE, "--expert-exe is required for combination")
        inputs, gate, bias, vl, experts = integration.fixture()
        requests = [RouteCase("production-text", inputs, gate, bias, vl),
                    RouteCase("production-image", inputs, gate, bias, vl, True)]
        expert_cases = []
        for case, response in zip(requests, execute(EXE, requests, EXE_ARGS)):
            want = oracle_trace(case)
            self.assertEqual(mismatches(case, response, want), [])
            for slot, (actual_id, actual_weight) in enumerate(zip(response["indices"], response["weights"])):
                expected = independent_expert.check_expert(self, (inputs, *experts[want["indices"][slot]]),
                                                            want["weights_bits"][slot])
                trace = integration.expert_trace(expected)
                for traced in (False, True):
                    expert_cases.append(expert_parity.ExpertCase(
                        f"{case.name}.{slot}.expert={actual_id}", inputs, *experts[actual_id],
                        route=actual_weight, traced=traced, expected_trace=trace))
        responses = expert_parity.execute(EXPERT_EXE, expert_cases)
        self.assertEqual(len(responses), 24)
        self.assertEqual([error for case, response in zip(expert_cases, responses)
                          if (error := expert_parity.check_expert(case, response))], [])

    def test_wrong_mapping_and_weight_complete_with_numeric_differences(self):
        inputs, gate, bias, vl, experts = integration.fixture()
        request = RouteCase("wiring-negative-control", inputs, gate, bias, vl)
        response = execute(EXE, [request], EXE_ARGS)[0]
        self.assertEqual(mismatches(request, response), [])
        cases = []
        for slot, (index, weight) in enumerate(zip(response["indices"], response["weights"])):
            other_id = response["indices"][(slot + 1) % 6]
            other_weight = response["weights"][(slot + 1) % 6]
            expected = independent_expert.check_expert(self, (inputs, *experts[index]), weight)
            for label, matrices, route in (("id", experts[other_id], weight),
                                           ("weight", experts[index], other_weight)):
                cases.append(expert_parity.ExpertCase(f"wrong-{label}-{slot}", inputs, *matrices,
                             route=route, expected_trace=integration.expert_trace(expected)))
        responses = expert_parity.execute(EXPERT_EXE, cases)
        for case, result in zip(cases, responses):
            self.assertEqual(result[0], "ok")
            self.assertIsNotNone(expert_parity.check_expert(case, result))


SUITES = {"parity": (Parity,), "contract": (Contract,), "harness": (Harness,),
          "combination": (Combination,)}


def main(argv=None):
    global EXE, EXPERT_EXE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument("--expert-exe", type=Path)
    parser.add_argument("--suite", choices=SUITES, default="parity")
    parser.add_argument("--exe-arg", action="append", default=[])
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--numeric-report", action="store_true")
    args = parser.parse_args(argv)
    EXE = args.exe.resolve()
    EXPERT_EXE = args.expert_exe.resolve() if args.expert_exe else None
    EXE_ARGS.extend(args.exe_arg)
    if args.numeric_report:
        results, errors = [], []
        cases = [case for case in directed_cases() if not args.case or
                 any(token in case.name for token in args.case)]
        if not cases:
            errors.append("no cases selected")
        try:
            for case, response in zip(cases, execute(EXE, cases, EXE_ARGS)):
                results.append({"case": case.name, "mismatches": mismatches(case, response)})
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
        print("P0024_NUMERIC " + json.dumps({"cases": results, "errors": errors, "skipped": 0}))
        return int(bool(errors or any(item["mismatches"] for item in results)))
    loader = unittest.TestLoader()
    suite = unittest.TestSuite(loader.loadTestsFromTestCase(cls) for cls in SUITES[args.suite])
    if args.case:
        suite = unittest.TestSuite(test for test in integration.iter_tests(suite)
                                   if any(token in test.id() for token in args.case))
    if not suite.countTestCases():
        print("no tests selected", file=sys.stderr)
        return 2
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() and result.testsRun and not result.skipped else 1


if __name__ == "__main__":
    sys.exit(main())
