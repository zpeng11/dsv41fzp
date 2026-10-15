#!/usr/bin/env python3
"""P0025 directed routed-branch oracle acceptance.

Only this module is collected.  Existing router/expert test modules are imported
for independent expectations and deterministic fixtures, not for their suites.
New aggregation expectations use exact Fraction arithmetic plus an independent
FP32 grid search and implement IEEE special-value classes locally.
"""

import argparse
import contextlib
import ctypes
import dataclasses
import decimal
import platform
import sys
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
QUANT_TESTS = ROOT / "tests" / "quant"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(QUANT_TESTS))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_expert as independent_expert
import test_formats as fmt
import test_router as independent_router
from ref import expert as expert_impl
from ref import moe as impl
from ref import router as router_impl

SIGN = 0x80000000
INF = 0x7F800000
QNAN = 0x7FC00000
D = 32
H = 32
N = 9


def _is_nan(bits):
    return (bits & 0x7FFFFFFF) > INF


def _finite_value(bits):
    value = fmt.expected_binary(bits, 23)
    if value is None:
        raise ValueError("finite FP32 bit pattern required")
    return value


def expected_add(left, right):
    """Independent IEEE binary32 RNE addition, including contracted specials."""
    if _is_nan(left) or _is_nan(right):
        return QNAN
    left_mag, right_mag = left & 0x7FFFFFFF, right & 0x7FFFFFFF
    if left_mag == INF or right_mag == INF:
        if left_mag == right_mag == INF and (left ^ right) & SIGN:
            return QNAN
        return left if left_mag == INF else right
    exact = _finite_value(left) + _finite_value(right)
    negative_zero = exact == 0 and left_mag == right_mag == 0 and bool(left & right & SIGN)
    return fmt.nearest_fp32_bits(exact, negative_zero)


def expected_accumulate(accumulator, output_bf16):
    """Widen BF16 exactly and perform one independent RNE add per component."""
    if len(accumulator) != len(output_bf16):
        raise ValueError("length mismatch in test expectation")
    return tuple(expected_add(left, right << 16)
                 for left, right in zip(accumulator, output_bf16))


def expected_partials(outputs, d=D):
    accumulator = (0,) * d
    partials = []
    for output in outputs:
        accumulator = expected_accumulate(accumulator, output)
        partials.append(accumulator)
    return tuple(partials), accumulator


def assert_fp32_vectors(test, actual, expected, label="FP32 vector"):
    """Compare finite/zero/infinity bits exactly and NaNs by classification."""
    test.assertIs(type(actual), tuple, label)
    test.assertEqual(len(actual), len(expected), label)
    for index, (got, want) in enumerate(zip(actual, expected)):
        if _is_nan(want):
            test.assertTrue(_is_nan(got), f"{label}[{index}] expected NaN, got {got:#x}")
        else:
            test.assertEqual(got, want, f"{label}[{index}]")


def assert_fp32_partials(test, actual, expected, label="partial sums"):
    test.assertIs(type(actual), tuple, label)
    test.assertEqual(len(actual), len(expected), label)
    for index, (got, want) in enumerate(zip(actual, expected)):
        assert_fp32_vectors(test, got, want, f"{label}[{index}]")


def gate_matrix():
    """Nine distinct rows whose all-one-input projections are one through nine."""
    row_codes = (0x3D00, 0x3D80, 0x3DC0, 0x3E00, 0x3E20,
                 0x3E40, 0x3E60, 0x3E80, 0x3E90)
    return router_impl.GateMatrix(N, D, tuple(code for code in row_codes for _ in range(D)))


def expert_matrices():
    """ID-distinct triples so wrong expert and weight wiring survive BF16 boundaries."""
    return tuple(
        (independent_expert.matrix(H, D, 1 + 3 * index, scale=119 + index),
         independent_expert.matrix(D, H, 2 + 5 * index, scale=120 + index),
         independent_expert.matrix(H, D, 7 + 7 * index, scale=121 + index))
        for index in range(N)
    )


def fixture():
    inputs = (0x3F80,) * D
    text_bias = (0x41800000,) * 6 + (0xC1800000,) * 3
    image_bias = (0xC1800000,) * 3 + (0x41800000,) * 6
    experts = tuple(impl.FP4Expert(*matrices) for matrices in expert_matrices())
    return inputs, gate_matrix(), text_bias, image_bias, experts


def zero_matrix(n, k):
    return independent_expert.constant_matrix(n, k, 0, 127)


def zero_fixture(n=6):
    inputs = (0,) * D
    gate = router_impl.GateMatrix(n, D, (0,) * (n * D))
    bias = (0,) * n
    experts = tuple(
        impl.FP4Expert(
            independent_expert.constant_matrix(H, D, 0, 118 + index),
            independent_expert.constant_matrix(D, H, 0, 128 + index),
            independent_expert.constant_matrix(H, D, 0, 138 + index),
        )
        for index in range(n)
    )
    return inputs, gate, bias, experts


def expert_trace(expected):
    bridge = expected["bridge"]
    return expert_impl.ExpertTrace(
        expected["input_codes"], expected["input_scales"],
        expected["gate_bf16"], expected["up_bf16"],
        expert_impl.BridgeTrace(
            bridge["gate_bits"], bridge["up_bits"], bridge["silu_bits"],
            bridge["gated_bits"], bridge["weighted_bits"], bridge["bf16_bits"]),
        expected["hidden_codes"], expected["hidden_scales"],
        expected["output_bf16"])


def expected_combination(inputs, gate, bias, experts, *, image_token=False,
                         bias_vl_bits=None):
    route = independent_router.expected_route(
        inputs, gate, bias, image_token=image_token, bias_vl_bits=bias_vl_bits)
    paired = sorted(zip(route["indices"], route["weights_bits"]))
    indices, traces, outputs = [], [], []
    for index, weight in paired:
        item = experts[index]
        expected = independent_expert.expected_expert(
            inputs, item.w1, item.w2, item.w3, weight)
        indices.append(index)
        traces.append(expert_trace(expected))
        outputs.append(expected["output_bf16"])
    partials, output = expected_partials(outputs, gate.k)
    return route, tuple(indices), tuple(traces), partials, output


def assert_route_trace(test, actual, expected):
    test.assertIsInstance(actual, router_impl.RouteTrace)
    for field in independent_router.TRACE_FIELDS:
        test.assertEqual(getattr(actual, field), expected[field], f"route.{field}")


def assert_expert_trace(test, actual, expected, label):
    test.assertIsInstance(actual, expert_impl.ExpertTrace, label)
    test.assertEqual(actual, expected, label)


def _aliases(module, original):
    return [name for name, value in vars(module).items() if value is original]


@contextlib.contextmanager
def patch_dependency(owner, name, replacement):
    """Patch a dependency whether moe retained its module or a direct function alias."""
    original = getattr(owner, name)
    aliases = _aliases(impl, original)
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(owner, name, replacement))
        for alias in aliases:
            stack.enter_context(patch.object(impl, alias, replacement))
        yield


@contextlib.contextmanager
def patched_upstream(*, route=None, trace_route=None, expert=None, trace_expert=None):
    replacements = ((router_impl, "route", route),
                    (router_impl, "trace_route", trace_route),
                    (expert_impl, "expert", expert),
                    (expert_impl, "trace_expert", trace_expert))
    with contextlib.ExitStack() as stack:
        for owner, name, replacement in replacements:
            if replacement is not None:
                stack.enter_context(patch_dependency(owner, name, replacement))
        yield


def expert_id(item, experts):
    fingerprint = (item.n, item.k, tuple(item.w_packed), tuple(item.scale_b))
    for index, candidate in enumerate(experts):
        matrix = candidate.w1
        if fingerprint == (matrix.n, matrix.k, tuple(matrix.w_packed),
                           tuple(matrix.scale_b)):
            return index
    raise AssertionError("unknown expert matrix")


def dummy_route_trace(indices, weights):
    zeros = (0,) * max(indices.__len__(), 6)
    return router_impl.RouteTrace(
        zeros, zeros, zeros, zeros, zeros, tuple(indices),
        tuple(weights), 0, 0x1E3CE508, tuple(weights), tuple(weights))


def dummy_expert_trace(output):
    empty = ()
    bridge = expert_impl.BridgeTrace(empty, empty, empty, empty, empty, empty)
    return expert_impl.ExpertTrace(empty, empty, empty, empty, bridge, empty, empty,
                                   tuple(output))


class PublicCombinationTests(unittest.TestCase):
    def test_public_types_are_frozen_and_shape_is_exact(self):
        _, _, _, _, experts = fixture()
        self.assertTrue(dataclasses.is_dataclass(experts[0]))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            experts[0].w1 = experts[0].w2

    def test_real_text_and_image_composition_trace_and_id_order(self):
        inputs, gate, bias, vision, experts = fixture()
        for image_token in (False, True):
            with self.subTest(image_token=image_token):
                route, indices, traces, partials, output = expected_combination(
                    inputs, gate, bias, experts, image_token=image_token,
                    bias_vl_bits=vision)
                actual = impl.routed(inputs, gate, bias, experts,
                                     image_token=image_token, bias_vl_bits=vision)
                traced = impl.trace_routed(inputs, gate, bias, experts,
                                           image_token=image_token,
                                           bias_vl_bits=vision)
                self.assertIs(type(actual), tuple)
                self.assertEqual(actual, output)
                self.assertIsInstance(traced, impl.RoutedTrace)
                self.assertTrue(dataclasses.is_dataclass(traced))
                with self.assertRaises(dataclasses.FrozenInstanceError):
                    traced.output_bits = ()
                assert_route_trace(self, traced.route, route)
                self.assertEqual(traced.expert_indices, indices)
                self.assertEqual(indices, tuple(sorted(route["indices"])))
                self.assertNotEqual(indices, route["indices"])
                self.assertEqual(len(traced.expert_traces), 6)
                for position, (got, want) in enumerate(zip(traced.expert_traces, traces)):
                    assert_expert_trace(self, got, want, f"expert[{position}]")
                self.assertEqual(traced.partial_sums_bits, partials)
                self.assertEqual(traced.output_bits, output)
                self.assertEqual(actual, traced.output_bits)

    def test_real_components_receive_router_pairs_and_run_in_id_order(self):
        inputs, gate, bias, vision, experts = fixture()
        expected_route = independent_router.expected_route(
            inputs, gate, bias, image_token=True, bias_vl_bits=vision)
        expected_pairs = tuple(sorted(zip(expected_route["indices"],
                                          expected_route["weights_bits"])))
        ordinary_calls, trace_calls = [], []
        original_expert, original_trace = expert_impl.expert, expert_impl.trace_expert

        def observe_expert(values, w1, w2, w3, *, route_weight_bits=None):
            ordinary_calls.append((expert_id(w1, experts), route_weight_bits))
            return original_expert(values, w1, w2, w3,
                                   route_weight_bits=route_weight_bits)

        def observe_trace(values, w1, w2, w3, *, route_weight_bits=None):
            trace_calls.append((expert_id(w1, experts), route_weight_bits))
            return original_trace(values, w1, w2, w3,
                                  route_weight_bits=route_weight_bits)

        with patched_upstream(expert=observe_expert, trace_expert=observe_trace):
            impl.routed(inputs, gate, bias, experts, image_token=True,
                        bias_vl_bits=vision)
            impl.trace_routed(inputs, gate, bias, experts, image_token=True,
                              bias_vl_bits=vision)
        self.assertEqual(tuple(ordinary_calls), expected_pairs)
        self.assertEqual(tuple(trace_calls), expected_pairs)


class AggregateTests(unittest.TestCase):
    def test_private_accumulate_exact_widen_specials_and_boundaries(self):
        cases = (
            ((0,), (0x0001,), (0x00010000,), "positive BF16 subnormal"),
            ((0,), (0x8001,), (0x80010000,), "negative BF16 subnormal"),
            ((0,), (0x8000,), (0,), "initial +0 plus -0"),
            ((SIGN,), (0x8000,), (SIGN,), "-0 plus -0"),
            ((0x3F800000,), (0x3380,), (0x3F800000,), "halfway even down"),
            ((0x3F800001,), (0x3380,), (0x3F800002,), "halfway odd up"),
            ((0x3F800000,), (0xBF80,), (0,), "exact cancellation"),
            ((0x7F7FFFFF,), (0x7F7F,), (INF,), "overflow"),
            ((INF,), (0xFF80,), (QNAN,), "opposite infinities"),
            ((0xFF800000,), (0x3F80,), (0xFF800000,), "infinity propagation"),
            ((0,), (0x7FC1,), (QNAN,), "NaN propagation"),
        )
        for accumulator, output, expected, label in cases:
            with self.subTest(label=label):
                assert_fp32_vectors(
                    self, expected_accumulate(accumulator, output), expected,
                    f"independent {label}")
                assert_fp32_vectors(
                    self, impl._accumulate(accumulator, output), expected,
                    f"implementation {label}")
        vector_acc = (0, SIGN, 0x3F800000, 0x7F7FFFFF)
        vector_out = (0x0001, 0x8000, 0xBF80, 0x7F7F)
        self.assertEqual(impl._accumulate(vector_acc, vector_out),
                         expected_accumulate(vector_acc, vector_out))

    def test_partial_sums_are_each_rne_step_not_exact_or_early_bf16(self):
        cancellation = ((0x4B80,), (0x3F80,), (0xCB80,))
        accumulator = (0,)
        actual_partials = []
        for output in cancellation:
            accumulator = impl._accumulate(accumulator, output)
            actual_partials.append(accumulator)
        expected_cancellation = ((0x4B800000,), (0x4B800000,), (0,))
        self.assertEqual(tuple(actual_partials), expected_cancellation)
        exact_once = fmt.nearest_fp32_bits(
            sum((_finite_value(code[0] << 16) for code in cancellation),
                Fraction(0)))
        self.assertEqual(exact_once, 0x3F800000)
        self.assertNotEqual(accumulator, (exact_once,))

        outputs = ((0x3F80,), (0x3B80,), (0x3B80,))
        partials, correct = expected_partials(outputs, 1)
        self.assertEqual(partials, ((0x3F800000,), (0x3F808000,), (0x3F810000,)))
        early = (0,)
        for output in outputs:
            early = expected_accumulate(early, output)
            early = tuple(fmt.nearest_bf16_bits(bits) << 16 for bits in early)
        self.assertEqual(early, (0x3F800000,))
        self.assertNotEqual(early, correct)

    def test_mock_boundary_final_nonfinite_and_overflow_are_returned(self):
        """Directed aggregate boundary: real composition is covered separately above."""
        inputs, gate, bias, experts = zero_fixture()
        indices, weights = tuple(range(6)), (0x3F800000,) * 6
        outputs = {
            0: (0x7F7F,) + (0,) * 31,
            1: (0x7F7F,) + (0,) * 31,
            2: (0xFF80,) + (0,) * 31,
            3: (0x7F80,) + (0,) * 31,
            4: (0x7FC1,) + (0,) * 31,
            5: (0,) * 32,
        }
        route_trace = dummy_route_trace(indices, weights)

        def fake_route(*args, **kwargs):
            return indices, weights

        def fake_trace_route(*args, **kwargs):
            return route_trace

        def fake_expert(values, w1, w2, w3, *, route_weight_bits=None):
            return outputs[expert_id(w1, experts)]

        def fake_trace(values, w1, w2, w3, *, route_weight_bits=None):
            return dummy_expert_trace(outputs[expert_id(w1, experts)])

        ordered = tuple(outputs[index] for index in indices)
        partials, expected = expected_partials(ordered)
        with patched_upstream(route=fake_route, trace_route=fake_trace_route,
                              expert=fake_expert, trace_expert=fake_trace):
            ordinary = impl.routed(inputs, gate, bias, experts)
            traced = impl.trace_routed(inputs, gate, bias, experts)
        assert_fp32_vectors(self, ordinary, expected, "nonfinite ordinary output")
        assert_fp32_partials(self, traced.partial_sums_bits, partials,
                             "nonfinite trace partials")
        assert_fp32_vectors(self, traced.output_bits, expected,
                            "nonfinite trace output")
        self.assertTrue(_is_nan(expected[0]))


class ValidationAndSnapshotTests(unittest.TestCase):
    def entries(self):
        return (impl.routed, impl.trace_routed)

    def test_expert_collection_and_cross_shape_validation_precede_route(self):
        inputs, gate, bias, _, experts = fixture()
        malformed = expert_impl.FP4Matrix(H, D, (), ())
        bad_unselected = list(experts)
        bad_unselected[-1] = impl.FP4Expert(malformed, experts[-1].w2,
                                            experts[-1].w3)
        wrong_h = impl.FP4Expert(
            independent_expert.matrix(64, D, 3),
            independent_expert.matrix(D, 64, 4),
            independent_expert.matrix(64, D, 5))
        mismatched = list(experts)
        mismatched[-1] = wrong_h

        def route_started(*args, **kwargs):
            raise AssertionError("route started before complete expert validation")

        for entry in self.entries():
            for collection, error in (
                    (experts[:-1], ValueError), (experts + (experts[0],), ValueError),
                    (iter(experts), TypeError), (5, TypeError),
                    (tuple(bad_unselected), ValueError), (tuple(mismatched), ValueError)):
                with self.subTest(entry=entry.__name__, kind=type(collection).__name__):
                    with patched_upstream(route=route_started,
                                          trace_route=route_started):
                        with self.assertRaises(error):
                            entry(inputs, gate, bias, collection)
        two_d = memoryview(bytearray(N * 2)).cast("H", shape=[N, 1])
        for entry in self.entries():
            with self.assertRaises(TypeError):
                entry(inputs, gate, bias, two_d)

    def test_all_matrix_encodings_and_unused_bias_are_validated(self):
        inputs, gate, bias, vision, experts = fixture()
        invalid_scale = dataclasses.replace(
            experts[-1].w2,
            scale_b=experts[-1].w2.scale_b[:-1] + (0xFF,))
        bad = experts[:-1] + (impl.FP4Expert(experts[-1].w1, invalid_scale,
                                             experts[-1].w3),)
        for entry in self.entries():
            with self.subTest(entry=entry.__name__, material="unselected-expert"):
                with self.assertRaises(ValueError):
                    entry(inputs, gate, bias, bad)
            with self.subTest(entry=entry.__name__, material="unused-vision-bias"):
                with self.assertRaises(ValueError):
                    entry(inputs, gate, bias, experts, image_token=False,
                          bias_vl_bits=vision[:-1] + (INF,))
            with self.subTest(entry=entry.__name__, material="unused-text-bias"):
                with self.assertRaises(ValueError):
                    entry(inputs, gate, bias[:-1] + (INF,), experts,
                          image_token=True, bias_vl_bits=vision)

    def test_gate_d_matrix_orientation_and_common_h_contract(self):
        inputs, gate, bias, _, experts = fixture()
        d64 = router_impl.GateMatrix(N, 64, (0,) * (N * 64))
        cases = []
        cases.append(((0,) * 64, d64, experts, "gate D differs from experts"))
        wrong_w2 = impl.FP4Expert(experts[0].w1,
                                  independent_expert.matrix(64, H, 11),
                                  experts[0].w3)
        cases.append((inputs, gate, (wrong_w2,) + experts[1:], "w2 is not D-by-H"))
        wrong_w3 = impl.FP4Expert(experts[0].w1, experts[0].w2,
                                  independent_expert.matrix(64, D, 12))
        cases.append((inputs, gate, (wrong_w3,) + experts[1:], "w3 H mismatch"))
        for entry in self.entries():
            for values, matrix, materials, label in cases:
                with self.subTest(entry=entry.__name__, label=label):
                    with self.assertRaises(ValueError):
                        entry(values, matrix, bias, materials)

    def test_every_caller_sequence_is_snapshotted_before_route_and_unchanged(self):
        base_input, base_gate, base_bias, base_vision, base_experts = fixture()
        inputs = list(base_input)
        gate_weights = list(base_gate.weight_bf16)
        bias, vision = list(base_bias), list(base_vision)
        matrix_storage = []
        mutable_experts = []
        for item in base_experts:
            matrices = []
            for matrix in (item.w1, item.w2, item.w3):
                packed, scales = list(matrix.w_packed), list(matrix.scale_b)
                matrix_storage.append((packed, scales))
                matrices.append(expert_impl.FP4Matrix(matrix.n, matrix.k,
                                                      packed, scales))
            mutable_experts.append(impl.FP4Expert(*matrices))
        gate = router_impl.GateMatrix(base_gate.n, base_gate.k, gate_weights)
        before = (inputs[:], gate_weights[:], bias[:], vision[:],
                  tuple((packed[:], scales[:]) for packed, scales in matrix_storage))
        baseline = impl.routed(tuple(base_input), base_gate, tuple(base_bias),
                               base_experts, image_token=True,
                               bias_vl_bits=tuple(base_vision))
        original_route = router_impl.route
        observed = False

        def mutate_at_route(values, matrix, corrections, **kwargs):
            nonlocal observed
            observed = True
            self.assertIs(type(values), tuple)
            self.assertIs(type(matrix.weight_bf16), tuple)
            self.assertIs(type(corrections), tuple)
            inputs[:] = [0] * len(inputs)
            gate_weights[:] = [0] * len(gate_weights)
            bias[:] = [0] * len(bias)
            vision[:] = [0] * len(vision)
            for packed, scales in matrix_storage:
                packed[:] = [0] * len(packed)
                scales[:] = [127] * len(scales)
            return original_route(values, matrix, corrections, **kwargs)

        with patched_upstream(route=mutate_at_route):
            actual = impl.routed(inputs, gate, bias, mutable_experts,
                                 image_token=True, bias_vl_bits=vision)
        self.assertTrue(observed)
        self.assertEqual(actual, baseline)

        inputs[:] = before[0]
        gate_weights[:] = before[1]
        bias[:] = before[2]
        vision[:] = before[3]
        for (packed, scales), (old_packed, old_scales) in zip(matrix_storage,
                                                              before[4]):
            packed[:] = old_packed
            scales[:] = old_scales
        snapshot = (inputs[:], gate_weights[:], bias[:], vision[:],
                    tuple((packed[:], scales[:]) for packed, scales in matrix_storage))
        impl.routed(inputs, gate, bias, mutable_experts, image_token=True,
                    bias_vl_bits=vision)
        self.assertEqual(
            (inputs, gate_weights, bias, vision,
             tuple((packed, scales) for packed, scales in matrix_storage)), snapshot)

    def test_basic_parameter_types_fail_without_partial_result(self):
        inputs, gate, bias, vision, experts = fixture()
        for entry in self.entries():
            for flag in (0, 1, None, "image"):
                with self.subTest(entry=entry.__name__, flag=flag):
                    with self.assertRaises(TypeError):
                        entry(inputs, gate, bias, experts, image_token=flag,
                              bias_vl_bits=vision)
            with self.assertRaises(TypeError):
                entry(inputs, gate, bias, experts[0])
            with self.assertRaises(TypeError):
                entry(inputs, object(), bias, experts)


class FailurePropagationTests(unittest.TestCase):
    def test_router_runtime_and_certification_failures_propagate(self):
        inputs, gate, bias, _, experts = fixture()
        for entry, dependency in ((impl.routed, "ordinary"),
                                  (impl.trace_routed, "trace")):
            for error in (RuntimeError("router runtime"),
                          ArithmeticError("router certification")):
                def fail(*args, _error=error, **kwargs):
                    raise _error
                replacements = ({"route": fail} if dependency == "ordinary"
                                else {"trace_route": fail})
                with self.subTest(entry=entry.__name__, error=type(error).__name__):
                    with patched_upstream(**replacements):
                        with self.assertRaisesRegex(type(error), str(error)):
                            entry(inputs, gate, bias, experts)

    def test_expert_runtime_and_certification_failures_propagate(self):
        inputs, gate, bias, _, experts = fixture()
        original_expert, original_trace = expert_impl.expert, expert_impl.trace_expert
        for entry, traced in ((impl.routed, False), (impl.trace_routed, True)):
            for error in (RuntimeError("expert runtime"),
                          ArithmeticError("expert certification")):
                calls = 0
                original = original_trace if traced else original_expert

                def fail_second(*args, _error=error, **kwargs):
                    nonlocal calls
                    calls += 1
                    if calls == 2:
                        raise _error
                    return original(*args, **kwargs)

                replacements = ({"trace_expert": fail_second} if traced
                                else {"expert": fail_second})
                with self.subTest(entry=entry.__name__, error=type(error).__name__):
                    with patched_upstream(**replacements):
                        with self.assertRaisesRegex(type(error), str(error)):
                            entry(inputs, gate, bias, experts)
                    self.assertEqual(calls, 2)


class SensitivityTests(unittest.TestCase):
    def test_wrong_id_wrong_weight_duplicate_and_post_w2_weight_mutations(self):
        inputs, gate, bias, vision, experts = fixture()
        baseline = impl.routed(inputs, gate, bias, experts,
                               image_token=True, bias_vl_bits=vision)
        route = independent_router.expected_route(
            inputs, gate, bias, image_token=True, bias_vl_bits=vision)
        original_route, original_expert = router_impl.route, expert_impl.expert

        def run_with_route(indices, weights, expert_call=None):
            def fake_route(*args, **kwargs):
                return tuple(indices), tuple(weights)
            with patched_upstream(route=fake_route,
                                  expert=expert_call if expert_call is not None
                                  else original_expert):
                return impl.routed(inputs, gate, bias, experts,
                                   image_token=True, bias_vl_bits=vision)

        wrong_ids = list(route["indices"])
        replacement = next(index for index in range(gate.n)
                           if index not in wrong_ids)
        wrong_ids[0] = replacement
        wrong_id = run_with_route(wrong_ids, route["weights_bits"])
        self.assertNotEqual(wrong_id, baseline)

        wrong_weights = list(route["weights_bits"])
        wrong_weights[0], wrong_weights[1] = wrong_weights[1], wrong_weights[0]
        wrong_weight = run_with_route(route["indices"], wrong_weights)
        self.assertNotEqual(wrong_weight, baseline)

        def duplicate(values, w1, w2, w3, *, route_weight_bits=None):
            output = original_expert(values, w1, w2, w3,
                                     route_weight_bits=route_weight_bits)
            return tuple(fmt.nearest_bf16_bits(
                independent_expert.expected_mul(code << 16,
                                                route_weight_bits))
                         for code in output)

        duplicate_result = run_with_route(route["indices"],
                                          route["weights_bits"], duplicate)
        self.assertNotEqual(duplicate_result, baseline)

        def post_w2(values, w1, w2, w3, *, route_weight_bits=None):
            output = original_expert(values, w1, w2, w3,
                                     route_weight_bits=None)
            return tuple(fmt.nearest_bf16_bits(
                independent_expert.expected_mul(code << 16,
                                                route_weight_bits))
                         for code in output)

        moved = run_with_route(route["indices"], route["weights_bits"], post_w2)
        self.assertNotEqual(moved, baseline)
        self.assertEqual(impl.routed(inputs, gate, bias, experts,
                                     image_token=True, bias_vl_bits=vision), baseline)

    def test_slot_order_mutation_and_early_bf16_have_completed_counterexamples(self):
        inputs, gate, bias, experts = zero_fixture()
        slot_indices = (0, 2, 1, 3, 4, 5)
        weights = (0x3F800000,) * 6
        values = {
            0: (0x60AD,) + (0,) * 31,
            1: (0xE0AD,) + (0,) * 31,
            2: (0x3F80,) + (0,) * 31,
            3: (0,) * 32, 4: (0,) * 32, 5: (0,) * 32,
        }

        def fake_route(*args, **kwargs):
            return slot_indices, weights

        def fake_expert(input_bf16, w1, w2, w3, *, route_weight_bits=None):
            return values[expert_id(w1, experts)]

        with patched_upstream(route=fake_route, expert=fake_expert):
            correct = impl.routed(inputs, gate, bias, experts)
            with patch.object(impl, "sorted",
                              side_effect=lambda pairs, key=None: tuple(pairs),
                              create=True):
                slot_order_result = impl.routed(inputs, gate, bias, experts)
        id_order = tuple(values[index] for index in sorted(slot_indices))
        slot_order = tuple(values[index] for index in slot_indices)
        _, id_result = expected_partials(id_order)
        _, slot_result = expected_partials(slot_order)
        self.assertEqual(correct, id_result)
        self.assertEqual(slot_order_result, slot_result)
        self.assertNotEqual(slot_order_result, correct)
        with patched_upstream(route=fake_route, expert=fake_expert):
            self.assertEqual(impl.routed(inputs, gate, bias, experts), correct)

        early_outputs = ((0x3F80,) * D, (0x3B80,) * D, (0x3B80,) * D,
                         (0,) * D, (0,) * D, (0,) * D)
        early_values = {index: output for index, output in enumerate(early_outputs)}

        def early_expert(input_bf16, w1, w2, w3, *, route_weight_bits=None):
            return early_values[expert_id(w1, experts)]

        with patched_upstream(route=fake_route, expert=early_expert):
            correct_early_case = impl.routed(inputs, gate, bias, experts)

            def early_accumulate(accumulator, output):
                added = expected_accumulate(accumulator, output)
                return tuple(fmt.nearest_bf16_bits(bits) << 16 for bits in added)

            with patch.object(impl, "_accumulate", side_effect=early_accumulate):
                early = impl.routed(inputs, gate, bias, experts)
        partials, expected_early_case = expected_partials(early_outputs)
        self.assertEqual(correct_early_case, expected_early_case)
        self.assertEqual(correct_early_case, partials[-1])
        self.assertNotEqual(early, correct_early_case)


class EnvironmentTests(unittest.TestCase):
    def test_decimal_ambient_context_is_unchanged_on_all_paths(self):
        inputs, gate, bias, vision, experts = fixture()
        baseline = impl.routed(inputs, gate, bias, experts,
                               image_token=True, bias_vl_bits=vision)
        for rounding in (decimal.ROUND_FLOOR, decimal.ROUND_CEILING):
            with decimal.localcontext() as context:
                context.prec, context.Emin, context.Emax = 2, -2, 2
                context.rounding = rounding
                for signal in context.traps:
                    context.traps[signal] = True
                context.clear_flags()
                context.flags[decimal.Inexact] = True
                paths = ("ordinary", "trace", "parameter", "runtime")
                for path in paths:
                    with self.subTest(rounding=rounding, path=path):
                        before = (context.prec, context.Emin, context.Emax,
                                  context.rounding, dict(context.traps),
                                  dict(context.flags))
                        if path == "ordinary":
                            self.assertEqual(impl.routed(
                                inputs, gate, bias, experts, image_token=True,
                                bias_vl_bits=vision), baseline)
                        elif path == "trace":
                            self.assertEqual(impl.trace_routed(
                                inputs, gate, bias, experts, image_token=True,
                                bias_vl_bits=vision).output_bits, baseline)
                        elif path == "parameter":
                            with self.assertRaises(ValueError):
                                impl.routed(inputs, gate, bias, experts,
                                            bias_vl_bits=vision[:-1] + (INF,))
                        else:
                            def fail(*args, **kwargs):
                                raise ArithmeticError("not certified")
                            with patched_upstream(expert=fail):
                                with self.assertRaises(ArithmeticError):
                                    impl.routed(inputs, gate, bias, experts)
                        after = (context.prec, context.Emin, context.Emax,
                                 context.rounding, dict(context.traps),
                                 dict(context.flags))
                        self.assertEqual(after, before)

    def test_env_a_all_masked_rounding_ftz_daz_sticky_and_restore(self):
        self.assertEqual(sys.platform, "linux", "ENV-A requires Linux")
        self.assertEqual(platform.machine(), "x86_64", "ENV-A requires x86_64")
        self.assertEqual(platform.libc_ver()[0], "glibc", "ENV-A requires glibc")

        class Fenv(ctypes.Structure):
            _fields_ = [("control", ctypes.c_ushort), ("reserved1", ctypes.c_ushort),
                        ("status", ctypes.c_ushort), ("reserved2", ctypes.c_ushort),
                        ("tags", ctypes.c_ushort), ("reserved3", ctypes.c_ushort),
                        ("eip", ctypes.c_uint), ("cs", ctypes.c_ushort),
                        ("opcode", ctypes.c_ushort), ("data", ctypes.c_uint),
                        ("selector", ctypes.c_ushort), ("reserved5", ctypes.c_ushort),
                        ("mxcsr", ctypes.c_uint)]

        self.assertEqual(ctypes.sizeof(Fenv), 32)
        self.assertEqual(Fenv.mxcsr.offset, 28)
        library = ctypes.CDLL("libm.so.6")
        for name in ("fegetenv", "fesetenv"):
            getattr(library, name).argtypes = (ctypes.POINTER(Fenv),)
            getattr(library, name).restype = ctypes.c_int

        injected_inputs, injected_gate, injected_bias, injected_experts = zero_fixture()
        indices, weights = tuple(range(6)), (0x3F800000,) * 6
        outputs = tuple(((0x0001 if index == 0 else 0),) * D
                        for index in range(6))
        route_trace = dummy_route_trace(indices, weights)

        def fake_route(*args, **kwargs):
            return indices, weights

        def fake_trace_route(*args, **kwargs):
            return route_trace

        def fake_expert(values, w1, w2, w3, *, route_weight_bits=None):
            return outputs[expert_id(w1, injected_experts)]

        def fake_trace(values, w1, w2, w3, *, route_weight_bits=None):
            return dummy_expert_trace(outputs[expert_id(w1, injected_experts)])

        injected_expected = expected_partials(outputs)[1]
        invalid_bias = injected_bias[:-1] + (INF,)
        real_inputs, real_gate, real_bias, real_vision, real_experts = fixture()
        real_trace = impl.trace_routed(
            real_inputs, real_gate, real_bias, real_experts,
            image_token=True, bias_vl_bits=real_vision)
        real_expected = real_trace.output_bits
        self.assertEqual(impl.routed(
            real_inputs, real_gate, real_bias, real_experts,
            image_token=True, bias_vl_bits=real_vision), real_expected)

        paths = ("injected-aggregate-ordinary", "injected-aggregate-trace",
                 "real-combination-ordinary", "real-combination-trace",
                 "parameter-reject", "runtime-failure")
        saved = Fenv()
        self.assertEqual(library.fegetenv(ctypes.byref(saved)), 0)
        failures = []
        try:
            for mode in (rounding | denormals
                         for rounding in (0, 0x2000, 0x4000, 0x6000)
                         for denormals in (0, 0x8000, 0x40, 0x8040)):
                for status in (0, 0x3F):
                    for path in paths:
                        setting = Fenv.from_buffer_copy(bytes(saved))
                        setting.mxcsr = 0x1F80 | mode | status
                        setting.control = ((setting.control & ~0xC00) |
                                           ((mode >> 3) & 0xC00) | 0x3F)
                        setting.status = (setting.status & ~0x3F) | status
                        if library.fesetenv(ctypes.byref(setting)):
                            failures.append((mode, status, path, "set failed"))
                            continue
                        label = (mode, status, path)
                        result = None
                        try:
                            if path.startswith("injected-"):
                                with patched_upstream(
                                        route=fake_route,
                                        trace_route=fake_trace_route,
                                        expert=fake_expert,
                                        trace_expert=fake_trace):
                                    if path.endswith("ordinary"):
                                        result = impl.routed(
                                            injected_inputs, injected_gate,
                                            injected_bias, injected_experts)
                                    else:
                                        result = impl.trace_routed(
                                            injected_inputs, injected_gate,
                                            injected_bias,
                                            injected_experts).output_bits
                            elif path == "real-combination-ordinary":
                                result = impl.routed(
                                    real_inputs, real_gate, real_bias, real_experts,
                                    image_token=True, bias_vl_bits=real_vision)
                            elif path == "real-combination-trace":
                                result = impl.trace_routed(
                                    real_inputs, real_gate, real_bias, real_experts,
                                    image_token=True, bias_vl_bits=real_vision)
                            elif path == "parameter-reject":
                                with self.assertRaises(ValueError):
                                    impl.routed(injected_inputs, injected_gate,
                                                invalid_bias, injected_experts)
                            else:
                                def runtime_fail(*args, **kwargs):
                                    raise RuntimeError("expert failed")
                                with patched_upstream(expert=runtime_fail):
                                    with self.assertRaises(RuntimeError):
                                        impl.routed(injected_inputs, injected_gate,
                                                    injected_bias,
                                                    injected_experts)
                        except Exception as error:  # collect all modes before reporting
                            failures.append((*label, type(error).__name__, str(error)))
                        after = Fenv()
                        library.fegetenv(ctypes.byref(after))
                        if (after.mxcsr & ~0x3F, after.control) != \
                                (setting.mxcsr & ~0x3F, setting.control):
                            failures.append((*label, "control changed"))
                        if (after.mxcsr & status) != status or \
                                (after.status & status) != status:
                            failures.append((*label, "existing sticky cleared"))
                        if path.startswith("injected-") and result != injected_expected:
                            failures.append((*label, "injected aggregate mismatch"))
                        if path == "real-combination-ordinary" and \
                                result != real_expected:
                            failures.append((*label, "real ordinary mismatch"))
                        if path == "real-combination-trace" and result != real_trace:
                            failures.append((*label, "real trace mismatch"))
        finally:
            restored = library.fesetenv(ctypes.byref(saved))
        self.assertEqual(restored, 0)
        after_restore = Fenv()
        self.assertEqual(library.fegetenv(ctypes.byref(after_restore)), 0)
        self.assertEqual(bytes(after_restore), bytes(saved))
        self.assertEqual(failures, [], repr(failures))


SUITES = {
    "all": (PublicCombinationTests, AggregateTests, ValidationAndSnapshotTests,
            FailurePropagationTests, SensitivityTests, EnvironmentTests),
    "combination": (PublicCombinationTests,),
    "aggregate": (AggregateTests,),
    "validation": (ValidationAndSnapshotTests, FailurePropagationTests),
    "sensitivity": (SensitivityTests,),
    "environment": (EnvironmentTests,),
}


def iter_tests(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from iter_tests(item)
        else:
            yield item


def main(argv=None):
    parser = argparse.ArgumentParser(description="P0025 routed-branch oracle acceptance")
    parser.add_argument("--suite", choices=tuple(SUITES), default="all")
    parser.add_argument("--case", action="append", default=[])
    args = parser.parse_args(argv)
    loader = unittest.TestLoader()
    suite = unittest.TestSuite(loader.loadTestsFromTestCase(cls)
                               for cls in SUITES[args.suite])
    if args.case:
        suite = unittest.TestSuite(
            test for test in iter_tests(suite)
            if any(token in test.id() for token in args.case))
    if suite.countTestCases() == 0:
        print("no tests selected", file=sys.stderr)
        return 2
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() and result.testsRun and not result.skipped else 1


if __name__ == "__main__":
    print("P0025 directed routed-branch acceptance; standard library only", flush=True)
    print(f"Python: {sys.version}; platform: {platform.platform()}; machine: {platform.machine()}",
          flush=True)
    sys.exit(main())
