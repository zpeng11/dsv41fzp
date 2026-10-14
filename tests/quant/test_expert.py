#!/usr/bin/env python3
"""P0020 directed CPU expert acceptance: python3 tests/quant/test_expert.py.

Only this module's tests are collected. Existing test modules supply independent
Fraction/grid-search activation and dot expectations, never production outputs.
SiLU uses positive Taylor sums with a geometric remainder bound, outward dyadic
interval squaring, reciprocal bounds, and FP32 grid searches at both endpoints;
it shares neither Decimal.exp nor host exp with the implementation. Expectations
flow exclusively from expected earlier stages. Sensitivity checks compare explicit
wrong-stage models numerically after the correct production baseline passes.
"""

import ctypes
import dataclasses
import decimal
import platform
import sys
import unittest
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_formats as fmt
import test_compute as dot
import test_act_quant as quant
from ref import expert as impl

NAN = dot.NAN_SENTINEL
BRIDGE_FIELDS = ("gate_bits", "up_bits", "silu_bits", "gated_bits", "weighted_bits", "bf16_bits")


def outward(value, precision, upper=False):
    """Enclose an exact rational on a fixed dyadic grid, never round inward."""
    scaled = value * (1 << precision)
    whole, remainder = divmod(scaled.numerator, scaled.denominator)
    return Fraction(whole + bool(upper and remainder), 1 << precision)


def exp_interval(magnitude, terms, precision):
    """Positive Taylor sum S_n <= exp(r) <= S_n+t_(n+1)/(1-r/(n+2))."""
    reduced, squares = magnitude, 0
    while reduced > Fraction(1, 2):
        reduced /= 2
        squares += 1
    term = total = Fraction(1)
    for index in range(1, terms + 1):
        term *= reduced / index
        total += term
    next_term = term * reduced / (terms + 1)
    low = total
    high = total + next_term / (1 - reduced / (terms + 2))
    for _ in range(squares):
        low = outward(low * low, precision)
        high = outward(high * high, precision, upper=True)
    return low, high


def silu_interval(value, terms=24, precision=384):
    """Transport exp(abs(g)) bounds through reciprocal and signed SiLU."""
    low_exp, high_exp = exp_interval(abs(value), terms, precision)
    low_recip, high_recip = 1 / high_exp, 1 / low_exp
    if value < 0:
        return value * high_recip / (1 + high_recip), value * low_recip / (1 + low_recip)
    return value / (1 + high_recip), value / (1 + low_recip)


@lru_cache(maxsize=None)
def expected_silu(bits):
    """Certify SiLU; for t>=128, d(t*exp(-t))/dt=exp(-t)*(1-t)<0.

    A Taylor lower bound on exp(128) proves 128/exp(128)<2**-150;
    monotonicity then bounds the whole negative tail below the zero midpoint.
    """
    magnitude = bits & 0x7FFFFFFF
    if magnitude > 0x7F800000 or bits == 0xFF800000:
        return NAN
    if magnitude == 0:
        return bits
    value = fmt.expected_binary(bits, 23)
    if value <= -128:
        low_exp, _ = exp_interval(Fraction(128), 24, 384)
        if not 128 / low_exp < fmt.power2(-150):
            raise AssertionError("negative-tail bound not established")
        return 0x80000000
    for terms in (24, 48, 96, 192):
        low, high = silu_interval(value, terms, 16 * terms)
        low_bits = fmt.nearest_fp32_bits(low, value < 0)
        high_bits = fmt.nearest_fp32_bits(high, value < 0)
        if low_bits == high_bits:
            return low_bits
    raise AssertionError(f"independent SiLU interval did not converge: {bits:#x}")


def expected_mul(left, right):
    """IEEE multiplication with independent exact-value rounding and NaN classes."""
    if left == NAN or right == NAN or dot.is_nan_fp32_bits(left) or dot.is_nan_fp32_bits(right):
        return NAN
    sign = (left ^ right) & 0x80000000
    a, b = left & 0x7FFFFFFF, right & 0x7FFFFFFF
    if a == 0x7F800000 or b == 0x7F800000:
        return NAN if a == 0 or b == 0 else sign | 0x7F800000
    return fmt.nearest_fp32_bits(fmt.expected_binary(left, 23) * fmt.expected_binary(right, 23), bool(sign))


def expected_clamp(bits, gate):
    if dot.is_nan_fp32_bits(bits):
        return NAN
    if bits == 0xFF800000:
        return bits if gate else 0xC1200000
    if bits == 0x7F800000:
        return 0x41200000
    value = fmt.expected_binary(bits, 23)
    if value > 10:
        return 0x41200000
    if not gate and value < -10:
        return 0xC1200000
    return bits


def expected_bridge(gates, ups, weight=None, *, raw_fp32=False, fault=None):
    """Independent bridge, with explicit wrong-stage counterfactuals for sensitivity."""
    result = {name: [] for name in BRIDGE_FIELDS}
    if fault == "swap":
        gates, ups = ups, gates
    for gate, up in zip(gates, ups):
        gate = gate if raw_fp32 else gate << 16
        up = up if raw_fp32 else up << 16
        if fault != "no-gate-clamp":
            gate = expected_clamp(gate, True)
        if fault != "no-up-clamp":
            up = expected_clamp(up, False)
        if fault == "gate-lower" and fmt.expected_binary(gate, 23) < -10:
            gate = 0xC1200000
        if fault == "upper-nine" and fmt.expected_binary(gate, 23) > 9:
            gate = fmt.nearest_fp32_bits(Fraction(9))
        if fault == "up-lower-nine" and fmt.expected_binary(up, 23) < -9:
            up = fmt.nearest_fp32_bits(Fraction(-9))
        if fault == "weight-before-silu":
            gate = expected_mul(gate, weight)
        silu = expected_silu(gate)
        if fault == "early-bf16":
            silu = fmt.nearest_bf16_bits(silu) << 16
        gated = expected_mul(silu, up)
        if fault == "weight-on-up":
            gated = expected_mul(silu, expected_mul(up, weight))
        weighted = gated if weight is None or fault in ("omit-weight", "weight-before-silu", "weight-on-up") else expected_mul(gated, weight)
        bf16 = NAN if weighted == NAN else fmt.nearest_bf16_bits(weighted)
        for name, value in zip(BRIDGE_FIELDS, (gate, up, silu, gated, weighted, bf16)):
            result[name].append(value)
    return {name: tuple(values) for name, values in result.items()}


def expected_linear(codes, scales, matrix, *, raw_fp32=False):
    values = []
    for row in range(matrix.n):
        packed_start, scale_start = row * matrix.k // 2, row * matrix.k // 32
        _, fp32, bf16 = dot.expected_trace(
            codes, matrix.w_packed[packed_start:packed_start + matrix.k // 2],
            scales, matrix.scale_b[scale_start:scale_start + matrix.k // 32],
        )
        values.append(fp32 if raw_fp32 else bf16)
    return tuple(values)


def expected_expert(inputs, w1, w2, w3, weight=None):
    _, codes, scales = quant.expected_trace(inputs)
    gate = expected_linear(codes, scales, w1)
    up = expected_linear(codes, scales, w3)
    bridge = expected_bridge(gate, up, weight)
    _, hidden_codes, hidden_scales = quant.expected_trace(bridge["bf16_bits"])
    return dict(input_codes=codes, input_scales=scales, gate_bf16=gate, up_bf16=up,
                bridge=bridge, hidden_codes=hidden_codes, hidden_scales=hidden_scales,
                output_bf16=expected_linear(hidden_codes, hidden_scales, w2))


def matrix(n, k, seed=0, *, scale=123):
    """Small deterministic row-major weights with distinct rows and scale groups."""
    nibbles = tuple((row * 5 + col * 3 + (col // 7) + seed) % 16
                    for row in range(n) for col in range(k))
    scales = tuple(scale + (row + group + seed) % 3
                   for row in range(n) for group in range(k // 32))
    return impl.FP4Matrix(n, k, fmt.independent_pack(nibbles), scales)


def constant_matrix(n, k, nibble, scale):
    return impl.FP4Matrix(n, k, fmt.independent_pack((nibble,) * (n * k)), (scale,) * (n * k // 32))


def fixture(d=32, h=32):
    anchors = (0x3F81, 0xBF42, 0x3EA7, 0x4001, 0xBE93, 0x3D81, 0x8000, 0x0001)
    inputs = tuple(anchors[i % len(anchors)] for i in range(d))
    return inputs, matrix(h, d, 1), matrix(d, h, 2), matrix(h, d, 7)


def check_sequence(test, actual, expected, label, width=32):
    test.assertIs(type(actual), tuple, label)
    test.assertEqual(len(actual), len(expected), label)
    for index, (got, want) in enumerate(zip(actual, expected)):
        dot.check_bits(test, got, want, f"{label}[{index}]", width)


def check_bridge(test, gates, ups, weight=None):
    expected = expected_bridge(gates, ups, weight)
    actual = impl.trace_swiglu(gates, ups, route_weight_bits=weight)
    test.assertIsInstance(actual, impl.BridgeTrace)
    for name in BRIDGE_FIELDS:
        check_sequence(test, getattr(actual, name), expected[name], name, 16 if name == "bf16_bits" else 32)
    direct = impl.swiglu(gates, ups, route_weight_bits=weight)
    check_sequence(test, direct, expected["bf16_bits"], "swiglu", 16)
    test.assertEqual(direct, actual.bf16_bits)
    return expected


def check_expert(test, args, weight=None):
    expected = expected_expert(*args, weight)
    actual = impl.trace_expert(*args, route_weight_bits=weight)
    test.assertIsInstance(actual, impl.ExpertTrace)
    for name, want in expected.items():
        if name == "bridge":
            test.assertIsInstance(actual.bridge, impl.BridgeTrace)
            for field in BRIDGE_FIELDS:
                check_sequence(test, getattr(actual.bridge, field), want[field], f"bridge.{field}", 16 if field == "bf16_bits" else 32)
        else:
            check_sequence(test, getattr(actual, name), want, name, 16)
    direct = impl.expert(*args, route_weight_bits=weight)
    check_sequence(test, direct, expected["output_bf16"], "expert", 16)
    test.assertEqual(direct, actual.output_bf16)
    return expected


class SiLUAndBridgeTests(unittest.TestCase):
    def test_independent_silu_tail_and_rounding_boundaries(self):
        bits = [0, 0x80000000, 1, 3, 0x80000001, 0x80000003,
                0x007FFFFF, 0x00800000, 0x807FFFFF, 0x80800000,
                0x3F800001, 0xBF800001, 0x41200000, 0xC1200000,
                0xC2FFFFFF, 0xC3000000, 0xC3000001, 0xFF7FFFFF]
        bits += [fmt.nearest_fp32_bits(-Fraction(n, 2)) for n in range(180, 225)]
        bits += [fmt.nearest_fp32_bits(-Fraction(n, 8)) for n in range(870, 885)]
        for value in bits:
            with self.subTest(bits=hex(value)):
                dot.check_bits(self, impl._silu_fp32_bits(value), expected_silu(value), "SiLU", 32)
        for value in (Fraction(-128), fmt.expected_binary(0xC2FFFFFF, 23)):
            low, high = silu_interval(value)
            self.assertLess(low, high)
            self.assertEqual(fmt.nearest_fp32_bits(low), 0x80000000)
            self.assertEqual(fmt.nearest_fp32_bits(high), 0x80000000)
        self.assertNotEqual(expected_silu(fmt.nearest_fp32_bits(Fraction(-100))), 0x80000000)
        self.assertEqual(expected_silu(fmt.nearest_fp32_bits(Fraction(-112))), 0x80000000)

    def test_clamps_no_gate_lower_bound_signed_zero_and_specials(self):
        gates = (0x411F, 0x4120, 0x4121, 0xC11F, 0xC120, 0xC121,
                 0xC200, 0xC2C8, 0xC2DC, 0xC300, 0, 0x8000,
                 0x7F80, 0xFF80, 0x7FC1, 0xFFC1, 0x3F80, 0xBF80,
                 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x0001, 0x8001)
        ups = (0x411F, 0x4120, 0x4121, 0xC11F, 0xC120, 0xC121,
               0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80,
               0x7F80, 0xFF80, 0x3F80, 0x3F80, 0, 0x8000,
               0x7FC1, 0xFFC1, 0x7F80, 0xFF80, 0x0001, 0x0001)
        for weight in (None, 0, 0x80000000, 0x3F000001):
            with self.subTest(weight=weight):
                check_bridge(self, gates, ups, weight)
        for bits in (0xFF800000, 0x7FC00001, 0xFFC00001):
            self.assertTrue(dot.is_nan_fp32_bits(impl._silu_fp32_bits(bits)))

    def test_none_route_skips_only_the_optional_multiply(self):
        gates, ups = (0x3F81, 0xC121), (0xBF81, 0x4121)
        expected = check_bridge(self, gates, ups)
        for entry in (impl.swiglu, impl.trace_swiglu):
            for weight, count in ((None, 2), (0x3F800000, 4)):
                with patch.object(impl, "_multiply_fp32_bits", wraps=impl._multiply_fp32_bits) as multiply:
                    result = entry(gates, ups, route_weight_bits=weight)
                self.assertEqual(multiply.call_count, count)
                output = result.bf16_bits if isinstance(result, impl.BridgeTrace) else result
                self.assertEqual(output, expected["bf16_bits"])

    def test_product_rne_midpoints_subnormal_overflow_and_ieee(self):
        pairs = ((1, 0x3F000000), (3, 0x3F000000), (0x80000001, 0x3F000000),
                 (0x007FFFFF, 0x3F800001), (0x00800000, 0x3F000000),
                 (0x3F800800, 0x3F800800), (0x7F7FFFFF, 0x3F800001),
                 (0, 0xFF800000), (0x7F800000, 0xBF800000),
                 (0xFF800000, 0xBF800000), (0x7FC00001, 0x3F800000),
                 (0x3F800000, 0xFFC00001), (0x80000000, 0x3F800000))
        for a, b in pairs:
            with self.subTest(a=hex(a), b=hex(b)):
                dot.check_bits(self, impl._multiply_fp32_bits(a, b), expected_mul(a, b), "multiply", 32)
        for weight in (1, 3, 0x007FFFFF, 0x00800000, 0x80000001, 0x3F808000, 0x7F7FFFFF):
            check_bridge(self, (0x0001, 0x3F80, 0x4120, 0xBF80), (0x3F00, 0x3F81, 0x4120, 0xBF81), weight)

    def test_bf16_midpoint_and_neighbors_from_weighted_bridge(self):
        gate, up = (0x4120,), (0x3F80,)
        silu = fmt.expected_binary(expected_silu(0x41200000), 23)
        target = Fraction(1) + fmt.power2(-8)
        weight = fmt.nearest_fp32_bits(target / silu)
        results = []
        for candidate in range(weight - 3, weight + 4):
            expected = check_bridge(self, gate, up, candidate)
            results.append(expected)
        self.assertIn(0x3F808000, [item["weighted_bits"][0] for item in results])
        self.assertEqual({item["bf16_bits"][0] for item in results}, {0x3F80, 0x3F81})


class ExpertTests(unittest.TestCase):
    def test_single_multi_group_and_rectangular_all_stages(self):
        for d, h, weight in ((32, 32, None), (32, 64, 0x3F000001), (64, 32, 0xBF400001)):
            with self.subTest(d=d, h=h, weight=weight):
                check_expert(self, fixture(d, h), weight)

    def test_zero_weights_and_zero_route(self):
        inputs = (0x8000, 0, 0x0001, 0x8001) * 8
        zero = constant_matrix(32, 32, 0, 127)
        check_expert(self, (inputs, zero, zero, zero))
        check_expert(self, fixture(), 0)

    def test_final_w2_infinity_and_nan_are_allowed(self):
        inputs = (0x3F80,) * 32
        first = constant_matrix(64, 32, 2, 122)
        positive = constant_matrix(32, 64, 7, 254)
        signs = (7,) * 32 + (15,) * 32
        opposite = impl.FP4Matrix(32, 64, fmt.independent_pack(signs * 32), (254,) * 64)
        for final, nan in ((positive, False), (opposite, True)):
            expected = check_expert(self, (inputs, first, final, first))
            if nan:
                self.assertEqual(expected["output_bf16"], (NAN,) * 32)
            else:
                self.assertEqual(expected["output_bf16"], (0x7F80,) * 32)

    def test_linear_infinities_rescued_by_bridge_clamps(self):
        inputs = (0x3F80,) * 32
        finite = constant_matrix(32, 32, 2, 122)
        positive = constant_matrix(32, 32, 7, 254)
        negative = constant_matrix(32, 32, 15, 254)
        final = matrix(32, 32, 2)
        cases = ((positive, finite, 0x7F80, 0x3F80),
                 (finite, positive, 0x3F80, 0x7F80),
                 (finite, negative, 0x3F80, 0xFF80),
                 (positive, positive, 0x7F80, 0x7F80),
                 (positive, negative, 0x7F80, 0xFF80))
        for gate, up, gate_code, up_code in cases:
            with self.subTest(gate=hex(gate_code), up=hex(up_code)):
                expected = check_expert(self, (inputs, gate, final, up))
                self.assertEqual(expected["gate_bf16"], (gate_code,) * 32)
                self.assertEqual(expected["up_bf16"], (up_code,) * 32)
                self.assertTrue(all((code & 0x7F80) != 0x7F80
                                    for code in expected["bridge"]["bf16_bits"]))

    def test_nonfinite_bridge_rejected_before_hidden_quant_and_w2(self):
        positive = (0x3F80,) * 32
        final = matrix(32, 32, 2)
        finite = constant_matrix(32, 32, 2, 122)
        negative = constant_matrix(32, 32, 15, 254)
        opposite = impl.FP4Matrix(32, 64, fmt.independent_pack(((7,) * 32 + (15,) * 32) * 32), (254,) * 64)
        cases = ((positive, negative, final, finite, None, 0xFF80, 0x3F80),
                 (positive, constant_matrix(32, 32, 2, 127), final,
                  constant_matrix(32, 32, 2, 127), 0x7F7FFFFF, 0x4200, 0x4200),
                 (positive * 2, opposite, matrix(64, 32),
                  constant_matrix(32, 64, 2, 121), None, NAN, 0x3F80))
        for x, gate, final, up, weight, gate_code, up_code in cases:
            _, codes, scales = quant.expected_trace(x)
            self.assertEqual(expected_linear(codes, scales, gate), (gate_code,) * 32)
            self.assertEqual(expected_linear(codes, scales, up), (up_code,) * 32)
            bridge = expected_bridge((0x7FC0 if gate_code == NAN else gate_code,) * 32,
                                     (up_code,) * 32, weight)
            self.assertTrue(all(code == NAN or (code & 0x7F80) == 0x7F80
                                for code in bridge["bf16_bits"]))
            for entry in (impl.expert, impl.trace_expert):
                with self.subTest(entry=entry.__name__, gate=gate_code, weight=weight):
                    with patch.object(impl, "_linear", wraps=impl._linear) as linear, \
                         patch.object(impl, "_bridge", wraps=impl._bridge) as bridge_call, \
                         patch.object(impl, "act_quant", wraps=impl.act_quant) as quant_call:
                        with self.assertRaises(ValueError):
                            entry(x, gate, final, up, route_weight_bits=weight)
                        self.assertEqual(bridge_call.call_count, 1)
                        self.assertEqual(quant_call.call_count, 1)
                        self.assertFalse(any(call.args[2] == final for call in linear.call_args_list))

    def test_validation_and_input_immutability(self):
        inputs, w1, w2, w3 = fixture()
        for entry in (impl.expert, impl.trace_expert):
            for bad, error in ((-1, ValueError), (65536, ValueError), (0x7F80, ValueError),
                               (0xFF80, ValueError), (0x7FC1, ValueError), (True, TypeError),
                               (1.0, TypeError), (None, TypeError)):
                with self.subTest(entry=entry.__name__, bad=bad):
                    with self.assertRaises(error):
                        entry(inputs[:-1] + (bad,), w1, w2, w3)
            for data in ((), inputs[:-1], inputs + (0,)):
                with self.assertRaises(ValueError):
                    entry(data, w1, w2, w3)
            with self.assertRaises(TypeError):
                entry(inputs, object(), w2, w3)
            for changes, error in ((dict(n=0), ValueError), (dict(k=31), ValueError),
                                   (dict(n=-32), ValueError), (dict(k=True), TypeError),
                                   (dict(n=32.0), TypeError), (dict(w_packed=b""), ValueError),
                                   (dict(scale_b=()), ValueError),
                                   (dict(scale_b=(255,) * 32), ValueError),
                                   (dict(w_packed=(256,) * 512), ValueError),
                                   (dict(w_packed=(True,) * 512), TypeError)):
                with self.subTest(entry=entry.__name__, changes=tuple(changes)):
                    with self.assertRaises(error):
                        entry(inputs, dataclasses.replace(w1, **changes), w2, w3)
            with self.assertRaises(ValueError):
                entry(inputs, w1, matrix(64, 32), w3)
            with self.assertRaises(ValueError):
                entry(inputs, w1, w2, matrix(64, 32))
        mutable_input = list(inputs)
        packed, scales = list(w1.w_packed), list(w1.scale_b)
        mutable_matrix = impl.FP4Matrix(32, 32, packed, scales)
        snapshot = (mutable_input[:], packed[:], scales[:])
        check_expert(self, (mutable_input, mutable_matrix, w2, w3))
        self.assertEqual((mutable_input, packed, scales), snapshot)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            mutable_matrix.n = 64

    def test_bridge_validation_and_weight_validation_all_entries(self):
        for entry in (impl.swiglu, impl.trace_swiglu):
            with self.assertRaises(ValueError):
                entry((0,), ())
            for bad, error in ((-1, ValueError), (65536, ValueError), (True, TypeError),
                               (1.0, TypeError), (None, TypeError)):
                for gates, ups in (((bad,), (0,)), ((0,), (bad,))):
                    with self.assertRaises(error):
                        entry(gates, ups)
            gates, ups = [0x4121, 0xC121], [0xC121, 0x4121]
            snapshot = (gates[:], ups[:])
            entry(gates, ups)
            self.assertEqual((gates, ups), snapshot)
            self.assertEqual(entry((), ()).bf16_bits if entry is impl.trace_swiglu else entry((), ()), ())
        for entry, args in ((impl.swiglu, ((0,), (0,))), (impl.trace_swiglu, ((0,), (0,))),
                            (impl.expert, fixture()), (impl.trace_expert, fixture())):
            for weight, error in ((0x7F800000, ValueError), (0xFF800000, ValueError),
                                  (0x7FC00001, ValueError), (-1, ValueError),
                                  (1 << 32, ValueError), (True, TypeError), (1.0, TypeError)):
                with self.subTest(entry=entry.__name__, weight=weight):
                    with self.assertRaises(error):
                        entry(*args, route_weight_bits=weight)


class SensitivityTests(unittest.TestCase):
    def test_wrong_bridge_stages_are_numeric_counterexamples(self):
        gates = (0x4121, 0xC121, 0x3F81, 0xBF42, 0x3EA7, 0x4001)
        ups = (0xC121, 0x4121, 0x3FA7, 0xBF81, 0x4001, 0x3E93)
        weight = 0x3F000001
        correct = check_bridge(self, gates, ups, weight)
        for fault in ("no-gate-clamp", "no-up-clamp", "gate-lower", "upper-nine", "up-lower-nine", "swap", "early-bf16",
                      "omit-weight", "weight-before-silu", "weight-on-up"):
            with self.subTest(fault=fault):
                wrong = expected_bridge(gates, ups, weight, fault=fault)
                field = "weighted_bits" if fault in ("omit-weight", "weight-before-silu", "weight-on-up") else "silu_bits" if fault == "early-bf16" else "gated_bits"
                self.assertNotEqual(wrong[field], correct[field],
                                    f"{fault} must produce a completed numeric {field} mismatch")
        check_bridge(self, gates, ups, weight)

    def test_route_weight_moved_after_w2_changes_final_numbers(self):
        args = fixture(32, 64)
        weight = fmt.nearest_fp32_bits(Fraction(7, 10))
        correct = check_expert(self, args, weight)
        unweighted = expected_expert(*args)
        wrong_output = tuple(fmt.nearest_bf16_bits(expected_mul(code << 16, weight))
                             for code in unweighted["output_bf16"])
        self.assertNotEqual(unweighted["bridge"]["bf16_bits"], correct["bridge"]["bf16_bits"])
        self.assertNotEqual(wrong_output, correct["output_bf16"],
                            "weight after w2 must cause a completed final BF16 mismatch")

    def test_bypassed_linear_bf16_is_detected_at_bridge(self):
        args = fixture(32, 64)
        expected = check_expert(self, args, 0x3F000001)
        raw_gate = expected_linear(expected["input_codes"], expected["input_scales"], args[1], raw_fp32=True)
        raw_up = expected_linear(expected["input_codes"], expected["input_scales"], args[3], raw_fp32=True)
        wrong = expected_bridge(raw_gate, raw_up, 0x3F000001, raw_fp32=True)
        self.assertNotEqual(wrong["gate_bits"], expected["bridge"]["gate_bits"])
        self.assertNotEqual(wrong["weighted_bits"], expected["bridge"]["weighted_bits"])


class EnvironmentTests(unittest.TestCase):
    def test_decimal_ambient_context_does_not_affect_new_entries(self):
        args = fixture()
        gates, ups, weight = (0x0001, 0xC2DC, 0x3F81), (0x3F80, 0x4120, 0xBF81), 0x3F000001
        bridge = expected_bridge(gates, ups, weight)
        expert = expected_expert(*args, weight)
        for rounding in (decimal.ROUND_FLOOR, decimal.ROUND_CEILING):
            with decimal.localcontext() as context:
                context.prec, context.Emin, context.Emax = 2, -2, 2
                context.rounding = rounding
                for signal in context.traps:
                    context.traps[signal] = True
                context.clear_flags()
                before = (context.prec, context.Emin, context.Emax, context.rounding,
                          dict(context.traps), dict(context.flags))
                self.assertEqual(impl.swiglu(gates, ups, route_weight_bits=weight), bridge["bf16_bits"])
                self.assertEqual(impl.trace_swiglu(gates, ups, route_weight_bits=weight).weighted_bits, bridge["weighted_bits"])
                self.assertEqual(impl.expert(*args, route_weight_bits=weight), expert["output_bf16"])
                self.assertEqual(impl.trace_expert(*args, route_weight_bits=weight).output_bf16, expert["output_bf16"])
                after = (context.prec, context.Emin, context.Emax, context.rounding,
                         dict(context.traps), dict(context.flags))
                self.assertEqual(after, before)

    @unittest.skipUnless(sys.platform == "linux" and platform.machine() == "x86_64"
                         and platform.libc_ver()[0] == "glibc", "fenv layout is glibc x86_64 only")
    def test_masked_fenv_numeric_control_and_sticky_preservation(self):
        """D0015: masked exceptions; allow added, never cleared, sticky bits.

        Numerical results ignore host rounding/FTZ/DAZ and controls stay intact.
        Decimal's backend may set inexact; this is not a no-side-effects claim.
        A no-op control must still preserve the complete recorded environment.
        """
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
        args = fixture()
        gates, ups = (0x0001, 0x8001, 0xC2DC, 0x3F81), (0x3F80, 0x3F80, 0x4120, 0x3FA7)
        bridge = expected_bridge(gates, ups, 0x3F000001)
        expert = expected_expert(*args, 0x3F000001)
        def control(*values, route_weight_bits=None):
            return bridge["bf16_bits"]

        calls = ((control, (), bridge["bf16_bits"]),
                 (impl.swiglu, (gates, ups), bridge["bf16_bits"]),
                 (impl.trace_swiglu, (gates, ups), bridge["bf16_bits"]),
                 (impl.expert, args, expert["output_bf16"]),
                 (impl.trace_expert, args, expert["output_bf16"]))
        saved = Fenv()
        self.assertEqual(library.fegetenv(ctypes.byref(saved)), 0)
        failures = []
        try:
            for mode in (rounding | denormals
                         for rounding in (0, 0x2000, 0x4000, 0x6000)
                         for denormals in (0, 0x8000, 0x40, 0x8040)):
                for status in (0, 0x3F):
                    for entry, values, expected in calls:
                        for reject in ((False,) if entry is control else (False, True)):
                            setting = Fenv.from_buffer_copy(bytes(saved))
                            setting.mxcsr = 0x1F80 | mode | status
                            setting.control = (setting.control & ~0xC00) | ((mode >> 3) & 0xC00) | 0x3F
                            setting.status = (setting.status & ~0x3F) | status
                            if library.fesetenv(ctypes.byref(setting)):
                                failures.append((mode, status, "set failed"))
                                continue
                            label = (mode, status, entry.__name__, reject)
                            try:
                                result = entry(*values, route_weight_bits=0x7F800000 if reject else 0x3F000001)
                            except ValueError:
                                if not reject:
                                    raise
                            else:
                                if reject:
                                    failures.append((*label, "invalid route accepted"))
                            after = Fenv()
                            library.fegetenv(ctypes.byref(after))
                            if entry is control and bytes(after) != bytes(setting):
                                failures.append((*label, "no-op changed environment"))
                            if (after.mxcsr & ~0x3F, after.control) != (setting.mxcsr & ~0x3F, setting.control):
                                failures.append((*label, "control changed"))
                            if (after.mxcsr & status) != status or (after.status & status) != status:
                                failures.append((*label, "existing sticky flag cleared"))
                            if not reject:
                                output = result.bf16_bits if isinstance(result, impl.BridgeTrace) else result.output_bf16 if isinstance(result, impl.ExpertTrace) else result
                                if output != expected:
                                    failures.append((*label, "numeric mismatch"))
        finally:
            restore = library.fesetenv(ctypes.byref(saved))
        self.assertEqual(restore, 0)
        restored = Fenv()
        self.assertEqual(library.fegetenv(ctypes.byref(restored)), 0)
        self.assertEqual(bytes(restored), bytes(saved))
        self.assertEqual(failures, [], repr(failures))


if __name__ == "__main__":
    print("P0020 directed expert acceptance; standard library only", flush=True)
    unittest.main(verbosity=2)
