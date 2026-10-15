#!/usr/bin/env python3
"""P0023 directed single-token router acceptance.

Only this module is collected by its CLI.  Expected stages use exact Fraction
arithmetic and the independent encoding/grid searches from test_formats.py.
Softplus is certified with Taylor bounds for exp and the atanh log1p series;
sqrt is rounded by an independent FP32 grid search and squared midpoints.
No expectation calls a router numerical helper, Decimal exp/ln, or libm.
"""

import argparse
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
sys.path.insert(0, str(ROOT / "tests" / "quant"))

import test_formats as fmt
from ref import router as impl

TRACE_FIELDS = (
    "linear_bits", "temp_bits", "softplus_bits", "score_bits",
    "selection_bits", "indices", "selected_bits", "sum_bits",
    "denominator_bits", "normalized_bits", "weights_bits",
)
TOPK = 6
EPS = Fraction(1, 10 ** 20)
SCALE = Fraction(3, 2)


def _finite32(bits):
    return (bits & 0x7F800000) != 0x7F800000


def _finite16(bits):
    return (bits & 0x7F80) != 0x7F80


def _round(exact, negative_zero=False):
    return fmt.nearest_fp32_bits(exact, negative_zero)


def _value(bits):
    if not _finite32(bits):
        raise ValueError("non-finite intermediate")
    return fmt.expected_binary(bits, 23)


def _mul(left, right):
    exact = _value(left) * _value(right)
    negative_zero = exact == 0 and bool((left ^ right) & 0x80000000)
    return _round(exact, negative_zero)


def _add(left, right):
    exact = _value(left) + _value(right)
    negative_zero = exact == 0 and bool(left & right & 0x80000000)
    return _round(exact, negative_zero)


def _div(left, right):
    exact = _value(left) / _value(right)
    negative_zero = exact == 0 and bool((left ^ right) & 0x80000000)
    return _round(exact, negative_zero)


def outward(value, precision, upper=False):
    scaled = value * (1 << precision)
    whole, remainder = divmod(scaled.numerator, scaled.denominator)
    return Fraction(whole + bool(upper and remainder), 1 << precision)


def exp_positive_interval(value, terms, precision):
    """Bound exp(value) with a positive Taylor sum and geometric remainder."""
    reduced, squares = value, 0
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


def log1p_interval(low_y, high_y, terms):
    """Bound log(1+y)=2*sum z**(2j+1)/(2j+1), z=y/(2+y)."""
    def bound(y, upper):
        z = y / (2 + y)
        power = z
        total = Fraction(0)
        for index in range(terms + 1):
            total += power / (2 * index + 1)
            power *= z * z
        result = 2 * total
        if upper:
            result += 2 * power / ((2 * terms + 3) * (1 - z * z))
        return result
    return bound(low_y, False), bound(high_y, True)


@lru_cache(maxsize=None)
def softplus_interval(bits):
    value = _value(bits)
    if value > 20:
        return value, value
    if value <= -104:
        low_exp, _ = exp_positive_interval(Fraction(104), 32, 512)
        if not 1 / low_exp < fmt.power2(-150):
            raise AssertionError("negative softplus tail bound not established")
        return Fraction(0), fmt.power2(-150)
    magnitude = abs(value)
    for terms in (24, 48, 96, 192):
        low_exp, high_exp = exp_positive_interval(magnitude, terms, 16 * terms)
        low_y, high_y = 1 / high_exp, 1 / low_exp
        low_log, high_log = log1p_interval(low_y, high_y, terms)
        if value >= 0:
            low_log += value
            high_log += value
        if _round(low_log) == _round(high_log):
            return low_log, high_log
    raise AssertionError(f"independent softplus interval did not converge: {bits:#x}")


@lru_cache(maxsize=None)
def expected_softplus(bits):
    value = _value(bits)
    if value > 20:
        return bits
    low, high = softplus_interval(bits)
    low_bits, high_bits = _round(low), _round(high)
    if low_bits != high_bits:
        raise AssertionError(f"softplus rounding not certified: {bits:#x}")
    return low_bits


def _sqrt_fraction_bits(exact):
    if exact < 0:
        raise ValueError("negative square root")
    if exact == 0:
        return 0
    low, high = 0, 0x7F7FFFFF
    while low < high:
        middle = (low + high) // 2
        if fmt.expected_binary(middle, 23) ** 2 < exact:
            low = middle + 1
        else:
            high = middle
    upper = low
    upper_value = fmt.expected_binary(upper, 23)
    if upper_value * upper_value == exact:
        return upper
    lower = upper - 1
    midpoint = (fmt.expected_binary(lower, 23) + upper_value) / 2
    midpoint_square = midpoint * midpoint
    if exact < midpoint_square:
        return lower
    if exact > midpoint_square:
        return upper
    return lower if lower % 2 == 0 else upper


@lru_cache(maxsize=None)
def expected_sqrt(bits):
    if bits & 0x7FFFFFFF == 0:
        return bits
    value = _value(bits)
    return _sqrt_fraction_bits(value)


def _fused_score(bits):
    low, high = softplus_interval(bits)
    low_bits, high_bits = _sqrt_fraction_bits(low), _sqrt_fraction_bits(high)
    if low_bits != high_bits:
        raise AssertionError(f"fused nonlinear rounding not certified: {bits:#x}")
    return low_bits


def bf16(exact, negative_zero=False):
    return fmt.nearest_bf16_bits(_round(exact, negative_zero))


def fp32(exact, negative_zero=False):
    return _round(exact, negative_zero)


def gate_from_rows(rows):
    """Build a small row-major GateMatrix from BF16 bit-pattern rows."""
    rows = tuple(tuple(row) for row in rows)
    if not rows or not rows[0]:
        raise ValueError("rows must be nonempty")
    if any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("ragged rows")
    return impl.GateMatrix(len(rows), len(rows[0]), tuple(v for row in rows for v in row))


def basic_fixture():
    """Reusable N=8, K=1 fixture with selection-changing finite biases."""
    linears = (-4, -2, -1, 0, 1, 2, 4, 8)
    gate = gate_from_rows([[bf16(Fraction(value))] for value in linears])
    inputs = (bf16(Fraction(1)),)
    biases = tuple(fp32(value) for value in (
        Fraction(0), Fraction(3), Fraction(0), Fraction(0),
        Fraction(-2), Fraction(0), Fraction(0), Fraction(-6),
    ))
    vl = tuple(fp32(value) for value in (
        Fraction(7), Fraction(0), Fraction(0), Fraction(0),
        Fraction(0), Fraction(0), Fraction(-3), Fraction(0),
    ))
    return inputs, gate, biases, vl


def environment_fixture():
    """BF16 subnormal input, FP32 subnormal products, and one tiny route weight."""
    inputs = (0x0001, bf16(Fraction(1)))
    small = bf16(fmt.power2(-16))
    rows = [[small, bf16(Fraction(-103))]] + [[small, 0x7F7F]] * 5
    gate = gate_from_rows(rows)
    return inputs, gate, (0,) * 6


def overflow_fixtures():
    """Finite inputs causing respectively product and sequential-sum overflow."""
    product = ((bf16(Fraction(2)),), gate_from_rows([[0x7F7F]] * 6), (0,) * 6)
    summed = ((bf16(Fraction(1)), bf16(Fraction(1))),
              gate_from_rows([[0x7F7F, 0x7F7F]] * 6), (0,) * 6)
    return product, summed


def _project(input_bf16, gate, *, exact_accumulation=False):
    values = []
    for row in range(gate.n):
        products = []
        acc = 0
        for col in range(gate.k):
            product = _mul(input_bf16[col] << 16,
                           gate.weight_bf16[row * gate.k + col] << 16)
            if not _finite32(product):
                raise ValueError("non-finite gate product")
            products.append(product)
            if not exact_accumulation:
                acc = _add(acc, product)
                if not _finite32(acc):
                    raise ValueError("non-finite gate sum")
        if exact_accumulation:
            exact = sum((_value(product) for product in products), Fraction(0))
            acc = _round(exact)
        values.append(acc)
    return tuple(values)


def _expected_route(input_bf16, gate, bias_bits, *, image_token=False,
                    bias_vl_bits=None, fault=None):
    linear = _project(input_bf16, gate, exact_accumulation=fault == "exact-projection")
    temp = tuple(_div(bits, fp32(Fraction(1))) for bits in linear)
    softplus = tuple(expected_softplus(bits) for bits in temp)
    if fault == "fused-nonlinear":
        score = tuple(_fused_score(bits) for bits in temp)
    else:
        score = tuple(expected_sqrt(bits) for bits in softplus)
    chosen_bias = bias_vl_bits if image_token and bias_vl_bits is not None else bias_bits
    selection = tuple(_add(bits, chosen_bias[index]) for index, bits in enumerate(score))
    if any(not _finite32(bits) for bits in selection):
        raise ValueError("non-finite selection")
    reverse_tie = fault == "reverse-tie"
    order = sorted(range(gate.n),
                   key=lambda index: (-_value(selection[index]),
                                      -index if reverse_tie else index))
    indices = tuple(order[:TOPK])
    if fault == "bias-in-weight":
        selected = tuple(selection[index] for index in indices)
    elif fault == "mismatched-weight":
        selected = tuple(score[indices[(position + 1) % TOPK]]
                         for position in range(TOPK))
    else:
        selected = tuple(score[index] for index in indices)
    if fault == "normalize-all":
        summands = score
    else:
        summands = selected
    if fault == "reverse-sum":
        summands = tuple(reversed(summands))
    total = 0
    for bits in summands:
        total = _add(total, bits)
    denominator = _add(total, fp32(EPS))
    normalized = tuple(_div(bits, denominator) for bits in selected)
    if fault == "omit-scale":
        weights = normalized
    else:
        scale = fp32(Fraction(5, 4) if fault == "wrong-scale" else SCALE)
        weights = tuple(_mul(bits, scale) for bits in normalized)
    return dict(
        linear_bits=linear, temp_bits=temp, softplus_bits=softplus,
        score_bits=score, selection_bits=selection, indices=indices,
        selected_bits=selected, sum_bits=total, denominator_bits=denominator,
        normalized_bits=normalized, weights_bits=weights,
    )


def expected_route(input_bf16, gate, bias_bits, *, image_token=False,
                   bias_vl_bits=None):
    """Independent complete expectation; keys exactly match RouteTrace fields."""
    return _expected_route(input_bf16, gate, bias_bits, image_token=image_token,
                           bias_vl_bits=bias_vl_bits)


def assert_route(test, input_bf16, gate, bias_bits, *, image_token=False,
                 bias_vl_bits=None):
    want = expected_route(input_bf16, gate, bias_bits, image_token=image_token,
                          bias_vl_bits=bias_vl_bits)
    trace = impl.trace_route(input_bf16, gate, bias_bits, image_token=image_token,
                             bias_vl_bits=bias_vl_bits)
    test.assertIsInstance(trace, impl.RouteTrace)
    test.assertTrue(dataclasses.is_dataclass(trace))
    for name in TRACE_FIELDS:
        got = getattr(trace, name)
        test.assertEqual(got, want[name], name)
        if name not in ("sum_bits", "denominator_bits"):
            test.assertIs(type(got), tuple, name)
            for value in got:
                test.assertIs(type(value), int, name)
        else:
            test.assertIs(type(got), int, name)
    direct = impl.route(input_bf16, gate, bias_bits, image_token=image_token,
                        bias_vl_bits=bias_vl_bits)
    test.assertEqual(direct, (want["indices"], want["weights_bits"]))
    test.assertEqual(direct, (trace.indices, trace.weights_bits))
    return want, trace


class ExpectationTests(unittest.TestCase):
    def test_constants_and_independent_rounding_anchors(self):
        self.assertEqual(fp32(EPS), 0x1E3CE508)
        self.assertEqual(fp32(SCALE), 0x3FC00000)
        self.assertEqual(expected_softplus(fp32(Fraction(-2))), 0x3E01F96B)
        self.assertEqual(expected_sqrt(0x3E01F96B), 0x3EB668F8)
        self.assertNotEqual(_fused_score(fp32(Fraction(-2))), 0x3EB668F8)

    def test_public_expected_route_signature_and_stage_flow(self):
        inputs, gate, bias, vl = basic_fixture()
        result = expected_route(inputs, gate, bias, bias_vl_bits=vl)
        self.assertEqual(tuple(result), TRACE_FIELDS)
        self.assertEqual(len(result["indices"]), TOPK)
        self.assertEqual(result["selected_bits"],
                         tuple(result["score_bits"][i] for i in result["indices"]))


class NonlinearTests(unittest.TestCase):
    def test_softplus_negative_tail_threshold_and_rounding(self):
        minus_104 = fp32(Fraction(-104))
        twenty = fp32(Fraction(20))
        cases = [
            minus_104 - 1, minus_104, minus_104 + 1,
            fp32(Fraction(-100)), fp32(Fraction(-20)), fp32(Fraction(-2)),
            0x80800000, 0x807FFFFF, 0x80000001, 0x80000000,
            0, 1, 0x007FFFFF, 0x00800000, fp32(Fraction(1)),
            twenty - 1, twenty, twenty + 1, 0x7F7FFFFF,
        ]
        for bits in cases:
            with self.subTest(bits=hex(bits)):
                self.assertEqual(impl._softplus_fp32_bits(bits), expected_softplus(bits))
        self.assertEqual(expected_softplus(minus_104), 0)
        self.assertEqual(expected_softplus(twenty + 1), twenty + 1)

    def test_sqrt_grid_midpoints_subnormals_and_extremes(self):
        cases = (0, 1, 2, 3, 0x007FFFFF, 0x00800000,
                 fp32(Fraction(1)), fp32(Fraction(2)), fp32(Fraction(4)),
                 0x7F7FFFFF)
        for bits in cases:
            with self.subTest(bits=hex(bits)):
                self.assertEqual(impl._sqrt_fp32_bits(bits), expected_sqrt(bits))
        lower, upper = fp32(Fraction(1)), fp32(Fraction(1)) + 1
        midpoint = (fmt.expected_binary(lower, 23) + fmt.expected_binary(upper, 23)) / 2
        self.assertEqual(_sqrt_fraction_bits(midpoint * midpoint), lower)


class RoutePipelineTests(unittest.TestCase):
    def test_complete_trace_ordinary_and_frozen_shape(self):
        inputs, gate, bias, vl = basic_fixture()
        _, trace = assert_route(self, inputs, gate, bias, bias_vl_bits=vl)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            trace.sum_bits = 0
        with self.assertRaises(dataclasses.FrozenInstanceError):
            gate.n = 9

    def test_projection_order_cancellation_and_no_fma(self):
        rows = [
            [bf16(fmt.power2(24)), bf16(Fraction(1)), bf16(-fmt.power2(24))],
            [bf16(Fraction(-4)), 0, 0], [bf16(Fraction(-2)), 0, 0],
            [bf16(Fraction(-1)), 0, 0], [0, 0, 0],
            [bf16(Fraction(1)), 0, 0], [bf16(Fraction(2)), 0, 0],
        ]
        gate = gate_from_rows(rows)
        inputs = (bf16(Fraction(1)),) * 3
        bias = (0,) * gate.n
        expected, _ = assert_route(self, inputs, gate, bias)
        self.assertEqual(expected["linear_bits"][0], 0)
        wrong = _expected_route(inputs, gate, bias, fault="exact-projection")
        self.assertEqual(wrong["linear_bits"][0], fp32(Fraction(1)))
        self.assertNotEqual(wrong["linear_bits"], expected["linear_bits"])

    def test_projection_product_subnormal_and_underflow(self):
        inputs = (0x0080, 0x0080)
        row = [bf16(fmt.power2(-23)), bf16(fmt.power2(-24))]
        gate = gate_from_rows([row] * 6)
        expected, _ = assert_route(self, inputs, gate, (0,) * 6)
        self.assertEqual(expected["linear_bits"], (1,) * 6)

    def test_product_and_sequential_sum_overflow_fail_closed(self):
        product, summed = overflow_fixtures()
        for entry in (impl.route, impl.trace_route):
            for boundary, args in (("product", product), ("sum", summed)):
                with self.subTest(entry=entry.__name__, boundary=boundary):
                    with self.assertRaises(ValueError):
                        entry(*args)

    def test_topk_cutoff_near_neighbors_ties_and_signed_zero(self):
        gate = gate_from_rows([[0]] * 9)
        ulp = fmt.power2(-24)
        bias = tuple(fp32(index * ulp) for index in range(9))
        expected, _ = assert_route(self, (0,), gate, bias)
        self.assertEqual(expected["indices"], (8, 7, 6, 5, 4, 3))
        tied, _ = assert_route(self, (0,), gate, (0, 0x80000000) + (0,) * 7)
        self.assertEqual(tied["indices"], (0, 1, 2, 3, 4, 5))

    def test_bias_changes_selection_but_not_selected_score_source(self):
        inputs, gate, bias, _ = basic_fixture()
        expected, _ = assert_route(self, inputs, gate, bias)
        unbiased = expected_route(inputs, gate, (0,) * gate.n)
        self.assertNotEqual(expected["indices"], unbiased["indices"])
        self.assertEqual(expected["selected_bits"],
                         tuple(expected["score_bits"][i] for i in expected["indices"]))
        self.assertNotEqual(expected["selected_bits"],
                            tuple(expected["selection_bits"][i] for i in expected["indices"]))

    def test_text_image_bias_and_image_fallback(self):
        inputs, gate, bias, vl = basic_fixture()
        text, _ = assert_route(self, inputs, gate, bias, image_token=False,
                               bias_vl_bits=vl)
        image, _ = assert_route(self, inputs, gate, bias, image_token=True,
                                bias_vl_bits=vl)
        fallback, _ = assert_route(self, inputs, gate, bias, image_token=True)
        self.assertNotEqual(text["indices"], image["indices"])
        self.assertEqual(fallback, expected_route(inputs, gate, bias))

    def test_zero_denominator_eps_negative_tail_and_zero_weights(self):
        gate = gate_from_rows([[bf16(Fraction(-128))]] * 6)
        expected, _ = assert_route(self, (bf16(Fraction(1)),), gate, (0,) * 6)
        self.assertEqual(expected["softplus_bits"], (0,) * 6)
        self.assertEqual(expected["score_bits"], (0,) * 6)
        self.assertEqual(expected["sum_bits"], 0)
        self.assertEqual(expected["denominator_bits"], fp32(EPS))
        self.assertEqual(expected["normalized_bits"], (0,) * 6)
        self.assertEqual(expected["weights_bits"], (0,) * 6)

    def test_tiny_selected_score_produces_subnormal_route_weight(self):
        rows = [[bf16(Fraction(-103))]] + [[0x7F7F]] * 5
        gate = gate_from_rows(rows)
        expected, _ = assert_route(self, (bf16(Fraction(1)),), gate, (0,) * 6)
        tiny = expected["indices"].index(0)
        self.assertEqual(expected["softplus_bits"][0], 1)
        self.assertGreater(expected["weights_bits"][tiny], 0)
        self.assertLess(expected["weights_bits"][tiny], 0x00800000)

    def test_injected_unreachable_nonfinite_selection_is_defensively_rejected(self):
        """Internal injection only: finite public inputs cannot naturally reach this."""
        gate = gate_from_rows([[0]] * 6)
        bias = (0x7F7FFFFF,) * 6
        original = impl._sqrt_fp32_bits
        with patch.object(impl, "_sqrt_fp32_bits", return_value=0x7F7FFFFF):
            for entry in (impl.route, impl.trace_route):
                with self.subTest(entry=entry.__name__):
                    with self.assertRaises(ValueError):
                        entry((0,), gate, bias)
        self.assertIs(impl._sqrt_fp32_bits, original)


class ValidationTests(unittest.TestCase):
    def entries(self):
        return (impl.route, impl.trace_route)

    def test_gate_metadata_input_and_encoding_validation(self):
        inputs, gate, bias, _ = basic_fixture()
        malformed = (
            (impl.GateMatrix(5, 1, (0,) * 5), ValueError),
            (impl.GateMatrix(6, 0, ()), ValueError),
            (impl.GateMatrix(True, 1, (0,) * 6), TypeError),
            (impl.GateMatrix(6.0, 1, (0,) * 6), TypeError),
            (impl.GateMatrix(6, True, (0,) * 6), TypeError),
            (impl.GateMatrix(6, 1, (0,) * 5), ValueError),
            (impl.GateMatrix(6, 1, (0,) * 7), ValueError),
            (impl.GateMatrix(6, 1, (0,) * 5 + (True,)), TypeError),
            (impl.GateMatrix(6, 1, (0,) * 5 + (-1,)), ValueError),
            (impl.GateMatrix(6, 1, (0,) * 5 + (0x10000,)), ValueError),
            (impl.GateMatrix(6, 1, (0,) * 5 + (0x7F80,)), ValueError),
        )
        for entry in self.entries():
            with self.assertRaises(TypeError):
                entry(inputs, object(), bias)
            for bad_gate, error in malformed:
                with self.subTest(entry=entry.__name__, gate=bad_gate):
                    with self.assertRaises(error):
                        entry((0,) * bad_gate.k if type(bad_gate.k) is int and bad_gate.k > 0 else (),
                              bad_gate, (0,) * max(6, bad_gate.n if type(bad_gate.n) is int else 6))
            for bad, error in ((True, TypeError), (1.0, TypeError), (-1, ValueError),
                               (0x10000, ValueError), (0x7F80, ValueError)):
                with self.subTest(entry=entry.__name__, input=bad):
                    with self.assertRaises(error):
                        entry((bad,), gate, bias)
            for bad_input in ((), (0, 0)):
                with self.assertRaises(ValueError):
                    entry(bad_input, gate, bias)

    def test_bias_image_flag_and_all_unused_content_are_validated(self):
        inputs, gate, bias, vl = basic_fixture()
        for entry in self.entries():
            for flag in (0, 1, None, "true"):
                with self.subTest(entry=entry.__name__, flag=flag):
                    with self.assertRaises(TypeError):
                        entry(inputs, gate, bias, image_token=flag)
            for bad, error in ((True, TypeError), (1.0, TypeError), (-1, ValueError),
                               (1 << 32, ValueError), (0x7F800000, ValueError),
                               (0x7FC00000, ValueError)):
                bad_bias = bias[:-1] + (bad,)
                with self.subTest(entry=entry.__name__, normal=bad):
                    with self.assertRaises(error):
                        entry(inputs, gate, bad_bias, image_token=True, bias_vl_bits=vl)
                with self.subTest(entry=entry.__name__, vl=bad):
                    with self.assertRaises(error):
                        entry(inputs, gate, bias, image_token=False,
                              bias_vl_bits=vl[:-1] + (bad,))
            for bad_bias in ((), bias[:-1], bias + (0,)):
                with self.assertRaises(ValueError):
                    entry(inputs, gate, bad_bias)
            with patch.object(impl, "_multiply_fp32_bits",
                              side_effect=AssertionError("projected before validation")):
                with self.assertRaises(ValueError):
                    entry(inputs, gate, bias, bias_vl_bits=vl[:-1] + (0x7F800000,))
                with self.assertRaises(ValueError):
                    entry(inputs, gate, bias[:-1] + (0x7F800000,),
                          image_token=True, bias_vl_bits=vl)
            overflow_input, overflow_gate, _ = overflow_fixtures()[0]
            invalid_unused = (0,) * 5 + (0x7F800000,)
            with self.assertRaisesRegex(ValueError, "vision bias contains nonfinite"):
                entry(overflow_input, overflow_gate, (0,) * 6,
                      image_token=False, bias_vl_bits=invalid_unused)

    def test_non_sequences_bool_encodings_and_2d_memoryviews(self):
        inputs, gate, bias, _ = basic_fixture()
        input_2d = memoryview(bytearray(2)).cast("B", shape=[1, 2])
        weight_2d = memoryview(bytearray(12)).cast("H", shape=[6, 1])
        bias_2d = memoryview(bytearray(24)).cast("I", shape=[6, 1])
        for entry in self.entries():
            for bad in (5, 1.5, None, iter((0,))):
                with self.subTest(entry=entry.__name__, bad=bad):
                    with self.assertRaises(TypeError):
                        entry(bad, gate, bias)
                    with self.assertRaises(TypeError):
                        entry(inputs, gate, bad)
            with self.assertRaises(TypeError):
                entry(input_2d, impl.GateMatrix(6, 2, (0,) * 12), (0,) * 6)
            with self.assertRaises(TypeError):
                entry(inputs, impl.GateMatrix(6, 1, weight_2d), (0,) * 6)
            with self.assertRaises(TypeError):
                entry(inputs, gate, bias_2d)

    def test_snapshot_before_first_multiply_and_normal_input_immutability(self):
        input_bits, base_gate, base_bias, base_vl = basic_fixture()
        original_multiply = impl._multiply_fp32_bits
        for entry in self.entries():
            for image_token in (False, True):
                with self.subTest(entry=entry.__name__, image_token=image_token):
                    inputs = list(input_bits)
                    weights = list(base_gate.weight_bf16)
                    bias, vl = list(base_bias), list(base_vl)
                    gate = impl.GateMatrix(base_gate.n, base_gate.k, weights)
                    snapshots = (inputs[:], weights[:], bias[:], vl[:])
                    immutable_gate = impl.GateMatrix(gate.n, gate.k, tuple(weights))
                    frozen = expected_route(tuple(inputs), immutable_gate, tuple(bias),
                                            image_token=image_token,
                                            bias_vl_bits=tuple(vl))
                    changed = False
                    def mutate(left, right):
                        nonlocal changed
                        if not changed:
                            changed = True
                            inputs[:] = [0] * len(inputs)
                            weights[:] = [0] * len(weights)
                            bias[:] = [0] * len(bias)
                            vl[:] = [0] * len(vl)
                        return original_multiply(left, right)
                    with patch.object(impl, "_multiply_fp32_bits", side_effect=mutate):
                        result = entry(inputs, gate, bias, image_token=image_token,
                                       bias_vl_bits=vl)
                    self.assertTrue(changed, "first projection multiply was not reached")
                    if entry is impl.trace_route:
                        for name in TRACE_FIELDS:
                            self.assertEqual(getattr(result, name), frozen[name], name)
                    else:
                        self.assertEqual(result,
                                         (frozen["indices"], frozen["weights_bits"]))

                    inputs, weights, bias, vl = (list(values) for values in snapshots)
                    gate = impl.GateMatrix(base_gate.n, base_gate.k, weights)
                    before = (inputs[:], weights[:], bias[:], vl[:])
                    entry(inputs, gate, bias, image_token=image_token, bias_vl_bits=vl)
                    self.assertEqual((inputs, weights, bias, vl), before)


class SensitivityTests(unittest.TestCase):
    def test_completed_wrong_stage_models_have_numeric_or_index_mismatches(self):
        inputs, gate, bias, vl = basic_fixture()
        correct, _ = assert_route(self, inputs, gate, bias, bias_vl_bits=vl)
        cases = (
            ("bias-in-weight", "selected_bits"),
            ("normalize-all", "sum_bits"),
            ("mismatched-weight", "selected_bits"),
            ("omit-scale", "weights_bits"),
            ("wrong-scale", "weights_bits"),
        )
        for fault, field in cases:
            with self.subTest(fault=fault):
                wrong = _expected_route(inputs, gate, bias, bias_vl_bits=vl, fault=fault)
                self.assertNotEqual(wrong[field], correct[field],
                                    f"{fault} must complete with a numeric mismatch")
        sum_gate = gate_from_rows([[bf16(Fraction(value))]
                                   for value in (-30, -10, -2, 1, 4, 16)])
        sum_input, sum_bias = (bf16(Fraction(1)),), (0,) * 6
        sum_correct, _ = assert_route(self, sum_input, sum_gate, sum_bias)
        sum_wrong = _expected_route(sum_input, sum_gate, sum_bias, fault="reverse-sum")
        self.assertNotEqual(sum_wrong["sum_bits"], sum_correct["sum_bits"],
                            "reverse-sum must complete with a numeric mismatch")
        assert_route(self, inputs, gate, bias, bias_vl_bits=vl)

    def test_projection_tie_and_fused_nonlinear_counterexamples(self):
        rows = [[bf16(fmt.power2(24)), bf16(Fraction(1)), bf16(-fmt.power2(24))]]
        rows += [[bf16(Fraction(value)), 0, 0] for value in (-4, -2, -1, 0, 1, 2)]
        projection_gate = gate_from_rows(rows)
        projection_input = (bf16(Fraction(1)),) * 3
        projection_bias = (0,) * 7
        correct, _ = assert_route(self, projection_input, projection_gate, projection_bias)
        wrong = _expected_route(projection_input, projection_gate, projection_bias,
                                fault="exact-projection")
        self.assertNotEqual(wrong["linear_bits"], correct["linear_bits"])

        tie_gate = gate_from_rows([[0]] * 7)
        tied, _ = assert_route(self, (0,), tie_gate, (0,) * 7)
        wrong_tie = _expected_route((0,), tie_gate, (0,) * 7, fault="reverse-tie")
        self.assertNotEqual(wrong_tie["indices"], tied["indices"])

        nonlinear_gate = gate_from_rows([[bf16(Fraction(-2))]] +
                                        [[bf16(Fraction(value))] for value in (-1, 0, 1, 2, 3)])
        nonlinear_bias = (0,) * 6
        nonlinear, _ = assert_route(self, (bf16(Fraction(1)),), nonlinear_gate,
                                    nonlinear_bias)
        fused = _expected_route((bf16(Fraction(1)),), nonlinear_gate,
                                nonlinear_bias, fault="fused-nonlinear")
        self.assertNotEqual(fused["score_bits"], nonlinear["score_bits"])
        assert_route(self, (bf16(Fraction(1)),), nonlinear_gate, nonlinear_bias)


class CertificationAndIndependenceTests(unittest.TestCase):
    def test_noncertification_propagates_and_preserves_inputs(self):
        inputs, gate, bias, vl = basic_fixture()
        for entry in (impl.route, impl.trace_route):
            mutable = (list(inputs), list(gate.weight_bf16), list(bias), list(vl))
            mutable_gate = impl.GateMatrix(gate.n, gate.k, mutable[1])
            snapshot = tuple(values[:] for values in mutable)
            with patch.object(impl, "_softplus_fp32_bits",
                              side_effect=ArithmeticError("not certified")):
                with self.assertRaisesRegex(ArithmeticError, "not certified"):
                    entry(mutable[0], mutable_gate, mutable[2],
                          image_token=True, bias_vl_bits=mutable[3])
            self.assertEqual(mutable, snapshot)
        with patch.object(impl, "_INITIAL_PRECISION", 1), \
             patch.object(impl, "_MAX_PRECISION", 1):
            with self.assertRaises(ArithmeticError):
                impl._softplus_fp32_bits(fp32(Fraction(-2)))

    def test_expected_route_does_not_call_router_or_expert_entries(self):
        inputs, gate, bias, vl = basic_fixture()
        forbidden = ("route", "trace_route", "_multiply_fp32_bits", "_add_fp32_bits",
                     "_divide_fp32_bits", "_softplus_fp32_bits", "_sqrt_fp32_bits")
        patches = [patch.object(impl, name, side_effect=AssertionError(name))
                   for name in forbidden]
        from ref import expert
        patches += [patch.object(expert, name, side_effect=AssertionError(name))
                    for name in ("expert", "trace_expert", "swiglu", "trace_swiglu")]
        for active in patches:
            active.start()
        try:
            result = expected_route(inputs, gate, bias, image_token=True,
                                    bias_vl_bits=vl)
        finally:
            for active in reversed(patches):
                active.stop()
        self.assertEqual(tuple(result), TRACE_FIELDS)
        self.assertEqual(len(result["indices"]), TOPK)


class EnvironmentTests(unittest.TestCase):
    def test_decimal_ambient_context_is_exactly_unchanged(self):
        inputs, gate, bias, vl = basic_fixture()
        want = expected_route(inputs, gate, bias, image_token=True, bias_vl_bits=vl)
        runtime_reject = overflow_fixtures()[0]
        for rounding in (decimal.ROUND_FLOOR, decimal.ROUND_CEILING):
            with decimal.localcontext() as context:
                context.prec, context.Emin, context.Emax = 2, -2, 2
                context.rounding = rounding
                for signal in context.traps:
                    context.traps[signal] = True
                context.clear_flags()
                context.flags[decimal.Inexact] = True
                for path in ("ordinary", "trace", "parameter-reject", "runtime-reject"):
                    with self.subTest(rounding=rounding, path=path):
                        before = (context.prec, context.Emin, context.Emax,
                                  context.rounding, dict(context.traps),
                                  dict(context.flags))
                        if path == "ordinary":
                            result = impl.route(inputs, gate, bias, image_token=True,
                                                bias_vl_bits=vl)
                            self.assertEqual(result,
                                             (want["indices"], want["weights_bits"]))
                        elif path == "trace":
                            result = impl.trace_route(inputs, gate, bias,
                                                      image_token=True,
                                                      bias_vl_bits=vl)
                            for name in TRACE_FIELDS:
                                self.assertEqual(getattr(result, name), want[name], name)
                        elif path == "parameter-reject":
                            with self.assertRaises(ValueError):
                                impl.route(inputs, gate, bias, image_token=False,
                                           bias_vl_bits=vl[:-1] + (0x7F800000,))
                        else:
                            with self.assertRaises(ValueError):
                                impl.route(*runtime_reject)
                        after = (context.prec, context.Emin, context.Emax,
                                 context.rounding, dict(context.traps),
                                 dict(context.flags))
                        self.assertEqual(after, before)

    def test_env_a_masked_fenv_control_and_sticky_contract(self):
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

        inputs, gate, bias = environment_fixture()
        want = expected_route(inputs, gate, bias)
        self.assertEqual(_mul(inputs[0] << 16, gate.weight_bf16[0] << 16), 1)
        tiny_position = want["indices"].index(0)
        self.assertLess(want["weights_bits"][tiny_position], 0x00800000)
        product_reject, sum_reject = overflow_fixtures()
        invalid_bias = (0,) * 5 + (0x7F800000,)
        paths = (("ordinary", impl.route, (inputs, gate, bias), False),
                 ("trace", impl.trace_route, (inputs, gate, bias), False),
                 ("parameter-route-reject", impl.route, (inputs, gate, bias), True),
                 ("parameter-trace-reject", impl.trace_route, (inputs, gate, bias), True),
                 ("product-reject", impl.route, product_reject, True),
                 ("sum-reject", impl.trace_route, sum_reject, True))
        saved = Fenv()
        self.assertEqual(library.fegetenv(ctypes.byref(saved)), 0)
        failures = []
        try:
            for mode in (rounding | denormals
                         for rounding in (0, 0x2000, 0x4000, 0x6000)
                         for denormals in (0, 0x8000, 0x40, 0x8040)):
                for status in (0, 0x3F):
                    for path, entry, arguments, reject in paths:
                        setting = Fenv.from_buffer_copy(bytes(saved))
                        setting.mxcsr = 0x1F80 | mode | status
                        setting.control = ((setting.control & ~0xC00) |
                                           ((mode >> 3) & 0xC00) | 0x3F)
                        setting.status = (setting.status & ~0x3F) | status
                        if library.fesetenv(ctypes.byref(setting)):
                            failures.append((mode, status, path, "set failed"))
                            continue
                        label = (mode, status, path)
                        try:
                            if path.startswith("parameter-"):
                                result = entry(*arguments, image_token=False,
                                               bias_vl_bits=invalid_bias)
                            else:
                                result = entry(*arguments)
                        except ValueError:
                            if not reject:
                                failures.append((*label, "unexpected rejection"))
                        else:
                            if reject:
                                failures.append((*label, "runtime overflow accepted"))
                        after = Fenv()
                        library.fegetenv(ctypes.byref(after))
                        if (after.mxcsr & ~0x3F, after.control) != \
                                (setting.mxcsr & ~0x3F, setting.control):
                            failures.append((*label, "control changed"))
                        if (after.mxcsr & status) != status or \
                                (after.status & status) != status:
                            failures.append((*label, "existing sticky cleared"))
                        if not reject and path == "ordinary":
                            if result != (want["indices"], want["weights_bits"]):
                                failures.append((*label, "ordinary mismatch"))
                        if not reject and path == "trace":
                            for name in TRACE_FIELDS:
                                if getattr(result, name) != want[name]:
                                    failures.append((*label, name, "trace mismatch"))
        finally:
            restored = library.fesetenv(ctypes.byref(saved))
        self.assertEqual(restored, 0)
        after_restore = Fenv()
        self.assertEqual(library.fegetenv(ctypes.byref(after_restore)), 0)
        self.assertEqual(bytes(after_restore), bytes(saved))
        self.assertEqual(failures, [], repr(failures))


SUITES = {
    "all": (ExpectationTests, NonlinearTests, RoutePipelineTests,
            ValidationTests, SensitivityTests, CertificationAndIndependenceTests,
            EnvironmentTests),
    "core": (ExpectationTests, NonlinearTests, RoutePipelineTests,
             CertificationAndIndependenceTests),
    "validation": (ValidationTests,),
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
    parser = argparse.ArgumentParser(description="P0023 router oracle acceptance")
    parser.add_argument("--suite", choices=tuple(SUITES), default="all")
    parser.add_argument("--case", action="append", default=[])
    args = parser.parse_args(argv)
    loader = unittest.TestLoader()
    suite = unittest.TestSuite(loader.loadTestsFromTestCase(cls) for cls in SUITES[args.suite])
    if args.case:
        suite = unittest.TestSuite(test for test in iter_tests(suite)
                                   if any(token in test.id() for token in args.case))
    if suite.countTestCases() == 0:
        print("no tests selected", file=sys.stderr)
        return 2
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() and not result.skipped else 1


if __name__ == "__main__":
    print("P0023 directed router acceptance; standard library only", flush=True)
    print(f"Python: {sys.version}; platform: {platform.platform()}; machine: {platform.machine()}",
          flush=True)
    sys.exit(main())
