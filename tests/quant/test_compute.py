#!/usr/bin/env python3
"""P0012 mandatory CPU compute acceptance: python3 tests/quant/test_compute.py.

Expected values are built independently of ref/compute.py: Fraction bit-field
decodes and packing anchors come from tests/quant/test_formats.py, every
rounding step goes through test_formats' ordered-encoding search as the
primary method, and a separate integer round-to-nearest-even constructed in
this file must agree with it on directed, boundary, and seeded random inputs.
Exact arithmetic is rational throughout. No expectation calls a ref helper or
host float arithmetic. Every pipeline stage is compared bit-exactly; NaN is
compared by classification.
The sensitivity suite mutates the agreed source anchors inside throwaway
copies (ref plus the two test files only) and requires numeric failures,
reported through a digested single-line JSON marker, plus crash and tamper
controls and a pristine-copy retest; repository sources are byte-checked.

Entry points:
    python3 tests/quant/test_compute.py                          # all mandatory
    python3 tests/quant/test_compute.py --suite core             # numeric + validation
    python3 tests/quant/test_compute.py --suite sensitivity
    python3 tests/quant/test_compute.py --suite directed --marker  # subprocess use
    python3 tests/quant/test_compute.py --case <name substring>
"""

import argparse
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_formats import (  # noqa: E402
    expected_binary,
    expected_e2m1,
    expected_e4m3fn,
    expected_e8m0,
    host_fp32_bits,
    host_fp32_value,
    independent_pack,
    nearest_bf16_bits,
    nearest_fp32_bits,
    power2,
)
from ref.compute import (  # noqa: E402
    BlockTrace,
    ComputeTrace,
    _add_fp32,
    expert_grouped_dot,
    trace_expert_grouped_dot,
)

ENTRY_POINTS = (expert_grouped_dot, trace_expert_grouped_dot)
VALID_A_CODES = tuple(code for code in range(256) if code not in (0x7F, 0xFF))
NAN_SENTINEL = -1
MARKER_PREFIX = "P0012_MARKER "


@lru_cache(maxsize=None)
def decode_a(code):
    return expected_e4m3fn(code)


@lru_cache(maxsize=None)
def decode_w(nibble):
    return expected_e2m1(nibble)


@lru_cache(maxsize=None)
def decode_scale(code):
    return expected_e8m0(code)


@lru_cache(maxsize=None)
def finite_value(bits):
    return expected_binary(bits, 23)


def is_nan_fp32_bits(bits):
    return (bits & 0x7F800000) == 0x7F800000 and (bits & 0x7FFFFF) != 0


def is_nan_bf16_bits(bits):
    return (bits & 0x7F80) == 0x7F80 and (bits & 0x7F) != 0


def is_inf_bits(bits):
    return (bits & 0x7FFFFFFF) == 0x7F800000


def rne_fp32_bits(magnitude, sign):
    """Round an exact nonnegative dyadic rational to binary32 bits, ties to even."""
    if magnitude == 0:
        return 0x80000000 if sign else 0x00000000
    numerator, denominator = magnitude.numerator, magnitude.denominator
    if denominator & (denominator - 1):
        raise AssertionError("exact value is not dyadic")
    exponent = 1 - denominator.bit_length()
    significand = numerator
    bit_length = significand.bit_length()
    top = exponent + bit_length - 1
    sign_bit = 0x80000000 if sign else 0
    if top > 127:
        return sign_bit | 0x7F800000
    if top >= -126:
        shift = bit_length - 24
        if shift > 0:
            dropped = significand & ((1 << shift) - 1)
            halfway = 1 << (shift - 1)
            quotient = significand >> shift
            if dropped > halfway or (dropped == halfway and quotient & 1):
                quotient += 1
            if quotient == 1 << 24:
                quotient = 1 << 23
                top += 1
        else:
            quotient = significand << -shift
        if top > 127:
            return sign_bit | 0x7F800000
        return sign_bit | ((top + 127) << 23) | (quotient & 0x7FFFFF)
    shift = -149 - exponent
    if shift > 0:
        dropped = significand & ((1 << shift) - 1)
        halfway = 1 << (shift - 1)
        quotient = significand >> shift
        if dropped > halfway or (dropped == halfway and quotient & 1):
            quotient += 1
    else:
        quotient = significand << -shift
    if quotient >= 1 << 23:
        return sign_bit | 0x00800000 | (quotient & 0x7FFFFF)
    return sign_bit | quotient


def _search_round(magnitude, sign):
    """Primary independent rounding through test_formats' ordered-encoding search."""
    if magnitude == 0:
        return nearest_fp32_bits(Fraction(0), sign)
    return nearest_fp32_bits(-magnitude if sign else magnitude)


def model_add(left_bits, right_bits, rounder=_search_round):
    """One binary32 addition on bit patterns under the COMPUTE special rules."""
    if left_bits == NAN_SENTINEL or right_bits == NAN_SENTINEL:
        return NAN_SENTINEL
    left_inf, right_inf = is_inf_bits(left_bits), is_inf_bits(right_bits)
    if left_inf and right_inf:
        return left_bits if (left_bits >> 31) == (right_bits >> 31) else NAN_SENTINEL
    if left_inf:
        return left_bits
    if right_inf:
        return right_bits
    exact = finite_value(left_bits) + finite_value(right_bits)
    if exact == 0:
        sign = bool(left_bits >> 31) and bool(right_bits >> 31)
    else:
        sign = exact < 0
    return rounder(abs(exact), sign)


def model_scale_mul(bits, scale, rounder=_search_round):
    """Multiply a binary32 bit pattern by an exact positive power-of-two scale."""
    if bits == NAN_SENTINEL:
        return NAN_SENTINEL
    if is_inf_bits(bits):
        return bits
    magnitude = finite_value(bits)
    if magnitude == 0:
        return bits
    return rounder(abs(magnitude) * scale, bool(bits >> 31))


_PRODUCT_BITS = {}


def model_product(code, nibble, rounder=_search_round):
    """Element product bits; sign follows the operand code signs."""
    cache = _PRODUCT_BITS.setdefault(rounder, {})
    key = (code, nibble)
    if key not in cache:
        sign = bool((code >> 7) ^ (nibble >> 3))
        cache[key] = rounder(abs(decode_a(code) * decode_w(nibble)), sign)
    return cache[key]


def model_weight_nibble(w_packed, index):
    byte = w_packed[index // 2]
    return (byte >> 4) & 0xF if index % 2 else byte & 0xF


def expected_trace(a_codes, w_packed, scale_a, scale_b, rounder=_search_round):
    """Independent stage-by-stage expectation for the whole pipeline."""
    blocks = []
    acc_bits = 0
    for j in range(len(a_codes) // 32):
        products, partial_sums = [], []
        dot_bits = 0
        for r in range(32):
            index = 32 * j + r
            product = model_product(a_codes[index], model_weight_nibble(w_packed, index), rounder)
            dot_bits = model_add(dot_bits, product, rounder)
            products.append(product)
            partial_sums.append(dot_bits)
        after_a_bits = model_scale_mul(dot_bits, decode_scale(scale_a[j]), rounder)
        block_bits = model_scale_mul(after_a_bits, decode_scale(scale_b[j]), rounder)
        acc_bits = model_add(acc_bits, block_bits, rounder)
        blocks.append({
            "products": tuple(products),
            "partial_sums": tuple(partial_sums),
            "dot_bits": dot_bits,
            "after_a_bits": after_a_bits,
            "block_bits": block_bits,
            "acc_bits": acc_bits,
        })
    bf16_bits = NAN_SENTINEL if acc_bits == NAN_SENTINEL else nearest_bf16_bits(acc_bits)
    return blocks, acc_bits, bf16_bits


_PRODUCT_PAIRS = None


def product_pair(value):
    """Find an (a_code, nibble) pair whose exact product equals the value."""
    global _PRODUCT_PAIRS
    if _PRODUCT_PAIRS is None:
        table = {}
        for code in VALID_A_CODES:
            a_value = decode_a(code)
            for nibble in range(16):
                table.setdefault(a_value * decode_w(nibble), (code, nibble))
        _PRODUCT_PAIRS = table
    pair = _PRODUCT_PAIRS.get(value)
    if pair is None:
        raise AssertionError(f"no encoded product equals {value}")
    return pair


def build_case(block_terms, scale_a, scale_b):
    """Encode per-element signed products and per-block scale codes into inputs."""
    a_codes, nibbles = [], []
    for terms in block_terms:
        if len(terms) > 32:
            raise AssertionError("at most 32 products per block")
        pairs = [product_pair(term) for term in terms]
        pairs.extend([(0, 0)] * (32 - len(pairs)))
        a_codes.extend(code for code, _ in pairs)
        nibbles.extend(nibble for _, nibble in pairs)
    return tuple(a_codes), independent_pack(nibbles), tuple(scale_a), tuple(scale_b)


def check_bits(test, actual, expected, label, width):
    """Compare one stage; NaN by classification, everything else by bits."""
    nan_checker = is_nan_fp32_bits if width == 32 else is_nan_bf16_bits
    if expected == NAN_SENTINEL:
        test.assertTrue(nan_checker(actual), f"{label}: NaN classification expected, got {actual:#010x}")
    else:
        test.assertIs(type(actual), int, label)
        test.assertEqual(actual, expected, f"{label}: bits")


def check_pipeline(test, a_codes, w_packed, scale_a, scale_b, *, tag):
    """Compare both entries and every trace stage against the expectations."""
    blocks, fp32_bits, bf16_bits = expected_trace(a_codes, w_packed, scale_a, scale_b)
    trace = trace_expert_grouped_dot(a_codes, w_packed, scale_a, scale_b)
    direct = expert_grouped_dot(a_codes, w_packed, scale_a, scale_b)
    bf16_result = expert_grouped_dot(a_codes, w_packed, scale_a, scale_b, output="bf16")
    test.assertIsInstance(trace, ComputeTrace)
    test.assertEqual(len(trace.blocks), len(blocks), f"{tag}: block count")
    for index, (got, want) in enumerate(zip(trace.blocks, blocks)):
        test.assertEqual(len(got.products), 32, f"{tag}: block {index} products")
        test.assertEqual(len(got.partial_sums), 32, f"{tag}: block {index} partial sums")
        for r in range(32):
            check_bits(test, got.products[r], want["products"][r], f"{tag}: b{index} product {r}", 32)
            check_bits(test, got.partial_sums[r], want["partial_sums"][r], f"{tag}: b{index} partial {r}", 32)
        check_bits(test, got.dot_bits, want["dot_bits"], f"{tag}: b{index} dot", 32)
        check_bits(test, got.after_a_bits, want["after_a_bits"], f"{tag}: b{index} after scale_a", 32)
        check_bits(test, got.block_bits, want["block_bits"], f"{tag}: b{index} block", 32)
        check_bits(test, got.acc_bits, want["acc_bits"], f"{tag}: b{index} acc", 32)
    check_bits(test, trace.fp32_bits, fp32_bits, f"{tag}: fp32", 32)
    check_bits(test, trace.bf16_bits, bf16_bits, f"{tag}: bf16", 16)
    test.assertIsInstance(trace.result, float, f"{tag}: fp32 result type")
    test.assertIs(type(direct), float, f"{tag}: direct result type")
    if fp32_bits == NAN_SENTINEL:
        test.assertTrue(math.isnan(trace.result), f"{tag}: fp32 result NaN")
        test.assertTrue(math.isnan(direct), f"{tag}: direct result NaN")
    else:
        test.assertEqual(host_fp32_bits(trace.result), fp32_bits, f"{tag}: fp32 result bits")
        test.assertEqual(host_fp32_value(fp32_bits), trace.result, f"{tag}: fp32 result on grid")
        test.assertEqual(host_fp32_bits(direct), fp32_bits, f"{tag}: direct result bits")
        test.assertEqual(direct, trace.result, f"{tag}: direct equals trace")
    test.assertIs(type(bf16_result), int, f"{tag}: bf16 result type")
    check_bits(test, bf16_result, bf16_bits, f"{tag}: bf16 result", 16)


class ExpectationCrossChecks(unittest.TestCase):
    """The integer RNE must agree with the primary search-based rounding."""

    def test_integer_rne_matches_neighbor_grid_search(self):
        directed = [
            Fraction(0), power2(-151), power2(-150), power2(-149), 3 * power2(-150),
            power2(-148), power2(-126), power2(-125), Fraction(1) + power2(-24),
            Fraction(1) + 3 * power2(-24), Fraction(2 ** 24) - 1, Fraction(2 ** 24) + 1,
            Fraction(2 ** 25 - 1) * power2(103), Fraction(2 ** 25 - 2) * power2(103),
            Fraction(2 ** 24 - 1) * power2(104), Fraction(2 ** 128) - power2(103),
        ]
        rng = random.Random(0x5812)
        for _ in range(256):
            directed.append(Fraction(rng.randrange(1, 1 << 40)) * power2(rng.randrange(-200, 131)))
        for magnitude in directed:
            for sign in (False, True):
                with self.subTest(magnitude=str(magnitude), sign=sign):
                    self.assertEqual(
                        rne_fp32_bits(magnitude, sign),
                        nearest_fp32_bits(-magnitude if sign and magnitude else magnitude, sign and magnitude == 0),
                    )

    def test_model_pipeline_matches_search_rounded_pipeline(self):
        rng = random.Random(0x9A12)
        cases = [
            build_case([[Fraction(2688)] * 7 + [power2(-10)] + [Fraction(-2688)] * 7], (0x7F,), (0x7F,)),
            build_case([[Fraction(2)]], (0xFE,), (0x00,)),
            build_case([[Fraction(1)], [Fraction(1)], [Fraction(-1)]], (0x7F, 0x97, 0x97), (0x7F,) * 3),
            build_case([[power2(-10)], [power2(-10)]], (0x00, 0x00), (0x73, 0x72)),
            build_case([[-power2(-10)]], (0x00,), (0x73,)),
            build_case([[Fraction(2688)] * 32, [Fraction(1)], [Fraction(-2688)] * 32, [Fraction(1)], [Fraction(-1)]],
                       (0xFE, 0x7F, 0xFE, 0x7F, 0x7F), (0x7F,) * 5),
        ]
        for _ in range(6):
            block_count = rng.choice((1, 2, 5))
            cases.append((
                tuple(rng.choice(VALID_A_CODES) for _ in range(32 * block_count)),
                independent_pack([rng.randrange(16) for _ in range(32 * block_count)]),
                tuple(rng.randrange(255) for _ in range(block_count)),
                tuple(rng.randrange(255) for _ in range(block_count)),
            ))
        for a_codes, w_packed, scale_a, scale_b in cases:
            with self.subTest(k=len(a_codes)):
                self.assertEqual(
                    expected_trace(a_codes, w_packed, scale_a, scale_b),
                    expected_trace(a_codes, w_packed, scale_a, scale_b, rounder=rne_fp32_bits),
                )


class ValidationTests(unittest.TestCase):
    """Input failure contract: metadata before content, exact errors per class."""

    def baseline(self):
        return (0x38,) * 32, bytes([0x22] * 16), (0x7F,), (0x7F,)

    def test_metadata_errors(self):
        a, w, sa, sb = self.baseline()
        cases = (
            ((), b"", (), ()),
            ((), b"\x00", (), ()),
            ((0x38,) * 16, bytes(8), (0x7F,), (0x7F,)),
            ((0x38,) * 33, bytes(16), (0x7F,), (0x7F,)),
            ((0x38,) * 96, bytes(47), (0x7F,) * 3, (0x7F,) * 3),
            ((0x38,) * 96, bytes(48), (0x7F,) * 2, (0x7F,) * 3),
            ((0x38,) * 96, bytes(48), (0x7F,) * 3, (0x7F,) * 4),
        )
        for entry in ENTRY_POINTS:
            for bad_a, bad_w, bad_sa, bad_sb in cases:
                with self.subTest(entry=entry.__name__, k=len(bad_a), w=len(bad_w)):
                    with self.assertRaises(ValueError):
                        entry(bad_a, bad_w, bad_sa, bad_sb)
            for mode in ("float32", "BF16", "fp64", "", None):
                with self.subTest(entry=entry.__name__, mode=mode):
                    with self.assertRaises(ValueError):
                        entry(a, w, sa, sb, output=mode)

    def test_metadata_validated_before_content(self):
        a, w, sa, sb = self.baseline()
        float_a = (1.0,) + a[1:]
        for entry in ENTRY_POINTS:
            with self.subTest(entry=entry.__name__, kind="scale length"):
                with self.assertRaises(ValueError):
                    entry(float_a, w, (0x7F, 0x7F), sb)
            with self.subTest(entry=entry.__name__, kind="output mode"):
                with self.assertRaises(ValueError):
                    entry(float_a, w, sa, sb, output="fp64")
            with self.subTest(entry=entry.__name__, kind="K grouping"):
                with self.assertRaises(ValueError):
                    entry((0x38, 1.0) + (0x38,) * 31, bytes(16), (0x7F,), (0x7F,))

    def test_content_errors(self):
        a, w, sa, sb = self.baseline()
        a_values = ((0x7F, ValueError), (0xFF, ValueError), (256, ValueError), (-1, ValueError),
                    (-128, ValueError), (10 ** 100, ValueError), (True, TypeError), (1.0, TypeError),
                    ("0", TypeError), (None, TypeError))
        scale_values = ((255, ValueError), (256, ValueError), (-1, ValueError), (True, TypeError),
                        (0.5, TypeError), ("1", TypeError))
        for entry in ENTRY_POINTS:
            for bad, error in a_values:
                with self.subTest(entry=entry.__name__, input="a_codes", bad=bad):
                    with self.assertRaises(error):
                        entry((a[0],) + (bad,) + a[2:], w, sa, sb)
            for name, values in (("scale_a", scale_values), ("scale_b", scale_values)):
                for bad, error in values:
                    with self.subTest(entry=entry.__name__, input=name, bad=bad):
                        scales = list(sa if name == "scale_a" else sb)
                        scales[0] = bad
                        with self.assertRaises(error):
                            entry(a, w, scales if name == "scale_a" else sa, scales if name == "scale_b" else sb)

    def test_packed_content_and_validate_before_compute(self):
        a, w, sa, sb = self.baseline()
        for entry in ENTRY_POINTS:
            for bad, error in ((-1, ValueError), (256, ValueError), (True, TypeError),
                               (1.0, TypeError), ("0", TypeError), (None, TypeError)):
                with self.subTest(entry=entry.__name__, bad=bad):
                    with self.assertRaises(error):
                        entry(a, list(w[:-1]) + [bad], sa, sb)
            with patch("ref.compute._add_fp32", side_effect=AssertionError("computed before validation")):
                with self.assertRaises(TypeError):
                    entry(a[:-1] + (True,), w, sa, sb)
                with self.assertRaises(TypeError):
                    entry(a, list(w[:-1]) + [True], sa, sb)
                with self.assertRaises(ValueError):
                    entry(a, w, sa, (0xFF,))

    def test_non_sequence_and_nested_inputs(self):
        a, w, sa, sb = self.baseline()
        for entry in ENTRY_POINTS:
            for bad in (5, 1.5, None, iter((1, 2))):
                with self.subTest(entry=entry.__name__, bad=bad):
                    with self.assertRaises(TypeError):
                        entry(bad, w, sa, sb)
                    with self.assertRaises(TypeError):
                        entry(a, bad, sa, sb)
            with self.subTest(entry=entry.__name__, nested=True):
                with self.assertRaises(TypeError):
                    entry([[0x38]] * 32, w, sa, sb)


class StructureTests(unittest.TestCase):
    """Trace shape, field types, and mode independence of the pipeline."""

    def test_trace_structure_types_and_mode_independence(self):
        a, w, sa, sb = build_case([[Fraction(1)]] * 5, (0x7F, 0x80, 0x81, 0x7E, 0x82), (0x7F,) * 5)
        trace = trace_expert_grouped_dot(a, w, sa, sb)
        bf16_trace = trace_expert_grouped_dot(a, w, sa, sb, output="bf16")
        fp32_value = expert_grouped_dot(a, w, sa, sb)
        bf16_value = expert_grouped_dot(a, w, sa, sb, output="bf16")
        self.assertIsInstance(trace, ComputeTrace)
        self.assertIs(type(trace.blocks), tuple)
        self.assertEqual(len(trace.blocks), 5)
        for block in trace.blocks:
            self.assertIsInstance(block, BlockTrace)
            for name in ("products", "partial_sums"):
                field = getattr(block, name)
                self.assertIs(type(field), tuple)
                self.assertEqual(len(field), 32)
                for bits in field:
                    self.assertIs(type(bits), int)
            for name in ("dot_bits", "after_a_bits", "block_bits", "acc_bits"):
                self.assertIs(type(getattr(block, name)), int)
        for name in ("fp32_bits", "bf16_bits"):
            self.assertIs(type(getattr(trace, name)), int)
        self.assertIsInstance(trace.result, float)
        self.assertIs(type(bf16_trace.result), int)
        self.assertIs(type(bf16_value), int)
        self.assertEqual(fp32_value, trace.result)
        self.assertEqual(bf16_value, bf16_trace.result)
        self.assertEqual(trace.blocks, bf16_trace.blocks)
        self.assertEqual(trace.fp32_bits, bf16_trace.fp32_bits)
        self.assertEqual(trace.bf16_bits, bf16_trace.bf16_bits)


class DirectedCases(unittest.TestCase):
    """Directed counterexamples and boundaries; the sensitivity mutation targets."""

    def test_directed_in_block_sequential_order(self):
        a, w, sa, sb = build_case([[Fraction(2688)] * 7 + [power2(-10)] + [Fraction(-2688)] * 7], (0x7F,), (0x7F,))
        check_pipeline(self, a, w, sa, sb, tag="sequential")
        self.assertEqual(trace_expert_grouped_dot(a, w, sa, sb).fp32_bits, 0x00000000)

    def test_directed_scale_application_order(self):
        a, w, sa, sb = build_case([[Fraction(2)]], (0xFE,), (0x00,))
        check_pipeline(self, a, w, sa, sb, tag="scale-order")
        trace = trace_expert_grouped_dot(a, w, sa, sb)
        self.assertEqual(trace.blocks[0].after_a_bits, 0x7F800000)
        self.assertEqual(trace.blocks[0].block_bits, 0x7F800000)
        self.assertEqual(trace.fp32_bits, 0x7F800000)
        self.assertEqual(trace.bf16_bits, 0x7F80)
        self.assertEqual(expert_grouped_dot(a, w, sa, sb, output="bf16"), 0x7F80)

    def test_directed_cross_block_accumulation_order(self):
        a, w, sa, sb = build_case([[Fraction(1)], [Fraction(1)], [Fraction(-1)]], (0x7F, 0x97, 0x97), (0x7F,) * 3)
        check_pipeline(self, a, w, sa, sb, tag="block-order")
        trace = trace_expert_grouped_dot(a, w, sa, sb)
        self.assertEqual([block.acc_bits for block in trace.blocks], [0x3F800000, 0x4B800000, 0x00000000])
        self.assertEqual(trace.fp32_bits, 0x00000000)

    def test_directed_scale_multiplication_rounding_no_fma(self):
        a, w, sa, sb = build_case([[power2(-10)], [power2(-10)]], (0x00, 0x00), (0x73, 0x72))
        check_pipeline(self, a, w, sa, sb, tag="no-fma")
        trace = trace_expert_grouped_dot(a, w, sa, sb)
        self.assertEqual([block.block_bits for block in trace.blocks], [0x00000001, 0x00000000])
        self.assertEqual(trace.fp32_bits, 0x00000001)

    def test_directed_final_bf16_only_at_end(self):
        a, w, sa, sb = build_case([[Fraction(1), power2(-8)], [Fraction(-1)]], (0x7F, 0x7F), (0x7F, 0x7F))
        check_pipeline(self, a, w, sa, sb, tag="bf16-end")
        trace = trace_expert_grouped_dot(a, w, sa, sb)
        self.assertEqual(trace.fp32_bits, 0x3B800000)
        self.assertEqual(trace.bf16_bits, 0x3B80)
        self.assertEqual(expert_grouped_dot(a, w, sa, sb), host_fp32_value(0x3B800000))

    def test_directed_negative_zero_stage_boundaries(self):
        a, w, sa, sb = build_case([[-power2(-10)]], (0x00,), (0x72,))
        check_pipeline(self, a, w, sa, sb, tag="neg-zero")
        trace = trace_expert_grouped_dot(a, w, sa, sb)
        self.assertEqual(trace.blocks[0].block_bits, 0x80000000)
        self.assertEqual(trace.blocks[0].acc_bits, 0x00000000)
        a2, w2, sa2, sb2 = build_case([[-power2(-10)]], (0x00,), (0x73,))
        check_pipeline(self, a2, w2, sa2, sb2, tag="neg-subnormal")
        trace2 = trace_expert_grouped_dot(a2, w2, sa2, sb2)
        self.assertEqual(trace2.blocks[0].block_bits, 0x80000001)
        self.assertEqual(trace2.fp32_bits, 0x80000001)
        self.assertEqual(trace2.bf16_bits, 0x8000)
        self.assertEqual(expert_grouped_dot(a2, w2, sa2, sb2, output="bf16"), 0x8000)

    def test_directed_five_block_scale_grouping(self):
        a, w, sa, sb = build_case([[Fraction(1)]] * 5, (0x7F, 0x80, 0x81, 0x7E, 0x82), (0x7F,) * 5)
        check_pipeline(self, a, w, sa, sb, tag="k160")
        trace = trace_expert_grouped_dot(a, w, sa, sb)
        self.assertEqual(
            [block.after_a_bits for block in trace.blocks],
            [0x3F800000, 0x40000000, 0x40800000, 0x3F000000, 0x41000000],
        )
        self.assertEqual(trace.fp32_bits, 0x41780000)

    def test_directed_both_scale_group_indices(self):
        a, w, sa, sb = build_case(
            [[Fraction(1)]] * 5,
            (0x7F, 0x80, 0x81, 0x7E, 0x82), (0x80, 0x7F, 0x7D, 0x82, 0x7E),
        )
        check_pipeline(self, a, w, sa, sb, tag="both-scale-groups")
        trace = trace_expert_grouped_dot(a, w, sa, sb)
        self.assertEqual([block.block_bits for block in trace.blocks],
                         [0x40000000, 0x40000000, 0x3F800000, 0x40800000, 0x40800000])
        self.assertEqual(trace.fp32_bits, 0x41500000)

    def test_directed_normal_subnormal_boundary_and_carry(self):
        terms = [Fraction(2688)] * 6 + [Fraction(256)]
        for extra, expected in ((-2 * power2(-10), 0x007FFFFF), (Fraction(0), 0x00800000),
                                (2 * power2(-10), 0x00800001)):
            with self.subTest(extra=extra):
                a, w, sa, sb = build_case([terms + [extra]], (0x00,), (0x72,))
                check_pipeline(self, a, w, sa, sb, tag="normal-subnormal")
                self.assertEqual(trace_expert_grouped_dot(a, w, sa, sb).fp32_bits, expected)
        a, w, sa, sb = build_case([terms + [-2 * power2(-10)], [power2(-10)]],
                                 (0x00, 0x00), (0x72, 0x73))
        check_pipeline(self, a, w, sa, sb, tag="subnormal-carry")
        self.assertEqual(trace_expert_grouped_dot(a, w, sa, sb).fp32_bits, 0x00800000)

    def test_directed_packed_nibble_order(self):
        a, w, sa, sb = build_case([[power2(-10), Fraction(448)] * 16, []], (0x7F, 0x7F), (0x7F, 0x7F))
        self.assertEqual((w[0] & 0xF, w[0] >> 4), (1, 6))
        check_pipeline(self, a, w, sa, sb, tag="nibbles")

    def test_directed_accumulator_midpoint_and_sides(self):
        cases = (
            ([Fraction(1), -power2(-10)], 0x4B800000),
            ([Fraction(1)], 0x4B800000),
            ([Fraction(1), power2(-10)], 0x4B800001),
        )
        for terms, fp32_bits in cases:
            with self.subTest(terms=terms):
                a, w, sa, sb = build_case([[Fraction(1)], terms], (0x97, 0x7F), (0x7F, 0x7F))
                check_pipeline(self, a, w, sa, sb, tag="acc-midpoint")
                self.assertEqual(trace_expert_grouped_dot(a, w, sa, sb).fp32_bits, fp32_bits)

    def test_directed_dot_midpoint_and_sides(self):
        cases = ((power2(-10), 0x46800000), (3 * power2(-10), 0x46800002))
        for extra, dot_bits in cases:
            with self.subTest(extra=extra):
                a, w, sa, sb = build_case(
                    [[Fraction(2688)] * 6 + [Fraction(256)] + [extra]], (0x7F,), (0x7F,))
                check_pipeline(self, a, w, sa, sb, tag="dot-midpoint")
                trace = trace_expert_grouped_dot(a, w, sa, sb)
                self.assertEqual(trace.blocks[0].dot_bits, dot_bits)
                self.assertEqual(trace.fp32_bits, dot_bits)

    def test_directed_bf16_final_tie_and_sides(self):
        cases = ((-power2(-10), 0x3F80), (None, 0x3F80), (power2(-10), 0x3F81))
        for extra, bf16_bits in cases:
            with self.subTest(extra=extra):
                terms = [Fraction(1), power2(-8)] + ([extra] if extra is not None else [])
                a, w, sa, sb = build_case([terms], (0x7F,), (0x7F,))
                check_pipeline(self, a, w, sa, sb, tag="bf16-tie")
                self.assertEqual(trace_expert_grouped_dot(a, w, sa, sb).bf16_bits, bf16_bits)
                self.assertEqual(expert_grouped_dot(a, w, sa, sb, output="bf16"), bf16_bits)

    def test_directed_inf_and_nan_propagation(self):
        max_block = [Fraction(2688)] * 32
        subcases = (
            ("inf plus finite", [max_block, [Fraction(1)]], (0xFE, 0x7F), (0x7F,) * 2, 0x7F800000, 0x7F80),
            ("same sign infinities", [max_block, max_block], (0xFE, 0xFE), (0x7F, 0x00), 0x7F800000, 0x7F80),
            ("negative infinity plus finite", [[-t for t in max_block], [Fraction(1)]], (0xFE, 0x7F), (0x00, 0x7F), 0xFF800000, 0xFF80),
        )
        for name, blocks, sa, sb, fp32_bits, bf16_bits in subcases:
            with self.subTest(case=name):
                a, w, sa_codes, sb_codes = build_case(blocks, sa, sb)
                check_pipeline(self, a, w, sa_codes, sb_codes, tag=name)
                trace = trace_expert_grouped_dot(a, w, sa_codes, sb_codes)
                self.assertEqual(trace.fp32_bits, fp32_bits)
                self.assertEqual(trace.bf16_bits, bf16_bits)
        a, w, sa, sb = build_case(
            [max_block, [Fraction(1)], [-t for t in max_block], [Fraction(1)], [Fraction(-1)]],
            (0xFE, 0x7F, 0xFE, 0x7F, 0x7F), (0x7F,) * 5)
        check_pipeline(self, a, w, sa, sb, tag="opposite infinities")
        trace = trace_expert_grouped_dot(a, w, sa, sb)
        self.assertTrue(is_nan_fp32_bits(trace.blocks[2].acc_bits))
        self.assertTrue(is_nan_fp32_bits(trace.blocks[3].acc_bits))
        self.assertTrue(is_nan_fp32_bits(trace.blocks[4].acc_bits))
        self.assertTrue(is_nan_fp32_bits(trace.fp32_bits))
        self.assertTrue(is_nan_bf16_bits(trace.bf16_bits))
        self.assertTrue(math.isnan(expert_grouped_dot(a, w, sa, sb)))
        self.assertTrue(is_nan_bf16_bits(expert_grouped_dot(a, w, sa, sb, output="bf16")))

    def test_directed_finite_block_accumulation_overflow(self):
        a, w, sa, sb = build_case([[Fraction(1)], [Fraction(1)]], (0xFE, 0xFE), (0x7F, 0x7F))
        check_pipeline(self, a, w, sa, sb, tag="acc-overflow")
        trace = trace_expert_grouped_dot(a, w, sa, sb)
        self.assertEqual([block.block_bits for block in trace.blocks], [0x7F000000, 0x7F000000])
        self.assertEqual([block.acc_bits for block in trace.blocks], [0x7F000000, 0x7F800000])

    def test_directed_underflow_zero_and_subnormal(self):
        for sb_code, block_bits in ((0x73, 0x00000001), (0x72, 0x00000000)):
            with self.subTest(scale_b=hex(sb_code)):
                a, w, sa, sb = build_case([[power2(-10)]], (0x00,), (sb_code,))
                check_pipeline(self, a, w, sa, sb, tag="underflow")
                trace = trace_expert_grouped_dot(a, w, sa, sb)
                self.assertEqual(trace.blocks[0].block_bits, block_bits)
                self.assertEqual(trace.fp32_bits, block_bits)
                self.assertEqual(trace.bf16_bits, 0x0000)


DIRECTED_TEST_NAMES = tuple(sorted(name for name in vars(DirectedCases) if name.startswith("test_")))

EARLY_BF16_HELPER = '''


def _p0012_mutant_bf16(block):
    """Round block to bf16 and back to fp32 (sensitivity mutation)."""
    import struct
    bits = struct.unpack("<I", struct.pack("<f", block))[0]
    if bits & 0x7F800000 == 0x7F800000:
        return block
    magnitude = bits & 0x7FFFFFFF
    dropped = magnitude & 0xFFFF
    quotient = magnitude >> 16
    if dropped > 0x8000 or (dropped == 0x8000 and quotient & 1):
        quotient += 1
    return struct.unpack("<f", struct.pack("<I", (bits & 0x80000000) | (quotient << 16)))[0]
'''

MUTATIONS = (
    {
        "name": "activation scale shared per four blocks",
        "slug": "scale-group",
        "anchor": "decode_e8m0(scale_a[j])",
        "replacement": "decode_e8m0(scale_a[j // 4])",
        "helper": None,
        "detector": "DirectedCases.test_directed_five_block_scale_grouping",
    },
    {
        "name": "early bf16 rounding before accumulation",
        "slug": "early-bf16",
        "anchor": "acc = _add_fp32(acc, block)",
        "replacement": "acc = _add_fp32(acc, _p0012_mutant_bf16(block))",
        "helper": EARLY_BF16_HELPER,
        "detector": "DirectedCases.test_directed_final_bf16_only_at_end",
    },
    {
        "name": "reversed cross-block accumulation order",
        "slug": "reverse-blocks",
        "anchor": "for j in range(k // 32):",
        "replacement": "for j in reversed(range(k // 32)):",
        "helper": None,
        "detector": "DirectedCases.test_directed_cross_block_accumulation_order",
    },
)


class ExhaustiveCodeTests(unittest.TestCase):
    """All 4064 finite codepoint pairs and both scale encoding boundaries."""

    def test_all_codepoint_pairs_and_scale_encoding_boundaries(self):
        boundaries = (0x00, 0x01, 0x02, 0x7D, 0x7E, 0x7F, 0x80, 0x81, 0xFC, 0xFD, 0xFE)
        pairs = [(code, nibble) for code in VALID_A_CODES for nibble in range(16)]
        self.assertEqual(len(pairs), 4064)
        for run in range(len(pairs) // 32):
            chunk = pairs[32 * run:32 * (run + 1)]
            a_codes = tuple(code for code, _ in chunk)
            w_packed = independent_pack([nibble for _, nibble in chunk])
            scale_a = (boundaries[run % len(boundaries)],)
            scale_b = (boundaries[(run * 5 + 3) % len(boundaries)],)
            with self.subTest(run=run):
                check_pipeline(self, a_codes, w_packed, scale_a, scale_b, tag=f"codepoints-{run}")


class RandomPipelineTests(unittest.TestCase):
    """Fixed-seed random pipelines as a supplement to the directed cases."""

    def test_fixed_seed_random_pipelines(self):
        rng = random.Random(0x12)
        for case in range(48):
            block_count = rng.choice((1, 2, 3, 4, 5, 7))
            a_codes = tuple(rng.choice(VALID_A_CODES) for _ in range(32 * block_count))
            w_packed = independent_pack([rng.randrange(16) for _ in range(32 * block_count)])
            scale_a = tuple(rng.randrange(255) for _ in range(block_count))
            scale_b = tuple(rng.randrange(255) for _ in range(block_count))
            with self.subTest(case=case, blocks=block_count):
                check_pipeline(self, a_codes, w_packed, scale_a, scale_b, tag=f"random-{case}")


class PrivateAddTests(unittest.TestCase):
    """Private adder coverage, not a substitute for public-path evidence.

    Same-sign negative-zero addition and arbitrary input-rounding boundaries
    are not all reachable from the encoded pipeline. Public accumulator
    Inf/NaN propagation is checked separately by DirectedCases.
    """

    def assert_add_bits(self, left, right, expected):
        if expected == NAN_SENTINEL:
            self.assertTrue(math.isnan(_add_fp32(left, right)), (left, right))
        else:
            self.assertIs(type(_add_fp32(left, right)), float, (left, right))
            self.assertEqual(host_fp32_bits(_add_fp32(left, right)), expected, (left, right))

    def test_unreachable_sign_zeros_and_special_values(self):
        min_sub = host_fp32_value(0x00000001)
        checks = (
            (-0.0, -0.0, 0x80000000),
            (-0.0, 0.0, 0x00000000),
            (0.0, -0.0, 0x00000000),
            (-1.0, 1.0, 0x00000000),
            (1.0, -1.0, 0x00000000),
            (-0.0, -min_sub, 0x80000001),
            (-0.0, min_sub, 0x00000001),
            (math.inf, -math.inf, NAN_SENTINEL),
            (math.inf, math.inf, 0x7F800000),
            (-math.inf, -math.inf, 0xFF800000),
            (math.inf, 1.0, 0x7F800000),
            (-math.inf, 1.0, 0xFF800000),
            (math.nan, 1.0, NAN_SENTINEL),
            (1.0, math.nan, NAN_SENTINEL),
            (math.nan, math.inf, NAN_SENTINEL),
            (math.nan, math.nan, NAN_SENTINEL),
        )
        for left, right, expected in checks:
            with self.subTest(left=left, right=right):
                self.assert_add_bits(left, right, expected)

    def test_directed_rne_ties_and_boundaries(self):
        min_sub = host_fp32_value(0x00000001)
        checks = (
            (2.0 ** 24, 1.0, 0x4B800000),
            (2.0 ** 24, 3.0, 0x4B800002),
            (2.0 ** 24 - 2.0, 2.0, 0x4B800000),
            (2.0 ** 24, min_sub, 0x4B800000),
            (1.0, 2.0 ** -25, 0x3F800000),
            (1.0, 2.0 ** -24, 0x3F800000),
            (1.0 + 2.0 ** -23, 2.0 ** -24, 0x3F800002),
            (1.0 + 2.0 ** -23, -(2.0 ** -24), 0x3F800000),
            (host_fp32_value(0x007FFFFF), min_sub, 0x00800000),
            (host_fp32_value(0x007FFFFF), host_fp32_value(0x007FFFFF), 0x00FFFFFE),
            (host_fp32_value(0x00000001), 1.5 * 2.0 ** -149, 0x00000003),
            (host_fp32_value(0x00000001), -(1.5 * 2.0 ** -149), 0x80000001),
            (0.0, -(2.0 ** -150), 0x00000000),
        )
        for left, right, expected in checks:
            with self.subTest(left=left, right=right):
                self.assert_add_bits(left, right, expected)

    def test_matches_model_and_neighbor_search_on_random_pairs(self):
        rng = random.Random(0xAD5)
        for case in range(384):
            left_bits = rng.randrange(0x7F800000) | (rng.randrange(2) << 31)
            right_bits = rng.randrange(0x7F800000) | (rng.randrange(2) << 31)
            expected = model_add(left_bits, right_bits, rne_fp32_bits)
            left, right = host_fp32_value(left_bits), host_fp32_value(right_bits)
            with self.subTest(case=case, left=hex(left_bits), right=hex(right_bits)):
                self.assert_add_bits(left, right, expected)
            if case % 4 == 0:
                exact = finite_value(left_bits) + finite_value(right_bits)
                sign = exact < 0 or (exact == 0 and (left_bits >> 31) and (right_bits >> 31))
                self.assertEqual(expected, nearest_fp32_bits(exact, bool(sign)))


class MarkerProtocolError(AssertionError):
    """Raised when a subprocess marker is missing, malformed, or tampered."""


def iter_tests(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from iter_tests(item)
        else:
            yield item


def test_key(test):
    parent = getattr(test, "test_case", test)
    return f"{parent.__class__.__name__}.{parent._testMethodName}"


def marker_payload(suite):
    """Run the suite quietly and build the digested single-line marker payload."""
    tests = list(iter_tests(suite))
    result = unittest.TestResult()
    suite.run(result)
    failed = {test_key(test) for test, _ in result.failures}
    errored = {test_key(test) for test, _ in result.errors}
    skipped = {test_key(test) for test, _ in result.skipped}
    cases = []
    for test in tests:
        key = test_key(test)
        if key in errored:
            status = "error"
        elif key in failed:
            status = "fail"
        elif key in skipped:
            status = "skip"
        else:
            status = "pass"
        cases.append([key, status])
    core = {
        "cases": cases,
        "errors": sum(1 for _, status in cases if status == "error"),
        "failures": sum(1 for _, status in cases if status == "fail"),
        "skipped": sum(1 for _, status in cases if status == "skip"),
        "tests": len(tests),
    }
    payload = dict(core)
    payload["digest"] = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def validate_marker_payload(payload):
    """Reject markers whose shape, counts, case list, or digest disagree."""
    required = {"cases", "errors", "failures", "skipped", "tests", "digest"}
    if not isinstance(payload, dict) or set(payload) != required:
        raise MarkerProtocolError("marker keys do not match the protocol")
    for name in ("errors", "failures", "skipped", "tests"):
        value = payload[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise MarkerProtocolError(f"marker field {name} is not a count")
    cases = payload["cases"]
    if not isinstance(cases, list) or len(cases) != payload["tests"]:
        raise MarkerProtocolError("marker case list does not match the test count")
    statuses = ("pass", "fail", "error", "skip")
    tallies = dict.fromkeys(statuses, 0)
    keys = set()
    for entry in cases:
        if not isinstance(entry, list) or len(entry) != 2 or not isinstance(entry[0], str) or entry[1] not in statuses:
            raise MarkerProtocolError("marker case entry is malformed")
        if entry[0] in keys:
            raise MarkerProtocolError("marker case id duplicated")
        keys.add(entry[0])
        tallies[entry[1]] += 1
    if (tallies["fail"], tallies["error"], tallies["skip"]) != (payload["failures"], payload["errors"], payload["skipped"]):
        raise MarkerProtocolError("marker counts disagree with the case list")
    core = {name: payload[name] for name in ("cases", "errors", "failures", "skipped", "tests")}
    digest = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if digest != payload["digest"]:
        raise MarkerProtocolError("marker digest does not match its content")


def run_marker_subprocess(script, extra_arguments, timeout=300):
    """Run a test copy and return its validated marker payload."""
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    process = subprocess.run(
        [sys.executable, str(script), *[str(argument) for argument in extra_arguments]],
        capture_output=True, text=True, env=environment, timeout=timeout,
    )
    if process.returncode not in (0, 1):
        raise MarkerProtocolError(f"subprocess exited {process.returncode}: {process.stderr.strip()[-400:]}")
    lines = [line for line in process.stdout.splitlines() if line.startswith(MARKER_PREFIX)]
    if len(lines) != 1:
        raise MarkerProtocolError(f"expected exactly one marker line, found {len(lines)}")
    try:
        payload = json.loads(lines[0][len(MARKER_PREFIX):])
    except json.JSONDecodeError as error:
        raise MarkerProtocolError(f"marker is not valid JSON: {error}") from error
    validate_marker_payload(payload)
    expected_exit = int(bool(payload["failures"] or payload["errors"] or payload["skipped"]))
    if process.returncode != expected_exit:
        raise MarkerProtocolError("subprocess exit disagrees with its test result")
    return payload


def build_copy(destination):
    """Copy ref plus the two quant test files into an isolated throwaway tree."""
    (destination / "tests" / "quant").mkdir(parents=True)
    shutil.copytree(ROOT / "ref", destination / "ref")
    shutil.copy(ROOT / "tests" / "quant" / "test_formats.py", destination / "tests" / "quant" / "test_formats.py")
    shutil.copy(Path(__file__).resolve(), destination / "tests" / "quant" / "test_compute.py")
    return destination


class SensitivityTests(unittest.TestCase):
    """Mutation sensitivity, marker failure propagation, and pristine retest."""

    def test_mutation_sensitivity_controls_and_retest(self):
        guarded = (ROOT / "ref" / "compute.py", ROOT / "tests" / "quant" / "test_formats.py", Path(__file__).resolve())
        digests = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in guarded}
        with tempfile.TemporaryDirectory(prefix="p0012-sensitivity-") as base:
            pristine = build_copy(Path(base) / "pristine")
            script = pristine / "tests" / "quant" / "test_compute.py"
            baseline = run_marker_subprocess(script, ["--suite", "directed", "--marker"])
            self.assertEqual(baseline["failures"] + baseline["errors"] + baseline["skipped"], 0)
            self.assertEqual(baseline["tests"], len(DIRECTED_TEST_NAMES))
            for mutation in MUTATIONS:
                with tempfile.TemporaryDirectory(prefix=f"p0012-{mutation['slug']}-") as directory:
                    mutated = build_copy(Path(directory) / "tree")
                    source = mutated / "ref" / "compute.py"
                    text = source.read_text()
                    self.assertEqual(
                        text.count(mutation["anchor"]), 1,
                        f"{mutation['name']}: anchor must occur exactly once",
                    )
                    text = text.replace(mutation["anchor"], mutation["replacement"])
                    if mutation["helper"]:
                        text += mutation["helper"]
                    source.write_text(text)
                    result = run_marker_subprocess(
                        mutated / "tests" / "quant" / "test_compute.py",
                        ["--suite", "directed", "--marker"],
                    )
                    self.assertEqual(result["errors"], 0, f"{mutation['name']}: failures must be numeric")
                    self.assertEqual(result["skipped"], 0, f"{mutation['name']}: skips are not allowed")
                    self.assertEqual(result["tests"], len(DIRECTED_TEST_NAMES))
                    self.assertGreater(result["failures"], 0, f"{mutation['name']}: directed suite must fail")
                    failed = {entry[0] for entry in result["cases"] if entry[1] == "fail"}
                    self.assertIn(mutation["detector"], failed, f"{mutation['name']}: detector case must fail")
            for anchor, replacement, outcome in (
                ("acc = _add_fp32(acc, block)", "raise RuntimeError('P0012 injected compute exception')", "errors"),
                ('result = acc if output == "fp32" else bf16_bits',
                 'result = 0.0 if output == "fp32" else 0', "failures"),
            ):
                with tempfile.TemporaryDirectory(prefix="p0012-propagation-") as directory:
                    mutated = build_copy(Path(directory) / "tree")
                    source = mutated / "ref" / "compute.py"
                    text = source.read_text()
                    self.assertEqual(text.count(anchor), 1)
                    source.write_text(text.replace(anchor, replacement))
                    mutated_script = mutated / "tests" / "quant" / "test_compute.py"
                    result = run_marker_subprocess(mutated_script, ["--suite", "directed", "--marker"])
                    self.assertGreater(result[outcome], 0)
                    self.assertEqual(result["skipped"], 0)
                    process = subprocess.run(
                        [sys.executable, str(mutated_script), "--suite", "directed"],
                        capture_output=True, text=True, timeout=300,
                    )
                    self.assertEqual(process.returncode, 1)
            with self.assertRaises(MarkerProtocolError):
                run_marker_subprocess(Path(__file__).resolve(), ["--control", "raise"])
            with self.assertRaises(MarkerProtocolError):
                run_marker_subprocess(Path(__file__).resolve(), ["--control", "tamper"])
            retest = run_marker_subprocess(script, ["--suite", "directed", "--marker"])
            self.assertEqual(retest["failures"] + retest["errors"] + retest["skipped"], 0)
            self.assertEqual(retest["tests"], len(DIRECTED_TEST_NAMES))
        for path, digest in digests.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest, f"{path} must stay untouched")


SUITE_CLASSES = {
    "all": (ExpectationCrossChecks, ValidationTests, StructureTests, DirectedCases,
            ExhaustiveCodeTests, RandomPipelineTests, PrivateAddTests, SensitivityTests),
    "core": (ExpectationCrossChecks, ValidationTests, StructureTests, DirectedCases,
             ExhaustiveCodeTests, RandomPipelineTests, PrivateAddTests),
    "directed": (DirectedCases,),
    "sensitivity": (SensitivityTests,),
}


def gather_suite(arguments):
    loader = unittest.TestLoader()
    suite = unittest.TestSuite(
        loader.loadTestsFromTestCase(case_class) for case_class in SUITE_CLASSES[arguments.suite]
    )
    if arguments.cases:
        suite = unittest.TestSuite(
            test for test in iter_tests(suite)
            if any(selector in test._testMethodName or selector in test.id() for selector in arguments.cases)
        )
    return suite


def main(argv=None):
    parser = argparse.ArgumentParser(description="P0012 compute oracle acceptance")
    parser.add_argument("--suite", choices=tuple(SUITE_CLASSES), default="all")
    parser.add_argument("--case", dest="cases", action="append", default=[])
    parser.add_argument("--marker", action="store_true")
    parser.add_argument("--control", choices=("raise", "tamper"))
    arguments = parser.parse_args(argv)
    if arguments.control == "raise":
        raise RuntimeError("P0012 sensitivity control: simulated runner crash")
    if arguments.control == "tamper":
        arguments.suite = "directed"
        payload = marker_payload(gather_suite(arguments))
        for entry in payload["cases"]:
            if entry[1] == "pass":
                entry[1] = "fail"
                payload["failures"] += 1
                break
        else:
            payload["cases"][0][1] = "pass"
            payload["failures"] -= 1
        print(MARKER_PREFIX + json.dumps(payload), flush=True)
        return 0
    suite = gather_suite(arguments)
    if suite.countTestCases() == 0:
        print("no tests selected", file=sys.stderr)
        return 2
    if arguments.marker:
        payload = marker_payload(suite)
        print(MARKER_PREFIX + json.dumps(payload), flush=True)
        return int(bool(payload["failures"] or payload["errors"] or payload["skipped"]))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() and not result.skipped else 1


if __name__ == "__main__":
    print("P0012 mandatory CPU compute acceptance; standard library only", flush=True)
    print(f"Python: {sys.version}; platform: {platform.platform()}; machine: {platform.machine()}", flush=True)
    sys.exit(main())
