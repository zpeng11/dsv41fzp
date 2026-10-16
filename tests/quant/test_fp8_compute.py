#!/usr/bin/env python3
"""P0027 FP8 block-scaled dot acceptance; Python standard library only.

Expectations use Fraction bit-field decodes and ordered binary32/BF16 grids from
``test_formats``, plus independently validated Fraction arithmetic components
from ``test_compute``. They never call a ref decode or compute helper.
Every product, partial sum, scale stage, block accumulator, and both public
outputs is checked.  Suites are independently reproducible with ``--suite``.
"""

import argparse
import ctypes
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import subprocess
import sys
import tempfile
import unittest
from collections.abc import Sequence
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_formats import (  # noqa: E402
    expected_e4m3fn,
    expected_e8m0,
    host_fp32_bits,
    nearest_bf16_bits,
    nearest_fp32_bits,
    power2,
)
from test_compute import (  # noqa: E402
    model_add as compute_model_add,
    model_scale_mul as compute_model_scale,
    rne_fp32_bits,
)
from ref.act_quant import act_quant  # noqa: E402
from ref.compute import BlockTrace, ComputeTrace  # noqa: E402
from ref.fp8_compute import (  # noqa: E402
    fp8_grouped_dot,
    trace_fp8_grouped_dot,
)

ENTRY_POINTS = (fp8_grouped_dot, trace_fp8_grouped_dot)
VALID_CODES = tuple(code for code in range(256) if code not in (0x7F, 0xFF))
NAN_SENTINEL = -1


@lru_cache(maxsize=None)
def decode_fp8(code):
    return expected_e4m3fn(code)


@lru_cache(maxsize=None)
def decode_scale(code):
    return expected_e8m0(code)


def is_nan_fp32_bits(bits):
    return (bits & 0x7F800000) == 0x7F800000 and (bits & 0x7FFFFF) != 0


def is_nan_bf16_bits(bits):
    return (bits & 0x7F80) == 0x7F80 and (bits & 0x7F) != 0


def search_round(magnitude, sign):
    if magnitude == 0:
        return nearest_fp32_bits(Fraction(0), bool(sign))
    return nearest_fp32_bits(-magnitude if sign else magnitude)


def model_add(left_bits, right_bits, rounder=search_round):
    return compute_model_add(left_bits, right_bits, rounder)


def model_scale(bits, scale, rounder=search_round):
    return compute_model_scale(bits, scale, rounder)


_PRODUCT_BITS = {}


def model_product(a_code, w_code, rounder=search_round):
    cache = _PRODUCT_BITS.setdefault(rounder, {})
    key = (a_code, w_code)
    if key not in cache:
        exact = decode_fp8(a_code) * decode_fp8(w_code)
        cache[key] = rounder(abs(exact), bool((a_code ^ w_code) >> 7))
    return cache[key]


def expected_trace(a_codes, w_codes, scale_a, scale_b, rounder=search_round):
    blocks = []
    acc_bits = 0
    for j in range(len(a_codes) // 32):
        products, partial_sums = [], []
        dot_bits = 0
        for index in range(32 * j, 32 * j + 32):
            product = model_product(a_codes[index], w_codes[index], rounder)
            dot_bits = model_add(dot_bits, product, rounder)
            products.append(product)
            partial_sums.append(dot_bits)
        after_a_bits = model_scale(dot_bits, decode_scale(scale_a[j]), rounder)
        block_bits = model_scale(after_a_bits, decode_scale(scale_b[j]), rounder)
        acc_bits = model_add(acc_bits, block_bits, rounder)
        blocks.append({
            "products": tuple(products), "partial_sums": tuple(partial_sums),
            "dot_bits": dot_bits, "after_a_bits": after_a_bits,
            "block_bits": block_bits, "acc_bits": acc_bits,
        })
    bf16_bits = NAN_SENTINEL if acc_bits == NAN_SENTINEL else nearest_bf16_bits(acc_bits)
    return tuple(blocks), acc_bits, bf16_bits


_PRODUCT_PAIRS = None


def product_pair(value):
    global _PRODUCT_PAIRS
    if _PRODUCT_PAIRS is None:
        pairs = {}
        for a_code in VALID_CODES:
            for w_code in VALID_CODES:
                pairs.setdefault(decode_fp8(a_code) * decode_fp8(w_code), (a_code, w_code))
        _PRODUCT_PAIRS = pairs
    try:
        return _PRODUCT_PAIRS[value]
    except KeyError as error:
        raise AssertionError(f"no FP8 product encodes {value}") from error


def build_case(block_terms, scale_a, scale_b):
    a_codes, w_codes = [], []
    for terms in block_terms:
        if len(terms) > 32:
            raise AssertionError("a block has more than 32 terms")
        pairs = [product_pair(term) for term in terms]
        pairs.extend([(0, 0)] * (32 - len(pairs)))
        a_codes.extend(a for a, _ in pairs)
        w_codes.extend(w for _, w in pairs)
    return tuple(a_codes), tuple(w_codes), tuple(scale_a), tuple(scale_b)


class NumericMismatch(AssertionError):
    pass


class ValidationOrderMismatch(AssertionError):
    pass


def numeric_equal(actual, expected, label):
    if actual != expected:
        raise NumericMismatch(f"{label}: {actual!r} != {expected!r}")


def check_bits(actual, expected, label, width):
    if expected == NAN_SENTINEL:
        checker = is_nan_fp32_bits if width == 32 else is_nan_bf16_bits
        if not checker(actual):
            raise NumericMismatch(f"{label}: expected NaN, got {actual:#x}")
    elif type(actual) is not int or actual != expected:
        raise NumericMismatch(f"{label}: {actual!r} != {expected:#x}")


def check_pipeline(test, a_codes, w_codes, scale_a, scale_b, *, tag):
    expected_blocks, fp32_bits, bf16_bits = expected_trace(a_codes, w_codes, scale_a, scale_b)
    trace32 = trace_fp8_grouped_dot(a_codes, w_codes, scale_a, scale_b)
    trace16 = trace_fp8_grouped_dot(a_codes, w_codes, scale_a, scale_b, output="bf16")
    direct32 = fp8_grouped_dot(a_codes, w_codes, scale_a, scale_b)
    direct16 = fp8_grouped_dot(a_codes, w_codes, scale_a, scale_b, output="bf16")
    test.assertIsInstance(trace32, ComputeTrace, tag)
    test.assertIsInstance(trace16, ComputeTrace, tag)
    test.assertIs(type(direct32), float, tag)
    test.assertIs(type(trace32.result), float, tag)
    test.assertIs(type(direct16), int, tag)
    test.assertIs(type(trace16.result), int, tag)
    numeric_equal(trace32.blocks, trace16.blocks, f"{tag}: mode stages")
    numeric_equal(trace32.fp32_bits, trace16.fp32_bits, f"{tag}: mode fp32 bits")
    numeric_equal(trace32.bf16_bits, trace16.bf16_bits, f"{tag}: mode bf16 bits")
    numeric_equal(len(trace32.blocks), len(expected_blocks), f"{tag}: blocks")
    for block_index, (got, want) in enumerate(zip(trace32.blocks, expected_blocks)):
        test.assertIsInstance(got, BlockTrace, tag)
        numeric_equal(len(got.products), 32, f"{tag}: products length")
        numeric_equal(len(got.partial_sums), 32, f"{tag}: partial length")
        for index in range(32):
            check_bits(got.products[index], want["products"][index], f"{tag}: b{block_index} p{index}", 32)
            check_bits(got.partial_sums[index], want["partial_sums"][index], f"{tag}: b{block_index} s{index}", 32)
        for field in ("dot_bits", "after_a_bits", "block_bits", "acc_bits"):
            check_bits(getattr(got, field), want[field], f"{tag}: b{block_index} {field}", 32)
    check_bits(trace32.fp32_bits, fp32_bits, f"{tag}: trace fp32", 32)
    check_bits(trace32.bf16_bits, bf16_bits, f"{tag}: trace bf16", 16)
    check_bits(direct16, bf16_bits, f"{tag}: direct bf16", 16)
    numeric_equal(trace16.result, direct16, f"{tag}: trace bf16 result")
    if fp32_bits == NAN_SENTINEL:
        if not (math.isnan(trace32.result) and math.isnan(direct32)):
            raise NumericMismatch(f"{tag}: fp32 results must be NaN")
    else:
        check_bits(host_fp32_bits(trace32.result), fp32_bits, f"{tag}: trace result", 32)
        check_bits(host_fp32_bits(direct32), fp32_bits, f"{tag}: direct result", 32)
        numeric_equal(trace32.result, direct32, f"{tag}: direct/trace result")


class ExpectationTests(unittest.TestCase):
    def test_independent_integer_rounder_matches_grid_search(self):
        values = [
            Fraction(0), power2(-151), power2(-150), power2(-149), 3 * power2(-150),
            power2(-126), Fraction(1) + power2(-24), Fraction(1) + 3 * power2(-24),
            power2(128) - power2(103), power2(128) - power2(104),
        ]
        rng = random.Random(0x27)
        for _ in range(256):
            values.append(Fraction(rng.randrange(1, 1 << 42)) * power2(rng.randrange(-200, 130)))
        for value in values:
            for sign in (False, True):
                self.assertEqual(rne_fp32_bits(value, sign), search_round(value, sign))

    def test_integer_and_grid_pipeline_expectations_match(self):
        vectors = [
            build_case([[Fraction(2688)] * 7 + [power2(-10)] + [Fraction(-2688)] * 7], (0x7F,), (0x7F,)),
            build_case([[Fraction(2)]], (0xFE,), (0x00,)),
            build_case([[Fraction(1)], [Fraction(1)], [Fraction(-1)]], (0x7F, 0x97, 0x97), (0x7F,) * 3),
        ]
        rng = random.Random(0x8F27)
        for _ in range(12):
            blocks = rng.choice((1, 2, 5))
            vectors.append((
                tuple(rng.choice(VALID_CODES) for _ in range(32 * blocks)),
                tuple(rng.choice(VALID_CODES) for _ in range(32 * blocks)),
                tuple(rng.randrange(255) for _ in range(blocks)),
                tuple(rng.randrange(255) for _ in range(blocks)),
            ))
        for vector in vectors:
            self.assertEqual(expected_trace(*vector), expected_trace(*vector, rounder=rne_fp32_bits))


class DirectedNumericTests(unittest.TestCase):
    def test_wrong_weight_decode_vector(self):
        a = (0x38, 0x40, 0x48, 0x50) + (0,) * 28
        w = (0x40, 0x38, 0x30, 0x28) + (0,) * 28
        check_pipeline(self, a, w, (0x7F,), (0x7F,), tag="weight-decode")
        self.assertNotEqual(a, w)

    def test_in_block_sequential_order(self):
        vector = build_case([[power2(-10), Fraction(200704), Fraction(-200704)]], (0x7F,), (0x7F,))
        check_pipeline(self, *vector, tag="in-block-order")
        self.assertEqual(trace_fp8_grouped_dot(*vector).fp32_bits, 0)

    def test_cross_block_sequential_order(self):
        vector = build_case([[Fraction(1)], [Fraction(1)], [Fraction(-1)]], (0x7F, 0x97, 0x97), (0x7F,) * 3)
        check_pipeline(self, *vector, tag="cross-block-order")
        trace = trace_fp8_grouped_dot(*vector)
        self.assertEqual([block.acc_bits for block in trace.blocks], [0x3F800000, 0x4B800000, 0])

    def test_scale_position_and_separation(self):
        moved = build_case([[Fraction(2)]], (0xFE,), (0x00,))
        check_pipeline(self, *moved, tag="scale-position")
        self.assertEqual(trace_fp8_grouped_dot(*moved).fp32_bits, 0x7F800000)
        separated = build_case([[power2(-10)], [power2(-10)]], (0x00, 0x00), (0x73, 0x72))
        check_pipeline(self, *separated, tag="separate-scales")
        self.assertEqual([b.block_bits for b in trace_fp8_grouped_dot(*separated).blocks], [1, 0])

    def test_final_bf16_only_and_midpoints(self):
        vector = build_case([[Fraction(1), power2(-8)], [Fraction(-1)]], (0x7F, 0x7F), (0x7F, 0x7F))
        check_pipeline(self, *vector, tag="late-bf16")
        self.assertEqual(trace_fp8_grouped_dot(*vector).fp32_bits, 0x3B800000)
        cases = (
            ([Fraction(1), power2(-8), -power2(-10)], (0x7F,), (0x7F,), 0x3F80),
            ([Fraction(1), power2(-8)], (0x7F,), (0x7F,), 0x3F80),
            ([Fraction(1), power2(-8), power2(-10)], (0x7F,), (0x7F,), 0x3F81),
            ([Fraction(1), 3 * power2(-8)], (0x7F,), (0x7F,), 0x3F82),
            ([Fraction(2), -power2(-8)], (0x7F,), (0x7F,), 0x4000),
            ([Fraction(1)], (0x01,), (0x77,), 0x0000),
            ([Fraction(1), Fraction(1), Fraction(1)], (0x01,), (0x77,), 0x0002),
        )
        for terms, scale_a, scale_b, expected in cases:
            vector = build_case([terms], scale_a, scale_b)
            check_pipeline(self, *vector, tag="bf16-midpoint")
            self.assertEqual(trace_fp8_grouped_dot(*vector).bf16_bits, expected)

    def test_fp32_rne_midpoints_and_carry(self):
        for extra, expected in ((power2(-10), 0x46800000), (3 * power2(-10), 0x46800002)):
            vector = build_case([[Fraction(2688)] * 6 + [Fraction(256), extra]], (0x7F,), (0x7F,))
            check_pipeline(self, *vector, tag="dot-rne")
            self.assertEqual(trace_fp8_grouped_dot(*vector).fp32_bits, expected)
        for terms, expected in (([Fraction(1), -power2(-10)], 0x4B800000),
                                ([Fraction(1)], 0x4B800000),
                                ([Fraction(1), power2(-10)], 0x4B800001)):
            vector = build_case([[Fraction(1)], terms], (0x97, 0x7F), (0x7F, 0x7F))
            check_pipeline(self, *vector, tag="acc-rne")
            self.assertEqual(trace_fp8_grouped_dot(*vector).fp32_bits, expected)

    def test_subnormal_underflow_overflow_zero_and_specials(self):
        for scale_b, expected in ((0x73, 1), (0x72, 0)):
            vector = build_case([[power2(-10)]], (0x00,), (scale_b,))
            check_pipeline(self, *vector, tag="underflow")
            self.assertEqual(trace_fp8_grouped_dot(*vector).blocks[0].block_bits, expected)
        negative = build_case([[-power2(-10)]], (0x00,), (0x72,))
        check_pipeline(self, *negative, tag="negative-zero")
        self.assertEqual(trace_fp8_grouped_dot(*negative).blocks[0].block_bits, 0x80000000)
        max_block = [Fraction(200704)] * 32
        positive = build_case([max_block, [Fraction(1)]], (0xFE, 0x7F), (0x7F, 0x7F))
        check_pipeline(self, *positive, tag="positive-inf")
        self.assertEqual(trace_fp8_grouped_dot(*positive).fp32_bits, 0x7F800000)
        opposite = build_case([max_block, [-term for term in max_block]], (0xFE, 0xFE), (0x7F, 0x7F))
        check_pipeline(self, *opposite, tag="opposite-inf")
        self.assertTrue(is_nan_fp32_bits(trace_fp8_grouped_dot(*opposite).fp32_bits))

    def test_scale_boundaries_and_multiblock(self):
        vector = build_case([[Fraction(1)]] * 5,
                            (0x00, 0x01, 0x7F, 0xFD, 0xFE),
                            (0xFE, 0xFD, 0x7F, 0x01, 0x00))
        check_pipeline(self, *vector, tag="scale-boundaries")


class ExhaustiveAndIntegrationTests(unittest.TestCase):
    def test_all_finite_fp8_code_pairs_and_scale_boundaries(self):
        pairs = [(a, w) for a in VALID_CODES for w in VALID_CODES]
        self.assertEqual(len(pairs), 64516)
        boundaries = (0x00, 0x01, 0x02, 0x7E, 0x7F, 0x80, 0xFC, 0xFD, 0xFE)
        pairs.extend([(0, 0)] * (-len(pairs) % 32))
        for start in range(0, len(pairs), 32):
            chunk = pairs[start:start + 32]
            run = start // 32
            vector = (tuple(a for a, _ in chunk), tuple(w for _, w in chunk),
                      (boundaries[run % len(boundaries)],),
                      (boundaries[(run * 5 + 3) % len(boundaries)],))
            check_pipeline(self, *vector, tag=f"codes-{run}")

    def test_two_dimensional_weight_and_scale_row_extraction(self):
        rows, k = 35, 96
        weights = tuple(VALID_CODES[(row * 19 + col * 7) % len(VALID_CODES)]
                        for row in range(rows) for col in range(k))
        scale_cols = k // 32
        scales = tuple((0x70 + scale_row * 11 + col * 3) % 0xFF
                       for scale_row in range((rows + 31) // 32) for col in range(scale_cols))
        a = tuple(VALID_CODES[(col * 13 + 5) % len(VALID_CODES)] for col in range(k))
        scale_a = (0x7D, 0x7F, 0x81)
        for row in (0, 31, 32, rows - 1):
            w_row = weights[row * k:(row + 1) * k]
            scale_row = row // 32
            scale_b = scales[scale_row * scale_cols:(scale_row + 1) * scale_cols]
            check_pipeline(self, a, w_row, scale_a, scale_b, tag=f"2d-row-{row}")

    def test_actual_act_quant_output_is_consumed_directly(self):
        bf16 = tuple((0x3F80 + (index * 17) % 0x400) | (0x8000 if index % 5 == 0 else 0)
                     for index in range(96))
        a_codes, scale_a = act_quant(bf16)
        w_codes = tuple(VALID_CODES[(index * 29 + 3) % len(VALID_CODES)] for index in range(96))
        scale_b = (0x7E, 0x80, 0x7C)
        check_pipeline(self, a_codes, w_codes, scale_a, scale_b, tag="act-quant-consumer")

    def test_fixed_seed_random_pipelines(self):
        rng = random.Random(0xF827)
        for case in range(40):
            blocks = rng.choice((1, 2, 3, 5, 7))
            vector = (
                tuple(rng.choice(VALID_CODES) for _ in range(32 * blocks)),
                tuple(rng.choice(VALID_CODES) for _ in range(32 * blocks)),
                tuple(rng.randrange(255) for _ in range(blocks)),
                tuple(rng.randrange(255) for _ in range(blocks)),
            )
            check_pipeline(self, *vector, tag=f"random-{case}")


class ValidationTests(unittest.TestCase):
    def baseline(self):
        return (0x38,) * 32, (0x40,) * 32, (0x7F,), (0x7F,)

    def test_metadata_type_width_length_group_and_mode_rejection(self):
        a, w, sa, sb = self.baseline()
        metadata = (
            ((), (), (), ()), ((0x38,) * 16, (0x40,) * 16, (0x7F,), (0x7F,)),
            ((0x38,) * 33, (0x40,) * 33, (0x7F,), (0x7F,)),
            ((0x38,) * 64, (0x40,) * 63, (0x7F,) * 2, (0x7F,) * 2),
            ((0x38,) * 64, (0x40,) * 64, (0x7F,), (0x7F,) * 2),
            ((0x38,) * 64, (0x40,) * 64, (0x7F,) * 2, (0x7F,) * 3),
        )
        for entry in ENTRY_POINTS:
            for vector in metadata:
                with self.assertRaises(ValueError):
                    entry(*vector)
            for mode in ("float32", "BF16", "fp64", "", None, True):
                with self.assertRaises(ValueError):
                    entry(a, w, sa, sb, output=mode)
            for position in range(4):
                for bad in (5, 1.5, None, True, iter((1, 2))):
                    vector = [a, w, sa, sb]
                    vector[position] = bad
                    with self.assertRaises(TypeError):
                        entry(*vector)

    def test_content_bool_width_nan_and_nested_rejection(self):
        a, w, sa, sb = self.baseline()
        byte_bad = ((-1, ValueError), (256, ValueError), (10 ** 100, ValueError),
                    (True, TypeError), (1.0, TypeError), ("1", TypeError), (None, TypeError))
        for entry in ENTRY_POINTS:
            for input_index in (0, 1):
                for bad, error in byte_bad + ((0x7F, ValueError), (0xFF, ValueError)):
                    vector = [list(a), list(w), list(sa), list(sb)]
                    vector[input_index][-1] = bad
                    with self.assertRaises(error):
                        entry(*vector)
            for input_index in (2, 3):
                for bad, error in byte_bad + ((0xFF, ValueError),):
                    vector = [list(a), list(w), list(sa), list(sb)]
                    vector[input_index][-1] = bad
                    with self.assertRaises(error):
                        entry(*vector)
            with self.assertRaises(TypeError):
                entry([[0x38]] * 32, w, sa, sb)

    def test_memoryview_dimensions_checked_before_content(self):
        a, w, sa, sb = self.baseline()
        valid = memoryview(bytes(a))
        for entry in ENTRY_POINTS:
            self.assertIsNotNone(entry(valid, memoryview(bytes(w)), memoryview(bytes(sa)), memoryview(bytes(sb))))
        invalid = (
            memoryview(bytes(32)).cast("B", shape=(32, 1)),
            memoryview(bytes(32)).cast("B", shape=(16, 2)),
            memoryview(bytes(32)).cast("B", shape=(8, 2, 2)),
        )
        with patch("ref.fp8_compute._byte", side_effect=ValidationOrderMismatch("content observed")) as byte:
            for entry in ENTRY_POINTS:
                for view in invalid:
                    with self.assertRaises(TypeError):
                        entry(view, w, sa, sb)
            self.assertEqual(byte.call_count, 0)

    def test_metadata_precedes_content_and_failure_precedes_compute(self):
        a, w, sa, sb = self.baseline()
        for entry in ENTRY_POINTS:
            with self.assertRaises(ValueError):
                entry(a[:-1] + (1.0, 0x38), w, sa, sb)
            with self.assertRaises(ValueError):
                entry(a[:-1] + (1.0, 0x38), w, sa, sb, output="fp64")
        with patch("ref.fp8_compute._product_bits", side_effect=ValidationOrderMismatch("calculation started")):
            for entry in ENTRY_POINTS:
                with self.assertRaises(TypeError):
                    entry(a, w[:-1] + (True,), sa, sb)
                with self.assertRaises(ValueError):
                    entry(a, w, sa, (0xFF,))

    def test_snapshot_length_change_rejected_before_compute(self):
        class ChangingSequence(Sequence):
            def __init__(self, reported_length, values):
                self.reported_length = reported_length
                self.values = tuple(values)
            def __len__(self):
                return self.reported_length
            def __getitem__(self, index):
                return self.values[index]

        a, w, sa, sb = self.baseline()
        cases = (
            (0, ChangingSequence(32, a[:-1])), (0, ChangingSequence(32, a + (0,))),
            (1, ChangingSequence(32, w[:-1])), (1, ChangingSequence(32, w + (0,))),
            (2, ChangingSequence(1, ())), (2, ChangingSequence(1, sa + (0,))),
            (3, ChangingSequence(1, ())), (3, ChangingSequence(1, sb + (0,))),
        )
        with patch("ref.fp8_compute._product_bits",
                   side_effect=ValidationOrderMismatch("calculation started")) as product:
            for entry in ENTRY_POINTS:
                for position, changing in cases:
                    vector = [a, w, sa, sb]
                    vector[position] = changing
                    with self.subTest(entry=entry.__name__, input=position, actual=len(changing.values)):
                        with self.assertRaises(ValueError):
                            entry(*vector)
            self.assertEqual(product.call_count, 0)

    def test_inputs_unchanged_on_success_and_failure(self):
        inputs = [list(values) for values in self.baseline()]
        snapshots = [list(values) for values in inputs]
        for entry in ENTRY_POINTS:
            entry(*inputs)
            self.assertEqual(inputs, snapshots)
            bad = [list(values) for values in inputs]
            bad[1][-1] = 0xFF
            bad_snapshot = [list(values) for values in bad]
            with self.assertRaises(ValueError):
                entry(*bad)
            self.assertEqual(bad, bad_snapshot)

    def test_trace_structure_and_shared_evaluate_core(self):
        vector = build_case([[Fraction(1)]] * 5, (0x7F,) * 5, (0x7F,) * 5)
        trace = trace_fp8_grouped_dot(*vector)
        self.assertIs(type(trace.blocks), tuple)
        self.assertEqual(len(trace.blocks), 5)
        for block in trace.blocks:
            self.assertIsInstance(block, BlockTrace)
            self.assertIs(type(block.products), tuple)
            self.assertIs(type(block.partial_sums), tuple)
            self.assertTrue(all(type(value) is int for value in block.products + block.partial_sums))
        with patch("ref.fp8_compute._evaluate", return_value="sentinel") as core:
            self.assertEqual(fp8_grouped_dot(*vector), "sentinel")
            self.assertIs(core.call_args.kwargs["observe"], False)
            self.assertEqual(trace_fp8_grouped_dot(*vector), "sentinel")
            self.assertIs(core.call_args.kwargs["observe"], True)


class Fenv(ctypes.Structure):
    _fields_ = [
        ("control", ctypes.c_ushort), ("reserved1", ctypes.c_ushort),
        ("status", ctypes.c_ushort), ("reserved2", ctypes.c_ushort),
        ("tags", ctypes.c_ushort), ("reserved3", ctypes.c_ushort),
        ("eip", ctypes.c_uint), ("cs", ctypes.c_ushort),
        ("opcode", ctypes.c_ushort), ("data", ctypes.c_uint),
        ("selector", ctypes.c_ushort), ("reserved5", ctypes.c_ushort),
        ("mxcsr", ctypes.c_uint),
    ]


class EnvironmentTests(unittest.TestCase):
    def test_eight_mxcsr_modes_clear_and_preset_success_and_rejection(self):
        self.assertEqual((sys.platform, platform.machine(), platform.libc_ver()[0]),
                         ("linux", "x86_64", "glibc"))
        self.assertEqual((ctypes.sizeof(Fenv), Fenv.mxcsr.offset), (32, 28))
        library = ctypes.CDLL("libm.so.6")
        for name in ("fegetenv", "fesetenv"):
            function = getattr(library, name)
            function.argtypes = (ctypes.POINTER(Fenv),)
            function.restype = ctypes.c_int
        saved = Fenv()
        self.assertEqual(library.fegetenv(ctypes.byref(saved)), 0)
        maximum = [Fraction(200704)] * 32
        vectors = (
            ("normal", build_case([[Fraction(1), power2(-8)], [Fraction(-1)]],
                                  (0x7F, 0x7F), (0x7F, 0x7F))),
            ("positive-subnormal", build_case([[power2(-10)]], (0x00,), (0x73,))),
            ("negative-subnormal", build_case([[-power2(-10)]], (0x00,), (0x73,))),
            ("negative-zero-stage", build_case([[-power2(-10)]], (0x00,), (0x72,))),
            ("opposite-infinities", build_case([maximum, [-term for term in maximum]],
                                                (0xFE, 0xFE), (0x7F, 0x7F))),
        )
        expectations = {name: expected_trace(*vector) for name, vector in vectors}
        modes = (
            ("nearest", 0), ("upward", 0x4000), ("downward", 0x2000),
            ("toward-zero", 0x6000), ("ftz", 0x8000), ("daz", 0x40),
            ("ftz-daz", 0x8040), ("unmasked-overflow", 0),
        )
        failures = []

        def setting_for(name, mode, status):
            setting = Fenv.from_buffer_copy(bytes(saved))
            setting.mxcsr = 0x1F80 | mode | status
            if name == "unmasked-overflow":
                setting.mxcsr &= ~0x400
            return setting

        def validate_result(result, expected, output, traced):
            blocks, fp32_bits, bf16_bits = expected
            if traced and not isinstance(result, ComputeTrace):
                raise NumericMismatch("environment trace type differs")
            formal_result = result.result if traced else result
            expected_type = float if output == "fp32" else int
            if type(formal_result) is not expected_type:
                raise NumericMismatch("environment result type differs")
            if traced:
                if len(result.blocks) != len(blocks):
                    raise NumericMismatch("environment trace block count differs")
                for block_index, (actual, wanted) in enumerate(zip(result.blocks, blocks)):
                    for index in range(32):
                        check_bits(actual.products[index], wanted["products"][index],
                                   f"environment b{block_index} product {index}", 32)
                        check_bits(actual.partial_sums[index], wanted["partial_sums"][index],
                                   f"environment b{block_index} partial {index}", 32)
                    for field in ("dot_bits", "after_a_bits", "block_bits", "acc_bits"):
                        check_bits(getattr(actual, field), wanted[field],
                                   f"environment b{block_index} {field}", 32)
                check_bits(result.fp32_bits, fp32_bits, "environment fp32", 32)
                check_bits(result.bf16_bits, bf16_bits, "environment bf16", 16)
                if output == "bf16":
                    check_bits(result.result, bf16_bits, "environment trace bf16 result", 16)
                elif fp32_bits == NAN_SENTINEL:
                    if not math.isnan(result.result):
                        raise NumericMismatch("environment trace fp32 result is not NaN")
                else:
                    check_bits(host_fp32_bits(result.result), fp32_bits,
                               "environment trace fp32 result", 32)
            elif output == "bf16":
                check_bits(result, bf16_bits, "environment direct bf16", 16)
            elif fp32_bits == NAN_SENTINEL:
                if not math.isnan(result):
                    raise NumericMismatch("environment direct fp32 result is not NaN")
            else:
                check_bits(host_fp32_bits(result), fp32_bits, "environment direct fp32", 32)

        try:
            for mode_name, mode in modes:
                for status in (0, 0x3F):
                    for vector_name, vector in vectors:
                        for entry in ENTRY_POINTS:
                            for output in ("fp32", "bf16"):
                                setting = setting_for(mode_name, mode, status)
                                set_result = library.fesetenv(ctypes.byref(setting))
                                before = Fenv()
                                before_result = library.fegetenv(ctypes.byref(before))
                                error = None
                                try:
                                    result = entry(*vector, output=output)
                                except Exception as caught:
                                    result, error = None, caught
                                after = Fenv()
                                after_result = library.fegetenv(ctypes.byref(after))
                                restore_result = library.fesetenv(ctypes.byref(saved))
                                label = (mode_name, status, vector_name, entry.__name__, output)
                                if set_result or before_result or after_result or restore_result:
                                    failures.append(label + ("fenv return", set_result, before_result,
                                                             after_result, restore_result))
                                    continue
                                if before.mxcsr != setting.mxcsr:
                                    failures.append(label + ("before differs from setting",))
                                if after.mxcsr != before.mxcsr:
                                    failures.append(label + ("mxcsr changed",))
                                if error is not None:
                                    failures.append(label + (type(error).__name__,))
                                    continue
                                try:
                                    validate_result(result, expectations[vector_name], output,
                                                    entry is trace_fp8_grouped_dot)
                                except NumericMismatch as mismatch:
                                    failures.append(label + (str(mismatch),))
                    reject_vector = vectors[0][1]
                    for entry in ENTRY_POINTS:
                        for output in ("fp32", "bf16"):
                            setting = setting_for(mode_name, mode, status)
                            set_result = library.fesetenv(ctypes.byref(setting))
                            before = Fenv()
                            before_result = library.fegetenv(ctypes.byref(before))
                            error = None
                            try:
                                entry(reject_vector[0], reject_vector[1][:-1] + (0xFF,),
                                      reject_vector[2], reject_vector[3], output=output)
                            except Exception as caught:
                                error = caught
                            after = Fenv()
                            after_result = library.fegetenv(ctypes.byref(after))
                            restore_result = library.fesetenv(ctypes.byref(saved))
                            label = (mode_name, status, "reject", entry.__name__, output)
                            if set_result or before_result or after_result or restore_result:
                                failures.append(label + ("fenv return", set_result, before_result,
                                                         after_result, restore_result))
                                continue
                            if before.mxcsr != setting.mxcsr:
                                failures.append(label + ("before differs from setting",))
                            if after.mxcsr != before.mxcsr:
                                failures.append(label + ("mxcsr changed",))
                            if type(error) is not ValueError:
                                failures.append(label + ("wrong rejection", type(error).__name__ if error else None))
        finally:
            final_restore = library.fesetenv(ctypes.byref(saved))
        self.assertEqual(final_restore, 0)
        restored = Fenv()
        self.assertEqual(library.fegetenv(ctypes.byref(restored)), 0)
        self.assertEqual(bytes(restored), bytes(saved))
        self.assertEqual(failures, [])


SENSITIVITY_VECTORS = {
    "wrong-weight-e2m1": lambda: ((0x38, 0x40, 0x48, 0x50) + (0,) * 28,
                                   (0x40, 0x38, 0x30, 0x28) + (0,) * 28,
                                   (0x7F,), (0x7F,)),
    "move-scale-per-product": lambda: build_case([[Fraction(2), Fraction(-2)]],
                                                   (0xFE,), (0x7F,)),
    "swap-scales": lambda: build_case([[Fraction(2)]], (0xFE,), (0x00,)),
    "merge-scales": lambda: build_case([[Fraction(2)]], (0xFE,), (0x00,)),
    "in-block-order": lambda: build_case(
        [[power2(-10), Fraction(200704), Fraction(-200704)]], (0x7F,), (0x7F,)),
    "cross-block-order": lambda: build_case(
        [[Fraction(1)], [Fraction(1)], [Fraction(-1)]],
        (0x7F, 0x97, 0x97), (0x7F,) * 3),
    "early-bf16": lambda: build_case(
        [[Fraction(1), power2(-8)], [Fraction(-1)]],
        (0x7F, 0x7F), (0x7F, 0x7F)),
}


MUTATIONS = (
    ("wrong-weight-e2m1", ((
        "product = _product_bits(a_codes[index], w_codes[index])",
        "product = _product_bits(a_codes[index], "
        "(0x00, 0x30, 0x38, 0x3C, 0x40, 0x44, 0x48, 0x4C, "
        "0x80, 0xB0, 0xB8, 0xBC, 0xC0, 0xC4, 0xC8, 0xCC)[w_codes[index] & 15])",
    ),)),
    ("move-scale-per-product", ((
        "product = _product_bits(a_codes[index], w_codes[index])",
        "product = _scale_bits(_product_bits(a_codes[index], w_codes[index]), scale_a[j])",
    ), (
        "after_a = _scale_bits(dot, scale_a[j])",
        "after_a = dot",
    ))),
    ("swap-scales", ((
        "after_a = _scale_bits(dot, scale_a[j])",
        "after_a = _scale_bits(dot, scale_b[j])",
    ), (
        "block = _scale_bits(after_a, scale_b[j])",
        "block = _scale_bits(after_a, scale_a[j])",
    ))),
    ("merge-scales", ((
        "after_a = _scale_bits(dot, scale_a[j])",
        "after_a = dot",
    ), (
        "block = _scale_bits(after_a, scale_b[j])",
        "block = _scale_bits(after_a, scale_a[j] + scale_b[j] - 127)",
    ))),
    ("in-block-order", (("for r in range(32):", "for r in reversed(range(32)):"),)),
    ("cross-block-order", ((
        "for j in range(len(a_codes) // 32):",
        "for j in reversed(range(len(a_codes) // 32)):",
    ),)),
    ("early-bf16", ((
        "acc = _add_bits(acc, block)",
        "acc = _add_bits(acc, fp32_bits_to_bf16(block) << 16)",
    ),)),
)


def iter_tests(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from iter_tests(item)
        else:
            yield item


def load_mutant(path, source):
    path.write_text(source)
    name = f"_p0027_mutant_{path.parent.name.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def module_probe(module, vector):
    trace32 = module.trace_fp8_grouped_dot(*vector)
    trace16 = module.trace_fp8_grouped_dot(*vector, output="bf16")
    direct32 = module.fp8_grouped_dot(*vector)
    direct16 = module.fp8_grouped_dot(*vector, output="bf16")
    return {
        "direct_fp32": host_fp32_bits(direct32), "direct_bf16": direct16,
        "trace_fp32_result": host_fp32_bits(trace32.result),
        "trace_bf16_result": trace16.result,
        "fp32": trace32.fp32_bits, "bf16": trace32.bf16_bits,
        "dot": [block.dot_bits for block in trace32.blocks],
        "after_a": [block.after_a_bits for block in trace32.blocks],
        "block": [block.block_bits for block in trace32.blocks],
        "acc": [block.acc_bits for block in trace32.blocks],
    }


class SensitivityTests(unittest.TestCase):
    def test_runner_errors_skips_and_missing_dependency_exit_nonzero(self):
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        for control in ("error", "skip"):
            process = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--control", control],
                capture_output=True, text=True, env=environment, timeout=60,
            )
            self.assertNotEqual(process.returncode, 0, control)
        with tempfile.TemporaryDirectory(prefix="p0027-missing-") as directory:
            script = Path(directory) / "test_fp8_compute.py"
            script.write_bytes(Path(__file__).read_bytes())
            process = subprocess.run(
                [sys.executable, str(script), "--suite", "numeric"],
                capture_output=True, text=True, env=environment, timeout=60,
            )
            self.assertNotEqual(process.returncode, 0)
            self.assertIn("ModuleNotFoundError", process.stderr)

    def test_source_mutations_and_2d_layout_mapping_are_numerically_detected(self):
        source_path = ROOT / "ref" / "fp8_compute.py"
        guarded = (source_path, Path(__file__).resolve())
        digests = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in guarded}
        pristine_source = source_path.read_text()
        evidence = []
        with tempfile.TemporaryDirectory(prefix="p0027-sensitivity-") as directory:
            for slug, replacements in MUTATIONS:
                vector = SENSITIVITY_VECTORS[slug]()
                before = module_probe(sys.modules[fp8_grouped_dot.__module__], vector)
                text = pristine_source
                for anchor, replacement in replacements:
                    self.assertEqual(text.count(anchor), 1, slug)
                    text = text.replace(anchor, replacement)
                mutant = load_mutant(Path(directory) / f"{slug}.py", text)
                after = module_probe(mutant, vector)
                self.assertNotEqual(after, before, slug)
                expected_blocks, expected_fp32, expected_bf16 = expected_trace(*vector)
                expected_fields = {
                    "fp32": expected_fp32, "bf16": expected_bf16,
                    "dot": [block["dot_bits"] for block in expected_blocks],
                    "after_a": [block["after_a_bits"] for block in expected_blocks],
                    "block": [block["block_bits"] for block in expected_blocks],
                    "acc": [block["acc_bits"] for block in expected_blocks],
                }
                for field, expected in expected_fields.items():
                    self.assertEqual(before[field], expected, (slug, field))
                self.assertTrue(any(after[field] != expected for field, expected in expected_fields.items()), slug)
                evidence.append({"class": slug, "before": before, "after": after})
            rows, k, scale_cols = 35, 64, 2
            a = (0x38,) * k
            weights = tuple(0x38 if row < 32 else 0x40 for row in range(rows) for _ in range(k))
            scales = (0x7F, 0x80, 0x81, 0x82)
            row = 32
            w_row = weights[row * k:(row + 1) * k]
            right = scales[(row // 32) * scale_cols:(row // 32 + 1) * scale_cols]
            wrong = scales[:scale_cols]
            correct_vector = (a, w_row, (0x7F, 0x7F), right)
            wrong_vector = (a, w_row, (0x7F, 0x7F), wrong)
            check_pipeline(self, *correct_vector, tag="sensitivity-2d-baseline")
            before = module_probe(sys.modules[fp8_grouped_dot.__module__], correct_vector)
            after = module_probe(sys.modules[fp8_grouped_dot.__module__], wrong_vector)
            expected_blocks, expected_fp32, expected_bf16 = expected_trace(*correct_vector)
            for field, expected in {
                "fp32": expected_fp32, "bf16": expected_bf16,
                "dot": [block["dot_bits"] for block in expected_blocks],
                "after_a": [block["after_a_bits"] for block in expected_blocks],
                "block": [block["block_bits"] for block in expected_blocks],
                "acc": [block["acc_bits"] for block in expected_blocks],
            }.items():
                self.assertEqual(before[field], expected, field)
            self.assertNotEqual(before, after)
            evidence.append({"class": "2d-scale-mapping", "before": before, "after": after})
        for path, digest in digests.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
        print("P0027_SENSITIVITY " + json.dumps(evidence, separators=(",", ":"), sort_keys=True), flush=True)


NUMERIC_CLASSES = (ExpectationTests, DirectedNumericTests, ExhaustiveAndIntegrationTests)
SUITE_CLASSES = {
    "numeric": NUMERIC_CLASSES,
    "validation": (ValidationTests,),
    "environment": (EnvironmentTests,),
    "sensitivity": (SensitivityTests,),
    "all": (*NUMERIC_CLASSES, ValidationTests, EnvironmentTests, SensitivityTests),
}


def gather_suite(arguments):
    loader = unittest.TestLoader()
    suite = unittest.TestSuite(loader.loadTestsFromTestCase(case) for case in SUITE_CLASSES[arguments.suite])
    if arguments.cases:
        suite = unittest.TestSuite(test for test in iter_tests(suite)
                                   if any(selector in test.id() for selector in arguments.cases))
    return suite


def main(argv=None):
    parser = argparse.ArgumentParser(description="P0027 FP8 compute acceptance")
    parser.add_argument("--suite", choices=tuple(SUITE_CLASSES), default="all")
    parser.add_argument("--case", dest="cases", action="append", default=[])
    parser.add_argument("--control", choices=("error", "skip"), help=argparse.SUPPRESS)
    arguments = parser.parse_args(argv)
    if arguments.control:
        class RunnerControl(unittest.TestCase):
            def runTest(self):
                if arguments.control == "skip":
                    self.skipTest("P0027 runner skip control")
                raise RuntimeError("P0027 runner error control")
        suite = unittest.TestSuite([RunnerControl()])
    else:
        suite = gather_suite(arguments)
    if suite.countTestCases() == 0:
        print("no tests selected", file=sys.stderr)
        return 2
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() and not result.skipped else 1


if __name__ == "__main__":
    print("P0027 FP8 grouped-dot acceptance; standard library only", flush=True)
    print(f"Python: {sys.version}; platform: {platform.platform()}; machine: {platform.machine()}", flush=True)
    sys.exit(main())
