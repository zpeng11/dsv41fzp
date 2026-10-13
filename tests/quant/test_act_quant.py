#!/usr/bin/env python3
"""P0014 mandatory CPU activation-quant acceptance: python3 tests/quant/test_act_quant.py.

Every numeric expectation is built independently of ref/act_quant.py: BF16
elements decode through test_formats' Fraction bit-field definition, the scale
exponent is the exact ceil(log2(max(amax, fl32(1e-4))/448)) and is cross-checked
against a second construction that multiplies the bounded amax by fl32(1/448)
with one binary32 RNE and reads the exponent back from the product bits;
quotients are one binary32 RNE of the exact x / 2^e with gradual underflow and
preserved zero signs; E4M3FN codes come from an ordered-grid nearest search
with ties to the even code. All acceptance rounding goes through
test_formats' ordered-encoding neighbor search (search_rounder), so no
expectation shares a method with the module under test and none constructs a
numeric expectation through host float arithmetic; a fast integer rounder is
kept only as a cross-checked aid. host_fp32_bits only converts the combination
consumer's float output for bit comparison.

Coverage: all 32640 finite BF16 amax codes drive full group-stage comparisons
and prove the +-448 clamp unreachable; all 65280 finite element codes are
checked twice inside legal 32-element groups (once within their own binade,
once under the maximum amax anchor); all 256 non-finite codes are rejected.
Directed cases pin the mutation-sensitive boundaries listed in
DIRECTED_TEST_NAMES. The combination test feeds act_quant output to
ref.compute.expert_grouped_dot against expectations rebuilt from the expected
quantized codes and scales, never from the raw BF16 dot product.

Sensitivity mutates only throwaway source copies and requires named numeric
or stage-bit failures. The supported-host environment probe checks both entry
points under altered rounding, FTZ, DAZ, and exception-status settings.

Entry points:
    python3 tests/quant/test_act_quant.py
    python3 tests/quant/test_act_quant.py --suite core
    python3 tests/quant/test_act_quant.py --suite directed --marker
    python3 tests/quant/test_act_quant.py --suite sensitivity
    python3 tests/quant/test_act_quant.py --case <name substring>
"""

import argparse
import ast
import bisect
import ctypes
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
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_formats import (  # noqa: E402
    expected_binary,
    expected_e4m3fn,
    host_fp32_bits,
    independent_pack,
    nearest_fp32_bits,
    power2,
)
from ref.act_quant import (  # noqa: E402
    ActQuantTrace,
    GroupTrace,
    act_quant,
    trace_act_quant,
)
from ref.compute import expert_grouped_dot  # noqa: E402
from test_compute import (  # noqa: E402
    NAN_SENTINEL,
    expected_trace as compute_expected_trace,
    is_nan_bf16_bits,
    is_nan_fp32_bits,
)

ENTRY_POINTS = (act_quant, trace_act_quant)
FL32_1E_4_BITS = nearest_fp32_bits(Fraction(1, 10000))
FL32_1_448_BITS = nearest_fp32_bits(Fraction(1, 448))
FL32_1E_4 = expected_binary(FL32_1E_4_BITS, 23)
FL32_1_448 = expected_binary(FL32_1_448_BITS, 23)
CLAMP_LIMIT_BITS = 0x43E00000
POSITIVE_AMAX_CODES = tuple(range(0x7F80))
FINITE_BF16_CODES = tuple(code for code in range(0x10000) if (code & 0x7F80) != 0x7F80)
E4M3_MAGNITUDES = tuple(expected_e4m3fn(code) for code in range(0x7F))


def dyadic_parts(bits):
    """Split a nonnegative finite binary32 pattern into significand * 2**exp2."""
    exponent = (bits >> 23) & 0xFF
    fraction = bits & 0x7FFFFF
    if exponent:
        return (1 << 23) | fraction, exponent - 150
    return fraction, -149


def rne_fp32_bits(numer, exp2, sign):
    """Round the exact dyadic numer * 2**exp2 to binary32 bits, RNE, gradual."""
    if numer == 0:
        return sign << 31
    bit_length = numer.bit_length()
    top = bit_length - 1 + exp2
    sign_bit = sign << 31
    if top > 127:
        return sign_bit | 0x7F800000
    if top >= -126:
        shift = bit_length - 24
        if shift > 0:
            dropped = numer & ((1 << shift) - 1)
            midpoint = 1 << (shift - 1)
            quotient = numer >> shift
            if dropped > midpoint or (dropped == midpoint and quotient & 1):
                quotient += 1
            if quotient == 1 << 24:
                quotient = 1 << 23
                top += 1
        else:
            quotient = numer << -shift
        if top > 127:
            return sign_bit | 0x7F800000
        return sign_bit | ((top + 127) << 23) | (quotient & 0x7FFFFF)
    shift = -149 - exp2
    if shift > 0:
        dropped = numer & ((1 << shift) - 1)
        midpoint = 1 << (shift - 1)
        quotient = numer >> shift
        if dropped > midpoint or (dropped == midpoint and quotient & 1):
            quotient += 1
    else:
        quotient = numer << -shift
    return sign_bit | quotient


@lru_cache(maxsize=None)
def search_rounder(numer, exp2, sign):
    """Independent rounding through test_formats' ordered-encoding search."""
    exact = Fraction(numer) * power2(exp2)
    if exact == 0:
        return nearest_fp32_bits(exact, bool(sign))
    return nearest_fp32_bits(-exact if sign else exact)


def ceil_log2(value):
    """Smallest integer e with 2**e >= value, for a positive exact Fraction."""
    center = value.numerator.bit_length() - value.denominator.bit_length()
    for exponent in (center - 1, center, center + 1):
        if power2(exponent) >= value:
            return exponent
    raise AssertionError("ceil_log2 bounds failed")


def encode_e4m3_from_bits(bits):
    """Nearest E4M3FN code for a clamped binary32 pattern, ties to even code."""
    sign = bits >> 31
    magnitude_bits = bits & 0x7FFFFFFF
    if magnitude_bits == 0:
        return sign << 7
    magnitude = expected_binary(magnitude_bits, 23)
    if magnitude > E4M3_MAGNITUDES[-1]:
        raise AssertionError("clamped quotient exceeds the E4M3FN maximum")
    index = bisect.bisect_left(E4M3_MAGNITUDES, magnitude)
    if E4M3_MAGNITUDES[index] == magnitude:
        code = index
    else:
        lower, upper = index - 1, index
        lower_distance = magnitude - E4M3_MAGNITUDES[lower]
        upper_distance = E4M3_MAGNITUDES[upper] - magnitude
        if lower_distance < upper_distance:
            code = lower
        elif upper_distance < lower_distance:
            code = upper
        else:
            code = lower if lower % 2 == 0 else upper
    return (sign << 7) | code


def brute_encode_e4m3(magnitude):
    """Linear-scan nearest E4M3FN magnitude code, ties to the even code."""
    best_code, best_distance = None, None
    for code in range(0x7F):
        distance = abs(magnitude - E4M3_MAGNITUDES[code])
        if best_distance is None or distance < best_distance or (
            distance == best_distance and code % 2 == 0 and best_code % 2 == 1
        ):
            best_code, best_distance = code, distance
    return best_code


def bf16_code(exact):
    """Exact Fraction to its BF16 code; the value must be representable."""
    sign = 0x8000 if exact < 0 else 0
    magnitude = abs(exact)
    if magnitude == 0:
        return sign
    center = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
    binade = next(e for e in (center - 1, center, center + 1)
                  if power2(e) <= magnitude < power2(e + 1))
    if binade >= -126:
        scaled = (magnitude / power2(binade) - 1) * 128
        if scaled.denominator != 1 or not 0 <= scaled.numerator <= 127:
            raise AssertionError(f"not BF16-representable: {exact}")
        code = ((binade + 127) << 7) | scaled.numerator
    else:
        subnormal = magnitude * power2(133)
        if subnormal.denominator != 1 or not 1 <= subnormal.numerator <= 127:
            raise AssertionError(f"not BF16-representable: {exact}")
        code = subnormal.numerator
    if (code & 0x7F80) == 0x7F80:
        raise AssertionError(f"not a finite BF16 value: {exact}")
    return sign | code


def compute_scale_stages(amax_code, rounder):
    """Both scale constructions for one amax code; they must always agree."""
    amax_bits = amax_code << 16
    amax_value = expected_binary(amax_bits, 23)
    bounded_bits = amax_bits if amax_value >= FL32_1E_4 else FL32_1E_4_BITS
    bounded_value = expected_binary(bounded_bits, 23)
    left_sig, left_exp = dyadic_parts(bounded_bits)
    right_sig, right_exp = dyadic_parts(FL32_1_448_BITS)
    product_bits = rounder(left_sig * right_sig, left_exp + right_exp, 0)
    exact_exponent = ceil_log2(bounded_value / 448)
    staged_exponent = ceil_log2(expected_binary(product_bits, 23))
    if exact_exponent != staged_exponent:
        raise AssertionError(
            f"scale exponent methods disagree for amax {amax_code:#06x}: "
            f"{exact_exponent} vs {staged_exponent}"
        )
    if bounded_value > 448 * power2(exact_exponent):
        raise AssertionError(f"clamp reachable for amax {amax_code:#06x}")
    return amax_bits, bounded_bits, product_bits, exact_exponent, exact_exponent + 127


_SCALE_STAGE_CACHE = {}


def group_scale_stages(amax_code):
    stages = _SCALE_STAGE_CACHE.get(amax_code)
    if stages is None:
        stages = compute_scale_stages(amax_code, search_rounder)
        _SCALE_STAGE_CACHE[amax_code] = stages
    return stages


def expected_group(codes, rounder=search_rounder):
    """Independent stage-by-stage expectation for one 32-element group."""
    amax_code = max(code & 0x7FFF for code in codes)
    if rounder is search_rounder:
        amax_bits, bounded_bits, product_bits, exponent, scale_code = group_scale_stages(amax_code)
    else:
        amax_bits, bounded_bits, product_bits, exponent, scale_code = compute_scale_stages(amax_code, rounder)
    quotient_bits, clamped_bits, a_codes = [], [], []
    for code in codes:
        sign = code >> 15
        magnitude = code & 0x7FFF
        if magnitude == 0:
            zero = sign << 31
            quotient_bits.append(zero)
            clamped_bits.append(zero)
            a_codes.append(sign << 7)
            continue
        exponent_field = magnitude >> 7
        fraction = magnitude & 0x7F
        significand = fraction if exponent_field == 0 else 128 + fraction
        exp2 = -133 if exponent_field == 0 else exponent_field - 134
        bits = rounder(significand, exp2 - exponent, sign)
        quotient_bits.append(bits)
        clamped = bits if (bits & 0x7FFFFFFF) <= CLAMP_LIMIT_BITS else (
            (bits & 0x80000000) | CLAMP_LIMIT_BITS
        )
        clamped_bits.append(clamped)
        a_codes.append(encode_e4m3_from_bits(clamped))
    return {
        "amax_bits": amax_bits,
        "bounded_amax_bits": bounded_bits,
        "product_bits": product_bits,
        "exponent": exponent,
        "scale_code": scale_code,
        "quotient_bits": tuple(quotient_bits),
        "clamped_bits": tuple(clamped_bits),
        "a_codes": tuple(a_codes),
    }


def expected_trace(bf16_bits, rounder=search_rounder):
    """Independent whole-row expectation: (groups, a_codes, scale_a)."""
    bits = tuple(bf16_bits)
    if not bits or len(bits) % 32:
        raise AssertionError("expectations require K positive and divisible by 32")
    groups, codes, scales = [], [], []
    for index in range(len(bits) // 32):
        group = expected_group(bits[32 * index:32 * index + 32], rounder)
        groups.append(group)
        codes.extend(group["a_codes"])
        scales.append(group["scale_code"])
    return tuple(groups), tuple(codes), tuple(scales)


class NumericMismatch(AssertionError):
    """Only a completed numeric or stage-bit comparison supplies semantic evidence."""


class ValidationOrderMismatch(AssertionError):
    """An input-validation sentinel distinguishes a contract violation from a crash."""


def assert_numeric_equal(actual, expected, label):
    if actual != expected:
        raise NumericMismatch(f"{label}: {actual!r} != {expected!r}")


def check_values(test, bf16_bits, *, tag):
    """Compare every observed stage value against the independent expectation."""
    bits = tuple(bf16_bits)
    groups, a_codes, scale_a = expected_trace(bits)
    trace = trace_act_quant(bits)
    direct = act_quant(bits)
    test.assertEqual(len(trace.groups), len(groups), f"{tag}: group count")
    for index, (got, want) in enumerate(zip(trace.groups, groups)):
        label = f"{tag}: group {index}"
        assert_numeric_equal(
            (got.amax_bits, got.bounded_amax_bits, got.product_bits, got.exponent, got.scale_code),
            (want["amax_bits"], want["bounded_amax_bits"], want["product_bits"],
             want["exponent"], want["scale_code"]),
            label,
        )
        assert_numeric_equal(got.quotient_bits, want["quotient_bits"], f"{label} quotient bits")
        assert_numeric_equal(got.clamped_bits, want["clamped_bits"], f"{label} clamped bits")
        assert_numeric_equal(got.a_codes, want["a_codes"], f"{label} codes")
    assert_numeric_equal(trace.a_codes, a_codes, f"{tag}: trace codes")
    assert_numeric_equal(trace.scale_a, scale_a, f"{tag}: trace scales")
    assert_numeric_equal(direct, (a_codes, scale_a), f"{tag}: direct result")


def check_pipeline(test, bf16_bits, *, tag):
    """Full comparison including structure, types, and entry consistency."""
    check_values(test, bf16_bits, tag=tag)
    bits = tuple(bf16_bits)
    trace = trace_act_quant(bits)
    direct = act_quant(bits)
    test.assertIsInstance(trace, ActQuantTrace, f"{tag}: trace type")
    test.assertIs(type(direct), tuple, f"{tag}: direct type")
    test.assertEqual(len(direct), 2, f"{tag}: direct arity")
    test.assertIs(type(direct[0]), tuple, f"{tag}: direct codes type")
    test.assertIs(type(direct[1]), tuple, f"{tag}: direct scales type")
    test.assertIs(type(trace.groups), tuple, f"{tag}: groups type")
    test.assertIs(type(trace.a_codes), tuple, f"{tag}: trace codes type")
    test.assertIs(type(trace.scale_a), tuple, f"{tag}: trace scales type")
    for group in trace.groups:
        test.assertIsInstance(group, GroupTrace, f"{tag}: group type")
        for name in ("amax_bits", "bounded_amax_bits", "product_bits", "exponent", "scale_code"):
            test.assertIs(type(getattr(group, name)), int, f"{tag}: group {name} type")
        for name in ("quotient_bits", "clamped_bits", "a_codes"):
            field = getattr(group, name)
            test.assertIs(type(field), tuple, f"{tag}: group {name} type")
            test.assertEqual(len(field), 32, f"{tag}: group {name} length")
            for value in field:
                test.assertIs(type(value), int, f"{tag}: group {name} element type")
    for value in trace.a_codes:
        test.assertIs(type(value), int, f"{tag}: trace code type")
    for value in trace.scale_a:
        test.assertIs(type(value), int, f"{tag}: trace scale type")
    test.assertTrue(all(0 <= code <= 0xFF and code not in (0x7F, 0xFF) for code in trace.a_codes),
                    f"{tag}: codes must be finite E4M3FN bytes")
    test.assertTrue(all(0 <= code <= 0xFE for code in trace.scale_a),
                    f"{tag}: scales must be finite E8M0 bytes")
    test.assertEqual(len(trace.a_codes), len(bits), f"{tag}: code count")
    test.assertEqual(len(trace.scale_a), len(bits) // 32, f"{tag}: scale count")


def five_group_bits():
    bits = []
    for k in (0, 1, 2, 3, 4):
        anchor = bf16_code(448 * power2(k))
        quarter = bf16_code(112 * power2(k))
        pattern = (anchor, quarter, quarter | 0x8000, 0x0000, 0x8000)
        bits.extend(pattern[position % 5] for position in range(32))
    return tuple(bits)


def below_floor_bits():
    zeros = (0x0000,) * 16 + (0x8000,) * 16
    mixed = (0x3880, 0x3800, 0x38D1) + (0x0000,) * 29
    return zeros + (0x3880,) * 32 + (0x3800,) * 32 + mixed


def subnormal_amax_bits():
    return (0x0001,) + (0x0000,) * 31 + (0x8001,) + (0x0000,) * 31


def quotient_boundary_bits():
    small = (0x0080, 0x8080, 0x0081, 0x007F, 0x8081, 0x807F)
    groups = []
    for anchor in (0x4F60, 0x4FE0):
        groups.append((anchor,) + tuple(small[position % 6] for position in range(31)))
    return tuple(code for group in groups for code in group)


def signed_zero_bits():
    pattern = (0x0000, 0x8000, 0x0001, 0x8001)
    near = (0x43E0,) + tuple(pattern[position % 4] for position in range(31))
    far = (0x7F7F,) + tuple(pattern[position % 4] for position in range(31))
    return near + far


def midpoint_bits():
    """Every E4M3FN midpoint with its BF16-representable neighbors, both signs."""
    targets = []
    for code in range(0x7E):
        center = bf16_code((E4M3_MAGNITUDES[code] + E4M3_MAGNITUDES[code + 1]) / 2)
        for candidate in (center - 1, center, center + 1):
            targets.append(candidate)
            targets.append(candidate | 0x8000)
    anchor = 0x43E0
    bits = []
    for start in range(0, len(targets), 31):
        group = [anchor] + targets[start:start + 31]
        group += [0x0000] * (32 - len(group))
        bits.extend(group)
    return tuple(bits)


class ExpectationCrossChecks(unittest.TestCase):
    """Test-only fast rounders, encoders, and constants against search baselines."""

    def test_reference_constants_from_neighbor_search(self):
        self.assertEqual(FL32_1E_4_BITS, 0x38D1B717)
        self.assertEqual(FL32_1_448_BITS, 0x3B124925)
        self.assertGreater(FL32_1_448, Fraction(1, 448))
        self.assertLess(448 * power2(-23), FL32_1E_4)
        self.assertLessEqual(FL32_1E_4, 448 * power2(-22))
        amax = expected_binary(0x38D2 << 16, 23)
        source_product = nearest_fp32_bits(amax * FL32_1_448)
        exact_division = nearest_fp32_bits(amax / 448)
        self.assertEqual(source_product, 0x34700001)
        self.assertEqual(exact_division, 0x34700000)
        self.assertEqual(ceil_log2(expected_binary(source_product, 23)), -22)
        self.assertEqual(ceil_log2(expected_binary(exact_division, 23)), -22)

    def test_ceil_log2_matches_step_search(self):
        values = [FL32_1E_4, FL32_1E_4 * FL32_1_448, Fraction(448), Fraction(440), Fraction(462)]
        values += [448 * power2(k) for k in (-22, -21, 0, 23, 120)]
        values += [power2(e) for e in (-149, -126, -22, -1, 0, 1, 120)]
        rng = random.Random(0xC11)
        for _ in range(256):
            values.append(Fraction(rng.randrange(1, 1 << 64), rng.randrange(1, 1 << 64)))
        for value in values:
            with self.subTest(value=str(value)):
                exponent = 0
                while power2(exponent - 1) >= value:
                    exponent -= 1
                while power2(exponent) < value:
                    exponent += 1
                self.assertEqual(ceil_log2(value), exponent)

    def test_integer_rne_matches_neighbor_grid_search(self):
        magnitudes = [
            Fraction(0), power2(-151), power2(-150), power2(-149), 3 * power2(-150),
            power2(-148), power2(-126), power2(-125), Fraction(1) + power2(-24),
            Fraction(1) + 3 * power2(-24), Fraction(112), Fraction(418), Fraction(448),
        ]
        rng = random.Random(0x5814)
        for _ in range(256):
            magnitudes.append(Fraction(rng.randrange(0, 1 << 40)) * power2(rng.randrange(-200, 131)))
        for magnitude in magnitudes:
            numer = magnitude.numerator
            exp2 = 1 - magnitude.denominator.bit_length()
            for sign in (0, 1):
                with self.subTest(magnitude=str(magnitude), sign=sign):
                    self.assertEqual(
                        rne_fp32_bits(numer, exp2, sign),
                        search_rounder(numer, exp2, sign),
                    )

    def test_e4m3_encoder_matches_brute_force_and_round_trips(self):
        magnitudes = list(E4M3_MAGNITUDES)
        for code in range(0x7E):
            magnitudes.append((E4M3_MAGNITUDES[code] + E4M3_MAGNITUDES[code + 1]) / 2)
        rng = random.Random(0xE4)
        while len(magnitudes) < 900:
            candidate = Fraction(rng.randrange(1, 1 << 24)) * power2(rng.randrange(-25, 9))
            if candidate <= 448:
                magnitudes.append(candidate)
        for magnitude in magnitudes:
            with self.subTest(magnitude=str(magnitude)):
                exp2 = 1 - magnitude.denominator.bit_length()
                bits = rne_fp32_bits(magnitude.numerator, exp2, 0)
                self.assertEqual(encode_e4m3_from_bits(bits), brute_encode_e4m3(magnitude))
        for code in range(0x7F):
            value = E4M3_MAGNITUDES[code]
            bits = rne_fp32_bits(value.numerator, 1 - value.denominator.bit_length(), 0)
            with self.subTest(code=hex(code)):
                self.assertEqual(encode_e4m3_from_bits(bits), code)
                self.assertEqual(encode_e4m3_from_bits(bits | 0x80000000), code | 0x80)

    def test_integer_rounder_pipeline_matches_search_rounded_pipeline(self):
        vectors = (
            five_group_bits(), below_floor_bits(), subnormal_amax_bits(),
            quotient_boundary_bits(), signed_zero_bits(),
        )
        for bits in vectors:
            with self.subTest(k=len(bits)):
                self.assertEqual(
                    expected_trace(bits, rne_fp32_bits),
                    expected_trace(bits, search_rounder),
                )


class ScaleExhaustiveTests(unittest.TestCase):
    """All finite BF16 amax: both exponent methods agree, clamp unreachable."""

    def test_all_finite_amax_group_stages(self):
        batch_groups = 256
        for start in range(0, len(POSITIVE_AMAX_CODES), batch_groups):
            bits = []
            for amax_code in POSITIVE_AMAX_CODES[start:start + batch_groups]:
                bits.append(amax_code)
                bits.extend([0x0000] * 31)
            with self.subTest(start=start):
                check_values(self, bits, tag=f"amax-{start}")
        spot = {
            0x0000: -22, 0x3880: -22, 0x38D1: -22, 0x38D2: -22,
            0x43E0: 0, 0x4F60: 23, 0x4FE0: 24, 0x7F60: 119, 0x7F7F: 120,
        }
        for code, exponent in spot.items():
            stages = group_scale_stages(code)
            self.assertEqual(stages[3], exponent, hex(code))
            self.assertEqual(stages[4], exponent + 127, hex(code))


class ElementExhaustiveTests(unittest.TestCase):
    """All 65280 finite BF16 element codes inside legal 32-element groups."""

    def test_all_finite_codes_within_their_own_binade(self):
        ordered = sorted(FINITE_BF16_CODES, key=lambda code: code & 0x7FFF)
        self.assertEqual(len(ordered), 65280)
        batch = 64 * 32
        for start in range(0, len(ordered), batch):
            with self.subTest(start=start):
                check_values(self, ordered[start:start + batch], tag=f"binade-{start}")

    def test_all_finite_codes_under_maximum_amax_anchor(self):
        ordered = sorted(FINITE_BF16_CODES, key=lambda code: code & 0x7FFF)
        anchor = 0x7F7F
        bits, group = [], [anchor]
        for code in ordered:
            group.append(code)
            if len(group) == 32:
                bits.extend(group)
                group = [anchor]
        group.extend([0x0000] * (32 - len(group)))
        bits.extend(group)
        batch = 64 * 32
        for start in range(0, len(bits), batch):
            with self.subTest(start=start):
                check_values(self, bits[start:start + batch], tag=f"anchored-{start}")


class DirectedCases(unittest.TestCase):
    """Directed counterexamples and boundaries; the sensitivity mutation targets."""

    def test_directed_scale_ceil_not_floor_or_round(self):
        exact_ks = (-21, -7, 0, 1, 23, 24, 119)
        contrast_ks = (-21, 0, 23, 119)
        amaxes = [448 * power2(k) for k in exact_ks]
        amaxes += [440 * power2(k) for k in contrast_ks]
        amaxes += [462 * power2(k - 1) for k in contrast_ks]
        expected_exponents = list(exact_ks) + list(contrast_ks) + list(contrast_ks)
        bits = []
        for value in amaxes:
            anchor = bf16_code(value)
            quarter = bf16_code(value / 4)
            pattern = (anchor, quarter, quarter | 0x8000, 0x0000, 0x8000)
            bits.extend(pattern[position % 5] for position in range(32))
        check_pipeline(self, bits, tag="scale-ceil")
        trace = trace_act_quant(bits)
        self.assertEqual([group.exponent for group in trace.groups], expected_exponents)
        self.assertEqual([group.scale_code for group in trace.groups],
                         [exponent + 127 for exponent in expected_exponents])
        for index in range(len(exact_ks)):
            self.assertEqual(trace.groups[index].quotient_bits[0], 0x43E00000, index)
            self.assertEqual(trace.groups[index].a_codes[0], 0x7E, index)

    def test_directed_amax_floor(self):
        bits = below_floor_bits()
        check_pipeline(self, bits, tag="amax-floor")
        trace = trace_act_quant(bits)
        self.assertEqual([group.exponent for group in trace.groups], [-22] * 4)
        self.assertEqual([group.scale_code for group in trace.groups], [0x69] * 4)
        self.assertEqual(trace.groups[0].amax_bits, 0x00000000)
        self.assertEqual(trace.groups[0].a_codes, (0,) * 16 + (0x80,) * 16)
        self.assertEqual(trace.groups[1].quotient_bits, (0x43800000,) * 32)
        self.assertEqual(trace.groups[1].a_codes, (0x78,) * 32)
        self.assertEqual(trace.groups[2].quotient_bits, (0x43000000,) * 32)
        self.assertEqual(trace.groups[2].a_codes, (0x70,) * 32)
        self.assertEqual(trace.groups[3].amax_bits, 0x38D1 << 16)
        self.assertEqual(trace.groups[3].quotient_bits[:3],
                         (0x43800000, 0x43000000, 0x43D10000))
        self.assertEqual(trace.groups[3].a_codes[:3], (0x78, 0x70, 0x7D))

    def test_directed_e4m3_rne_midpoints(self):
        bits = midpoint_bits()
        check_pipeline(self, bits, tag="e4m3-midpoints")
        probes = [
            0x43E0,
            bf16_code(power2(-10)), bf16_code(3 * power2(-10)), bf16_code(Fraction(432)),
            bf16_code(Fraction(17, 16)),
            bf16_code(power2(-10)) | 0x8000, bf16_code(3 * power2(-10)) | 0x8000,
            bf16_code(Fraction(432)) | 0x8000,
            bf16_code(power2(-10)) - 1, bf16_code(power2(-10)) + 1,
        ]
        probes += [0x0000] * (32 - len(probes))
        check_pipeline(self, probes, tag="midpoint-anchors")
        trace = trace_act_quant(probes)
        self.assertEqual(trace.groups[0].quotient_bits[1], bf16_code(power2(-10)) << 16)
        self.assertEqual(trace.a_codes[1], 0x00)
        self.assertEqual(trace.a_codes[2], 0x02)
        self.assertEqual(trace.a_codes[3], 0x7E)
        self.assertEqual(trace.a_codes[4], 0x38)
        self.assertEqual(trace.a_codes[5], 0x80)
        self.assertEqual(trace.a_codes[6], 0x82)
        self.assertEqual(trace.a_codes[7], 0xFE)
        self.assertEqual(trace.a_codes[8], 0x00)
        self.assertEqual(trace.a_codes[9], 0x01)

    def test_directed_signed_zero(self):
        bits = signed_zero_bits()
        check_pipeline(self, bits, tag="signed-zero")
        trace = trace_act_quant(bits)
        near, far = trace.groups
        self.assertEqual(near.exponent, 0)
        self.assertEqual(far.exponent, 120)
        self.assertEqual(near.quotient_bits[:5],
                         (0x43E00000, 0x00000000, 0x80000000, 0x00010000, 0x80010000))
        self.assertEqual(near.clamped_bits[:5], near.quotient_bits[:5])
        self.assertEqual(near.a_codes[:5], (0x7E, 0x00, 0x80, 0x00, 0x80))
        self.assertEqual(far.quotient_bits[:5],
                         (0x437F0000, 0x00000000, 0x80000000, 0x00000000, 0x80000000))
        self.assertEqual(far.a_codes[:5], (0x78, 0x00, 0x80, 0x00, 0x80))

    def test_directed_bf16_subnormal_decode(self):
        bits = subnormal_amax_bits()
        check_pipeline(self, bits, tag="bf16-subnormal")
        trace = trace_act_quant(bits)
        positive, negative = trace.groups
        for group in (positive, negative):
            self.assertEqual(group.amax_bits, 0x00010000)
            self.assertEqual(group.bounded_amax_bits, FL32_1E_4_BITS)
            self.assertEqual(group.exponent, -22)
            self.assertEqual(group.scale_code, 0x69)
        self.assertEqual(positive.quotient_bits, (0x08000000,) + (0x00000000,) * 31)
        self.assertEqual(positive.a_codes, (0x00,) + (0x00,) * 31)
        self.assertEqual(negative.quotient_bits, (0x88000000,) + (0x00000000,) * 31)
        self.assertEqual(negative.a_codes, (0x80,) + (0x00,) * 31)

    def test_directed_quotient_subnormal(self):
        bits = quotient_boundary_bits()
        check_pipeline(self, bits, tag="quotient-subnormal")
        group = trace_act_quant(bits).groups[0]
        self.assertEqual(group.exponent, 23)
        self.assertEqual(group.scale_code, 23 + 127)
        self.assertEqual(group.quotient_bits[:7],
                         (0x43E00000, 0x00000001, 0x80000001, 0x00000001,
                          0x00000001, 0x80000001, 0x80000001))
        self.assertEqual(group.clamped_bits, group.quotient_bits)
        self.assertEqual(group.a_codes[:7], (0x7E, 0x00, 0x80, 0x00, 0x00, 0x80, 0x80))

    def test_directed_quotient_halfway(self):
        bits = quotient_boundary_bits()
        check_pipeline(self, bits, tag="quotient-halfway")
        group = trace_act_quant(bits).groups[1]
        self.assertEqual(group.exponent, 24)
        self.assertEqual(group.scale_code, 24 + 127)
        self.assertEqual(group.quotient_bits[:7],
                         (0x43E00000, 0x00000000, 0x80000000, 0x00000001,
                          0x00000000, 0x80000001, 0x80000000))
        self.assertEqual(group.a_codes[:7], (0x7E, 0x00, 0x80, 0x00, 0x00, 0x80, 0x80))

    def test_directed_five_group_layout(self):
        bits = five_group_bits()
        self.assertEqual(len(bits), 160)
        check_pipeline(self, bits, tag="five-group-layout")
        trace = trace_act_quant(bits)
        self.assertEqual([group.exponent for group in trace.groups], [0, 1, 2, 3, 4])
        self.assertEqual(list(trace.scale_a), [0x7F, 0x80, 0x81, 0x82, 0x83])
        self.assertEqual([group.quotient_bits[0] for group in trace.groups],
                         [0x43E00000] * 5)
        self.assertEqual([group.a_codes[0] for group in trace.groups], [0x7E] * 5)
        self.assertEqual([group.a_codes[1] for group in trace.groups], [0x6E] * 5)
        self.assertEqual([group.a_codes[2] for group in trace.groups], [0xEE] * 5)
        self.assertEqual([group.a_codes[3] for group in trace.groups], [0x00] * 5)
        self.assertEqual([group.a_codes[4] for group in trace.groups], [0x80] * 5)


DIRECTED_TEST_NAMES = tuple(
    f"DirectedCases.{name}"
    for name in sorted(name for name in vars(DirectedCases) if name.startswith("test_"))
)


class ValidationTests(unittest.TestCase):
    """Input failure contract: metadata before content, exact errors per class."""

    def baseline(self):
        return (0x3F80,) * 32

    def test_metadata_errors(self):
        cases = ((), [], (0x3F80,) * 16, (0x3F80,) * 31, (0x3F80,) * 33,
                 (0x3F80,) * 63, (0x3F80,) * 97, [0x3F80] * 15)
        for entry in ENTRY_POINTS:
            for bad in cases:
                with self.subTest(entry=entry.__name__, length=len(bad)):
                    with self.assertRaises(ValueError):
                        entry(bad)

    def test_valid_lengths_and_containers(self):
        for length in (32, 64, 96, 160):
            for container in (tuple, list):
                data = container((0x3F80,) * length)
                codes, scales = act_quant(data)
                self.assertEqual(len(codes), length)
                self.assertEqual(len(scales), length // 32)
        self.assertEqual(act_quant(list(self.baseline())), act_quant(self.baseline()))

    def test_content_errors(self):
        values = ((-1, ValueError), (65536, ValueError), (-65536, ValueError),
                  (1 << 99, ValueError), (True, TypeError), (1.0, TypeError),
                  ("0x3F80", TypeError), (None, TypeError), ((0x3F80,), TypeError))
        for entry in ENTRY_POINTS:
            for bad, error in values:
                for position in (0, 31, 32, 63):
                    with self.subTest(entry=entry.__name__, bad=repr(bad), position=position):
                        data = [0x3F80] * 64
                        data[position] = bad
                        with self.assertRaises(error):
                            entry(data)

    def test_non_sequence_and_nested_inputs(self):
        generator = (code for code in (0x3F80,) * 32)
        for entry in ENTRY_POINTS:
            for bad in (5, 1.5, None, True, iter((0x3F80,) * 32), generator,
                        {(0x3F80,) * 32}, "0" * 32):
                with self.subTest(entry=entry.__name__, bad=type(bad).__name__):
                    with self.assertRaises(TypeError):
                        entry(bad)
            for nested in ([[0x3F80]] * 32, (0x3F80,) * 31 + ((0x3F80,),)):
                with self.subTest(entry=entry.__name__, nested=True):
                    with self.assertRaises(TypeError):
                        entry(nested)

    def test_memoryview_dimensions(self):
        contiguous = memoryview(bytearray(64)).cast("H")
        for index, code in enumerate(self.baseline()):
            contiguous[index] = code
        strided = memoryview(bytearray(128)).cast("H")[::2]
        for index, code in enumerate(self.baseline()):
            strided[index] = code
        expected = act_quant(self.baseline())
        expected_trace_result = trace_act_quant(self.baseline())
        for view in (contiguous, strided):
            snapshot = view.tobytes()
            self.assertEqual(act_quant(view), expected)
            self.assertEqual(trace_act_quant(view), expected_trace_result)
            self.assertEqual(view.tobytes(), snapshot)
        invalid = (
            memoryview(bytes(2)).cast("H", shape=()),
            memoryview(bytes(64)).cast("H", shape=(32, 1)),
            memoryview(bytes(128)).cast("H", shape=(32, 2)),
            memoryview(bytes(64)).cast("H", shape=(16, 2)),
            memoryview(bytes(64)).cast("H", shape=(32, 1, 1)),
        )
        with patch("ref.act_quant._bf16",
                   side_effect=ValidationOrderMismatch("content checked before dimensions")) as checked:
            for entry in ENTRY_POINTS:
                for view in invalid:
                    with self.subTest(entry=entry.__name__, shape=view.shape):
                        with self.assertRaises(TypeError):
                            entry(view)
            self.assertEqual(checked.call_count, 0)

    def test_metadata_validated_before_content(self):
        for entry in ENTRY_POINTS:
            with self.subTest(entry=entry.__name__, kind="length beats float"):
                with self.assertRaises(ValueError):
                    entry((1.0,) + (0x3F80,) * 32)
            with self.subTest(entry=entry.__name__, kind="short beats float"):
                with self.assertRaises(ValueError):
                    entry((0x3F80,) * 15 + (1.0,))

    def test_validation_precedes_computation(self):
        bad_content = {
            "float": (0x3F80,) * 63 + (1.0,),
            "bool": (0x3F80,) * 32 + (True,) + (0x3F80,) * 31,
            "wide": (0x3F80,) * 63 + (65536,),
            "nonfinite-tail": (0x3F80,) * 95 + (0xFFC0,),
        }
        with patch("ref.act_quant._group_scale",
                   side_effect=ValidationOrderMismatch("computed before validation")):
            for entry in ENTRY_POINTS:
                for name, data in bad_content.items():
                    with self.subTest(entry=entry.__name__, bad=name):
                        with self.assertRaises((TypeError, ValueError)):
                            entry(data)
        with patch("ref.act_quant._bf16",
                   side_effect=ValidationOrderMismatch("content checked before metadata")) as checked:
            for entry in ENTRY_POINTS:
                for data in ((1.0,) * 33, (1.0,) * 16, (0x7F80,) * 33):
                    with self.subTest(entry=entry.__name__, metadata="length"):
                        with self.assertRaises(ValueError):
                            entry(data)
            self.assertEqual(checked.call_count, 0)

    def test_input_not_rewritten(self):
        data = [0x3F80, 0x8000, 0x0001, 0x8001, 0x3880, 0x43E0, 0x7F7F, 0xFF7F] * 4
        snapshot = list(data)
        for entry in ENTRY_POINTS:
            entry(data)
            self.assertEqual(data, snapshot, entry.__name__)
            self.assertIs(type(data), list, entry.__name__)


class NonFiniteTests(unittest.TestCase):
    """All 256 non-finite BF16 codes are rejected at every element position."""

    def test_all_nonfinite_codes_rejected(self):
        baseline = (0x3F80,) * 32
        codes = [code for code in range(0x7F80, 0x10000) if (code & 0x7F80) == 0x7F80]
        self.assertEqual(len(codes), 256)
        for code in codes:
            for position in range(32):
                data = list(baseline)
                data[position] = code
                for entry in ENTRY_POINTS:
                    with self.assertRaises(ValueError):
                        entry(data)


class StructureTests(unittest.TestCase):
    """Trace shape, field types, and the shared evaluation core."""

    def test_trace_structure_types_and_entry_consistency(self):
        bits = five_group_bits()
        trace = trace_act_quant(bits)
        direct = act_quant(bits)
        self.assertIsInstance(trace, ActQuantTrace)
        self.assertIs(type(trace.groups), tuple)
        self.assertEqual(len(trace.groups), 5)
        self.assertEqual(len(trace.a_codes), 160)
        self.assertEqual(len(trace.scale_a), 5)
        self.assertEqual(direct, (trace.a_codes, trace.scale_a))
        for group in trace.groups:
            self.assertIsInstance(group, GroupTrace)
            for name in ("amax_bits", "bounded_amax_bits", "product_bits",
                         "exponent", "scale_code"):
                self.assertIs(type(getattr(group, name)), int)
            for name in ("quotient_bits", "clamped_bits", "a_codes"):
                field = getattr(group, name)
                self.assertIs(type(field), tuple)
                self.assertEqual(len(field), 32)
                self.assertTrue(all(type(value) is int for value in field))
        self.assertTrue(all(type(code) is int for code in trace.a_codes))
        self.assertTrue(all(type(code) is int for code in trace.scale_a))
        self.assertTrue(all(code not in (0x7F, 0xFF) for code in trace.a_codes))
        self.assertTrue(all(0 <= code <= 0xFE for code in trace.scale_a))

    def test_both_entries_share_the_evaluate_core(self):
        def fake_evaluate(bf16_bits, *, observe):
            if observe:
                return ActQuantTrace((), (0x11,), (0x22,))
            return (0x11,), (0x22,)

        with patch("ref.act_quant._evaluate", side_effect=fake_evaluate) as core:
            plain = act_quant((0x3F80,) * 32)
            self.assertEqual(core.call_count, 1)
            self.assertIs(core.call_args.kwargs["observe"], False)
            self.assertEqual(plain, ((0x11,), (0x22,)))
            traced = trace_act_quant((0x3F80,) * 32)
            self.assertEqual(core.call_count, 2)
            self.assertIs(core.call_args.kwargs["observe"], True)
            self.assertEqual(traced, ActQuantTrace((), (0x11,), (0x22,)))


class RandomTests(unittest.TestCase):
    """Fixed-seed random rows as a supplement to the exhaustive passes."""

    def test_fixed_seed_random_groups(self):
        rng = random.Random(0x14)
        specials = (0x0000, 0x8000, 0x0001, 0x8001, 0x007F, 0x0080, 0x3880, 0x3800,
                    0x38D1, 0x38D2, 0x43E0, 0x4F60, 0x4FE0, 0x7F60, 0x7F7F, 0xFF7F)
        for case in range(24):
            group_count = rng.choice((1, 2, 3, 4, 5, 7))
            bits = []
            for _ in range(group_count):
                mode = rng.choice(("uniform", "coherent", "specials", "tiny"))
                for _ in range(32):
                    if mode == "uniform":
                        code = rng.randrange(0x10000)
                        if (code & 0x7F80) == 0x7F80:
                            code = 0x3F80
                    elif mode == "coherent":
                        code = ((rng.randrange(-40, 121) + 127) << 7) | rng.randrange(128)
                        code |= rng.randrange(2) << 15
                    elif mode == "specials":
                        code = rng.choice(specials)
                    else:
                        code = rng.randrange(0x0100) | (rng.randrange(2) << 15)
                    bits.append(code)
            with self.subTest(case=case, groups=group_count):
                check_pipeline(self, bits, tag=f"random-{case}")


class ComputeCombinationTests(unittest.TestCase):
    """act_quant output drives the compute oracle; expectations from expected codes."""

    def test_act_quant_output_drives_expert_grouped_dot(self):
        rng = random.Random(0xC0FFEE)
        vectors = (
            (five_group_bits(), True), (below_floor_bits(), True),
            (subnormal_amax_bits(), False), (quotient_boundary_bits(), True),
            (signed_zero_bits(), True), (midpoint_bits(), True),
        )
        for bits, nonzero in vectors:
            with self.subTest(k=len(bits)):
                groups, a_codes, scale_a = expected_trace(bits)
                self.assertEqual(any(code & 0x7F for code in a_codes), nonzero)
                ref_codes, ref_scales = act_quant(bits)
                self.assertEqual((ref_codes, ref_scales), (a_codes, scale_a))
                w_packed = independent_pack(
                    [rng.randrange(16) for _ in range(len(a_codes))]
                )
                scale_b = tuple(rng.randrange(255) for _ in range(len(scale_a)))
                blocks, fp32_bits, bf16_bits = compute_expected_trace(
                    a_codes, w_packed, scale_a, scale_b
                )
                result = expert_grouped_dot(ref_codes, w_packed, ref_scales, scale_b)
                self.assertIs(type(result), float)
                bf16_result = expert_grouped_dot(
                    ref_codes, w_packed, ref_scales, scale_b, output="bf16"
                )
                self.assertIs(type(bf16_result), int)
                if fp32_bits == NAN_SENTINEL:
                    self.assertTrue(math.isnan(result))
                    self.assertTrue(is_nan_bf16_bits(bf16_result))
                else:
                    self.assertEqual(host_fp32_bits(result), fp32_bits)
                    self.assertEqual(bf16_result, bf16_bits)
                    self.assertFalse(is_nan_fp32_bits(fp32_bits))


CORE_CLASSES = (
    ExpectationCrossChecks,
    ScaleExhaustiveTests,
    ElementExhaustiveTests,
    DirectedCases,
    ValidationTests,
    NonFiniteTests,
    StructureTests,
    RandomTests,
    ComputeCombinationTests,
)


MARKER_PREFIX = "P0014_MARKER "


class ExecutionTests(unittest.TestCase):
    """Guard the integer-only dependency boundary and the supported host environment."""

    def test_integer_only_module_dependencies(self):
        tree = ast.parse((ROOT / "ref" / "act_quant.py").read_text())
        imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        self.assertTrue(all(isinstance(node, ast.ImportFrom) for node in imports))
        self.assertEqual({node.module for node in imports}, {"collections.abc", "dataclasses"})
        self.assertFalse(any(
            isinstance(node, ast.Constant) and isinstance(node.value, float)
            for node in ast.walk(tree)
        ))
        self.assertFalse(any(
            isinstance(node, (ast.BinOp, ast.AugAssign))
            and isinstance(node.op, (ast.Div, ast.Pow))
            for node in ast.walk(tree)
        ))
        self.assertFalse(any(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in ("float", "pow", "eval", "exec", "__import__")
            for node in ast.walk(tree)
        ))

    def test_host_rounding_ftz_daz_and_status_preserved(self):
        class Fenv(ctypes.Structure):
            """glibc x86_64 fenv_t from the supported host's bits/fenv.h."""

            _fields_ = [
                ("control", ctypes.c_ushort), ("reserved1", ctypes.c_ushort),
                ("status", ctypes.c_ushort), ("reserved2", ctypes.c_ushort),
                ("tags", ctypes.c_ushort), ("reserved3", ctypes.c_ushort),
                ("eip", ctypes.c_uint), ("cs", ctypes.c_ushort),
                ("opcode", ctypes.c_ushort), ("data", ctypes.c_uint),
                ("selector", ctypes.c_ushort), ("reserved5", ctypes.c_ushort),
                ("mxcsr", ctypes.c_uint),
            ]

        self.assertEqual(sys.platform, "linux", "floating-environment probe needs the supported Linux host")
        self.assertEqual(platform.machine(), "x86_64")
        self.assertEqual(platform.libc_ver()[0], "glibc")
        self.assertEqual(ctypes.sizeof(Fenv), 32)
        self.assertEqual(Fenv.mxcsr.offset, 28)
        library = ctypes.CDLL("libm.so.6")
        for name in ("fegetenv", "fesetenv"):
            function = getattr(library, name)
            function.argtypes = (ctypes.POINTER(Fenv),)
            function.restype = ctypes.c_int
        saved = Fenv()
        self.assertEqual(library.fegetenv(ctypes.byref(saved)), 0)
        inputs = (
            (0x0001, 0x8001, 0x8000) + (0,) * 29
            + (0x4F60, 0x0080, 0x8080) + (0,) * 29
            + (0x4FE0, 0x007F, 0x0080, 0x0081, 0x807F, 0x8080, 0x8081) + (0,) * 25
            + (0x43E0, 0x3F88, 0x3F98, 0x8000) + (0,) * 28
        )
        expected_groups, expected_codes, expected_scales = expected_trace(inputs)
        failures = []
        environments = (
            ("nearest", 0), ("upward", 0x4000), ("downward", 0x2000),
            ("toward-zero", 0x6000), ("ftz", 0x8000), ("daz", 0x40),
            ("ftz-daz", 0x8040), ("unmasked-overflow", 0),
        )
        try:
            for name, mode in environments:
                for status in (0, 0x3F):
                    setting = Fenv.from_buffer_copy(bytes(saved))
                    setting.mxcsr = 0x1F80 | mode | status
                    if name == "unmasked-overflow":
                        setting.mxcsr &= ~0x400
                    if library.fesetenv(ctypes.byref(setting)):
                        failures.append((name, status, "set failed"))
                        continue
                    before = Fenv()
                    library.fegetenv(ctypes.byref(before))
                    for entry in ENTRY_POINTS:
                        try:
                            result = entry(inputs)
                        except Exception as error:
                            after = Fenv()
                            library.fegetenv(ctypes.byref(after))
                            failures.append((name, status, entry.__name__, type(error).__name__))
                            if after.mxcsr != before.mxcsr:
                                failures.append((name, status, "exception changed MXCSR"))
                            continue
                        after = Fenv()
                        library.fegetenv(ctypes.byref(after))
                        if after.mxcsr != before.mxcsr:
                            failures.append((name, status, entry.__name__, "MXCSR changed"))
                        if isinstance(result, ActQuantTrace):
                            if (result.a_codes, result.scale_a) != (expected_codes, expected_scales):
                                failures.append((name, status, "trace output changed"))
                            if len(result.groups) != len(expected_groups):
                                failures.append((name, status, "trace group count changed"))
                            for actual, expected in zip(result.groups, expected_groups):
                                for field, value in expected.items():
                                    if getattr(actual, field) != value:
                                        failures.append((name, status, field, "stage changed"))
                        elif result != (expected_codes, expected_scales):
                            failures.append((name, status, "normal output changed"))
                    for entry in ENTRY_POINTS:
                        try:
                            entry((0,) * 63 + (0x7F80,))
                        except ValueError:
                            pass
                        else:
                            failures.append((name, status, "nonfinite accepted"))
                        after = Fenv()
                        library.fegetenv(ctypes.byref(after))
                        if after.mxcsr != before.mxcsr:
                            failures.append((name, status, "rejection changed MXCSR"))
                    if before.mxcsr != setting.mxcsr:
                        failures.append((name, status, "environment not installed"))
        finally:
            restore_result = library.fesetenv(ctypes.byref(saved))
        self.assertEqual(restore_result, 0)
        restored = Fenv()
        self.assertEqual(library.fegetenv(ctypes.byref(restored)), 0)
        self.assertEqual(bytes(restored), bytes(saved))
        self.assertEqual(failures, [])


class MarkerProtocolError(AssertionError):
    """A subprocess result must identify actual comparison failures without tampering."""


class MarkerResult(unittest.TestResult):
    """Retain typed evidence for tests and subtests; untyped exceptions are errors."""

    def __init__(self):
        super().__init__()
        self.numeric_failures = set()
        self.validation_failures = set()

    def record_failure(self, test, error):
        if error[0] is NumericMismatch:
            self.numeric_failures.add(test_key(test))
        elif error[0] is ValidationOrderMismatch:
            self.validation_failures.add(test_key(test))
        else:
            self.addError(test, error)

    def addFailure(self, test, error):
        super().addFailure(test, error)
        self.record_failure(test, error)

    def addSubTest(self, test, subtest, error):
        super().addSubTest(test, subtest, error)
        if error is not None and issubclass(error[0], test.failureException):
            self.record_failure(subtest, error)


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
    tests = list(iter_tests(suite))
    result = MarkerResult()
    suite.run(result)
    errored = {test_key(test) for test, _ in result.errors}
    skipped = {test_key(test) for test, _ in result.skipped}
    cases = []
    for test in tests:
        key = test_key(test)
        if key in errored:
            status = "error"
        elif key in skipped:
            status = "skip"
        elif key in result.validation_failures:
            status = "validation-fail"
        elif key in result.numeric_failures:
            status = "numeric-fail"
        else:
            status = "pass"
        cases.append([key, status])
    core = {
        "cases": cases,
        "errors": sum(status == "error" for _, status in cases),
        "failures": sum(status in ("numeric-fail", "validation-fail") for _, status in cases),
        "skipped": sum(status == "skip" for _, status in cases),
        "tests": len(tests),
    }
    return dict(core, digest=hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest())


def validate_marker_payload(payload):
    required = {"cases", "errors", "failures", "skipped", "tests", "digest"}
    if not isinstance(payload, dict) or set(payload) != required:
        raise MarkerProtocolError("marker keys do not match the protocol")
    for name in ("errors", "failures", "skipped", "tests"):
        value = payload[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise MarkerProtocolError(f"marker field {name} is not a count")
    cases = payload["cases"]
    if not isinstance(cases, list) or len(cases) != payload["tests"]:
        raise MarkerProtocolError("marker case list does not match its count")
    tallies = dict.fromkeys(("pass", "numeric-fail", "validation-fail", "error", "skip"), 0)
    keys = set()
    for entry in cases:
        if not isinstance(entry, list) or len(entry) != 2 or not isinstance(entry[0], str) or entry[1] not in tallies:
            raise MarkerProtocolError("malformed marker case")
        if entry[0] in keys:
            raise MarkerProtocolError("duplicate marker case")
        keys.add(entry[0])
        tallies[entry[1]] += 1
    failures = tallies["numeric-fail"] + tallies["validation-fail"]
    if (failures, tallies["error"], tallies["skip"]) != (payload["failures"], payload["errors"], payload["skipped"]):
        raise MarkerProtocolError("marker counts disagree with its cases")
    core = {name: payload[name] for name in ("cases", "errors", "failures", "skipped", "tests")}
    digest = hashlib.sha256(json.dumps(core, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    if digest != payload["digest"]:
        raise MarkerProtocolError("marker digest disagrees with its content")


def run_marker_subprocess(script, arguments, timeout=300):
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    process = subprocess.run(
        [sys.executable, str(script), *arguments], capture_output=True, text=True,
        env=environment, timeout=timeout,
    )
    if process.returncode not in (0, 1):
        raise MarkerProtocolError(f"subprocess exited {process.returncode}: {process.stderr[-400:]}")
    lines = [line for line in process.stdout.splitlines() if line.startswith(MARKER_PREFIX)]
    if len(lines) != 1:
        raise MarkerProtocolError("subprocess must emit exactly one result marker")
    try:
        payload = json.loads(lines[0][len(MARKER_PREFIX):])
    except json.JSONDecodeError as error:
        raise MarkerProtocolError("invalid result JSON") from error
    validate_marker_payload(payload)
    if process.returncode != int(bool(payload["failures"] or payload["errors"] or payload["skipped"])):
        raise MarkerProtocolError("subprocess exit disagrees with its result")
    return payload


def require_numeric_failure(payload, detector):
    validate_marker_payload(payload)
    if payload["errors"] or payload["skipped"]:
        raise MarkerProtocolError("semantic evidence must not contain exceptions or skips")
    if any(status == "validation-fail" for _, status in payload["cases"]):
        raise MarkerProtocolError("validation-order failures are not numeric evidence")
    if [detector, "numeric-fail"] not in payload["cases"]:
        raise MarkerProtocolError("the named detector must fail a numeric comparison")


def build_copy(destination):
    (destination / "tests" / "quant").mkdir(parents=True)
    shutil.copytree(ROOT / "ref", destination / "ref", ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("test_formats.py", "test_compute.py", "test_act_quant.py"):
        shutil.copy(ROOT / "tests" / "quant" / name, destination / "tests" / "quant" / name)
    return destination


_AWAY_SHIFT = '''

def _away_shift(significand, shift):
    if shift <= 0:
        return significand << -shift
    whole, remainder = divmod(significand, 1 << shift)
    return whole + (remainder >= 1 << (shift - 1))
'''


MUTATIONS = (
    {
        "slug": "scale-floor", "detector": "test_directed_scale_ceil_not_floor_or_round",
        "replacements": ((
            "scale_exponent = ((product_bits >> 23) & 0xFF) - 127 + bool(product_bits & 0x7FFFFF)",
            "scale_exponent = ((product_bits >> 23) & 0xFF) - 127",
        ),), "helper": "",
    },
    {
        "slug": "scale-round", "detector": "test_directed_scale_ceil_not_floor_or_round",
        "replacements": ((
            "scale_exponent = ((product_bits >> 23) & 0xFF) - 127 + bool(product_bits & 0x7FFFFF)",
            "scale_exponent = ((product_bits >> 23) & 0xFF) - 127 + (((1 << 23) | (product_bits & 0x7FFFFF)) ** 2 > (1 << 47))",
        ),), "helper": "",
    },
    {
        "slug": "four-group-scale", "detector": "test_directed_five_group_layout",
        "replacements": ((
            "amax, bounded, product, exponent, scale_code = _group_scale(group)",
            "amax, bounded, product, exponent, scale_code = _group_scale(inputs[(offset // 128) * 128:(offset // 128 + 1) * 128])",
        ),), "helper": "",
    },
    {
        "slug": "fp8-truncate", "detector": "test_directed_e4m3_rne_midpoints",
        "replacements": (
            ("return sign | _round_shift(significand, -9 - exponent)", "return sign | _truncate_shift(significand, -9 - exponent)"),
            ("rounded = _round_shift(significand, significand.bit_length() - 4)", "rounded = _truncate_shift(significand, significand.bit_length() - 4)"),
        ), "helper": '''

def _truncate_shift(significand, shift):
    return significand << -shift if shift <= 0 else significand >> shift
''',
    },
    {
        "slug": "fp8-tie-away", "detector": "test_directed_e4m3_rne_midpoints",
        "replacements": (
            ("return sign | _round_shift(significand, -9 - exponent)", "return sign | _away_shift(significand, -9 - exponent)"),
            ("rounded = _round_shift(significand, significand.bit_length() - 4)", "rounded = _away_shift(significand, significand.bit_length() - 4)"),
        ), "helper": _AWAY_SHIFT,
    },
    {
        "slug": "negative-zero", "detector": "test_directed_signed_zero",
        "replacements": ((
            "sign = (bits >> 24) & 0x80",
            "sign = 0 if (bits & 0x7FFFFFFF) == 0 else (bits >> 24) & 0x80",
        ),), "helper": "",
    },
    {
        "slug": "remove-floor", "detector": "test_directed_amax_floor",
        "replacements": (("bounded_amax_bits = max(amax_bits, _AMAX_FLOOR_BITS)", "bounded_amax_bits = amax_bits"),),
        "helper": "",
    },
    {
        "slug": "bf16-daz", "detector": "test_directed_bf16_subnormal_decode",
        "replacements": (
            ("amax_bits = max((code & 0x7FFF) << 16 for code in group)", "amax_bits = max(_daz_bf16(code) & 0x7FFFFFFF for code in group)"),
            ("bits = code << 16", "bits = _daz_bf16(code)"),
        ), "helper": '''

def _daz_bf16(code):
    return (code & 0x8000) << 16 if (code & 0x7F80) == 0 else code << 16
''',
    },
    {
        "slug": "quotient-ftz", "detector": "test_directed_quotient_subnormal",
        "replacements": ((
            "return _fp32_bits(bits & 0x80000000, significand, exponent - scale_exponent)",
            "return _flush_quotient(_fp32_bits(bits & 0x80000000, significand, exponent - scale_exponent))",
        ),), "helper": '''

def _flush_quotient(bits):
    return bits & 0x80000000 if (bits & 0x7FFFFFFF) < 0x00800000 else bits
''',
    },
    {
        "slug": "quotient-tie-away", "detector": "test_directed_quotient_halfway",
        "replacements": ((
            "return _fp32_bits(bits & 0x80000000, significand, exponent - scale_exponent)",
            "return _away_quotient(bits & 0x80000000, significand, exponent - scale_exponent)",
        ),), "helper": _AWAY_SHIFT + '''

def _away_quotient(sign, significand, exponent):
    if significand == 0:
        return sign
    if significand.bit_length() - 1 + exponent < -126:
        return sign | _away_shift(significand, -149 - exponent)
    return _fp32_bits(sign, significand, exponent)
''',
    },
)


class SensitivityTests(unittest.TestCase):
    """Semantic mutations must fail named numeric cases, never just crash or skip."""

    def test_marker_failure_categories(self):
        class MarkerProbe(unittest.TestCase):
            def runTest(self):
                for action in actions:
                    with self.subTest(action=action):
                        if action == "numeric":
                            assert_numeric_equal(0, 1, "P0014 marker regression")
                        elif action != "pass":
                            errors = {
                                "validation": ValidationOrderMismatch,
                                "assertion": AssertionError,
                                "runtime": RuntimeError,
                                "skip": unittest.SkipTest,
                            }
                            raise errors[action]("P0014 marker regression")

        cases = (
            (("pass",), "pass"),
            (("numeric",), "numeric-fail"),
            (("validation",), "validation-fail"),
            (("assertion",), "error"),
            (("runtime",), "error"),
            (("skip",), "skip"),
            (("numeric", "assertion"), "error"),
            (("numeric", "runtime"), "error"),
            (("numeric", "skip"), "skip"),
            (("numeric", "validation"), "validation-fail"),
        )
        for actions, status in cases:
            with self.subTest(actions=actions):
                payload = marker_payload(unittest.TestSuite([MarkerProbe()]))
                self.assertEqual(payload["cases"], [["MarkerProbe.runTest", status]])
                validate_marker_payload(payload)
                if status == "numeric-fail":
                    require_numeric_failure(payload, "MarkerProbe.runTest")
                    with self.assertRaises(MarkerProtocolError):
                        require_numeric_failure(payload, "OtherProbe.runTest")
                else:
                    with self.assertRaises(MarkerProtocolError):
                        require_numeric_failure(payload, "MarkerProbe.runTest")

    def test_marker_rejects_entry_and_expectation_assertions(self):
        for name in ("trace_act_quant", "act_quant", "expected_trace"):
            with self.subTest(source=name):
                with patch(f"{__name__}.{name}",
                           side_effect=AssertionError("P0014 injected assertion")):
                    suite = unittest.TestSuite([DirectedCases("test_directed_signed_zero")])
                    payload = marker_payload(suite)
                self.assertEqual(payload["cases"], [["DirectedCases.test_directed_signed_zero", "error"]])
                self.assertEqual(payload["failures"], 0)
                self.assertEqual(payload["errors"], 1)
                with self.assertRaises(MarkerProtocolError):
                    require_numeric_failure(payload, "DirectedCases.test_directed_signed_zero")

    def test_mutations_failure_propagation_and_pristine_retest(self):
        guarded = tuple((ROOT / "ref").glob("*.py")) + tuple(
            ROOT / "tests" / "quant" / name for name in ("test_formats.py", "test_compute.py", "test_act_quant.py")
        )
        digests = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in guarded}
        with tempfile.TemporaryDirectory(prefix="p0014-sensitivity-") as directory:
            pristine = build_copy(Path(directory) / "pristine")
            script = pristine / "tests" / "quant" / "test_act_quant.py"
            baseline = run_marker_subprocess(script, ["--suite", "directed", "--marker"])
            expected_cases = set(DIRECTED_TEST_NAMES)
            self.assertEqual({case for case, _ in baseline["cases"]}, expected_cases)
            self.assertEqual(baseline["failures"] + baseline["errors"] + baseline["skipped"], 0)
            for mutation in MUTATIONS:
                mutated = build_copy(Path(directory) / mutation["slug"])
                source = mutated / "ref" / "act_quant.py"
                text = source.read_text()
                for anchor, replacement in mutation["replacements"]:
                    self.assertEqual(text.count(anchor), 1, mutation["slug"])
                    text = text.replace(anchor, replacement)
                source.write_text(text + mutation["helper"])
                result = run_marker_subprocess(mutated / "tests" / "quant" / "test_act_quant.py", ["--suite", "directed", "--marker"])
                self.assertEqual({case for case, _ in result["cases"]}, expected_cases)
                require_numeric_failure(result, f"DirectedCases.{mutation['detector']}")
                source.write_bytes((pristine / "ref" / "act_quant.py").read_bytes())
                restored = run_marker_subprocess(mutated / "tests" / "quant" / "test_act_quant.py", ["--suite", "directed", "--marker"])
                self.assertEqual(restored, baseline, mutation["slug"])
            for name, replacement, category in (
                ("exception", "raise RuntimeError('P0014 injected exception')", "errors"),
                ("assertion-exception", "raise AssertionError('P0014 injected implementation assertion')", "errors"),
                ("wrong-output", "result = tuple(0 for _ in a_codes), tuple(scale_a)", "failures"),
            ):
                mutated = build_copy(Path(directory) / name)
                source = mutated / "ref" / "act_quant.py"
                text = source.read_text()
                anchor = "result = tuple(a_codes), tuple(scale_a)"
                self.assertEqual(text.count(anchor), 1)
                source.write_text(text.replace(anchor, replacement))
                mutated_script = mutated / "tests" / "quant" / "test_act_quant.py"
                result = run_marker_subprocess(mutated_script, ["--suite", "directed", "--marker"])
                self.assertGreater(result[category], 0)
                self.assertEqual(result["skipped"], 0)
                if category == "errors":
                    self.assertEqual(result["failures"], 0)
                    with self.assertRaises(MarkerProtocolError):
                        require_numeric_failure(result, "DirectedCases.test_directed_signed_zero")
                else:
                    require_numeric_failure(result, "DirectedCases.test_directed_signed_zero")
                process = subprocess.run([sys.executable, str(mutated_script), "--suite", "directed"], capture_output=True, text=True, timeout=300)
                self.assertEqual(process.returncode, 1)
            mutated = build_copy(Path(directory) / "interleaved-validation")
            source = mutated / "ref" / "act_quant.py"
            text = source.read_text()
            for anchor, replacement in (
                ("return tuple(_bf16(code) for code in bf16_bits)", "return tuple(bf16_bits)"),
                ("group = inputs[offset:offset + 32]", "group = tuple(_bf16(code) for code in inputs[offset:offset + 32])"),
            ):
                self.assertEqual(text.count(anchor), 1)
                text = text.replace(anchor, replacement)
            source.write_text(text)
            validation_arguments = ["--suite", "core", "--case", "test_validation_precedes_computation", "--marker"]
            result = run_marker_subprocess(mutated / "tests" / "quant" / "test_act_quant.py", validation_arguments)
            self.assertEqual(result["errors"], 0)
            self.assertEqual(result["skipped"], 0)
            self.assertEqual(result["cases"], [["ValidationTests.test_validation_precedes_computation", "validation-fail"]])
            with self.assertRaises(MarkerProtocolError):
                require_numeric_failure(result, "ValidationTests.test_validation_precedes_computation")
            source.write_bytes((pristine / "ref" / "act_quant.py").read_bytes())
            result = run_marker_subprocess(mutated / "tests" / "quant" / "test_act_quant.py", validation_arguments)
            self.assertEqual(result["cases"], [["ValidationTests.test_validation_precedes_computation", "pass"]])
            for control in ("raise", "tamper", "missing", "duplicate", "malformed", "exit"):
                with self.assertRaises(MarkerProtocolError, msg=control):
                    run_marker_subprocess(Path(__file__).resolve(), ["--control", control])
            retest = run_marker_subprocess(script, ["--suite", "directed", "--marker"])
            self.assertEqual(retest, baseline)
        for path, digest in digests.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest, str(path))


SUITE_CLASSES = {
    "all": (*CORE_CLASSES, ExecutionTests, SensitivityTests),
    "core": (*CORE_CLASSES, ExecutionTests),
    "directed": (DirectedCases,),
    "sensitivity": (SensitivityTests,),
}


def gather_suite(arguments):
    loader = unittest.TestLoader()
    suite = unittest.TestSuite(loader.loadTestsFromTestCase(case) for case in SUITE_CLASSES[arguments.suite])
    if arguments.cases:
        suite = unittest.TestSuite(
            test for test in iter_tests(suite)
            if any(selector in test.id() for selector in arguments.cases)
        )
    return suite


def main(argv=None):
    parser = argparse.ArgumentParser(description="P0014 activation quantization acceptance")
    parser.add_argument("--suite", choices=tuple(SUITE_CLASSES), default="all")
    parser.add_argument("--case", dest="cases", action="append", default=[])
    parser.add_argument("--marker", action="store_true")
    parser.add_argument("--control", choices=("raise", "tamper", "missing", "duplicate", "malformed", "exit"))
    arguments = parser.parse_args(argv)
    if arguments.control == "raise":
        raise RuntimeError("P0014 simulated runner crash")
    if arguments.control:
        payload = {"cases": [], "errors": 0, "failures": 0, "skipped": 0, "tests": 0}
        payload["digest"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        line = MARKER_PREFIX + json.dumps(payload)
        if arguments.control == "tamper":
            payload["tests"] = 1
            line = MARKER_PREFIX + json.dumps(payload)
        if arguments.control == "malformed":
            line = MARKER_PREFIX + "not-json"
        if arguments.control != "missing":
            print(line, flush=True)
        if arguments.control == "duplicate":
            print(line, flush=True)
        return 1 if arguments.control == "exit" else 0
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
    print("P0014 mandatory CPU activation quantization acceptance; standard library only", flush=True)
    print(f"Python: {sys.version}; platform: {platform.platform()}; machine: {platform.machine()}", flush=True)
    sys.exit(main())
