#!/usr/bin/env python3
"""P0006 mandatory CPU format acceptance: python3 tests/quant/test_formats.py.

Expected values use independent rational bit-field definitions, nearest-neighbor
rounding, and fixed official anchors, not the oracle's LUT or rounding helpers.
Decode basis: OCP MX v1.0 Tables 2/5/7 and FP8 Formats for Deep Learning,
arXiv:2209.05433v2 Table 1; BF16 RNE basis: PyTorch BFloat16.h,
detail::round_to_nearest_even (2026-10-10 snapshot). No source is fetched at run time.
"""

import ast
import bisect
import math
import platform
import random
import struct
import sys
import unittest
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ref import formats as f
from ref.dequant import (
    Tensor,
    dequantize_engram,
    dequantize_expert,
    dequantize_fp8_weights,
    dequantize_index_k,
    dequantize_main_kv,
    dequantize_swa,
)


def power2(exponent):
    return Fraction(2 ** exponent) if exponent >= 0 else Fraction(1, 2 ** -exponent)


def expected_e2m1(code):
    sign, exponent, fraction = code >> 3, (code >> 1) & 3, code & 1
    magnitude = Fraction(fraction, 2) if exponent == 0 else (1 + Fraction(fraction, 2)) * power2(exponent - 1)
    return -magnitude if sign else magnitude


def expected_e4m3fn(code):
    sign, exponent, fraction = code >> 7, (code >> 3) & 15, code & 7
    if exponent == 15 and fraction == 7:
        return None
    magnitude = Fraction(fraction, 8) * power2(-6) if exponent == 0 else (1 + Fraction(fraction, 8)) * power2(exponent - 7)
    return -magnitude if sign else magnitude


def expected_e8m0(code):
    return None if code == 255 else power2(code - 127)


def expected_binary(bits, fraction_width, exponent_width=8, bias=127):
    fraction_mask = (1 << fraction_width) - 1
    fraction = bits & fraction_mask
    exponent = (bits >> fraction_width) & ((1 << exponent_width) - 1)
    negative = bits >> (fraction_width + exponent_width)
    if exponent == (1 << exponent_width) - 1:
        return None
    magnitude = (
        Fraction(fraction, 1 << fraction_width) * power2(1 - bias)
        if exponent == 0 else
        (1 + Fraction(fraction, 1 << fraction_width)) * power2(exponent - bias)
    )
    return -magnitude if negative else magnitude


def host_fp32_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def host_fp32_value(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def nearest_fp32_bits(exact, negative_zero=False):
    """Binary search all nonnegative finite FP32 encodings using exact distances."""
    sign = 0x80000000 if exact < 0 or (exact == 0 and negative_zero) else 0
    magnitude = abs(exact)
    maximum = expected_binary(0x7F7FFFFF, 23)
    overflow_midpoint = maximum + power2(103)
    if magnitude >= overflow_midpoint:
        return sign | 0x7F800000
    if magnitude > maximum:
        return sign | 0x7F7FFFFF
    low, high = 0, 0x7F7FFFFF
    while low < high:
        middle = (low + high) // 2
        if expected_binary(middle, 23) < magnitude:
            low = middle + 1
        else:
            high = middle
    upper = low
    if upper == 0 or expected_binary(upper, 23) == magnitude:
        return sign | upper
    lower = upper - 1
    lower_distance = magnitude - expected_binary(lower, 23)
    upper_distance = expected_binary(upper, 23) - magnitude
    chosen = lower if lower_distance < upper_distance else upper
    if lower_distance == upper_distance:
        chosen = lower if lower % 2 == 0 else upper
    return sign | chosen


BF16_POSITIVE = tuple(expected_binary(code, 7) for code in range(0x7F80))


def nearest_bf16_bits(bits):
    sign = (bits >> 16) & 0x8000
    exponent, fraction = (bits >> 23) & 255, bits & 0x7FFFFF
    if exponent == 255:
        return sign | (0x7FC0 if fraction else 0x7F80)
    magnitude = abs(expected_binary(bits, 23))
    if magnitude >= BF16_POSITIVE[-1] + power2(119):
        return sign | 0x7F80
    index = bisect.bisect_left(BF16_POSITIVE, magnitude)
    if index == len(BF16_POSITIVE):
        return sign | 0x7F7F
    if index == 0 or BF16_POSITIVE[index] == magnitude:
        return sign | index
    lower_distance = magnitude - BF16_POSITIVE[index - 1]
    upper_distance = BF16_POSITIVE[index] - magnitude
    lower = index - 1
    chosen = lower if lower_distance < upper_distance else index
    if lower_distance == upper_distance:
        chosen = lower if lower % 2 == 0 else index
    return sign | chosen


def independent_pack(codes):
    return bytes(sum(code * 16 ** offset for offset, code in enumerate(codes[i:i + 2])) for i in range(0, len(codes), 2))


class CodecTests(unittest.TestCase):
    def assert_finite_bits(self, actual, expected, *, negative_zero=False):
        self.assertEqual(host_fp32_bits(actual), nearest_fp32_bits(expected, negative_zero))

    def test_e2m1_all_codes_and_grid(self):
        for code in range(16):
            with self.subTest(code=code):
                expected = expected_e2m1(code)
                self.assert_finite_bits(
                    f.decode_e2m1(code), expected, negative_zero=code == 8
                )
                self.assertEqual(f.e2m1_to_grid(code), int(expected * 2))
                self.assertEqual(f.grid_to_e2m1(int(expected * 2)), 0 if code == 8 else code)
        for invalid in (-13, -11, -10, -9, -7, -5, 5, 7, 9, 10, 11, 13, 10 ** 1000):
            with self.assertRaises(ValueError):
                f.grid_to_e2m1(invalid)
        for invalid in (0.0, 1.5, "2", True):
            with self.assertRaises(TypeError):
                f.grid_to_e2m1(invalid)

    def test_e4m3fn_all_bytes(self):
        for code in range(256):
            with self.subTest(code=code):
                expected = expected_e4m3fn(code)
                if expected is None:
                    self.assertTrue(math.isnan(f.decode_e4m3fn(code)))
                else:
                    self.assert_finite_bits(f.decode_e4m3fn(code), expected, negative_zero=code == 128)
        self.assertEqual(f.decode_e4m3fn(0x7E), 448.0)
        self.assertEqual(f.decode_e4m3fn(1), 2 ** -9)
        self.assertEqual(f.decode_e4m3fn(0x38), 1.0)

    def test_e8m0_all_bytes(self):
        for code in range(256):
            with self.subTest(code=code):
                expected = expected_e8m0(code)
                if expected is None:
                    self.assertTrue(math.isnan(f.decode_e8m0(code)))
                else:
                    self.assert_finite_bits(f.decode_e8m0(code), expected)
        self.assertEqual(host_fp32_bits(f.decode_e8m0(0)), 0x00400000)
        self.assertEqual(f.decode_e8m0(127), 1.0)
        self.assertEqual(f.decode_e8m0(254), 2 ** 127)

    def test_bf16_all_bit_patterns(self):
        for bits in range(65536):
            actual = f.decode_bf16(bits)
            exponent, fraction = (bits >> 7) & 255, bits & 127
            if exponent == 255:
                if fraction:
                    self.assertTrue(math.isnan(actual), hex(bits))
                else:
                    self.assertEqual(actual, -math.inf if bits & 0x8000 else math.inf)
            else:
                exact = expected_binary(bits, 7)
                self.assertEqual(actual, float(exact), hex(bits))
                self.assertEqual(host_fp32_bits(actual), bits << 16, hex(bits))
                self.assertEqual(f.encode_bf16(actual), bits, hex(bits))

    def test_bf16_every_finite_midpoint_and_neighbors(self):
        for code in range(0x7F7F):
            exact = (BF16_POSITIVE[code] + BF16_POSITIVE[code + 1]) / 2
            midpoint_bits = host_fp32_bits(float(exact))
            tie = code if code % 2 == 0 else code + 1
            for sign in (0, 0x80000000):
                bf_sign = sign >> 16
                for bits, expected in ((midpoint_bits - 1, code), (midpoint_bits, tie), (midpoint_bits + 1, code + 1)):
                    signed = bits | sign
                    self.assertEqual(f.fp32_bits_to_bf16(signed), expected | bf_sign, hex(signed))
                    self.assertEqual(f.encode_bf16(host_fp32_value(signed)), expected | bf_sign, hex(signed))

    def test_bf16_rounds_through_fp32_not_direct_binary64(self):
        midpoint = 1.0 + 2 ** -8
        above = math.nextafter(midpoint, math.inf)
        self.assertGreater(Fraction(above), (BF16_POSITIVE[0x3F80] + BF16_POSITIVE[0x3F81]) / 2)
        self.assertEqual(f.encode_bf16(above), 0x3F80)
        self.assertEqual(f.encode_bf16(-above), 0xBF80)
        odd_midpoint = 1.0 + 3 * 2 ** -8
        below = math.nextafter(odd_midpoint, -math.inf)
        self.assertLess(Fraction(below), (BF16_POSITIVE[0x3F81] + BF16_POSITIVE[0x3F82]) / 2)
        self.assertEqual(f.encode_bf16(below), 0x3F82)
        self.assertEqual(f.encode_bf16(-below), 0xBF82)

    def test_bf16_overflow_and_specials(self):
        for sign in (0, 0x80000000):
            for bits in (0, 1, 0x7F7F7FFF, 0x7F7F8000, 0x7F7F8001, 0x7F7FFFFF, 0x7F800000):
                signed = bits | sign
                self.assertEqual(f.fp32_bits_to_bf16(signed), nearest_bf16_bits(signed), hex(signed))
            for bits in (0x7F800001, 0x7FA00000, 0x7FC00000, 0x7FFFFFFF):
                bf = f.fp32_bits_to_bf16(bits | sign)
                self.assertEqual(bf & 0x7F80, 0x7F80)
                self.assertNotEqual(bf & 127, 0)
                self.assertTrue(math.isnan(f.decode_bf16(bf)))
        self.assertEqual(f.encode_bf16(-0.0), 0x8000)
        self.assertEqual(f.encode_bf16(math.inf), 0x7F80)
        self.assertEqual(f.encode_bf16(-math.inf), 0xFF80)
        self.assertTrue(math.isnan(f.decode_bf16(f.encode_bf16(math.nan))))

    def test_packing_all_bytes_and_row_layout(self):
        for raw in range(256):
            low, high = raw % 16, raw // 16
            self.assertEqual(f.unpack_e2m1(bytes([raw]), (1, 2)), (low, high))
            self.assertEqual(f.pack_e2m1((low, high), (1, 2)), bytes([raw]))
        codes = (1, 14, 2, 13, 3, 12, 4, 11, 5, 10, 6, 9)
        packed = bytes((0xE1, 0xD2, 0xC3, 0xB4, 0xA5, 0x96))
        self.assertEqual(f.pack_e2m1(codes, (2, 6)), packed)
        self.assertEqual(f.unpack_e2m1(packed, (2, 6)), codes)
        self.assertEqual(f.pack_e2m1((), (2, 0)), b"")
        for shape in ((2, 3), (1, 5)):
            with self.assertRaises(ValueError):
                f.pack_e2m1((0,) * math.prod(shape), shape)
            with self.assertRaises(ValueError):
                f.unpack_e2m1(b"\x00" * (math.prod(shape) // 2), shape)
        for action in (
            lambda: f.pack_e2m1((0, 16), (1, 2)),
            lambda: f.pack_e2m1((0, -1), (1, 2)),
            lambda: f.pack_e2m1((0, 1), (1, 4)),
            lambda: f.unpack_e2m1((256,), (1, 2)),
            lambda: f.unpack_e2m1((-1,), (1, 2)),
            lambda: f.unpack_e2m1(b"\x00", (2, 2)),
        ):
            with self.assertRaises(ValueError):
                action()

    def test_official_fp4_constants_without_gpu_import(self):
        source = ROOT / "third_party/reference/inference/convert.py"
        tree = ast.parse(source.read_text())
        assignment = next(
            node for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "FP4_TABLE"
                for target in node.targets
            )
        )
        table = ast.literal_eval(assignment.value.args[0])
        self.assertEqual(len(table), 16)
        for code, official in enumerate(table):
            self.assertEqual(f.decode_e2m1(code), official)
            if code != 8:
                self.assertEqual(host_fp32_bits(f.decode_e2m1(code)), host_fp32_bits(official))
        self.assertEqual(host_fp32_bits(table[8]), 0)
        self.assertEqual(host_fp32_bits(f.decode_e2m1(8)), 0x80000000)

    def test_invalid_bit_patterns(self):
        decoders = (
            (f.decode_e2m1, 4), (f.decode_e4m3fn, 8), (f.decode_e8m0, 8),
            (f.decode_bf16, 16), (f.fp32_from_bits, 32), (f.fp32_bits_to_bf16, 32),
        )
        for decoder, width in decoders:
            for invalid in (-1, 1 << width):
                with self.assertRaises(ValueError):
                    decoder(invalid)
            for invalid in (0.0, "0", True):
                with self.assertRaises(TypeError):
                    decoder(invalid)
        for shape in ((), (-1, 2)):
            with self.assertRaises(ValueError):
                f.checked_shape(shape)
        for shape in ((True, 2), (1.0, 2)):
            with self.assertRaises(TypeError):
                f.checked_shape(shape)


class FP32Tests(unittest.TestCase):
    def test_fp32_rounding_against_exact_search(self):
        cases = [
            Fraction(0), power2(-150), power2(-149), power2(-126),
            power2(-126) - power2(-150), power2(128) - power2(103),
            power2(128) - power2(104), Fraction(1) + power2(-24),
            Fraction(1) + 3 * power2(-24),
        ]
        for exact in cases:
            for sign in (1, -1):
                value = float(sign * exact)
                self.assertEqual(f.fp32_bits(value), nearest_fp32_bits(sign * exact, math.copysign(1.0, value) < 0))
        self.assertEqual(f.fp32_bits(10 ** 1000), 0x7F800000)
        self.assertEqual(f.fp32_bits(-(10 ** 1000)), 0xFF800000)
        self.assertEqual(f.fp32_bits(-0.0), 0x80000000)
        for value in (True, "1", Fraction(1)):
            with self.assertRaises(TypeError):
                f.fp32_bits(value)

    def test_product_boundaries_and_reproducible_random(self):
        pairs = [
            (1, 0x3F000000), (3, 0x3F000000),
            (0x007FFFFF, 0x3F800001), (0x00800000, 0x3F000000),
            (0x7F7FFFFF, 0x3F800001), (0x3FFFFFFF, 0x3F800001),
            (0x80000000, 0x3F800000),
        ]
        rng = random.Random(6)
        for _ in range(512):
            pairs.append((
                rng.randrange(0x7F800000) | (rng.randrange(2) << 31),
                rng.randrange(0x7F800000) | (rng.randrange(2) << 31),
            ))
        for left, right in pairs:
            exact = expected_binary(left, 23) * expected_binary(right, 23)
            negative = bool((left ^ right) & 0x80000000)
            actual = f.multiply_fp32(host_fp32_value(left), host_fp32_value(right))
            self.assertEqual(host_fp32_bits(actual), nearest_fp32_bits(exact, negative), (hex(left), hex(right)))

    def test_product_specials(self):
        for left, right in ((math.nan, 1.0), (1.0, math.nan), (math.inf, 0.0), (0.0, -math.inf)):
            self.assertTrue(math.isnan(f.multiply_fp32(left, right)))
        self.assertEqual(f.multiply_fp32(math.inf, -1.0), -math.inf)
        self.assertEqual(f.multiply_fp32(-math.inf, -1.0), math.inf)
        self.assertEqual(host_fp32_bits(f.multiply_fp32(-0.0, 2.0)), 0x80000000)
        self.assertEqual(f.multiply_fp32(1 + 2 ** -24, 1.0), 1.0)


class DequantTests(unittest.TestCase):
    def assert_dequant_values(
        self, operation, data, scales, logical_shape,
        expected_data, expected_scale, scale_index, zero_sign,
    ):
        result = operation(data, scales)
        bf16 = operation(data, scales, output="bf16")
        self.assertEqual(result.shape, logical_shape)
        self.assertEqual(bf16.shape, logical_shape)
        for i, (value, bf) in enumerate(zip(result.data, bf16.data)):
            exact = expected_data(i) * expected_scale(scale_index(i))
            expected_bits = nearest_fp32_bits(exact, zero_sign(i))
            self.assertEqual(host_fp32_bits(value), expected_bits, i)
            self.assertEqual(bf, nearest_bf16_bits(expected_bits), i)

    def test_grouped_paths_with_distinct_scales_and_batches(self):
        specs = (
            (dequantize_expert, (3, 96), 32, True, False),
            (dequantize_engram, (3, 96), 32, False, False),
            (dequantize_swa, (2, 3, 96), 32, False, False),
            (dequantize_main_kv, (2, 3, 48), 16, True, True),
            (dequantize_index_k, (2, 3, 96), 32, True, False),
        )
        for operation, logical_shape, group, packed, fp8_scale in specs:
            with self.subTest(path=operation.__name__):
                n = math.prod(logical_shape)
                if packed:
                    codes = tuple((i * 7 + i // logical_shape[-1] * 3) % 16 for i in range(n))
                    data = Tensor(independent_pack(codes), logical_shape[:-1] + (logical_shape[-1] // 2,))
                    expected_data = lambda i: expected_e2m1(codes[i])
                    zero_sign = lambda i: codes[i] == 8
                else:
                    codes = tuple((i * 13 + i // logical_shape[-1] * 5) % 127 | (0x80 if i % 3 else 0) for i in range(n))
                    data = Tensor(codes, logical_shape)
                    expected_data = lambda i: expected_e4m3fn(codes[i])
                    zero_sign = lambda i: codes[i] == 128
                scale_shape = logical_shape[:-1] + (logical_shape[-1] // group,)
                scale_codes = tuple(0x31 + i % 19 if fp8_scale else 120 + i % 15 for i in range(math.prod(scale_shape)))
                scales = Tensor(scale_codes, scale_shape)
                scale_values = tuple(expected_e4m3fn(code) if fp8_scale else expected_e8m0(code) for code in scale_codes)
                self.assert_dequant_values(
                    operation, data, scales, logical_shape, expected_data,
                    lambda j: scale_values[j], lambda i: i // group, zero_sign,
                )

    def test_2d_blocks_with_both_tails(self):
        rows, cols = 35, 67
        codes = tuple((r * 17 + c * 3) % 127 | (128 if (r + c) % 2 else 0) for r in range(rows) for c in range(cols))
        data = Tensor(codes, (rows, cols))
        scale_codes = (123, 124, 125, 126, 127, 128)
        scales = Tensor(scale_codes, (2, 3))
        self.assert_dequant_values(
            dequantize_fp8_weights, data, scales, (rows, cols),
            lambda i: expected_e4m3fn(codes[i]),
            lambda j: expected_e8m0(scale_codes[j]),
            lambda i: ((i // cols) // 32) * 3 + (i % cols) // 32,
            lambda i: codes[i] == 128,
        )
        with self.assertRaises(ValueError):
            dequantize_fp8_weights(data, Tensor(scale_codes, (3, 2)))
        with self.assertRaises(ValueError):
            dequantize_fp8_weights(data, Tensor((127, 127), (1, 2)))

    def test_2d_complete_blocks_extreme_scales(self):
        rows, cols = 96, 64
        anchors = (1, 0x7E, 0xFE, 0x80, 0x38, 0x78)
        codes = tuple(anchors[(r // 32) * 2 + c // 32] for r in range(rows) for c in range(cols))
        scale_codes = (0, 254, 254, 0, 1, 250)
        data = Tensor(codes, (rows, cols))
        scales = Tensor(scale_codes, (3, 2))
        self.assert_dequant_values(
            dequantize_fp8_weights, data, scales, (rows, cols),
            lambda i: expected_e4m3fn(codes[i]),
            lambda j: expected_e8m0(scale_codes[j]),
            lambda i: ((i // cols) // 32) * 2 + (i % cols) // 32,
            lambda i: codes[i] == 128,
        )

    def test_all_finite_data_scale_pairs(self):
        specs = (
            (dequantize_index_k, True, False, 32),
            (dequantize_swa, False, False, 32),
            (dequantize_main_kv, True, True, 16),
        )
        for operation, packed, fp8_scale, group in specs:
            with self.subTest(path=operation.__name__):
                codes = tuple(range(16)) if packed else tuple(
                    code for code in range(256) if code not in (127, 255)
                )
                row_codes = codes + (0,) * (-len(codes) % group)
                cols = len(row_codes)
                scale_codes = tuple(range(1, 127)) if fp8_scale else tuple(range(255))
                rows = len(scale_codes)
                logical_codes = row_codes * rows
                data = Tensor(
                    independent_pack(logical_codes) if packed else logical_codes,
                    (rows, cols // 2 if packed else cols),
                )
                scales = Tensor(
                    tuple(code for code in scale_codes for _ in range(cols // group)),
                    (rows, cols // group),
                )
                result = operation(data, scales)
                bf16 = operation(data, scales, output="bf16")
                for row, scale in enumerate(scale_codes):
                    exact_scale = expected_e4m3fn(scale) if fp8_scale else expected_e8m0(scale)
                    for col, code in enumerate(row_codes):
                        exact_data = expected_e2m1(code) if packed else expected_e4m3fn(code)
                        expected_bits = nearest_fp32_bits(
                            exact_data * exact_scale, code == (8 if packed else 128)
                        )
                        index = row * cols + col
                        self.assertEqual(host_fp32_bits(result.data[index]), expected_bits, (code, scale))
                        self.assertEqual(bf16.data[index], nearest_bf16_bits(expected_bits), (code, scale))

    def test_reject_nan_nonpositive_scales_and_invalid_codes(self):
        specs = (
            (dequantize_expert, True, False, 32),
            (dequantize_fp8_weights, False, False, 32),
            (dequantize_engram, False, False, 32),
            (dequantize_swa, False, False, 32),
            (dequantize_main_kv, True, True, 16),
            (dequantize_index_k, True, False, 32),
        )
        for operation, packed, fp8_scale, group in specs:
            data_shape = (1, group // 2 if packed else group)
            data = Tensor((0x11 if packed else 0x38,) * math.prod(data_shape), data_shape)
            good_scale = Tensor((0x38 if fp8_scale else 127,), (1, 1))
            for bad_scale in ((0, 128, 184, 127, 255) if fp8_scale else (255,)):
                with self.assertRaises(ValueError):
                    operation(data, Tensor((bad_scale,), (1, 1)))
            for bad_code in ((-1, 256) if packed else (-1, 256, 127, 255)):
                bad = Tensor((bad_code,) + data.data[1:], data_shape)
                with self.assertRaises(ValueError):
                    operation(bad, good_scale)
            for invalid_mode in ("float32", "BF16", None):
                with self.assertRaises(ValueError):
                    operation(data, good_scale, output=invalid_mode)
            with self.assertRaises(ValueError):
                operation(data, Tensor((good_scale.data[0],), (1,)))

    def test_invalid_shapes_and_empty_tensors(self):
        with self.assertRaises(ValueError):
            Tensor((1, 2), (2, 2))
        for operation in (dequantize_expert, dequantize_engram, dequantize_fp8_weights):
            for shape in ((32,), (1, 1, 32)):
                with self.assertRaises(ValueError):
                    operation(Tensor((1,) * 32, shape), Tensor((127,), (1,)))
        specs = (
            (dequantize_expert, True), (dequantize_engram, False),
            (dequantize_swa, False), (dequantize_main_kv, True),
            (dequantize_index_k, True),
        )
        for operation, packed in specs:
            with self.assertRaises(ValueError):
                operation(Tensor((1,) * 15, (1, 15)), Tensor((127,), (1, 1)))
            self.assertEqual(operation(Tensor((), (2, 0)), Tensor((), (2, 0))).shape, (2, 0))
            expected_cols = 64 if packed else 32
            group = 16 if operation == dequantize_main_kv else 32
            empty = operation(
                Tensor((), (0, 32)), Tensor((), (0, expected_cols // group))
            )
            self.assertEqual(empty.shape, (0, expected_cols))
        self.assertEqual(dequantize_fp8_weights(Tensor((), (0, 35)), Tensor((), (0, 2))).shape, (0, 35))
        self.assertEqual(dequantize_fp8_weights(Tensor((), (35, 0)), Tensor((), (2, 0))).shape, (35, 0))


if __name__ == "__main__":
    print("P0006 mandatory CPU acceptance; standard library only", flush=True)
    print(f"Python: {sys.version}; platform: {platform.platform()}; machine: {platform.machine()}", flush=True)
    unittest.main(verbosity=2)
