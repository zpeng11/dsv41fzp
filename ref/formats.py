"""P0006: CPU format oracle for the encoded formats in docs/MODEL.md §2.3.

Decode definitions: OCP MX v1.0 (September 2023), Tables 2, 5, and 7:
https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
BF16 RNE: PyTorch BFloat16.h, detail::round_to_nearest_even (2026-10-10 snapshot):
https://raw.githubusercontent.com/pytorch/pytorch/main/torch/headeronly/util/BFloat16.h

This reference chooses RNE, gradual FP32 underflow, and overflow to infinity.
NaN payloads and GPU cast/FTZ behavior are not part of its parity contract.
"""

import math
import struct
from collections.abc import Sequence


_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _unsigned(value: int, width: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("an integer bit pattern is required")
    if not 0 <= value < 1 << width:
        raise ValueError(f"bit pattern must fit in {width} bits")
    return value


def checked_shape(shape: Sequence[int]) -> tuple[int, ...]:
    dims = tuple(shape)
    if not dims:
        raise ValueError("a tensor must have at least one dimension")
    for dim in dims:
        if not isinstance(dim, int) or isinstance(dim, bool):
            raise TypeError("shape dimensions must be integers")
        if dim < 0:
            raise ValueError("shape dimensions must be nonnegative")
    return dims


def fp32_from_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", _unsigned(bits, 32)))[0]


def _round_shift(significand: int, shift: int) -> int:
    if shift <= 0:
        return significand << -shift
    whole, remainder = divmod(significand, 1 << shift)
    midpoint = 1 << (shift - 1)
    return whole + (remainder > midpoint or (remainder == midpoint and whole & 1))


def _binary_to_fp32_bits(sign: int, significand: int, exponent: int) -> int:
    if significand == 0:
        return sign
    top_exponent = significand.bit_length() - 1 + exponent
    if top_exponent < -126:
        return sign | _round_shift(significand, -149 - exponent)
    rounded = _round_shift(significand, significand.bit_length() - 24)
    if rounded == 1 << 24:
        rounded >>= 1
        top_exponent += 1
    if top_exponent > 127:
        return sign | 0x7F800000
    return sign | ((top_exponent + 127) << 23) | (rounded & 0x7FFFFF)


def fp32_bits(value: float | int) -> int:
    """Round a Python number to binary32, ties to even, with gradual underflow."""
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError("a Python float or integer is required")
    if isinstance(value, int):
        return _binary_to_fp32_bits(0x80000000 if value < 0 else 0, abs(value), 0)
    sign = 0x80000000 if math.copysign(1.0, value) < 0 else 0
    if math.isnan(value):
        return sign | 0x7FC00000
    if math.isinf(value):
        return sign | 0x7F800000
    numerator, denominator = abs(value).as_integer_ratio()
    return _binary_to_fp32_bits(sign, numerator, 1 - denominator.bit_length())


def _finite_parts(bits: int) -> tuple[int, int]:
    exponent = (bits >> 23) & 0xFF
    fraction = bits & 0x7FFFFF
    return (fraction, -149) if exponent == 0 else ((1 << 23) | fraction, exponent - 150)


def multiply_fp32(left: float, right: float) -> float:
    """One binary32 multiplication, RNE; no host FTZ, clamp, or BF16 conversion."""
    a, b = fp32_bits(left), fp32_bits(right)
    sign = (a ^ b) & 0x80000000
    mag_a, mag_b = a & 0x7FFFFFFF, b & 0x7FFFFFFF
    if mag_a > 0x7F800000 or mag_b > 0x7F800000:
        return fp32_from_bits(0x7FC00000)
    if mag_a == 0x7F800000 or mag_b == 0x7F800000:
        return fp32_from_bits(0x7FC00000 if mag_a == 0 or mag_b == 0 else sign | 0x7F800000)
    sig_a, exp_a = _finite_parts(a)
    sig_b, exp_b = _finite_parts(b)
    return fp32_from_bits(_binary_to_fp32_bits(sign, sig_a * sig_b, exp_a + exp_b))


def decode_e2m1(code: int) -> float:
    """Decode native E2M1, preserving both zeros; integer grids cannot retain -0."""
    code = _unsigned(code, 4)
    magnitude = _E2M1_MAGNITUDES[code & 7]
    return -magnitude if code & 8 else magnitude


def decode_e4m3fn(code: int) -> float:
    code = _unsigned(code, 8)
    exponent, fraction = (code >> 3) & 15, code & 7
    if exponent == 15 and fraction == 7:
        return math.copysign(math.nan, -1.0 if code & 0x80 else 1.0)
    magnitude = math.ldexp(fraction, -9) if exponent == 0 else math.ldexp(8 + fraction, exponent - 10)
    return -magnitude if code & 0x80 else magnitude


def decode_e8m0(code: int) -> float:
    """Decode unsigned E8M0 (E8M0FNU): 0..254 are powers of two; 255 is NaN."""
    code = _unsigned(code, 8)
    return math.nan if code == 255 else math.ldexp(1.0, code - 127)


def decode_bf16(bits: int) -> float:
    return fp32_from_bits(_unsigned(bits, 16) << 16)


def fp32_bits_to_bf16(bits: int) -> int:
    """Convert binary32 bits to BF16 using RNE; NaNs become canonical quiet NaNs."""
    bits = _unsigned(bits, 32)
    if (bits & 0x7FFFFFFF) > 0x7F800000:
        return ((bits >> 16) & 0x8000) | 0x7FC0
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF


def encode_bf16(value: float | int) -> int:
    """Round to FP32 first, then BF16; this is not a direct binary64 cast."""
    return fp32_bits_to_bf16(fp32_bits(value))


def e2m1_to_grid(code: int) -> int:
    return int(decode_e2m1(code) * 2)


def grid_to_e2m1(value: int) -> int:
    """Encode an integer grid point (twice E2M1); zero canonically maps to code 0."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("an integer grid point is required")
    for code, magnitude in enumerate(_E2M1_MAGNITUDES):
        if abs(value) == int(magnitude * 2):
            return code | (8 if value < 0 else 0)
    raise ValueError("not an E2M1 integer grid point")


def pack_e2m1(nibbles: Sequence[int], logical_shape: Sequence[int]) -> bytes:
    shape = checked_shape(logical_shape)
    if shape[-1] % 2:
        raise ValueError("the last logical dimension must be even")
    codes = tuple(_unsigned(code, 4) for code in nibbles)
    if len(codes) != math.prod(shape):
        raise ValueError("nibble count does not match logical shape")
    return bytes(codes[i] | (codes[i + 1] << 4) for i in range(0, len(codes), 2))


def unpack_e2m1(packed: Sequence[int], logical_shape: Sequence[int]) -> tuple[int, ...]:
    shape = checked_shape(logical_shape)
    if shape[-1] % 2:
        raise ValueError("the last logical dimension must be even")
    raw = tuple(_unsigned(code, 8) for code in packed)
    if len(raw) * 2 != math.prod(shape):
        raise ValueError("packed byte count does not match logical shape")
    return tuple(nibble for code in raw for nibble in (code & 15, code >> 4))
