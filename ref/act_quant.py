"""P0014: single-row BF16 activation quantization under docs/ACTQUANT.md.

All numeric stages construct integer bit patterns, including binary32 RNE and
signed gradual underflow. This CPU contract does not imply GPU cast parity.
"""

from collections.abc import Sequence
from dataclasses import dataclass


_AMAX_FLOOR_BITS = 0x38D1B717
_FP8_MAX_INV_BITS = 0x3B124925
_FP8_MAX_BITS = 0x43E00000


@dataclass(frozen=True)
class GroupTrace:
    amax_bits: int
    bounded_amax_bits: int
    product_bits: int
    exponent: int
    scale_code: int
    quotient_bits: tuple[int, ...]
    clamped_bits: tuple[int, ...]
    a_codes: tuple[int, ...]


@dataclass(frozen=True)
class ActQuantTrace:
    groups: tuple[GroupTrace, ...]
    a_codes: tuple[int, ...]
    scale_a: tuple[int, ...]


def _round_shift(significand: int, shift: int) -> int:
    if shift <= 0:
        return significand << -shift
    whole, remainder = divmod(significand, 1 << shift)
    midpoint = 1 << (shift - 1)
    return whole + (remainder > midpoint or (remainder == midpoint and whole & 1))


def _fp32_bits(sign: int, significand: int, exponent: int) -> int:
    """Round an exact signed dyadic value to binary32 with gradual underflow."""
    if significand == 0:
        return sign
    top = significand.bit_length() - 1 + exponent
    if top < -126:
        return sign | _round_shift(significand, -149 - exponent)
    rounded = _round_shift(significand, significand.bit_length() - 24)
    if rounded == 1 << 24:
        rounded >>= 1
        top += 1
    if top > 127:
        return sign | 0x7F800000
    return sign | ((top + 127) << 23) | (rounded & 0x7FFFFF)


def _finite_parts(bits: int) -> tuple[int, int]:
    exponent = (bits >> 23) & 0xFF
    fraction = bits & 0x7FFFFF
    return (fraction, -149) if exponent == 0 else ((1 << 23) | fraction, exponent - 150)


def _group_scale(group: tuple[int, ...]) -> tuple[int, int, int, int, int]:
    """Retain the source's lower bound, binary32 product, and bit-field ceil."""
    amax_bits = max((code & 0x7FFF) << 16 for code in group)
    bounded_amax_bits = max(amax_bits, _AMAX_FLOOR_BITS)
    significand, exponent = _finite_parts(bounded_amax_bits)
    inv_significand, inv_exponent = _finite_parts(_FP8_MAX_INV_BITS)
    product_bits = _fp32_bits(0, significand * inv_significand, exponent + inv_exponent)
    scale_exponent = ((product_bits >> 23) & 0xFF) - 127 + bool(product_bits & 0x7FFFFF)
    return amax_bits, bounded_amax_bits, product_bits, scale_exponent, scale_exponent + 127


def _quotient_bits(code: int, scale_exponent: int) -> int:
    bits = code << 16
    significand, exponent = _finite_parts(bits)
    return _fp32_bits(bits & 0x80000000, significand, exponent - scale_exponent)


def _clamp_bits(bits: int) -> int:
    return (bits & 0x80000000) | min(bits & 0x7FFFFFFF, _FP8_MAX_BITS)


def _encode_e4m3fn(bits: int) -> int:
    """Encode a clamped binary32 bit pattern by E4M3FN RNE, preserving zero sign."""
    sign = (bits >> 24) & 0x80
    significand, exponent = _finite_parts(bits)
    if significand == 0:
        return sign
    top = significand.bit_length() - 1 + exponent
    if top < -6:
        return sign | _round_shift(significand, -9 - exponent)
    rounded = _round_shift(significand, significand.bit_length() - 4)
    if rounded == 16:
        rounded >>= 1
        top += 1
    return sign | ((top + 7) << 3) | (rounded & 7)


def _bf16(code: int) -> int:
    if not isinstance(code, int) or isinstance(code, bool):
        raise TypeError("an integer BF16 bit pattern is required")
    if not 0 <= code <= 0xFFFF:
        raise ValueError("bit pattern must fit in 16 bits")
    if (code & 0x7F80) == 0x7F80:
        raise ValueError("nonfinite BF16 activations are not accepted")
    return code


def _inputs(bf16_bits: Sequence[int]) -> tuple[int, ...]:
    if not isinstance(bf16_bits, Sequence):
        raise TypeError("a one-dimensional BF16 encoded sequence is required")
    if isinstance(bf16_bits, memoryview) and bf16_bits.ndim != 1:
        raise TypeError("a one-dimensional BF16 encoded sequence is required")
    k = len(bf16_bits)
    if k == 0 or k % 32:
        raise ValueError("K must be positive and divisible by 32")
    return tuple(_bf16(code) for code in bf16_bits)


def _evaluate(bf16_bits: Sequence[int], *, observe: bool):
    inputs = _inputs(bf16_bits)
    a_codes, scale_a, groups = [], [], []
    for offset in range(0, len(inputs), 32):
        group = inputs[offset:offset + 32]
        amax, bounded, product, exponent, scale_code = _group_scale(group)
        scale_a.append(scale_code)
        quotients, clamped, codes = [], [], []
        for code in group:
            quotient = _quotient_bits(code, exponent)
            before_encode = _clamp_bits(quotient)
            encoded = _encode_e4m3fn(before_encode)
            codes.append(encoded)
            if observe:
                quotients.append(quotient)
                clamped.append(before_encode)
        a_codes.extend(codes)
        if observe:
            groups.append(GroupTrace(
                amax, bounded, product, exponent, scale_code,
                tuple(quotients), tuple(clamped), tuple(codes),
            ))
    result = tuple(a_codes), tuple(scale_a)
    if observe:
        return ActQuantTrace(tuple(groups), *result)
    return result


def act_quant(bf16_bits: Sequence[int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Quantize one BF16 encoded row to E4M3FN and one E8M0 scale per 32 elements."""
    return _evaluate(bf16_bits, observe=False)


def trace_act_quant(bf16_bits: Sequence[int]) -> ActQuantTrace:
    """Observe the same execution path, including otherwise invisible FP32 stages."""
    return _evaluate(bf16_bits, observe=True)
