"""P0012: single-output expert grouped dot under docs/COMPUTE.md.

This CPU contract is independent of the official GPU reduction and fusion rules.
All arithmetic steps round to binary32; BF16 conversion occurs only at the end.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

from ref.formats import (
    decode_e2m1,
    decode_e4m3fn,
    decode_e8m0,
    encode_bf16,
    fp32_bits,
    multiply_fp32,
)


@dataclass(frozen=True)
class BlockTrace:
    products: tuple[int, ...]
    partial_sums: tuple[int, ...]
    dot_bits: int
    after_a_bits: int
    block_bits: int
    acc_bits: int


@dataclass(frozen=True)
class ComputeTrace:
    blocks: tuple[BlockTrace, ...]
    fp32_bits: int
    bf16_bits: int
    result: float | int


def _from_bits(bits: int) -> float:
    """Construct an exact binary64 carrier without a DAZ-sensitive FP32 cast."""
    sign = bits & 0x80000000
    magnitude = bits & 0x7FFFFFFF
    if magnitude >= 0x7F800000:
        value = math.inf if magnitude == 0x7F800000 else math.nan
    else:
        exponent, significand = (bits >> 23) & 0xFF, bits & 0x7FFFFF
        if exponent:
            significand |= 1 << 23
        value = math.ldexp(significand, exponent - 150 if exponent else -149)
    return -value if sign else value


def _round_shift(significand: int, shift: int) -> int:
    if shift <= 0:
        return significand << -shift
    whole, remainder = divmod(significand, 1 << shift)
    midpoint = 1 << (shift - 1)
    return whole + (remainder > midpoint or (remainder == midpoint and whole & 1))


def _rounded_sum(sign: int, significand: int, exponent: int) -> int:
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


def _add_fp32(left: float, right: float) -> float:
    """One binary32 RNE addition using exact integer alignment, including specials."""
    a, b = fp32_bits(left), fp32_bits(right)
    mag_a, mag_b = a & 0x7FFFFFFF, b & 0x7FFFFFFF
    if mag_a > 0x7F800000 or mag_b > 0x7F800000:
        return _from_bits(0x7FC00000)
    if mag_a == 0x7F800000 or mag_b == 0x7F800000:
        if mag_a == mag_b and (a ^ b) & 0x80000000:
            return _from_bits(0x7FC00000)
        return _from_bits(a if mag_a == 0x7F800000 else b)
    exp_a, exp_b = (a >> 23) & 0xFF, (b >> 23) & 0xFF
    sig_a, sig_b = a & 0x7FFFFF, b & 0x7FFFFF
    if exp_a:
        sig_a |= 1 << 23
    if exp_b:
        sig_b |= 1 << 23
    exp_a = exp_a - 150 if exp_a else -149
    exp_b = exp_b - 150 if exp_b else -149
    exponent = min(exp_a, exp_b)
    aligned_a = sig_a << (exp_a - exponent)
    aligned_b = sig_b << (exp_b - exponent)
    total = (-aligned_a if a & 0x80000000 else aligned_a) + (
        -aligned_b if b & 0x80000000 else aligned_b
    )
    if total == 0:
        return _from_bits((a & b & 0x80000000) if mag_a == mag_b == 0 else 0)
    sign = 0x80000000 if total < 0 else 0
    return _from_bits(_rounded_sum(sign, abs(total), exponent))


def _scaled_fp32(value: float, scale: float) -> float:
    """Multiply by a validated E8M0 power of two with integer FP32 rounding."""
    bits = fp32_bits(value)
    if (bits & 0x7FFFFFFF) >= 0x7F800000:
        return _from_bits(bits)
    exponent, significand = (bits >> 23) & 0xFF, bits & 0x7FFFFF
    if exponent:
        significand |= 1 << 23
    if significand == 0:
        return _from_bits(bits)
    numerator, denominator = scale.as_integer_ratio()
    scale_exponent = numerator.bit_length() - denominator.bit_length()
    exponent = (exponent - 150 if exponent else -149) + scale_exponent
    return _from_bits(_rounded_sum(bits & 0x80000000, significand, exponent))


def _byte(code: int) -> int:
    if not isinstance(code, int) or isinstance(code, bool):
        raise TypeError("an integer bit pattern is required")
    if not 0 <= code <= 0xFF:
        raise ValueError("bit pattern must fit in 8 bits")
    return code


def _inputs(a_codes, w_packed, scale_a, scale_b, output):
    if output not in ("fp32", "bf16"):
        raise ValueError("output must be fp32 or bf16")
    sequences = (a_codes, w_packed, scale_a, scale_b)
    if not all(isinstance(values, Sequence) for values in sequences):
        raise TypeError("one-dimensional encoded sequences are required")
    k = len(a_codes)
    if k == 0 or k % 32:
        raise ValueError("K must be positive and divisible by 32")
    if len(w_packed) != k // 2:
        raise ValueError("packed weight length must be K/2")
    if len(scale_a) != k // 32 or len(scale_b) != k // 32:
        raise ValueError("each scale length must be K/32")
    a = tuple(_byte(code) for code in a_codes)
    w = tuple(_byte(code) for code in w_packed)
    sa = tuple(_byte(code) for code in scale_a)
    sb = tuple(_byte(code) for code in scale_b)
    if any(code in (0x7F, 0xFF) for code in a):
        raise ValueError("NaN activation codes are not accepted")
    if 0xFF in sa or 0xFF in sb:
        raise ValueError("NaN scale codes are not accepted")
    return a, w, sa, sb


def _evaluate(a_codes, w_packed, scale_a, scale_b, output, *, observe):
    a_codes, w_packed, scale_a, scale_b = _inputs(
        a_codes, w_packed, scale_a, scale_b, output
    )
    k = len(a_codes)
    acc = _from_bits(0)
    blocks = []
    for j in range(k // 32):
        dot = _from_bits(0)
        products, partial_sums = [], []
        for r in range(32):
            index = 32 * j + r
            weight = (w_packed[index // 2] >> (4 * (index % 2))) & 15
            product = multiply_fp32(decode_e4m3fn(a_codes[index]), decode_e2m1(weight))
            dot = _add_fp32(dot, product)
            if observe:
                products.append(fp32_bits(product))
                partial_sums.append(fp32_bits(dot))
        after_a = _scaled_fp32(dot, decode_e8m0(scale_a[j]))
        block = _scaled_fp32(after_a, decode_e8m0(scale_b[j]))
        acc = _add_fp32(acc, block)
        if observe:
            blocks.append(BlockTrace(
                tuple(products), tuple(partial_sums), fp32_bits(dot),
                fp32_bits(after_a), fp32_bits(block), fp32_bits(acc),
            ))
    bf16_bits = encode_bf16(acc)
    result = acc if output == "fp32" else bf16_bits
    if observe:
        return ComputeTrace(tuple(blocks), fp32_bits(acc), bf16_bits, result)
    return result


def expert_grouped_dot(
    a_codes: Sequence[int], w_packed: Sequence[int],
    scale_a: Sequence[int], scale_b: Sequence[int], *, output: str = "fp32",
) -> float | int:
    """Compute one output from encoded inputs with both K groups fixed at 32."""
    return _evaluate(a_codes, w_packed, scale_a, scale_b, output, observe=False)


def trace_expert_grouped_dot(
    a_codes: Sequence[int], w_packed: Sequence[int],
    scale_a: Sequence[int], scale_b: Sequence[int], *, output: str = "fp32",
) -> ComputeTrace:
    """Observe the same execution path without changing its arithmetic steps."""
    return _evaluate(a_codes, w_packed, scale_a, scale_b, output, observe=True)
