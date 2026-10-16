"""P0027: single-output FP8 block-scaled dot under docs/COMPUTE.md §8.

This is a CPU specification, not a claim about official GPU reduction or fusion.
All stages use integer bit patterns; only the final FP32 result needs a carrier.
"""

from collections.abc import Sequence

from ref.compute import BlockTrace, ComputeTrace, _byte, _from_bits
from ref.formats import _binary_to_fp32_bits, _finite_parts, fp32_bits_to_bf16


def _inputs(a_codes, w_codes, scale_a, scale_b, output):
    if output not in ("fp32", "bf16"):
        raise ValueError("output must be fp32 or bf16")
    sequences = (a_codes, w_codes, scale_a, scale_b)
    if not all(isinstance(values, Sequence) for values in sequences):
        raise TypeError("one-dimensional encoded sequences are required")
    if any(isinstance(values, memoryview) and values.ndim != 1 for values in sequences):
        raise TypeError("one-dimensional encoded sequences are required")
    k = len(a_codes)
    if k == 0 or k % 32:
        raise ValueError("K must be positive and divisible by 32")
    if len(w_codes) != k:
        raise ValueError("weight length must be K")
    if len(scale_a) != k // 32 or len(scale_b) != k // 32:
        raise ValueError("each scale length must be K/32")
    a = tuple(_byte(code) for code in a_codes)
    w = tuple(_byte(code) for code in w_codes)
    sa = tuple(_byte(code) for code in scale_a)
    sb = tuple(_byte(code) for code in scale_b)
    if (len(a), len(w), len(sa), len(sb)) != (k, k, k // 32, k // 32):
        raise ValueError("input lengths changed while taking snapshots")
    if any(code in (0x7F, 0xFF) for code in a):
        raise ValueError("NaN activation codes are not accepted")
    if any(code in (0x7F, 0xFF) for code in w):
        raise ValueError("NaN weight codes are not accepted")
    if 0xFF in sa or 0xFF in sb:
        raise ValueError("NaN scale codes are not accepted")
    return a, w, sa, sb


def _product_bits(a_code: int, w_code: int) -> int:
    """Multiply two validated finite E4M3FN codes without a float conversion."""
    exponent_a, fraction_a = (a_code >> 3) & 15, a_code & 7
    exponent_w, fraction_w = (w_code >> 3) & 15, w_code & 7
    significand_a = 8 + fraction_a if exponent_a else fraction_a
    significand_w = 8 + fraction_w if exponent_w else fraction_w
    exponent_a = exponent_a - 10 if exponent_a else -9
    exponent_w = exponent_w - 10 if exponent_w else -9
    return _binary_to_fp32_bits(
        ((a_code ^ w_code) & 0x80) << 24,
        significand_a * significand_w, exponent_a + exponent_w,
    )


def _add_bits(left: int, right: int) -> int:
    """Round one aligned integer sum, including the contract's zero and Inf rules."""
    mag_left, mag_right = left & 0x7FFFFFFF, right & 0x7FFFFFFF
    if mag_left > 0x7F800000 or mag_right > 0x7F800000:
        return 0x7FC00000
    if mag_left == 0x7F800000 or mag_right == 0x7F800000:
        if mag_left == mag_right and (left ^ right) & 0x80000000:
            return 0x7FC00000
        return left if mag_left == 0x7F800000 else right
    sig_left, exp_left = _finite_parts(left)
    sig_right, exp_right = _finite_parts(right)
    exponent = min(exp_left, exp_right)
    aligned_left = sig_left << (exp_left - exponent)
    aligned_right = sig_right << (exp_right - exponent)
    total = (-aligned_left if left & 0x80000000 else aligned_left) + (
        -aligned_right if right & 0x80000000 else aligned_right
    )
    if total == 0:
        return (left & right & 0x80000000) if mag_left == mag_right == 0 else 0
    return _binary_to_fp32_bits(0x80000000 if total < 0 else 0, abs(total), exponent)


def _scale_bits(bits: int, scale_code: int) -> int:
    """Apply one validated positive E8M0 scale with a distinct FP32 rounding."""
    if (bits & 0x7FFFFFFF) >= 0x7F800000:
        return bits
    significand, exponent = _finite_parts(bits)
    return _binary_to_fp32_bits(
        bits & 0x80000000, significand, exponent + scale_code - 127
    )


def _evaluate(a_codes, w_codes, scale_a, scale_b, output, *, observe):
    a_codes, w_codes, scale_a, scale_b = _inputs(
        a_codes, w_codes, scale_a, scale_b, output
    )
    acc = 0
    blocks = []
    for j in range(len(a_codes) // 32):
        dot = 0
        products, partial_sums = [], []
        for r in range(32):
            index = 32 * j + r
            product = _product_bits(a_codes[index], w_codes[index])
            dot = _add_bits(dot, product)
            if observe:
                products.append(product)
                partial_sums.append(dot)
        after_a = _scale_bits(dot, scale_a[j])
        block = _scale_bits(after_a, scale_b[j])
        acc = _add_bits(acc, block)
        if observe:
            blocks.append(BlockTrace(
                tuple(products), tuple(partial_sums), dot, after_a, block, acc,
            ))
    bf16_bits = fp32_bits_to_bf16(acc)
    result = _from_bits(acc) if output == "fp32" else bf16_bits
    if observe:
        return ComputeTrace(tuple(blocks), acc, bf16_bits, result)
    return result


def fp8_grouped_dot(
    a_codes: Sequence[int], w_codes: Sequence[int],
    scale_a: Sequence[int], scale_b: Sequence[int], *, output: str = "fp32",
) -> float | int:
    """Compute one FP8×FP8 output with two separately applied K-block scales."""
    return _evaluate(a_codes, w_codes, scale_a, scale_b, output, observe=False)


def trace_fp8_grouped_dot(
    a_codes: Sequence[int], w_codes: Sequence[int],
    scale_a: Sequence[int], scale_b: Sequence[int], *, output: str = "fp32",
) -> ComputeTrace:
    """Observe the same execution core, including every rounded stage."""
    return _evaluate(a_codes, w_codes, scale_a, scale_b, output, observe=True)
