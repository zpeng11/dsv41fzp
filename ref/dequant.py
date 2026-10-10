"""P0006: given-scale CPU dequantization; layouts follow docs/MODEL.md §2.1–2.3.

FP32 results and optional BF16 bit patterns are separate outputs. No scale is
inferred, and an optional BF16 result does not assert a GEMM execution order.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

from ref.formats import (
    checked_shape,
    decode_e2m1,
    decode_e4m3fn,
    decode_e8m0,
    encode_bf16,
    multiply_fp32,
    unpack_e2m1,
)


T = TypeVar("T", int, float)
Output = Literal["fp32", "bf16"]


@dataclass(frozen=True)
class Tensor(Generic[T]):
    """Flat row-major values with an explicit, exact shape; no implicit broadcasting."""

    data: tuple[T, ...]
    shape: tuple[int, ...]

    def __post_init__(self):
        object.__setattr__(self, "data", tuple(self.data))
        object.__setattr__(self, "shape", checked_shape(self.shape))
        if len(self.data) != math.prod(self.shape):
            raise ValueError("data count does not match tensor shape")


def _output_mode(output: Output) -> None:
    if output not in ("fp32", "bf16"):
        raise ValueError("output must be 'fp32' or 'bf16'")


def _decoded(
    data: tuple[int, ...],
    decoder: Callable[[int], float],
    *,
    scale: bool = False,
) -> tuple[float, ...]:
    values = tuple(decoder(code) for code in data)
    if any(math.isnan(value) for value in values):
        raise ValueError("model dequantization does not accept NaN data or scales")
    if scale and any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("scales must be finite and strictly positive")
    return values


def _finish(
    values: tuple[float, ...], shape: tuple[int, ...], output: Output
) -> Tensor[float] | Tensor[int]:
    if output == "fp32":
        return Tensor(values, shape)
    return Tensor(tuple(encode_bf16(value) for value in values), shape)


def _rowwise(
    data: Tensor[int],
    scales: Tensor[int],
    group: int,
    decoder: Callable[[int], float],
    scale_decoder: Callable[[int], float],
    output: Output,
    *,
    packed: bool = False,
) -> Tensor[float] | Tensor[int]:
    _output_mode(output)
    shape = data.shape[:-1] + (data.shape[-1] * (2 if packed else 1),)
    if shape[-1] % group:
        raise ValueError(f"last logical dimension must be divisible by {group}")
    if scales.shape != shape[:-1] + (shape[-1] // group,):
        raise ValueError("scale shape must exactly match row groups")
    raw = unpack_e2m1(data.data, shape) if packed else data.data
    values = _decoded(raw, decoder)
    factors = _decoded(scales.data, scale_decoder, scale=True)
    products = tuple(
        multiply_fp32(value, factors[i // group]) for i, value in enumerate(values)
    )
    return _finish(products, shape, output)


def dequantize_expert(
    data: Tensor[int], scales: Tensor[int], *, output: Output = "fp32"
) -> Tensor[float] | Tensor[int]:
    """Expert E2M1 bytes [out, K/2] and E8M0 scales [out, K/32]."""
    if len(data.shape) != 2:
        raise ValueError("expert weights must be a matrix")
    return _rowwise(data, scales, 32, decode_e2m1, decode_e8m0, output, packed=True)


def dequantize_fp8_weights(
    data: Tensor[int], scales: Tensor[int], *, output: Output = "fp32"
) -> Tensor[float] | Tensor[int]:
    """E4M3FN [out, K], with ceil(out/32) by ceil(K/32) E8M0 scales."""
    _output_mode(output)
    if len(data.shape) != 2:
        raise ValueError("FP8 weights must be a matrix")
    rows, cols = data.shape
    block_rows, block_cols = (rows + 31) // 32, (cols + 31) // 32
    if scales.shape != (block_rows, block_cols):
        raise ValueError("scale shape must cover the 32 by 32 blocks, including tails")
    values = _decoded(data.data, decode_e4m3fn)
    factors = _decoded(scales.data, decode_e8m0, scale=True)
    products = tuple(
        multiply_fp32(
            values[row * cols + col],
            factors[(row // 32) * block_cols + col // 32],
        )
        for row in range(rows) for col in range(cols)
    )
    return _finish(products, data.shape, output)


def dequantize_engram(
    data: Tensor[int], scales: Tensor[int], *, output: Output = "fp32"
) -> Tensor[float] | Tensor[int]:
    """Engram E4M3FN rows with E8M0 scales along the last dimension, per 32."""
    if len(data.shape) != 2:
        raise ValueError("Engram rows must be a matrix")
    return _rowwise(data, scales, 32, decode_e4m3fn, decode_e8m0, output)


def dequantize_swa(
    data: Tensor[int], scales: Tensor[int], *, output: Output = "fp32"
) -> Tensor[float] | Tensor[int]:
    return _rowwise(data, scales, 32, decode_e4m3fn, decode_e8m0, output)


def dequantize_main_kv(
    data: Tensor[int], scales: Tensor[int], *, output: Output = "fp32"
) -> Tensor[float] | Tensor[int]:
    """Compressed main KV: packed E2M1, positive E4M3FN scale per 16."""
    return _rowwise(data, scales, 16, decode_e2m1, decode_e4m3fn, output, packed=True)


def dequantize_index_k(
    data: Tensor[int], scales: Tensor[int], *, output: Output = "fp32"
) -> Tensor[float] | Tensor[int]:
    return _rowwise(data, scales, 32, decode_e2m1, decode_e8m0, output, packed=True)
