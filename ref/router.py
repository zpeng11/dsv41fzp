"""P0023: deterministic single-token MoE routing CPU oracle.

The fixed mainline algorithm uses binary32 RNE at every stated stage and keeps
all public inputs and outputs as encoded bit patterns. Decimal transcendental
work uses private contexts and certified intervals; failure to certify is an
error rather than a host-libm fallback.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import (
    Context,
    Decimal,
    DivisionByZero,
    InvalidOperation,
    Overflow,
    ROUND_CEILING,
    ROUND_FLOOR,
    ROUND_HALF_EVEN,
)
from fractions import Fraction
from math import isqrt

from ref.compute import _add_fp32, _from_bits
from ref.expert import _fraction_to_fp32_bits, _multiply_fp32_bits
from ref.formats import _finite_parts, _unsigned, fp32_bits


_SIGN = 0x80000000
_INF = 0x7F800000
_ONE = 0x3F800000
_SCALE = 0x3FC00000
_EPSILON = 0x1E3CE508
_TOPK = 6
_INITIAL_PRECISION = 192
_MAX_PRECISION = 3072


@dataclass(frozen=True)
class GateMatrix:
    """Encoded N-by-K BF16 gate weights snapshotted and validated at entry."""

    n: int
    k: int
    weight_bf16: Sequence[int]


@dataclass(frozen=True)
class RouteTrace:
    """Encoded boundary values from the same execution used by ``route``."""

    linear_bits: tuple[int, ...]
    temp_bits: tuple[int, ...]
    softplus_bits: tuple[int, ...]
    score_bits: tuple[int, ...]
    selection_bits: tuple[int, ...]
    indices: tuple[int, ...]
    selected_bits: tuple[int, ...]
    sum_bits: int
    denominator_bits: int
    normalized_bits: tuple[int, ...]
    weights_bits: tuple[int, ...]


def _encoded_sequence(values: Sequence[int], width: int) -> tuple[int, ...]:
    """Snapshot one-dimensional encoded input before validating its elements."""
    if not isinstance(values, Sequence):
        raise TypeError("a one-dimensional encoded sequence is required")
    if isinstance(values, memoryview) and values.ndim != 1:
        raise TypeError("a one-dimensional encoded sequence is required")
    snapshot = tuple(values)
    return tuple(_unsigned(value, width) for value in snapshot)


def _dimension(value: int, name: str, minimum: int) -> int:
    """Validate a matrix dimension without accepting booleans as integers."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _require_finite_bf16(values: tuple[int, ...], name: str) -> None:
    """Reject BF16 infinities and NaNs while allowing both signed zeros."""
    if any((value & 0x7F80) == 0x7F80 for value in values):
        raise ValueError(f"{name} contains nonfinite BF16 values")


def _require_finite_fp32(values: tuple[int, ...], name: str) -> None:
    """Reject FP32 infinities and NaNs while allowing both signed zeros."""
    if any((value & 0x7FFFFFFF) >= _INF for value in values):
        raise ValueError(f"{name} contains nonfinite FP32 values")


def _finite_fraction(bits: int) -> Fraction:
    """Decode a finite binary32 bit pattern to its exact rational value."""
    bits = _unsigned(bits, 32)
    if (bits & 0x7FFFFFFF) >= _INF:
        raise ValueError("a finite FP32 bit pattern is required")
    significand, exponent = _finite_parts(bits)
    magnitude = (
        Fraction(significand << exponent)
        if exponent >= 0
        else Fraction(significand, 1 << -exponent)
    )
    return -magnitude if bits & _SIGN else magnitude


def _add_fp32_bits(left: int, right: int) -> int:
    """Perform one binary32 RNE addition through the established integer core."""
    return fp32_bits(_add_fp32(_from_bits(left), _from_bits(right)))


def _divide_fp32_bits(numerator: int, denominator: int) -> int:
    """Correctly round one finite binary32 division using exact rationals."""
    numerator = _unsigned(numerator, 32)
    denominator = _unsigned(denominator, 32)
    if (numerator & 0x7FFFFFFF) >= _INF or (denominator & 0x7FFFFFFF) >= _INF:
        raise ValueError("division operands must be finite FP32 values")
    if denominator & 0x7FFFFFFF == 0:
        raise ValueError("division denominator must be nonzero")
    if numerator & 0x7FFFFFFF == 0:
        return (numerator ^ denominator) & _SIGN
    return _fraction_to_fp32_bits(
        _finite_fraction(numerator) / _finite_fraction(denominator)
    )


def _decimal_context(precision: int, rounding: str = ROUND_HALF_EVEN) -> Context:
    """Create an isolated wide-range context for one certification attempt."""
    return Context(
        prec=precision,
        rounding=rounding,
        Emin=-999999999,
        Emax=999999999,
        capitals=1,
        clamp=0,
        traps=[InvalidOperation, DivisionByZero, Overflow],
    )


def _decimal_fraction_bound(value: Fraction, precision: int, rounding: str) -> Decimal:
    """Convert a positive rational to a Decimal bound in the requested direction."""
    context = _decimal_context(precision, rounding)
    return context.divide(Decimal(value.numerator), Decimal(value.denominator))


def _softplus_fp32_bits(bits: int) -> int:
    """Correctly round ``ln(1 + exp(x))`` for one finite binary32 input.

    Inputs strictly above 20 use the specified linear branch. For x <= -128,
    ``softplus(x) < exp(x) < (3/8)**128 < 2**-150`` because e > 8/3, so the
    result rounds to positive zero. Other inputs use outward Decimal exp/add/ln
    intervals, with precision doubled through 3072 decimal digits.
    """
    bits = _unsigned(bits, 32)
    if (bits & 0x7FFFFFFF) >= _INF:
        raise ValueError("softplus input must be finite FP32")
    value = _finite_fraction(bits)
    if value > 20:
        return bits
    if value <= -128:
        return 0
    precision = _INITIAL_PRECISION
    while precision <= _MAX_PRECISION:
        context = _decimal_context(precision)
        base = Fraction(0) if value < 0 else value
        exponent_argument = value if value < 0 else -value
        argument = context.divide(
            Decimal(exponent_argument.numerator),
            Decimal(exponent_argument.denominator),
        )
        if Fraction(argument) != exponent_argument:
            precision *= 2
            continue
        lower_context = _decimal_context(precision, ROUND_FLOOR)
        upper_context = _decimal_context(precision, ROUND_CEILING)
        exponential_lower = lower_context.exp(argument)
        exponential_upper = upper_context.exp(argument)
        exp_lower = Fraction(lower_context.next_minus(exponential_lower))
        exp_upper = Fraction(upper_context.next_plus(exponential_upper))
        add_lower = 1 + exp_lower
        add_upper = 1 + exp_upper
        decimal_lower = _decimal_fraction_bound(add_lower, precision, ROUND_FLOOR)
        decimal_upper = _decimal_fraction_bound(add_upper, precision, ROUND_CEILING)
        logarithm_lower = lower_context.next_minus(lower_context.ln(decimal_lower))
        logarithm_upper = upper_context.next_plus(upper_context.ln(decimal_upper))
        lower_bits = _fraction_to_fp32_bits(base + Fraction(logarithm_lower))
        upper_bits = _fraction_to_fp32_bits(base + Fraction(logarithm_upper))
        if lower_bits == upper_bits:
            return lower_bits
        precision *= 2
    raise ArithmeticError("softplus binary32 rounding interval did not converge")


def _sqrt_fp32_bits(bits: int) -> int:
    """Correctly round a nonnegative finite binary32 square root.

    Integer ``isqrt`` supplies a high-precision lower approximation. Exact
    rational square comparisons locate adjacent binary32 values, and an exact
    squared-midpoint comparison applies ties-to-even without host sqrt.
    """
    bits = _unsigned(bits, 32)
    if bits & 0x7FFFFFFF == 0:
        return bits
    if bits & _SIGN or (bits & 0x7FFFFFFF) >= _INF:
        raise ValueError("sqrt input must be finite and nonnegative")

    value = _finite_fraction(bits)
    scale = 200
    approximation = Fraction(
        isqrt((value.numerator << (2 * scale)) // value.denominator),
        1 << scale,
    )
    candidate = _fraction_to_fp32_bits(approximation)
    candidate = min(candidate, 0x7F7FFFFF)
    while candidate and _finite_fraction(candidate) ** 2 > value:
        candidate -= 1
    while candidate < 0x7F7FFFFF and _finite_fraction(candidate + 1) ** 2 <= value:
        candidate += 1
    lower = _finite_fraction(candidate)
    upper_bits = candidate + 1
    upper = _finite_fraction(upper_bits)
    midpoint = (lower + upper) / 2
    midpoint_square = midpoint * midpoint
    if value < midpoint_square:
        return candidate
    if value > midpoint_square:
        return upper_bits
    return candidate if candidate & 1 == 0 else upper_bits


def _matrix(matrix: GateMatrix) -> GateMatrix:
    """Snapshot and validate one gate matrix without retaining mutable sequences."""
    if not isinstance(matrix, GateMatrix):
        raise TypeError("gate must be a GateMatrix")
    n = _dimension(matrix.n, "N", _TOPK)
    k = _dimension(matrix.k, "K", 1)
    weights = _encoded_sequence(matrix.weight_bf16, 16)
    if len(weights) != n * k:
        raise ValueError("gate weight length must be N*K")
    _require_finite_bf16(weights, "gate weights")
    return GateMatrix(n, k, weights)


def _inputs(input_bf16, gate, bias_bits, image_token, bias_vl_bits):
    """Snapshot and completely validate every public input before computation."""
    if not isinstance(image_token, bool):
        raise TypeError("image_token must be bool")
    inputs = _encoded_sequence(input_bf16, 16)
    matrix = _matrix(gate)
    bias = _encoded_sequence(bias_bits, 32)
    bias_vl = None if bias_vl_bits is None else _encoded_sequence(bias_vl_bits, 32)
    if len(inputs) != matrix.k:
        raise ValueError("input length must equal K")
    if len(bias) != matrix.n:
        raise ValueError("bias length must equal N")
    if bias_vl is not None and len(bias_vl) != matrix.n:
        raise ValueError("vision bias length must equal N")
    _require_finite_bf16(inputs, "input")
    _require_finite_fp32(bias, "bias")
    if bias_vl is not None:
        _require_finite_fp32(bias_vl, "vision bias")
    selected_bias = bias_vl if image_token and bias_vl is not None else bias
    return inputs, matrix, selected_bias


def _evaluate(input_bf16, gate, bias_bits, image_token, bias_vl_bits, *, observe):
    """Run routing once so ordinary and trace entry points cannot diverge."""
    inputs, gate, bias = _inputs(
        input_bf16, gate, bias_bits, image_token, bias_vl_bits
    )
    linear = []
    for row in range(gate.n):
        accumulator = 0
        offset = row * gate.k
        for column in range(gate.k):
            product = _multiply_fp32_bits(
                inputs[column] << 16,
                gate.weight_bf16[offset + column] << 16,
            )
            if (product & 0x7FFFFFFF) >= _INF:
                raise ValueError("nonfinite projection product")
            accumulator = _add_fp32_bits(accumulator, product)
            if (accumulator & 0x7FFFFFFF) >= _INF:
                raise ValueError("nonfinite projection accumulator")
        linear.append(accumulator)
    linear_bits = tuple(linear)
    temp_bits = tuple(_divide_fp32_bits(value, _ONE) for value in linear_bits)
    softplus_bits = tuple(_softplus_fp32_bits(value) for value in temp_bits)
    score_bits = tuple(_sqrt_fp32_bits(value) for value in softplus_bits)
    selection = []
    for score, correction in zip(score_bits, bias):
        selected = _add_fp32_bits(score, correction)
        if (selected & 0x7FFFFFFF) >= _INF:
            raise ValueError("nonfinite selection score")
        selection.append(selected)
    selection_bits = tuple(selection)
    indices = tuple(sorted(
        range(gate.n),
        key=lambda index: (-_finite_fraction(selection_bits[index]), index),
    )[:_TOPK])
    selected_bits = tuple(score_bits[index] for index in indices)
    sum_bits = 0
    for value in selected_bits:
        sum_bits = _add_fp32_bits(sum_bits, value)
    denominator_bits = _add_fp32_bits(sum_bits, _EPSILON)
    normalized_bits = tuple(
        _divide_fp32_bits(value, denominator_bits) for value in selected_bits
    )
    weights_bits = tuple(
        _multiply_fp32_bits(value, _SCALE) for value in normalized_bits
    )
    if observe:
        return RouteTrace(
            linear_bits,
            temp_bits,
            softplus_bits,
            score_bits,
            selection_bits,
            indices,
            selected_bits,
            sum_bits,
            denominator_bits,
            normalized_bits,
            weights_bits,
        )
    return indices, weights_bits


def route(
    input_bf16: Sequence[int],
    gate: GateMatrix,
    bias_bits: Sequence[int],
    *,
    image_token: bool = False,
    bias_vl_bits: Sequence[int] | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Select six experts and return their indices and scaled FP32 weight bits."""
    return _evaluate(
        input_bf16, gate, bias_bits, image_token, bias_vl_bits, observe=False
    )


def trace_route(
    input_bf16: Sequence[int],
    gate: GateMatrix,
    bias_bits: Sequence[int],
    *,
    image_token: bool = False,
    bias_vl_bits: Sequence[int] | None = None,
) -> RouteTrace:
    """Observe all contracted routing stages on the ordinary execution core."""
    return _evaluate(
        input_bf16, gate, bias_bits, image_token, bias_vl_bits, observe=True
    )
