"""P0020: single-token FP4 expert oracle under docs/EXPERT.md.

Composes the existing activation quantizer and grouped-dot oracle. Nonlinear
arithmetic uses bit patterns and certified Decimal exp intervals. Hardware
floating-point exceptions must be masked; Decimal may add sticky flags, but
controls and pre-existing sticky flags are preserved.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Context, Decimal, DivisionByZero, InvalidOperation, Overflow, ROUND_HALF_EVEN
from fractions import Fraction

from ref.act_quant import act_quant
from ref.compute import expert_grouped_dot
from ref.formats import _binary_to_fp32_bits, _finite_parts, _unsigned, fp32_bits_to_bf16


_SIGN = 0x80000000
_INF = 0x7F800000
_NAN = 0x7FC00000
_TEN = 0x41200000
_SILU_INITIAL_PRECISION = 192
_SILU_MAX_PRECISION = 3072


@dataclass(frozen=True)
class FP4Matrix:
    """Encoded N-by-K weights; validation and sequence snapshots happen at entry."""

    n: int
    k: int
    w_packed: Sequence[int]
    scale_b: Sequence[int]


@dataclass(frozen=True)
class BridgeTrace:
    gate_bits: tuple[int, ...]
    up_bits: tuple[int, ...]
    silu_bits: tuple[int, ...]
    gated_bits: tuple[int, ...]
    weighted_bits: tuple[int, ...]
    bf16_bits: tuple[int, ...]


@dataclass(frozen=True)
class ExpertTrace:
    input_codes: tuple[int, ...]
    input_scales: tuple[int, ...]
    gate_bf16: tuple[int, ...]
    up_bf16: tuple[int, ...]
    bridge: BridgeTrace
    hidden_codes: tuple[int, ...]
    hidden_scales: tuple[int, ...]
    output_bf16: tuple[int, ...]


def _multiply_fp32_bits(a: int, b: int) -> int:
    """One binary32 RNE multiplication, preserving signed gradual underflow."""
    a, b = _unsigned(a, 32), _unsigned(b, 32)
    sign = (a ^ b) & _SIGN
    mag_a, mag_b = a & 0x7FFFFFFF, b & 0x7FFFFFFF
    if mag_a > _INF or mag_b > _INF:
        return _NAN
    if mag_a == _INF or mag_b == _INF:
        return _NAN if mag_a == 0 or mag_b == 0 else sign | _INF
    sig_a, exp_a = _finite_parts(a)
    sig_b, exp_b = _finite_parts(b)
    return _binary_to_fp32_bits(sign, sig_a * sig_b, exp_a + exp_b)


def _fraction_to_fp32_bits(value: Fraction) -> int:
    """Round a rational by integer quotient/remainder, then reuse dyadic encoding."""
    sign = _SIGN if value < 0 else 0
    numerator, denominator = abs(value.numerator), value.denominator
    if numerator == 0:
        return sign
    top = numerator.bit_length() - denominator.bit_length()
    if top >= 0:
        top -= numerator < (denominator << top)
    else:
        top -= (numerator << -top) < denominator
    exponent = max(top - 23, -149)
    if exponent < 0:
        numerator <<= -exponent
    else:
        denominator <<= exponent
    whole, remainder = divmod(numerator, denominator)
    twice = remainder << 1
    rounded = whole + (twice > denominator or (twice == denominator and whole & 1))
    return _binary_to_fp32_bits(sign, rounded, exponent)


def _silu_fp32_bits(bits: int) -> int:
    """Correctly round real g/(1+exp(-g)) once, for already-clamped g <= 10.

    Signed zeros survive; NaN and -infinity produce NaN. For finite g <= -128,
    t*exp(-t) decreases for t >= 128 and 128*exp(-128) < 2**-150, so the
    correctly rounded result is negative zero. Otherwise exp(-abs(g)) avoids
    unstable positive exponents. Decimal exp is correctly rounded; its adjacent
    representable values enclose the exact exponential. Exact Fraction arithmetic
    transports that interval through SiLU, and only agreement of its binary32
    RNE endpoints permits a result. Precision is bounded; failure to certify
    raises ArithmeticError rather than guessing. This is not a general SiLU API.
    """
    bits = _unsigned(bits, 32)
    magnitude = bits & 0x7FFFFFFF
    negative = bool(bits & _SIGN)
    if magnitude > _INF or (negative and magnitude == _INF):
        return _NAN
    if not negative and magnitude > _TEN:
        raise ValueError("SiLU requires a gate already clamped to at most 10")
    if magnitude == 0:
        return bits
    if negative and magnitude >= 0x43000000:
        return _SIGN
    significand, exponent = _finite_parts(bits)
    magnitude_value = (
        Fraction(significand << exponent) if exponent >= 0
        else Fraction(significand, 1 << -exponent)
    )
    gate = -magnitude_value if negative else magnitude_value
    precision = _SILU_INITIAL_PRECISION
    while precision <= _SILU_MAX_PRECISION:
        context = Context(
            prec=precision, rounding=ROUND_HALF_EVEN, Emin=-9999, Emax=9999,
            capitals=1, clamp=0, traps=[InvalidOperation, DivisionByZero, Overflow],
        )
        argument = context.divide(
            Decimal(-magnitude_value.numerator), Decimal(magnitude_value.denominator)
        )
        if Fraction(argument) != -magnitude_value:
            precision *= 2
            continue
        exponential = context.exp(argument)
        lower_exp = Fraction(context.next_minus(exponential))
        upper_exp = Fraction(context.next_plus(exponential))
        if negative:
            lower = gate * upper_exp / (1 + upper_exp)
            upper = gate * lower_exp / (1 + lower_exp)
        else:
            lower = gate / (1 + upper_exp)
            upper = gate / (1 + lower_exp)
        lower_bits = _fraction_to_fp32_bits(lower)
        upper_bits = _fraction_to_fp32_bits(upper)
        if lower_bits == upper_bits:
            return lower_bits
        precision *= 2
    raise ArithmeticError("SiLU binary32 rounding interval did not converge")


def _encoded_sequence(values: Sequence[int], width: int) -> tuple[int, ...]:
    if not isinstance(values, Sequence):
        raise TypeError("a one-dimensional encoded sequence is required")
    if isinstance(values, memoryview) and values.ndim != 1:
        raise TypeError("a one-dimensional encoded sequence is required")
    snapshot = tuple(values)
    return tuple(_unsigned(value, width) for value in snapshot)


def _route_weight(bits: int | None) -> int | None:
    if bits is None:
        return None
    bits = _unsigned(bits, 32)
    if (bits & 0x7FFFFFFF) >= _INF:
        raise ValueError("route_weight_bits must encode a finite FP32 value")
    return bits


def _require_finite_bf16(values: tuple[int, ...], name: str) -> None:
    if any((value & 0x7F80) == 0x7F80 for value in values):
        raise ValueError(f"{name} contains nonfinite BF16 values")


def _clamp_bits(bits: int, *, gate: bool) -> int:
    magnitude = bits & 0x7FFFFFFF
    if magnitude > _INF:
        return bits
    if gate and bits & _SIGN:
        return bits
    return (bits & _SIGN) | min(magnitude, _TEN)


def _bridge_inputs(gate_bf16, up_bf16, route_weight_bits):
    weight = _route_weight(route_weight_bits)
    gate = _encoded_sequence(gate_bf16, 16)
    up = _encoded_sequence(up_bf16, 16)
    if len(gate) != len(up):
        raise ValueError("gate and up must have the same length")
    return gate, up, weight


def _bridge(gate_bf16, up_bf16, route_weight_bits, *, observe):
    result = []
    stages = tuple([] for _ in range(5)) if observe else None
    for gate_code, up_code in zip(gate_bf16, up_bf16):
        gate = _clamp_bits(gate_code << 16, gate=True)
        up = _clamp_bits(up_code << 16, gate=False)
        silu = _silu_fp32_bits(gate)
        gated = _multiply_fp32_bits(silu, up)
        weighted = (
            gated if route_weight_bits is None
            else _multiply_fp32_bits(gated, route_weight_bits)
        )
        result.append(fp32_bits_to_bf16(weighted))
        if observe:
            for stage, value in zip(stages, (gate, up, silu, gated, weighted)):
                stage.append(value)
    output = tuple(result)
    if observe:
        return BridgeTrace(*(tuple(stage) for stage in stages), output)
    return output


def swiglu(
    gate_bf16: Sequence[int], up_bf16: Sequence[int], *,
    route_weight_bits: int | None = None,
) -> tuple[int, ...]:
    """Clamp, SiLU, multiply up, optionally weight, then BF16; specials are allowed."""
    return _bridge(*_bridge_inputs(gate_bf16, up_bf16, route_weight_bits), observe=False)


def trace_swiglu(
    gate_bf16: Sequence[int], up_bf16: Sequence[int], *,
    route_weight_bits: int | None = None,
) -> BridgeTrace:
    """Observe the same bridge, including nonfinite inputs and results."""
    return _bridge(*_bridge_inputs(gate_bf16, up_bf16, route_weight_bits), observe=True)


def _dimension(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("matrix dimensions must be integers")
    if value <= 0 or value % 32:
        raise ValueError("matrix dimensions must be positive multiples of 32")
    return value


def _matrix(matrix: FP4Matrix) -> FP4Matrix:
    if not isinstance(matrix, FP4Matrix):
        raise TypeError("weights must be FP4Matrix instances")
    n, k = _dimension(matrix.n), _dimension(matrix.k)
    packed = _encoded_sequence(matrix.w_packed, 8)
    scales = _encoded_sequence(matrix.scale_b, 8)
    if len(packed) != n * (k // 2):
        raise ValueError("packed weight length must be N*K/2")
    if len(scales) != n * (k // 32):
        raise ValueError("weight scale length must be N*K/32")
    if 0xFF in scales:
        raise ValueError("NaN weight scale codes are not accepted")
    return FP4Matrix(n, k, packed, scales)


def _expert_inputs(input_bf16, w1, w2, w3, route_weight_bits):
    weight = _route_weight(route_weight_bits)
    inputs = _encoded_sequence(input_bf16, 16)
    _require_finite_bf16(inputs, "input")
    w1, w2, w3 = _matrix(w1), _matrix(w2), _matrix(w3)
    d, h = w1.k, w1.n
    if len(inputs) != d:
        raise ValueError("input length must equal D")
    if (w3.n, w3.k) != (h, d) or (w2.n, w2.k) != (d, h):
        raise ValueError("expected w1/w3 H-by-D and w2 D-by-H")
    return inputs, w1, w2, w3, weight


def _linear(codes, scales, matrix):
    packed_stride, scale_stride = matrix.k // 2, matrix.k // 32
    return tuple(
        expert_grouped_dot(
            codes,
            matrix.w_packed[row * packed_stride:(row + 1) * packed_stride],
            scales,
            matrix.scale_b[row * scale_stride:(row + 1) * scale_stride],
            output="bf16",
        )
        for row in range(matrix.n)
    )


def _evaluate(input_bf16, w1, w2, w3, route_weight_bits, *, observe):
    inputs, w1, w2, w3, weight = _expert_inputs(
        input_bf16, w1, w2, w3, route_weight_bits
    )
    input_codes, input_scales = act_quant(inputs)
    gate = _linear(input_codes, input_scales, w1)
    up = _linear(input_codes, input_scales, w3)
    bridge = _bridge(gate, up, weight, observe=observe)
    hidden = bridge.bf16_bits if observe else bridge
    _require_finite_bf16(hidden, "bridge output")
    hidden_codes, hidden_scales = act_quant(hidden)
    output = _linear(hidden_codes, hidden_scales, w2)
    if observe:
        return ExpertTrace(
            input_codes, input_scales, gate, up, bridge,
            hidden_codes, hidden_scales, output,
        )
    return output


def expert(
    input_bf16: Sequence[int], w1: FP4Matrix, w2: FP4Matrix, w3: FP4Matrix, *,
    route_weight_bits: int | None = None,
) -> tuple[int, ...]:
    """Evaluate one FP4 expert; require finite BF16 at activation-quantizer inputs.

    All external sequences are validated and snapshotted before computation.
    D and H are positive multiples of 32; w1/w3 are H-by-D and w2 is D-by-H.
    Every linear output is BF16. The bridge may clamp infinite w1/w3 outputs to
    finite values; only a nonfinite bridge BF16 result is rejected before the
    second activation quantization. Final w2 output may be nonfinite. The bridge
    rounds SiLU and each multiplication to FP32, then BF16. None skips the route
    multiplication entirely. This is a slow CPU oracle, not an engine path.
    Decimal may add hardware sticky exception flags under masked exceptions;
    numeric results and rounding/FTZ/DAZ control remain unaffected.
    """
    return _evaluate(input_bf16, w1, w2, w3, route_weight_bits, observe=False)


def trace_expert(
    input_bf16: Sequence[int], w1: FP4Matrix, w2: FP4Matrix, w3: FP4Matrix, *,
    route_weight_bits: int | None = None,
) -> ExpertTrace:
    """Observe expert boundary tensors on the same core and with the same checks."""
    return _evaluate(input_bf16, w1, w2, w3, route_weight_bits, observe=True)
