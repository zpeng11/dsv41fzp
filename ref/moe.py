"""P0025: single-token routed MoE branch oracle under docs/MOE.md.

The fixed composition reuses the established router and FP4 expert contracts,
then accumulates weighted expert outputs in ascending global expert order.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from ref.expert import FP4Matrix, ExpertTrace, _matrix, expert, trace_expert
from ref.router import (
    GateMatrix,
    RouteTrace,
    _add_fp32_bits,
    _inputs as _router_inputs,
    route,
    trace_route,
)


@dataclass(frozen=True)
class FP4Expert:
    w1: FP4Matrix
    w2: FP4Matrix
    w3: FP4Matrix


@dataclass(frozen=True)
class RoutedTrace:
    route: RouteTrace
    expert_indices: tuple[int, ...]
    expert_traces: tuple[ExpertTrace, ...]
    partial_sums_bits: tuple[tuple[int, ...], ...]
    output_bits: tuple[int, ...]


def _experts(values: Sequence[FP4Expert], count: int, d: int) -> tuple[FP4Expert, ...]:
    """Snapshot and validate all globally indexed experts before evaluation."""
    if not isinstance(values, Sequence):
        raise TypeError("a one-dimensional expert sequence is required")
    if isinstance(values, memoryview) and values.ndim != 1:
        raise TypeError("a one-dimensional expert sequence is required")
    snapshot = tuple(values)
    if len(snapshot) != count:
        raise ValueError("expert count must equal gate N")

    validated = []
    expected_h = None
    for value in snapshot:
        if not isinstance(value, FP4Expert):
            raise TypeError("experts must be FP4Expert instances")
        w1, w2, w3 = _matrix(value.w1), _matrix(value.w2), _matrix(value.w3)
        h = w1.n
        if w1.k != d or (w3.n, w3.k) != (h, d) or (w2.n, w2.k) != (d, h):
            raise ValueError("expected w1/w3 H-by-D and w2 D-by-H with D equal to gate K")
        if expected_h is None:
            expected_h = h
        elif h != expected_h:
            raise ValueError("all experts must have the same H")
        validated.append(FP4Expert(w1, w2, w3))
    return tuple(validated)


def _inputs(input_bf16, gate, bias_bits, experts, image_token, bias_vl_bits):
    """Snapshot and completely validate every routed-branch input."""
    inputs, matrix, selected_bias = _router_inputs(
        input_bf16, gate, bias_bits, image_token, bias_vl_bits
    )
    return inputs, matrix, selected_bias, _experts(experts, matrix.n, matrix.k)


def _accumulate(
    accumulator: tuple[int, ...], output_bf16: tuple[int, ...]
) -> tuple[int, ...]:
    """Add one validated BF16 expert vector to a validated FP32 accumulator."""
    return tuple(
        _add_fp32_bits(partial, value << 16)
        for partial, value in zip(accumulator, output_bf16)
    )


def _evaluate(
    input_bf16,
    gate,
    bias_bits,
    experts,
    image_token,
    bias_vl_bits,
    *,
    observe,
):
    """Run routing, experts, and ordered accumulation on one shared core."""
    inputs, gate, bias, experts = _inputs(
        input_bf16, gate, bias_bits, experts, image_token, bias_vl_bits
    )
    routed = (
        trace_route(inputs, gate, bias)
        if observe
        else route(inputs, gate, bias)
    )
    if observe:
        paired = zip(routed.indices, routed.weights_bits)
    else:
        paired = zip(*routed)
    ordered = tuple(sorted(paired, key=lambda pair: pair[0]))

    accumulator = (0,) * gate.k
    traces = []
    partials = []
    for index, weight in ordered:
        selected = experts[index]
        if observe:
            evaluation = trace_expert(
                inputs,
                selected.w1,
                selected.w2,
                selected.w3,
                route_weight_bits=weight,
            )
            traces.append(evaluation)
            output = evaluation.output_bf16
        else:
            output = expert(
                inputs,
                selected.w1,
                selected.w2,
                selected.w3,
                route_weight_bits=weight,
            )
        accumulator = _accumulate(accumulator, output)
        if observe:
            partials.append(accumulator)

    if observe:
        return RoutedTrace(
            routed,
            tuple(index for index, _ in ordered),
            tuple(traces),
            tuple(partials),
            accumulator,
        )
    return accumulator


def routed(
    input_bf16: Sequence[int],
    gate: GateMatrix,
    bias_bits: Sequence[int],
    experts: Sequence[FP4Expert],
    *,
    image_token: bool = False,
    bias_vl_bits: Sequence[int] | None = None,
) -> tuple[int, ...]:
    """Evaluate and FP32-accumulate the selected routed FP4 experts."""
    return _evaluate(
        input_bf16,
        gate,
        bias_bits,
        experts,
        image_token,
        bias_vl_bits,
        observe=False,
    )


def trace_routed(
    input_bf16: Sequence[int],
    gate: GateMatrix,
    bias_bits: Sequence[int],
    experts: Sequence[FP4Expert],
    *,
    image_token: bool = False,
    bias_vl_bits: Sequence[int] | None = None,
) -> RoutedTrace:
    """Observe routing, expert boundaries, and each ordered FP32 partial sum."""
    return _evaluate(
        input_bf16,
        gate,
        bias_bits,
        experts,
        image_token,
        bias_vl_bits,
        observe=True,
    )
