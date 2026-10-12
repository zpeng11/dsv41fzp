#!/usr/bin/env python3
"""P0013 C++ parity for the production expert grouped dot against ref/compute.py.

Usage: test_compute_parity.py --exe <quant_exec> --suite <parity|contract|add|harness>

Expected values come from ref.compute (trace_expert_grouped_dot for every stage,
_add_fp32 for the private adder probe); the production library is reached only
through the test executor. Comparison, response parsing and fault detection reuse
the P0010/P0011 driver helpers: finite values (signed zeros included) and Inf
bit-exact, NaN by class only, and any executor fault fails the run.

Compute request protocol, one per line; the response grammar is the P0010 one:
  <op> cap=<n> [overlap=a|w|sa|sb [in_offset=<b>] [out_offset=<b>]] [poison]
       : <a> ... ; <w bytes> ... ; <scale_a> ... ; <scale_b> ...
- Ops: expert_grouped_dot_fp32 / _bf16 (public overloads, uint32 / uint16 output)
  and expert_grouped_dot_trace (private, cap exactly 2K + 4K/32 + 1: products[K],
  partial_sums[K], dot, after_a, block, acc per block, then the FP32 result).
- overlap= shares one exact allocation between the named input (at byte in_offset)
  and the output (at byte out_offset); poison makes every input unreadable under
  ASan, so metadata rejections prove that no input was read first.
- add_fp32 cap=<n> : <left bits> ... ; <right bits> ... runs the production adder.
- Every environment runs twice, with the MXCSR status flags initially clear and set,
  so a call that raises or clears any flag breaks the full-MXCSR window.
- Rejections follow D0009: std::invalid_argument for K zero or not divisible by 32,
  weight/scale length mismatch, empty output, NaN activation or scale codes and any
  input overlapping the output; lengths and overlap are validated before any input
  is read, contents before the output is written, and failed calls leave the output
  untouched. Over-wide tokens, unknown ops and grammar violations kill the executor.
"""

import argparse
import random
import subprocess
import sys
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ref import formats as f
from ref.compute import _add_fp32, expert_grouped_dot, trace_expert_grouped_dot

from test_codec_parity import BF16, FP32, Case, HarnessError, check, failures, parse_response, run_executor
from test_dequant_parity import ENVIRONMENTS, MXCSR_STATUS, RawCase, mxcsr_probe


EXE = None
EXE_ARGS = []

STATUS_STATES = ("clear", "set")
A_CODES = tuple(code for code in range(256) if code not in (0x7F, 0xFF))
SCALE_CODES = tuple(range(0xFF))
SCALE_EDGES = (0x00, 0x01, 0x7E, 0x7F, 0x80, 0xFD, 0xFE)

# Encoded operands for single-product blocks: E4M3FN code, E2M1 nibble.
ONE = (0x38, 0x2)
NEG_ONE = (0x38, 0xA)
TWO = (0x38, 0x4)
NEG_TWO = (0x38, 0xC)
ONE_HALF = (0x3C, 0x2)
MAX_PRODUCT = (0x7E, 0x7)    # 448 * 6 = 2688
MIN_PRODUCT = (0x01, 0x1)    # 2^-9 * 0.5 = 2^-10
NEG_MIN_PRODUCT = (0x01, 0x9)
ZERO = (0x00, 0x0)


class ComputeCase(Case):
    """One compute request: four raw input sections plus an optional overlap target."""

    def __init__(self, name, op, sections, cap, out_width, expect, kind, overlap=None,
                 in_offset=None, out_offset=None, poison=False):
        super().__init__(name, op, (), [], cap, out_width, expect, kind)
        self.sections = [list(s) for s in sections]
        self.overlap = overlap
        self.in_offset = in_offset
        self.out_offset = out_offset
        self.poison = poison

    def line(self):
        head = [self.op, f"cap={self.cap}"]
        if self.overlap:
            head.append(f"overlap={self.overlap}")
        if self.in_offset is not None:
            head.append(f"in_offset={self.in_offset}")
        if self.out_offset is not None:
            head.append(f"out_offset={self.out_offset}")
        if self.poison:
            head.append("poison")
        return (" ".join(head) + " : "
                + " ; ".join(" ".join(f"{t:x}" for t in s) for s in self.sections))


def pack(nibbles):
    return list(f.pack_e2m1(list(nibbles), (len(nibbles),)))


def trace_words(trace):
    blocks = trace.blocks
    return ([p for b in blocks for p in b.products] + [s for b in blocks for s in b.partial_sums]
            + [b.dot_bits for b in blocks] + [b.after_a_bits for b in blocks]
            + [b.block_bits for b in blocks] + [b.acc_bits for b in blocks] + [trace.fp32_bits])


def vector(name, a, w, sa, sb):
    """Trace plus both public outputs for one valid input, all expected from ref."""
    trace = trace_expert_grouped_dot(a, w, sa, sb)
    sections = (a, w, sa, sb)
    words = trace_words(trace)
    return [
        ComputeCase(f"{name}.trace", "expert_grouped_dot_trace", sections, len(words), 4,
                    ("ok", words), FP32),
        ComputeCase(f"{name}.fp32", "expert_grouped_dot_fp32", sections, 2, 4,
                    ("ok", [trace.fp32_bits]), FP32),
        ComputeCase(f"{name}.bf16", "expert_grouped_dot_bf16", sections, 2, 2,
                    ("ok", [trace.bf16_bits]), BF16),
    ]


def blocks_vector(name, blocks):
    """Blocks given as (terms, scale_a, scale_b); terms are up to 32 (code, nibble) pairs."""
    a, nibbles, sa, sb = [], [], [], []
    for terms, scale_a, scale_b in blocks:
        terms = list(terms) + [ZERO] * (32 - len(terms))
        a += [code for code, _ in terms]
        nibbles += [nibble for _, nibble in terms]
        sa.append(scale_a)
        sb.append(scale_b)
    return vector(name, a, pack(nibbles), sa, sb)


def single(term, scale_a, scale_b):
    return ([term], scale_a, scale_b)


def trace_of(cases):
    return trace_expert_grouped_dot(*cases[0].sections)


# ---------------------------------------------------------------- parity suite


def exhaustive_cases():
    """Every finite E4M3FN x E2M1 pair once, under unit scales and under scale sweeps."""
    a = [code for code in A_CODES for _ in range(16)]
    nibbles = list(range(16)) * len(A_CODES)
    assert len(a) == 4064 and len(a) % 32 == 0
    blocks = len(a) // 32
    unit = vector("exhaustive.unit_scales", a, pack(nibbles), [127] * blocks, [127] * blocks)
    sweep_a = [(2 * j) % 0xFF for j in range(blocks)]
    sweep = vector("exhaustive.scale_sweep", a, pack(nibbles), sweep_a,
                   [0xFE - code for code in sweep_a])
    return unit + sweep


def scale_edge_cases():
    """Both scales over their encoding edges, one block each so Inf cannot mask later blocks."""
    terms = [MAX_PRODUCT, ONE_HALF, MIN_PRODUCT, NEG_ONE, (0x08, 0x3), (0x77, 0x6)]
    cases = []
    for scale_a in SCALE_EDGES:
        for scale_b in SCALE_EDGES:
            cases += blocks_vector(f"scale_edge.{scale_a:02x}.{scale_b:02x}",
                                   [(terms, scale_a, scale_b)])
    return cases


def random_vector(name, rng, k, scale_lo, scale_hi):
    a = [rng.choice(A_CODES) for _ in range(k)]
    w = [rng.randrange(256) for _ in range(k // 2)]
    sa = [rng.randrange(scale_lo, scale_hi) for _ in range(k // 32)]
    sb = [rng.randrange(scale_lo, scale_hi) for _ in range(k // 32)]
    return vector(name, a, w, sa, sb)


def layout_cases():
    """Adjacent distinct scales across many blocks and across 128-element boundaries."""
    rng = random.Random(1300)
    cases = []
    sa = [100 + 3 * j for j in range(16)]
    sb = [150 - 2 * j for j in range(16)]
    a = [rng.choice(A_CODES) for _ in range(512)]
    w = [rng.randrange(256) for _ in range(256)]
    cases += vector("layout.distinct_scales_512", a, w, sa, sb)
    # one nonzero term per block with alternating scales: a 128-element group would misscale three of four
    group = [single(ONE, 127 + (j % 4), 127 - 2 * (j % 4)) for j in range(8)]
    cases += blocks_vector("layout.group128_sensitive", group)
    trace = trace_of(cases[-3:])
    assert len({b.block_bits for b in trace.blocks[:4]}) == 4
    # distinct nibbles in one byte under distinct activations: 1*1 and 2*4, swapped 1*4 and 2*1
    nibble = blocks_vector("layout.nibble_order", [([ONE, (0x40, 0x6)], 127, 127)])
    assert trace_of(nibble).blocks[0].products[:2] == (0x3F800000, 0x41000000)
    return cases + nibble


def order_cases():
    """Vectors whose results change under tree reduction, exact summation or reordering."""
    cases = []
    # sequential: 16 * 2688 absorbs every later 2^-10; adjacent pairing keeps 16 * 2^-10
    tree = blocks_vector("order.tree_sensitive", [([MAX_PRODUCT] * 16 + [MIN_PRODUCT] * 16, 127, 127)])
    assert f.fp32_from_bits(trace_of(tree).fp32_bits) == 16 * 2688
    cases += tree
    # cross-block: 2^24 + 1 rounds to 2^24 then cancels; reversed order leaves 1
    reverse = blocks_vector("order.cross_block_sensitive",
                            [single(ONE, 151, 127), single(ONE, 127, 127), single(NEG_ONE, 151, 127)])
    assert trace_of(reverse).fp32_bits == 0
    cases += reverse
    # dot=2, scale_a=2^127, scale_b=2^-127: Inf in the specified order, 2 when swapped or folded
    swap = blocks_vector("order.scale_order", [single(TWO, 0xFE, 0x00)])
    block = trace_of(swap).blocks[0]
    assert (block.dot_bits, block.after_a_bits, block.block_bits) == (0x40000000, 0x7F800000, 0x7F800000)
    cases += swap
    # in-block terms that round differently in reverse order
    in_block = blocks_vector("order.in_block_sensitive",
                             [([MAX_PRODUCT] * 12 + [MIN_PRODUCT] * 3 + [(0xFE, 0x7)] * 12, 127, 127)])
    cases += in_block
    return cases


def rounding_cases():
    """Accumulator ties, carries, subnormal boundaries, underflow, overflow and specials."""
    cases = []
    directed = {
        # 2^24 + 1 ties down to even, 2^24 + 3 ties up to 2^24 + 4
        "tie_down": [single(ONE, 151, 127), single(ONE, 127, 127)],
        "tie_up": [single(ONE, 151, 127), single((0x40, 0x3), 127, 127), single(ONE, 127, 127)],
        "neighbors": [single(ONE, 151, 127), single(ONE_HALF, 127, 127)],
        # 2^24 - 1 + 0.5 carries into the exponent
        "carry": [single(ONE, 151, 127), single(NEG_ONE, 127, 127), single(ONE, 126, 127)],
        # 2^-149 steps, the largest subnormal plus 2^-149 reaching 2^-126
        "subnormal_steps": [single(ONE, 0x00, 105)] * 3,
        "subnormal_to_normal": [single(ONE, 0x00, 128), single(NEG_ONE, 0x00, 105),
                                single(ONE, 0x00, 105)],
        "negative_subnormal_chain": [single(NEG_ONE, 0x00, 105)] * 4 + [single(NEG_ONE, 0x00, 106)],
        # 2^-10 * 2^-127 * 2^-127 underflows to zero with the product's sign
        "underflow_zero": [single(MIN_PRODUCT, 0x00, 0x00)],
        "underflow_negative_zero": [single(NEG_MIN_PRODUCT, 0x00, 0x00)],
        # 3 * 2^-150 and 5 * 2^-150 are subnormal ties after the second scale
        "scale_subnormal_tie_up": [single((0x01, 0x5), 0x00, 113)],
        "scale_subnormal_tie_down": [single((0x05, 0x1), 0x00, 114)],
        # 1.5 * 2^127 twice overflows the accumulator; scale overflow keeps sign
        "acc_overflow": [single(ONE_HALF, 0xFE, 127), single(ONE_HALF, 0xFE, 127)],
        "scale_overflow_negative": [single(NEG_TWO, 0xFE, 0xFE)],
        "inf_survives_second_scale": [single(NEG_TWO, 0xFE, 0x00), single(ONE, 127, 127)],
        "nan_from_opposite_inf": [single(TWO, 0xFE, 0xFE), single(NEG_TWO, 0xFE, 0xFE),
                                  single(ONE, 127, 127), single(MAX_PRODUCT, 0xFE, 127)],
        "inf_plus_finite": [single(TWO, 0xFE, 0xFE), single(NEG_ONE, 0xFE, 127)],
        # 2^127 + 2^-149: the widest exponent difference reachable through the public path
        "max_exponent_gap": [single(ONE, 0xFE, 127), single(ONE, 0x00, 105)],
        "sign_zeros": [([(0x80, 0x0), (0x00, 0x8), (0x80, 0x8)], 127, 127)],
        "exact_cancellation": [single(ONE, 140, 127), single(NEG_ONE, 140, 127)],
        # 1 + 2^-8 and 1 + 3 * 2^-8 are exact FP32 results that tie in BF16
        "bf16_tie_down": [([ONE, (0x02, 0x2)], 127, 127)],
        "bf16_tie_up": [([ONE] + [(0x02, 0x2)] * 3, 127, 127)],
    }
    for name, blocks in directed.items():
        cases += blocks_vector(f"rounding.{name}", blocks)
    traces = {name: trace_of(cases[3 * i:3 * i + 3]) for i, name in enumerate(directed)}
    assert traces["tie_down"].fp32_bits == 0x4B800000
    assert traces["tie_up"].fp32_bits == 0x4B800002
    assert traces["carry"].fp32_bits == 0x4B800000
    assert traces["subnormal_to_normal"].fp32_bits == 0x00800000
    assert traces["underflow_negative_zero"].blocks[0].block_bits == 0x80000000
    assert traces["underflow_negative_zero"].fp32_bits == 0x00000000
    assert traces["acc_overflow"].fp32_bits == 0x7F800000
    assert traces["inf_survives_second_scale"].blocks[0].block_bits == 0xFF800000
    nan = traces["nan_from_opposite_inf"]
    assert all((b.acc_bits & 0x7FFFFFFF) > 0x7F800000 for b in nan.blocks[1:])
    assert traces["max_exponent_gap"].blocks[0].acc_bits == 0x7F000000
    assert traces["max_exponent_gap"].blocks[1].block_bits == 0x00000001
    assert traces["scale_subnormal_tie_up"].blocks[0].block_bits == 0x00000002
    assert traces["scale_subnormal_tie_down"].blocks[0].block_bits == 0x00000002
    assert (traces["bf16_tie_down"].bf16_bits, traces["bf16_tie_up"].bf16_bits) == (0x3F80, 0x3F82)
    return cases


def random_cases():
    rng = random.Random(1313)
    cases = []
    for i in range(24):
        k = 32 * rng.randrange(1, 17)
        lo, hi = ((0, 0xFF) if i % 3 == 0 else (110, 145))
        cases += random_vector(f"random.{i}", rng, k, lo, hi)
    return cases


def parity_cases():
    return (exhaustive_cases() + scale_edge_cases() + layout_cases() + order_cases()
            + rounding_cases() + random_cases())


# -------------------------------------------------------------- contract suite


def valid_input(k=64, seed=50):
    rng = random.Random(seed)
    a = [rng.choice(A_CODES) for _ in range(k)]
    w = [rng.randrange(256) for _ in range(k // 2)]
    return a, w, [127] * (k // 32), [126] * (k // 32)


def rejected(name, out, sections, cap=1, overlap=None, **placement):
    op = f"expert_grouped_dot_{out}"
    return ComputeCase(name, op, sections, cap, 4 if out == FP32 else 2,
                       ("err", "invalid_argument"), out, overlap, **placement)


def accepted(name, out, sections, cap=1, **placement):
    expect = expert_grouped_dot(*sections, output=out)
    bits = f.fp32_bits(expect) if out == FP32 else expect
    return ComputeCase(name, f"expert_grouped_dot_{out}", sections, cap, 4 if out == FP32 else 2,
                       ("ok", [bits]), out, **placement)


INPUT_NAMES = ("a", "w", "sa", "sb")


def offset_overlap_cases(out):
    """Partial overlaps in both directions are rejected; exact adjacency on either side is not."""
    sections = valid_input(256, 51)
    width = 4 if out == FP32 else 2
    cases = []
    for index, name in enumerate(INPUT_NAMES):
        size = len(sections[index])
        assert size > width and size % width == 0
        tag = f"{out}.offset_overlap_{name}"
        cases += [
            rejected(f"{tag}.output_inside_input", out, sections, overlap=name,
                     in_offset=0, out_offset=width),
            rejected(f"{tag}.output_on_input_last_byte", out, sections, overlap=name,
                     in_offset=0, out_offset=size - width),
            rejected(f"{tag}.input_inside_output", out, sections, overlap=name,
                     in_offset=width - 1, out_offset=0),
            accepted(f"{tag}.adjacent_after_input", out, sections, overlap=name,
                     in_offset=0, out_offset=size),
            accepted(f"{tag}.adjacent_before_input", out, sections, overlap=name,
                     in_offset=width, out_offset=0),
        ]
    return cases


def contract_cases():
    """Every library rejection class per output; metadata errors use short buffers."""
    cases = []
    a, w, sa, sb = valid_input()
    for out in (FP32, BF16):
        metadata = [
            rejected(f"{out}.k_zero", out, ([], [], [], [])),
            rejected(f"{out}.k_16", out, (a[:16], w[:8], [], [])),
            rejected(f"{out}.k_48", out, (a[:48], w[:24], sa[:1], sb[:1])),
            rejected(f"{out}.k_33", out, (a[:33], w[:16], sa[:1], sb[:1])),
            rejected(f"{out}.short_w_before_read", out, (a, w[:1], sa, sb)),
            rejected(f"{out}.long_w", out, (a, w + [0], sa, sb)),
            rejected(f"{out}.short_scale_a_before_read", out, (a, w, sa[:1], sb)),
            rejected(f"{out}.long_scale_a", out, (a, w, sa + [127], sb)),
            rejected(f"{out}.short_scale_b_before_read", out, (a, w, sa, sb[:1])),
            rejected(f"{out}.long_scale_b", out, (a, w, sa, sb + [127])),
            rejected(f"{out}.empty_output", out, (a, w, sa, sb), cap=0),
        ]
        cases += metadata
        cases += [rejected(c.name + ".poisoned", out, c.sections, c.cap, poison=True)
                  for c in metadata]
        cases += offset_overlap_cases(out)
        for name in ("a", "w", "sa", "sb"):
            cases.append(rejected(f"{out}.overlap_{name}", out, (a, w, sa, sb), cap=2, overlap=name))
        for bad in (0x7F, 0xFF):
            cases.append(rejected(f"{out}.nan_a_last.{bad:x}", out, (a[:-1] + [bad], w, sa, sb), cap=3))
        cases.append(rejected(f"{out}.nan_scale_a_last", out, (a, w, sa[:-1] + [0xFF], sb), cap=3))
        cases.append(rejected(f"{out}.nan_scale_b_last", out, (a, w, sa, sb[:-1] + [0xFF]), cap=3))
        cases.append(accepted(f"{out}.ok_cap_1", out, (a, w, sa, sb)))
    trace_k48 = ComputeCase("trace.k_48", "expert_grouped_dot_trace", (a[:48], w[:24], sa[:1], sb[:1]),
                            2 * 48 + 4 + 1, 4, ("err", "invalid_argument"), FP32)
    trace_nan = ComputeCase("trace.nan_a_last", "expert_grouped_dot_trace", (a[:-1] + [0x7F], w, sa, sb),
                            2 * 64 + 8 + 1, 4, ("err", "invalid_argument"), FP32)
    return cases + [trace_k48, trace_nan]


# ------------------------------------------------------------------- add suite


def add_expect(left, right):
    return f.fp32_bits(_add_fp32(f.fp32_from_bits(left), f.fp32_from_bits(right)))


def add_case(name, pairs):
    lefts, rights = [l for l, _ in pairs], [r for _, r in pairs]
    expect = [add_expect(l, r) for l, r in pairs]
    return ComputeCase(name, "add_fp32", (lefts, rights), len(pairs) + 3, 4, ("ok", expect), FP32)


ADD_DIRECTED = (
    ("exponent_gap", [
        (0x7F000000, 0x00000001), (0x7F7FFFFF, 0x80000001), (0x3F800000, 0x00000001),
        (0x00800000, 0x7F000000), (0x80000001, 0x7F7FFFFF), (0x3F800000, 0x0B800000)]),
    ("ties", [
        (0x3F800000, 0x33800000), (0x3F800001, 0x33800000), (0x4B800000, 0x3F800000),
        (0x4B800001, 0x3F800000), (0x3F800000, 0xB3000000), (0x3F800001, 0xB3800000)]),
    ("tie_neighbors", [
        (0x3F800000, 0x337FFFFF), (0x3F800000, 0x33800001), (0x3F800001, 0x337FFFFF),
        (0x3F800001, 0x33800001), (0x3F800000, 0xB2FFFFFF), (0x3F800000, 0xB3000001)]),
    ("sticky", [
        (0x3F800000, 0x33800001), (0x3F800000, 0xB3000001), (0x3F800001, 0x33800001),
        (0x40000000, 0x34000001), (0x3F800000, 0x32800001), (0x3F800000, 0xB2800001)]),
    ("near_cancellation", [
        (0x3F800000, 0xBF7FFFFF), (0x3F800001, 0xBF800000), (0x3FFFFFFF, 0xBFFFFFFE),
        (0x00800000, 0x807FFFFF), (0x3F800000, 0xBF800001), (0x4B000001, 0xCB000000)]),
    ("carry", [
        (0x3FFFFFFF, 0x33FFFFFF), (0x4B7FFFFF, 0x3F000000), (0x7F7FFFFF, 0x73800000),
        (0x7F7FFFFF, 0x737FFFFF), (0x7F7FFFFF, 0x7F7FFFFF), (0xFF7FFFFF, 0xF3800000)]),
    ("subnormal", [
        (0x00000001, 0x00000001), (0x007FFFFF, 0x00000001), (0x80000001, 0x80000001),
        (0x007FFFFF, 0x807FFFFF), (0x00800000, 0x80000001), (0x00400000, 0x00400000)]),
    ("sign_zero", [
        (0x00000000, 0x00000000), (0x80000000, 0x80000000), (0x00000000, 0x80000000),
        (0x80000000, 0x00000000), (0x3F800000, 0xBF800000), (0x80000000, 0x00000001),
        (0x80000000, 0x80000001), (0xBF800000, 0x3F800000)]),
    ("specials", [
        (0x7F800000, 0xFF800000), (0xFF800000, 0x7F800000), (0x7F800000, 0x7F800000),
        (0xFF800000, 0xFF800000), (0x7F800000, 0xFF7FFFFF), (0x00000000, 0xFF800000),
        (0x7FC00000, 0x3F800000), (0x3F800000, 0xFFC00001), (0x7F800001, 0x7F800000),
        (0xFF800000, 0x7FC00000)]),
)


def add_cases():
    cases = [add_case(f"add.{name}", pairs) for name, pairs in ADD_DIRECTED]
    sticky = cases[3].expect[1]
    assert sticky[0] == 0x3F800001, "1 + (2^-24 + 2^-47) must round up only through the sticky bit"
    assert cases[0].expect[1][0] == 0x7F000000
    rng = random.Random(1331)

    def finite():
        r = rng.random()
        if r < 0.1:
            magnitude = rng.randrange(0x800000)
        elif r < 0.2:
            magnitude = rng.randrange(0x7F000000, 0x7F800000)
        else:
            magnitude = rng.randrange(1, 0x7F800000)
        return magnitude | (rng.randrange(2) << 31)

    pairs = []
    for _ in range(20000):
        left = finite()
        if rng.random() < 0.5:
            exponent = (left >> 23) & 0xFF
            shift = rng.randrange(0, 40)
            right = (max(exponent - shift, 0) << 23) | rng.randrange(0x800000) | (rng.randrange(2) << 31)
        else:
            right = finite()
        pairs.append((left, right))
    cases.append(add_case("add.random", pairs))
    return cases


# ----------------------------------------------------------------- environments


class EnvironmentProbed(unittest.TestCase):
    """Runs one executor per environment, probing the control word before and after."""

    label = "suite"

    def run_probed(self, cases, env, status):
        probe = mxcsr_probe()
        start = time.perf_counter()
        first, *middle, last = run_executor(EXE, [probe, *cases, probe],
                                            [f"--mxcsr={env}", f"--status={status}", *EXE_ARGS])
        flags = MXCSR_STATUS if status == "set" else 0
        for response in (first, last):
            csr = response[1][0]
            self.assertEqual(csr, ENVIRONMENTS[env] | flags,
                             f"MXCSR {csr:#x} does not reflect {env} with status {status}")
        bad = [msg for case, resp in zip(cases, middle) if (msg := check(case, resp))]
        self.assertEqual(bad, [])
        print(f"{self.label}[{env},{status}]: {len(cases)} requests in "
              f"{time.perf_counter() - start:.2f}s", file=sys.stderr)

    def run_all(self, build):
        start = time.perf_counter()
        cases = build()
        print(f"{self.label}: {len(cases)} cases, expected values in "
              f"{time.perf_counter() - start:.2f}s", file=sys.stderr)
        for env in ENVIRONMENTS:
            for status in STATUS_STATES:
                with self.subTest(env=env, status=status):
                    self.run_probed(cases, env, status)


class Parity(EnvironmentProbed):
    label = "parity"

    def test_environments(self):
        self.run_all(parity_cases)


class Contract(EnvironmentProbed):
    label = "contract"

    def test_rejections(self):
        self.run_all(contract_cases)

    def test_oracle_rejects_the_same_inputs(self):
        """ref/ raises ValueError for every content and length class the executor can express.

        Output capacity and overlap are C++-only span concepts.
        """
        a, w, sa, sb = valid_input()
        bad_inputs = [
            ([], [], [], []), (a[:16], w[:8], [], []), (a[:48], w[:24], sa[:1], sb[:1]),
            (a, w[:1], sa, sb), (a, w + [0], sa, sb), (a, w, sa[:1], sb), (a, w, sa, sb + [127]),
            (a[:-1] + [0x7F], w, sa, sb), (a[:-1] + [0xFF], w, sa, sb),
            (a, w, sa[:-1] + [0xFF], sb), (a, w, sa, sb[:-1] + [0xFF]),
        ]
        for inputs in bad_inputs:
            for out in (FP32, BF16):
                with self.subTest(inputs=[len(s) for s in inputs], out=out), self.assertRaises(ValueError):
                    expert_grouped_dot(*inputs, output=out)


class Add(EnvironmentProbed):
    label = "add"

    def test_environments(self):
        self.run_all(add_cases)


def sample_cases():
    a, w, sa, sb = valid_input(32, 60)
    return vector("harness.sample", a, w, sa, sb)


class Harness(unittest.TestCase):
    """Every executor or protocol fault must fail the driver, never pass silently."""

    def test_faults_are_detected(self):
        cases = sample_cases()
        for fault in ("crash", "exit", "garbage", "drop"):
            with self.subTest(fault=fault), self.assertRaises(HarnessError):
                failures(EXE, cases, [f"--fault={fault}"])
        self.assertNotEqual(failures(EXE, cases, ["--fault=mismatch"]), [])

    def test_missing_executor_and_unknown_op(self):
        with self.assertRaises(HarnessError):
            failures(EXE.with_name("df_quant_exec_missing"), sample_cases())
        case = sample_cases()[1]
        unknown = ComputeCase("unknown_output_mode", "expert_grouped_dot_fp16", case.sections,
                              2, 2, case.expect, BF16)
        with self.assertRaises(HarnessError):
            failures(EXE, [unknown])

    def test_too_wide_tokens_die(self):
        a, w, sa, sb = valid_input(32, 61)
        for index in range(4):
            sections = [list(a), list(w), list(sa), list(sb)]
            sections[index][-1] = 0x100
            case = ComputeCase(f"too_wide.{index}", "expert_grouped_dot_fp32", sections, 1, 4,
                               ("ok", [0]), FP32)
            with self.subTest(section=index), self.assertRaises(HarnessError):
                failures(EXE, [case])
        wide = ComputeCase("add_too_wide", "add_fp32", ([0x100000000], [0]), 1, 4, ("ok", [0]), FP32)
        with self.assertRaises(HarnessError):
            failures(EXE, [wide])

    def test_trace_shape_is_checked(self):
        trace = sample_cases()[0]
        short = ComputeCase("trace_short", trace.op, trace.sections, trace.cap,
                            4, ("ok", trace.expect[1][:-1]), FP32)
        self.assertNotEqual(failures(EXE, [short]), [])
        for cap in (trace.cap - 1, trace.cap + 1):
            bad = ComputeCase("trace_cap", trace.op, trace.sections, cap, 4, ("ok", []), FP32)
            with self.subTest(cap=cap), self.assertRaises(HarnessError):
                failures(EXE, [bad])
        with self.assertRaises(HarnessError):
            parse_response(trace, "ok " + " ".join(["0000"] * trace.cap))
        with self.assertRaises(HarnessError):
            parse_response(trace, "ok " + " ".join(["00000000"] * (trace.cap - 1)))

    def test_malformed_requests_die(self):
        valid = sample_cases()[1].line()
        head, payload = valid.split(" : ")
        sections = payload.split(" ; ")
        lines = [
            ("no_colon", head),
            ("three_sections", head + " : " + " ; ".join(sections[:3])),
            ("five_sections", valid + " ; 7f"),
            ("no_cap", head.replace(" cap=2", "") + " : " + payload),
            ("duplicate_cap", head + " cap=2 : " + payload),
            ("shape_field", head + " shape=32 : " + payload),
            ("scale_shape_field", head + " scale_shape=1 : " + payload),
            ("alias_field", head + " alias : " + payload),
            ("unknown_overlap", head + " overlap=out : " + payload),
            ("duplicate_overlap", head + " overlap=a overlap=w : " + payload),
            ("trace_overlap", valid.replace("expert_grouped_dot_fp32 cap=2",
                                            "expert_grouped_dot_trace cap=69 overlap=a")),
            ("add_three_sections", "add_fp32 cap=3 : 0 ; 0 ; 0"),
            ("add_unpaired", "add_fp32 cap=3 : 0 0 ; 0"),
            ("add_short_cap", "add_fp32 cap=1 : 0 0 ; 0 0"),
            ("add_overlap", "add_fp32 cap=1 overlap=a : 0 ; 0"),
            ("dequant_overlap", "dequantize_expert_fp32 shape=1,16 scale_shape=1,1 cap=32 "
                                "overlap=a : " + " ".join(["0"] * 16) + " ; 7f"),
            ("dequant_three_sections", "dequantize_expert_fp32 shape=1,16 scale_shape=1,1 cap=32 : "
                                       + " ".join(["0"] * 16) + " ; 7f ; 7f"),
            ("codec_overlap", "decode_e2m1 shape=1 cap=1 overlap=a : 0"),
            ("multiply_overlap", "multiply_finite shape=1 scale_shape=1 cap=1 overlap=a : 0 ; 0"),
            ("in_offset_without_overlap", head + " in_offset=1 : " + payload),
            ("out_offset_without_overlap", head + " out_offset=4 : " + payload),
            ("misaligned_out_offset", head + " overlap=a out_offset=2 : " + payload),
            ("duplicate_in_offset", head + " overlap=a in_offset=0 in_offset=1 : " + payload),
            ("duplicate_out_offset", head + " overlap=a out_offset=0 out_offset=4 : " + payload),
            ("duplicate_poison", head + " poison poison : " + payload),
            ("poison_with_overlap", head + " overlap=a poison : " + payload),
            ("trace_poison", valid.replace("expert_grouped_dot_fp32 cap=2",
                                           "expert_grouped_dot_trace cap=69 poison")),
            ("add_poison", "add_fp32 cap=1 poison : 0 ; 0"),
            ("codec_poison", "decode_e2m1 shape=1 cap=1 poison : 0"),
            ("dequant_in_offset", "dequantize_expert_fp32 shape=1,16 scale_shape=1,1 cap=32 "
                                  "in_offset=0 : " + " ".join(["0"] * 16) + " ; 7f"),
        ]
        for name, text in lines:
            with self.subTest(case=name), self.assertRaises(HarnessError):
                failures(EXE, [RawCase(name, text)])
        with self.assertRaises(HarnessError):
            failures(EXE, sample_cases(), ["--status=maybe"])

    def test_poison_blocks_reads_under_asan(self):
        """Under ASan a poisoned valid request must die; elsewhere poison is a no-op."""
        asan = run_executor(EXE, [Case("asan", "asan", (), [], 1, 4, ("ok", []))])[0][1][0]
        self.assertIn(asan, (0, 1))
        valid = sample_cases()[1]
        poisoned = ComputeCase("poisoned_valid", valid.op, valid.sections, valid.cap, 4,
                               valid.expect, FP32, poison=True)
        if asan:
            with self.assertRaises(HarnessError):
                failures(EXE, [poisoned])
        else:
            self.assertEqual(failures(EXE, [poisoned]), [])
        print(f"harness: executor ASan={asan}", file=sys.stderr)

    def test_failures_reach_process_exit(self):
        for suite, fault in (("parity", "mismatch"), ("contract", "crash"), ("add", "exit")):
            with self.subTest(suite=suite):
                proc = subprocess.run(
                    [sys.executable, __file__, "--exe", str(EXE), "--suite", suite,
                     f"--exe-arg=--fault={fault}"],
                    capture_output=True, text=True, timeout=1800)
                self.assertNotEqual(proc.returncode, 0)


SUITES = {"parity": Parity, "contract": Contract, "add": Add, "harness": Harness}


def main():
    global EXE
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True, type=Path)
    ap.add_argument("--suite", required=True, choices=sorted(SUITES))
    ap.add_argument("--exe-arg", action="append", default=[],
                    help="extra executor argument; used by the harness suite only")
    args = ap.parse_args()
    EXE = args.exe
    EXE_ARGS.extend(args.exe_arg)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SUITES[args.suite])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() and result.testsRun else 1


if __name__ == "__main__":
    sys.exit(main())
