#!/usr/bin/env python3
"""P0011 C++ parity for the production given-scale dequantization against ref/dequant.py.

Usage: test_dequant_parity.py --exe <quant_exec> --suite <parity|contract|multiply|harness>

Expected values come from ref.dequant (plus ref.formats.multiply_fp32 for the
scale-multiplication helper); the production library is reached only through the
test executor. Comparison, response parsing and the fault path reuse the P0010
driver helpers in test_codec_parity.py: bit-exact, NaN by class only, and any
executor fault fails the run. Group sizes and scale formats follow the ref
oracle (docs/MODEL.md 2.1-2.3) for testing only; they are not a second
production configuration source.

Dequant request protocol, one per line; the response grammar is the P0010 one:
  <op> shape=<d0,d1,...> scale_shape=<d0,d1,...> cap=<n> [alias] [scale_alias]
       : <data hex token> ... ; <scale hex token> ...
- shape is the STORED data shape; packed E2M1 entries store K/2 bytes in the
  last dimension. Data and scale elements are uint8 (uint32 for multiply_finite);
  output elements are uint32 for *_fp32 and uint16 for *_bf16.
- alias overlaps the data buffer with the output, scale_alias the scale buffer;
  both must be rejected. Every field is mandatory; tensor requests carry exactly
  one ':' and one ';' (the P0010 mxcsr probe keeps its scale-free grammar);
  grammar violations, unknown ops and over-wide tokens kill the executor.
- Ops: dequantize_{expert,fp8_weights,engram,swa,main_kv,index_k}_{fp32,bf16}
  and multiply_finite (elementwise uint32 x uint32 -> uint32; scale_shape must
  equal shape). The P0010 `mxcsr` op reports the control word unchanged.
- Rejections follow D0007: std::overflow_error when a stored or logical element
  count, the packed last-dimension doubling (checked even for zero-element
  tensors), or the output byte count (out_count * sizeof(out), checked before
  capacity) exceeds size_t; otherwise leading zero dimensions short-circuit the
  element count to zero and the request is accepted.
  std::invalid_argument covers rank 0, wrong matrix rank, length mismatch, scale
  shape (rank, leading dims and group count must all match exactly), capacity,
  overlap and content (NaN data, NaN or non-positive scales). Metadata is
  validated before any data is read, content before any output is written, and
  failed calls leave the output buffer untouched.
"""

import argparse
import math
import random
import subprocess
import sys
import time
import unittest
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

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

from test_codec_parity import (
    BF16,
    FP32,
    Case,
    HarnessError,
    check,
    failures,
    parse_response,
    run_executor,
)


EXE = None
EXE_ARGS = []


class DequantCase(Case):
    """One dequant request: a stored-shape data tensor plus a separate scale tensor."""

    def __init__(self, name, op, shape, scale_shape, data, scales, cap, out_width,
                 expect, kind, data_alias=False, scale_alias=False):
        super().__init__(name, op, shape, list(data), cap, out_width, expect, kind)
        self.scale_shape = tuple(scale_shape)
        self.scales = list(scales)
        self.data_alias = data_alias
        self.scale_alias = scale_alias

    def line(self):
        head = [self.op, "shape=" + ",".join(map(str, self.shape)),
                "scale_shape=" + ",".join(map(str, self.scale_shape)), f"cap={self.cap}"]
        if self.data_alias:
            head.append("alias")
        if self.scale_alias:
            head.append("scale_alias")
        return (" ".join(head) + " : " + " ".join(f"{t:x}" for t in self.tokens)
                + " ; " + " ".join(f"{t:x}" for t in self.scales))


class RawCase(Case):
    """A deliberate grammar violation; the executor must die instead of answering."""

    def __init__(self, name, text):
        super().__init__(name, "raw", (), [], 1, 4, ("ok", []))
        self.text = text

    def line(self):
        return self.text


@dataclass(frozen=True)
class Entry:
    key: str
    fn: object
    group: int | None    # per-row group size; None selects the 32x32 block path
    packed: bool         # the stored last dimension holds K/2 bytes
    e4m3_scale: bool     # strictly positive E4M3FN scales instead of E8M0
    matrix: bool         # the ref entry requires rank 2


ENTRIES = (
    Entry("expert", dequantize_expert, 32, True, False, True),
    Entry("fp8_weights", dequantize_fp8_weights, None, False, False, True),
    Entry("engram", dequantize_engram, 32, False, False, True),
    Entry("swa", dequantize_swa, 32, False, False, False),
    Entry("main_kv", dequantize_main_kv, 16, True, True, False),
    Entry("index_k", dequantize_index_k, 32, True, False, False),
)
EXPERT, FP8_WEIGHTS, ENGRAM, SWA, MAIN_KV, INDEX_K = ENTRIES

E2M1_CODES = tuple(range(16))
E4M3FN_CODES = tuple(code for code in range(256) if code not in (0x7F, 0xFF))
E8M0_SCALES = tuple(range(0xFF))       # 0x00..0xFE are finite powers of two; 0xFF is NaN
E4M3FN_SCALES = tuple(range(1, 0x7F))  # strictly positive finite E4M3FN codes

HUGE = 1 << 32
MAX_SIZE = (1 << 64) - 1


def op_name(entry, out):
    return f"{entry.fn.__name__}_{out}"


def stored_shape(entry, logical):
    return logical[:-1] + (logical[-1] // 2,) if entry.packed else tuple(logical)


def scale_shape_of(entry, logical):
    if entry.group is None:
        return ((logical[0] + 31) // 32, (logical[1] + 31) // 32)
    return logical[:-1] + (logical[-1] // entry.group,)


def build(name, entry, out, data, logical, scales):
    """Expected bits for one valid request, straight from the ref oracle."""
    data, scales = tuple(data), tuple(scales)
    shape, scale_shape = stored_shape(entry, logical), scale_shape_of(entry, logical)
    result = entry.fn(Tensor(data, shape), Tensor(scales, scale_shape), output=out)
    if out == FP32:
        expect, width = ("ok", [f.fp32_bits(v) for v in result.data]), 4
    else:
        expect, width = ("ok", list(result.data)), 2
    return DequantCase(name, op_name(entry, out), shape, scale_shape, data, scales,
                       len(result.data) + 3, width, expect, out)


def both(name, entry, data, logical, scales):
    return [build(f"{name}.{out}", entry, out, data, logical, scales) for out in (FP32, BF16)]


def rejected(name, entry, out, shape, scale_shape, data, scales, cap,
             err="invalid_argument", data_alias=False, scale_alias=False):
    return DequantCase(name, op_name(entry, out), shape, scale_shape, data, scales, cap,
                       4 if out == FP32 else 2, ("err", err), out, data_alias, scale_alias)


def make_data(entry, logical, seed):
    pool = E2M1_CODES if entry.packed else E4M3FN_CODES
    rng = random.Random(seed)
    codes = [pool[rng.randrange(len(pool))] for _ in range(math.prod(logical))]
    codes[0] = 0
    codes[1] = 8 if entry.packed else 0x80
    return f.pack_e2m1(codes, logical) if entry.packed else codes


def good_scale_code(entry):
    return 0x38 if entry.e4m3_scale else 127


def valid_request(entry, logical, seed):
    data = make_data(entry, logical, seed)
    scales = [good_scale_code(entry)] * math.prod(scale_shape_of(entry, logical))
    return data, scales


def wrong_rank_request(entry, logical, seed):
    tall = (2, 2, logical[-1])
    data = make_data(entry, tall, seed)
    return tall, data, [good_scale_code(entry)] * 4


def wrong_scale_shapes(entry, logical):
    """Same-count shapes with wrong axes, plus one plain shape mismatch."""
    right = scale_shape_of(entry, logical)
    count = math.prod(right)
    wrongs = [(count, 1), (1, count)]
    if right[1] != right[0]:
        wrongs.append((right[1], right[0]))
    return wrongs, right[:-1] + (right[-1] + 1,)


def make_scales(entry, count):
    lo, hi = (1, 0x7E) if entry.e4m3_scale else (0, 0xFE)
    return [lo + i // 2 if i % 2 == 0 else hi - i // 2 for i in range(count)]


# ---------------------------------------------------------------- parity suite


def enumerated_rowwise(entry, logical, scale_codes):
    """Every finite data code under every legal scale code.

    Codes are zero-padded up to a whole number of groups and tiled per row; all
    groups of a row share the row's scale, so padding never removes a pair and
    one row per scale code covers the full data x scale product.
    """
    codes = E2M1_CODES if entry.packed else E4M3FN_CODES
    row = list(codes) + [0] * (-len(codes) % entry.group)
    groups = len(row) // entry.group
    assert logical[-1] == len(row) and math.prod(logical[:-1]) == len(scale_codes)
    rows = math.prod(logical[:-1])
    data = f.pack_e2m1(row * rows, logical) if entry.packed else row * rows
    scales = [code for code in scale_codes for _ in range(groups)]
    return both(f"{entry.key}.enumerate", entry, data, logical, scales)


def enumerated_fp8():
    """Every finite E4M3FN code under every finite E8M0 scale.

    A full combination needs each 32x32 block single-scaled, so one tail block
    row of eight rows tiles all 254 codes into each per-scale block column.
    """
    block = list(E4M3FN_CODES) + [0] * (-len(E4M3FN_CODES) % 256)
    data = []
    for row in range(8):
        for _ in range(len(E8M0_SCALES)):
            data += block[row * 32:(row + 1) * 32]
    return both("fp8_weights.enumerate", FP8_WEIGHTS, data,
                (8, 32 * len(E8M0_SCALES)), list(E8M0_SCALES))


LAYOUTS = (
    (EXPERT, (5, 96)),
    (FP8_WEIGHTS, (35, 67)),
    (FP8_WEIGHTS, (96, 64)),
    (ENGRAM, (7, 160)),
    (SWA, (2, 3, 96)),
    (MAIN_KV, (2, 3, 48)),
    (INDEX_K, (2, 3, 96)),
    (SWA, (96,)),
    (MAIN_KV, (48,)),
    (INDEX_K, (96,)),
    (SWA, (2, 2, 3, 96)),
    (MAIN_KV, (2, 2, 3, 48)),
    (INDEX_K, (2, 2, 3, 96)),
)


def layout_cases():
    """Multi-row multi-group requests with adjacent distinct scales and zeros."""
    cases = []
    for seed, (entry, logical) in enumerate(LAYOUTS, start=100):
        data = make_data(entry, logical, seed)
        scales = make_scales(entry, math.prod(scale_shape_of(entry, logical)))
        cases += both(f"{entry.key}.layout{logical}", entry, data, logical, scales)
    return cases


def zero_cases():
    cases = []
    for entry, cols in ((EXPERT, 32), (FP8_WEIGHTS, 67), (ENGRAM, 32),
                        (SWA, 32), (MAIN_KV, 16), (INDEX_K, 32)):
        cases += both(f"{entry.key}.zero_rows", entry, [], (0, cols), [])
    return cases


def sensitivity_cases():
    """Directed counters for the failure modes MXCSR tricks or tie bugs could hide."""
    cases = []
    # -448 (E4M3FN 0xFE) x 2^127 (E8M0 0xFE) is -inf even under round-up
    cases += both("sensitivity.neg448_overflow", ENGRAM, [0xFE] * 32, (1, 32), [0xFE])
    assert cases[0].expect[1][0] == 0xFF800000, "ref no longer overflows -448 x 2^127 to -inf"
    assert cases[1].expect[1][0] == 0xFF80
    pos = both("sensitivity.pos448_overflow", FP8_WEIGHTS, [0x7E] * 1024, (32, 32), [0xFE])
    assert pos[0].expect[1][0] == 0x7F800000 and pos[0].expect[1][-1] == 0x7F800000
    cases += pos
    neg = both("sensitivity.neg448_overflow_blocks", FP8_WEIGHTS, [0xFE] * 1024, (32, 32), [0xFE])
    assert neg[0].expect[1][0] == 0xFF800000 and neg[0].expect[1][-1] == 0xFF800000
    cases += neg
    # E8M0 0x00 is the subnormal scale 2^-127: products keep subnormal bit patterns
    sub = both("sensitivity.e8m0_zero_subnormal", ENGRAM, list(E4M3FN_CODES[:32]), (1, 32), [0x00])
    assert sub[0].expect[1][1] == 0x00002000  # 2^-9 x 2^-127
    cases += sub
    packed = both("sensitivity.e8m0_zero_packed", EXPERT,
                  f.pack_e2m1(list(E2M1_CODES) + [0] * 16, (32,)), (1, 32), [0x00])
    assert packed[0].expect[1][3] == 0x00600000  # 1.5 x 2^-127
    assert packed[1].expect[1][3] == 0x0060      # exactly representable in BF16, not a tie
    cases += packed
    # BF16 subnormal ties: 2^-9 and 3x2^-9 times 2^-125 hit 0x00008000 and 0x00018000
    ties = both("sensitivity.bf16_subnormal_tie", ENGRAM, [0x01, 0x03] * 16, (1, 32), [0x02])
    assert ties[0].expect[1][0] == 0x00008000 and ties[0].expect[1][1] == 0x00018000
    assert ties[1].expect[1][0] == 0x0000 and ties[1].expect[1][1] == 0x0002
    cases += ties
    nbr = both("sensitivity.bf16_subnormal_neighbors", ENGRAM, [0x01] * 64, (2, 32), [0x01, 0x03])
    assert nbr[0].expect[1][0] == 0x00004000 and nbr[0].expect[1][32] == 0x00010000
    assert nbr[1].expect[1][0] == 0x0000 and nbr[1].expect[1][32] == 0x0001
    cases += nbr
    return cases


def parity_cases():
    cases = []
    cases += enumerated_rowwise(EXPERT, (255, 32), E8M0_SCALES)
    cases += enumerated_fp8()
    cases += enumerated_rowwise(ENGRAM, (255, 256), E8M0_SCALES)
    cases += enumerated_rowwise(SWA, (3, 85, 256), E8M0_SCALES)
    cases += enumerated_rowwise(MAIN_KV, (126, 16), E4M3FN_SCALES)
    cases += enumerated_rowwise(INDEX_K, (255, 32), E8M0_SCALES)
    cases += layout_cases()
    cases += zero_cases()
    cases += sensitivity_cases()
    return cases


# -------------------------------------------------------------- contract suite


def overflow_cases(entry, out, tag):
    """size_t arithmetic that must fail before any length or capacity check."""
    cases = []
    big = (HUGE, HUGE)
    big_scales = scale_shape_of(entry, (HUGE, HUGE * (2 if entry.packed else 1)))
    cases.append(rejected(f"{tag}.element_overflow", entry, out, big, big_scales,
                          [], [], 0, "overflow_error"))
    if entry.packed:
        rows = 1 << 58 if entry.group == 32 else 1 << 59
        cases.append(rejected(f"{tag}.packed_double_overflow", entry, out, (rows, entry.group),
                              (rows, 2), [], [], 0, "overflow_error"))
        cases.append(rejected(f"{tag}.packed_last_dim_overflow", entry, out, (0, 1 << 63),
                              (0, 1 << 59), [], [], 0, "overflow_error"))
    rows, cols = 1 << 31, (1 << 31) if out == FP32 else (1 << 32)
    if entry.packed:
        byte_shape, byte_scales = (rows, cols // 2), (rows, cols // entry.group)
    elif entry.group is None:
        byte_shape, byte_scales = (rows, cols), ((rows + 31) // 32, (cols + 31) // 32)
    else:
        byte_shape, byte_scales = (rows, cols), (rows, cols // entry.group)
    cases.append(rejected(f"{tag}.output_bytes_overflow", entry, out, byte_shape, byte_scales,
                          [], [], 0, "overflow_error"))
    if entry.group is None:
        # maximal rows: the block ceil itself stays safe, the output bytes overflow
        cases.append(rejected(f"{tag}.max_rows_overflow", entry, out, (MAX_SIZE, 1),
                              ((MAX_SIZE + 31) // 32, 1), [], [], 0, "overflow_error"))
    return cases


def empty_ok(name, entry, out, logical):
    return DequantCase(name, op_name(entry, out), stored_shape(entry, logical),
                       scale_shape_of(entry, logical), [], [], 2,
                       4 if out == FP32 else 2, ("ok", []), out)


def empty_cases(entry, out, tag):
    cols = entry.group or 32
    empties = [(0, cols), (2, 0), (1 << 62, 0)]
    if not entry.matrix:
        empties.append((0,))
        empties.append((1 << 63, 1 << 63, 0, cols))  # huge leading dims, legal last dim
    if entry.group is None:
        empties.append((0, MAX_SIZE))                # zero rows tolerate maximal columns
    return [empty_ok(f"{tag}.empty{logical}", entry, out, logical) for logical in empties]


def content_cases(entry, out, tag, shape, scale_shape, data, scales, n_out):
    """The bad value sits in the last element; the output must stay untouched.

    Packed E2M1 data has no rejectable code and E8M0/E4M3FN scales cannot encode
    Inf, so those C++-impossible classes are not fabricated here.
    """
    cases = []
    if not entry.packed:
        for bad in (0x7F, 0xFF):
            cases.append(rejected(f"{tag}.nan_data_last.{bad:x}", entry, out, shape, scale_shape,
                                  list(data[:-1]) + [bad], scales, n_out))
    bad_scales = (0x00, 0x7F, 0x80, 0x81, 0xFF) if entry.e4m3_scale else (0xFF,)
    for bad in bad_scales:
        cases.append(rejected(f"{tag}.bad_scale_last.{bad:x}", entry, out, shape, scale_shape,
                              data, list(scales[:-1]) + [bad], n_out))
    return cases


def contract_cases():
    """Every rejection class per entry and output; metadata errors use short buffers."""
    cases = []
    for entry in ENTRIES:
        for out in (FP32, BF16):
            tag = f"{entry.key}.{out}"
            logical = (2, entry.group or 32)
            data, scales = valid_request(entry, logical, 50)
            shape, scale_shape = stored_shape(entry, logical), scale_shape_of(entry, logical)
            n_data, n_scale, n_out = len(data), len(scales), math.prod(logical)
            good = scales[0]
            cases += [
                rejected(f"{tag}.rank0", entry, out, (), (1,), [], [good], n_out),
                rejected(f"{tag}.short_data", entry, out, shape, scale_shape,
                         data[:-1], scales, n_out),
                rejected(f"{tag}.long_data", entry, out, shape, scale_shape,
                         list(data) + [0], scales, n_out),
                rejected(f"{tag}.short_scale", entry, out, shape, scale_shape,
                         data, scales[:-1], n_out),
                rejected(f"{tag}.long_scale", entry, out, shape, scale_shape,
                         data, list(scales) + [good], n_out),
                rejected(f"{tag}.short_output", entry, out, shape, scale_shape,
                         data, scales, n_out - 1),
                rejected(f"{tag}.data_alias", entry, out, shape, scale_shape,
                         data, scales, n_out, data_alias=True),
                rejected(f"{tag}.scale_alias", entry, out, shape, scale_shape,
                         data, scales, n_out, scale_alias=True),
                rejected(f"{tag}.scale_rank0_before_read", entry, out, shape, (),
                         data[:1], [], n_out),
                rejected(f"{tag}.scale_rank_extra_before_read", entry, out, shape, (1,) + scale_shape,
                         data[:1], scales[:1], n_out),
            ]
            if entry.matrix:
                tall, tall_data, tall_scales = wrong_rank_request(entry, logical, 51)
                cases.append(rejected(f"{tag}.wrong_rank", entry, out, stored_shape(entry, tall),
                                      (2, 2, 1), tall_data, tall_scales, math.prod(tall)))
            if entry.group is not None:
                bad_shape = (1, entry.group // 4) if entry.packed else (1, entry.group + 1)
                cases.append(rejected(f"{tag}.indivisible_group", entry, out, bad_shape, (1, 1),
                                      [0], [good], entry.group))
            wide = (35, 67) if entry.group is None else (2, 64)
            wide_data, wide_scales = valid_request(entry, wide, 52)
            wrongs, mismatch = wrong_scale_shapes(entry, wide)
            for wrong in wrongs:
                cases.append(rejected(f"{tag}.scale_axes{wrong}", entry, out,
                                      stored_shape(entry, wide), wrong,
                                      wide_data, wide_scales, math.prod(wide)))
                cases.append(rejected(f"{tag}.scale_axes_before_read{wrong}", entry, out,
                                      stored_shape(entry, wide), wrong,
                                      wide_data[:1], wide_scales[:1], math.prod(wide)))
            cases.append(rejected(f"{tag}.scale_shape_mismatch", entry, out,
                                  stored_shape(entry, wide), mismatch, wide_data,
                                  [good] * math.prod(mismatch), math.prod(wide)))
            cases += overflow_cases(entry, out, tag)
            cases += empty_cases(entry, out, tag)
            cases += content_cases(entry, out, tag, shape, scale_shape, data, scales, n_out)
    return cases


# -------------------------------------------------------------- multiply suite


def multiply_case(name, lefts, rights):
    lefts, rights = list(lefts), list(rights)
    expect = [f.fp32_bits(f.multiply_fp32(f.fp32_from_bits(l), f.fp32_from_bits(r)))
              for l, r in zip(lefts, rights)]
    return DequantCase(name, "multiply_finite", (len(lefts),), (len(lefts),),
                       lefts, rights, len(lefts) + 3, 4, ("ok", expect), FP32)


MULTIPLY_DIRECTED = (
    ("exact", [
        (0x3F800000, 0x3F800000), (0x3FC00000, 0x40000000), (0x40400000, 0x3E800000),
        (0x4B000000, 0x33000000), (0x3F000000, 0x40400000)]),
    ("tie_rounds_up", [(0x3FC00000, 0x3F800001), (0x00000003, 0x3F000000)]),
    ("tie_rounds_down", [(0x3FA00000, 0x3F800002), (0x00000001, 0x3F000000)]),
    ("neighbors", [
        (0x3FC00000, 0x3F800000), (0x3FC00000, 0x3F800002), (0x3FBFFFFF, 0x3F800001),
        (0x3FC00001, 0x3F800001), (0x3FA00000, 0x3F800001), (0x3FA00000, 0x3F800003),
        (0x3F9FFFFF, 0x3F800002), (0x3FA00001, 0x3F800002), (0x00000002, 0x3F000000),
        (0x00000001, 0x3F000001)]),
    ("carry_top", [
        (0x3F900001, 0x3FE38E37), (0x7F100001, 0x3FE38E37),
        (0x7F7FFFFF, 0x3F800001), (0x7F7FFFFF, 0x3F7FFFFF), (0x7F7FFFFF, 0x3F800000),
        (0x3FFFFFFF, 0x3FFFFFFF), (0x3F800001, 0x3F800001)]),
    ("underflow_to_zero", [
        (0x00000001, 0x3F000000), (0x80000001, 0x3F000000), (0x00000001, 0x3EFFFFFF),
        (0x00000001, 0x00000001), (0x80000001, 0x80000001)]),
    ("normal_subnormal", [
        (0x007FFFFF, 0x40000000), (0x00800000, 0x3F000000), (0x00800000, 0x3F000001),
        (0x007FFFFF, 0x3F800001), (0x00FFFFFF, 0x3F000000),
        (0x007FFFFF, 0x007FFFFF), (0x00000001, 0x7F7FFFFF)]),
    ("overflow", [
        (0x7F7FFFFF, 0x40000000), (0xFF7FFFFF, 0x40000000), (0xFF7FFFFF, 0xBF800000),
        (0x7F7FFFFF, 0x7F7FFFFF)]),
    ("sign_zero", [
        (0x00000000, 0x7F7FFFFF), (0x80000000, 0x7F7FFFFF), (0x80000000, 0xFF7FFFFF),
        (0x00000000, 0x80000000), (0x80000000, 0x3F800000), (0x80000000, 0x80000000)]),
)


def multiply_cases():
    cases = [multiply_case(f"multiply.{name}", [l for l, _ in pairs], [r for _, r in pairs])
             for name, pairs in MULTIPLY_DIRECTED]
    up, down = cases[1].expect[1], cases[2].expect[1]
    assert up[0] == 0x3FC00002 and up[1] == 0x00000002, "tie inputs no longer round up to even"
    assert down[0] == 0x3FA00002 and down[1] == 0x00000000, "tie inputs no longer round down to even"
    assert cases[4].expect[1][:2] == [0x40000000, 0x7F800000], "rounding carry anchors changed"
    rng = random.Random(21)

    def finite():
        magnitude = rng.randrange(0x800000) if rng.random() < 0.15 else rng.randrange(1, 0x7F800000)
        return magnitude | (rng.randrange(2) << 31)

    pairs = [(finite(), finite()) for _ in range(20000)]
    cases.append(multiply_case("multiply.random", [l for l, _ in pairs], [r for _, r in pairs]))
    return cases


def multiply_contract_cases():
    data = [0x3F800000] * 4

    def rejected_mul(name, shape, scale_shape, d, s, cap, err="invalid_argument",
                     data_alias=False, scale_alias=False):
        return DequantCase(name, "multiply_finite", shape, scale_shape, d, s, cap,
                           4, ("err", err), FP32, data_alias, scale_alias)

    return [
        rejected_mul("multiply.rank0", (), (), [], [], 4),
        rejected_mul("multiply.short_data", (4,), (4,), data[:-1], data, 4),
        rejected_mul("multiply.long_data", (4,), (4,), list(data) + [0], data, 4),
        rejected_mul("multiply.short_scale", (4,), (4,), data, data[:-1], 4),
        rejected_mul("multiply.long_scale", (4,), (4,), data, list(data) + [0], 4),
        rejected_mul("multiply.scale_axes", (4,), (2, 2), data, data, 4),
        rejected_mul("multiply.short_output", (4,), (4,), data, data, 3),
        rejected_mul("multiply.data_alias", (4,), (4,), data, data, 4, data_alias=True),
        rejected_mul("multiply.scale_alias", (4,), (4,), data, data, 4, scale_alias=True),
        rejected_mul("multiply.overflow", (HUGE, HUGE), (HUGE, HUGE), [], [], 0, "overflow_error"),
        DequantCase("multiply.zero_ok", "multiply_finite", (0,), (0,), [], [], 2, 4,
                    ("ok", []), FP32),
    ]


# ----------------------------------------------------------------- environments

MXCSR_STATUS, MXCSR_FTZ, MXCSR_DAZ, MXCSR_OVERFLOW = 0x003F, 0x8000, 0x0040, 0x0400
BASE_CONTROL = 0x1F80  # RNE, FTZ/DAZ off, all exception masks set
# environment -> the full control word (status flags excluded) the probes must report
ENVIRONMENTS = {
    "default": BASE_CONTROL,
    "up": BASE_CONTROL | 0x4000,
    "down": BASE_CONTROL | 0x2000,
    "zero": BASE_CONTROL | 0x6000,
    "ftz": BASE_CONTROL | MXCSR_FTZ,
    "daz": BASE_CONTROL | MXCSR_DAZ,
    "ftz_daz": BASE_CONTROL | MXCSR_FTZ | MXCSR_DAZ,
    "overflow_unmasked": BASE_CONTROL & ~MXCSR_OVERFLOW,
}


def mxcsr_probe():
    return Case("mxcsr", "mxcsr", (1,), [], 1, 4, ("ok", []))


class EnvironmentProbed(unittest.TestCase):
    """Runs one executor per environment, probing the control word before and after."""

    label = "suite"

    def run_probed(self, cases, env):
        probe = mxcsr_probe()
        start = time.perf_counter()
        first, *middle, last = run_executor(EXE, [probe, *cases, probe],
                                            [f"--mxcsr={env}", *EXE_ARGS])
        for response in (first, last):
            csr = response[1][0]
            self.assertEqual(csr & ~MXCSR_STATUS, ENVIRONMENTS[env],
                             f"MXCSR {csr:#x} does not reflect {env}")
        bad = [msg for case, resp in zip(cases, middle) if (msg := check(case, resp))]
        self.assertEqual(bad, [])
        print(f"{self.label}[{env}]: {len(cases)} requests in "
              f"{time.perf_counter() - start:.2f}s", file=sys.stderr)


class Parity(EnvironmentProbed):
    label = "parity"

    def test_environments(self):
        start = time.perf_counter()
        cases = parity_cases()
        print(f"parity: {len(cases)} cases, expected values in "
              f"{time.perf_counter() - start:.2f}s", file=sys.stderr)
        for env in ENVIRONMENTS:
            with self.subTest(env=env):
                self.run_probed(cases, env)


class Contract(unittest.TestCase):
    def test_rejections(self):
        start = time.perf_counter()
        cases = contract_cases()
        self.assertEqual(failures(EXE, cases, EXE_ARGS), [])
        print(f"contract: {len(cases)} requests in {time.perf_counter() - start:.2f}s",
              file=sys.stderr)

    def test_oracle_rejects_the_same_cases(self):
        """ref/ mirrors every rejection class the executor can express.

        Capacity, overlap and size_t arithmetic have no Python counterpart:
        span concepts are C++-only and Python integers cannot overflow.
        """
        with self.assertRaises(ValueError):
            Tensor((), ())
        for entry in ENTRIES:
            logical = (2, entry.group or 32)
            data, scales = valid_request(entry, logical, 50)
            shape, scale_shape = stored_shape(entry, logical), scale_shape_of(entry, logical)
            with self.assertRaises(ValueError):
                Tensor(tuple(data[:-1]), shape)
            with self.assertRaises(ValueError):
                Tensor(tuple(scales[:-1]), scale_shape)
            if entry.matrix:
                tall, tall_data, _ = wrong_rank_request(entry, logical, 51)
                with self.assertRaises(ValueError):
                    entry.fn(Tensor(tuple(tall_data), stored_shape(entry, tall)),
                             Tensor((scales[0],) * 4, (2, 2, 1)))
            if entry.group is not None:
                bad = (1, entry.group // 4) if entry.packed else (1, entry.group + 1)
                with self.assertRaises(ValueError):
                    entry.fn(Tensor((0,), bad), Tensor((scales[0],), (1, 1)))
            wide = (35, 67) if entry.group is None else (2, 64)
            wide_data, wide_scales = valid_request(entry, wide, 52)
            wide_shape = stored_shape(entry, wide)
            wrongs, mismatch = wrong_scale_shapes(entry, wide)
            for wrong in wrongs:
                with self.assertRaises(ValueError):
                    entry.fn(Tensor(tuple(wide_data), wide_shape), Tensor(tuple(wide_scales), wrong))
            with self.assertRaises(ValueError):
                entry.fn(Tensor(tuple(wide_data), wide_shape),
                         Tensor(tuple([scales[0]] * math.prod(mismatch)), mismatch))
            if not entry.packed:
                for bad_code in (0x7F, 0xFF):
                    with self.assertRaises(ValueError):
                        entry.fn(Tensor(tuple(list(data[:-1]) + [bad_code]), shape),
                                 Tensor(tuple(scales), scale_shape))
            for bad_code in ((0x00, 0x7F, 0x80, 0x81, 0xFF) if entry.e4m3_scale else (0xFF,)):
                with self.assertRaises(ValueError):
                    entry.fn(Tensor(tuple(data), shape),
                             Tensor(tuple(list(scales[:-1]) + [bad_code]), scale_shape))
            empty_logical = (0, logical[-1])
            empty = entry.fn(Tensor((), stored_shape(entry, empty_logical)),
                             Tensor((), scale_shape_of(entry, empty_logical)))
            self.assertEqual(len(empty.data), 0)


class Multiply(EnvironmentProbed):
    label = "multiply"

    def test_environments(self):
        start = time.perf_counter()
        cases = multiply_cases()
        print(f"multiply: {len(cases)} cases, expected values in "
              f"{time.perf_counter() - start:.2f}s", file=sys.stderr)
        for env in ENVIRONMENTS:
            with self.subTest(env=env):
                self.run_probed(cases, env)

    def test_empty(self):
        self.assertEqual(failures(EXE, [multiply_contract_cases()[-1]], EXE_ARGS), [])


def sample_cases():
    logical = (2, MAIN_KV.group)
    data, scales = valid_request(MAIN_KV, logical, 60)
    return both("harness.sample", MAIN_KV, data, logical, scales)


class Harness(unittest.TestCase):
    """Every executor or protocol fault must fail the driver, never pass silently."""

    def test_private_multiply_bad_requests_are_protocol_faults(self):
        for case in multiply_contract_cases()[:-1]:
            with self.subTest(case=case.name), self.assertRaises(HarnessError):
                failures(EXE, [case])
        for value in (0x7F800000, 0xFF800000, 0x7FC00000, 0x100000000):
            case = DequantCase("non_finite_or_wide", "multiply_finite", (1,), (1,),
                               [value], [0x3F800000], 1, 4, ("ok", []), FP32)
            with self.subTest(value=value), self.assertRaises(HarnessError):
                failures(EXE, [case])

    def test_faults_are_detected(self):
        cases = sample_cases()
        for fault in ("crash", "exit", "garbage", "drop"):
            with self.subTest(fault=fault), self.assertRaises(HarnessError):
                failures(EXE, cases, [f"--fault={fault}"])
        self.assertNotEqual(failures(EXE, cases, ["--fault=mismatch"]), [])

    def test_missing_executor_and_unknown_op(self):
        with self.assertRaises(HarnessError):
            failures(EXE.with_name("df_quant_exec_missing"), sample_cases())
        unknown = DequantCase("unknown_op", "dequantize_nothing_fp32", (1, 16), (1, 1),
                              [0], [127], 32, 4, ("ok", [0] * 32), FP32)
        with self.assertRaises(HarnessError):
            failures(EXE, [unknown])

    def test_too_wide_tokens_die(self):
        cases = [
            DequantCase("data_too_wide", op_name(EXPERT, FP32), (1, 16), (1, 1),
                        [0x100], [127], 32, 4, ("ok", []), FP32),
            DequantCase("scale_too_wide", op_name(EXPERT, FP32), (1, 16), (1, 1),
                        [0] * 16, [0x100], 32, 4, ("ok", []), FP32),
        ]
        for case in cases:
            with self.subTest(case=case.name), self.assertRaises(HarnessError):
                failures(EXE, [case])

    def test_malformed_requests_die(self):
        lines = [
            ("no_colon", "dequantize_expert_fp32 shape=2,16 scale_shape=2,1 cap=40"),
            ("no_scale_separator", "dequantize_expert_fp32 shape=2,16 scale_shape=2,1 cap=40 : 00 00"),
            ("no_cap", "dequantize_expert_fp32 shape=2,16 scale_shape=2,1 : 00 00 ; 7f"),
            ("no_shape", "dequantize_expert_fp32 scale_shape=2,1 cap=40 : 00 00 ; 7f"),
            ("no_scale_shape", "dequantize_expert_fp32 shape=2,16 cap=40 : 00 00 ; 7f"),
            ("unknown_field", "dequantize_expert_fp32 shape=2,16 scale_shape=2,1 cap=40 stride=1 : 00 00 ; 7f"),
        ]
        valid = sample_cases()[0].line()
        for malformed in ("2,,8", "2,8,", ",2,8"):
            lines.append((f"empty_shape_component.{malformed}",
                          valid.replace("shape=2,8 ", f"shape={malformed} ")))
        lines += [
            ("empty_scale_shape_component", valid.replace("scale_shape=2,1", "scale_shape=2,,1")),
            ("duplicate_shape", valid.replace("cap=", "shape=2,8 cap=")),
            ("duplicate_scale_shape", valid.replace("cap=", "scale_shape=2,1 cap=")),
            ("duplicate_cap", valid.replace("cap=", "cap=0 cap=")),
            ("duplicate_alias", valid.replace(" : ", " alias alias : ")),
            ("duplicate_scale_alias", valid.replace(" : ", " scale_alias scale_alias : ")),
            ("double_alias", valid.replace(" : ", " alias scale_alias : ")),
        ]
        for name, text in lines:
            with self.subTest(case=name), self.assertRaises(HarnessError):
                failures(EXE, [RawCase(name, text)])

    def test_malformed_tokens_are_protocol_errors(self):
        fp32 = DequantCase("fp32", op_name(ENGRAM, FP32), (1,), (1,), [0], [127], 1, 4,
                           ("ok", [0]), FP32)
        bf16 = DequantCase("bf16", op_name(ENGRAM, BF16), (1,), (1,), [0], [127], 1, 2,
                           ("ok", [0]), BF16)
        for case, tokens in ((fp32, ("-0000001", "+0000001", "0x000001", "0000_001",
                                     "7FC00000", "7fc0000", "7fc000000")),
                             (bf16, ("-001", "+001", "0x01", "0_01", "7FC0", "7fc", "7fc00"))):
            for token in tokens:
                with self.subTest(case=case.name, token=token), self.assertRaises(HarnessError):
                    parse_response(case, "ok " + token)

    def test_failures_reach_process_exit(self):
        for suite, fault in (("parity", "mismatch"), ("contract", "crash"),
                             ("multiply", "exit")):
            with self.subTest(suite=suite):
                proc = subprocess.run(
                    [sys.executable, __file__, "--exe", str(EXE), "--suite", suite,
                     f"--exe-arg=--fault={fault}"],
                    capture_output=True, text=True, timeout=1800)
                self.assertNotEqual(proc.returncode, 0)


SUITES = {"parity": Parity, "contract": Contract, "multiply": Multiply, "harness": Harness}


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
