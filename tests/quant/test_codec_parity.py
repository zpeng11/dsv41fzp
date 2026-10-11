#!/usr/bin/env python3
"""P0010 C++ parity for the production quant codec against the ref/ format oracle.

Usage: test_codec_parity.py --exe <quant_exec> --suite <parity>

Expected values come from ref/formats.py; the production library is reached only
through the test executor (tests/quant/quant_exec.cpp). Results are compared as bit
patterns; NaN compares by class only. Any executor fault fails the run.
"""

import argparse
import random
import subprocess
import sys
import time
import unittest
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ref import formats as f


FILL_BYTE = 0xA5
FP32, BF16, RAW = "fp32", "bf16", "raw"
EXE = None


class HarnessError(Exception):
    """The executor or the protocol failed; never a library verdict."""


@dataclass
class Case:
    name: str
    op: str
    shape: tuple
    tokens: list
    cap: int
    out_width: int
    expect: tuple
    kind: str = RAW
    alias: bool = False

    def line(self):
        head = [self.op, "shape=" + ",".join(map(str, self.shape)), f"cap={self.cap}"]
        if self.alias:
            head.append("alias")
        return " ".join(head) + " : " + " ".join(f"{t:x}" for t in self.tokens)


def fill_value(width):
    return int.from_bytes(bytes([FILL_BYTE]) * width, "little")


def is_nan(bits, kind):
    if kind == FP32:
        return (bits & 0x7FFFFFFF) > 0x7F800000
    if kind == BF16:
        return (bits & 0x7FFF) > 0x7F80
    return False


def run_executor(exe, cases, args=()):
    stdin = "".join(case.line() + "\n" for case in cases)
    try:
        proc = subprocess.run([str(exe), *args], input=stdin, capture_output=True,
                              text=True, timeout=900)
    except OSError as exc:
        raise HarnessError(f"cannot launch executor {exe}: {exc}") from exc
    if proc.returncode != 0:
        raise HarnessError(f"executor exited {proc.returncode}: {proc.stderr.strip()[-2000:]}")
    lines = proc.stdout.splitlines()
    if len(lines) != len(cases):
        raise HarnessError(f"expected {len(cases)} responses, got {len(lines)}")
    return [parse_response(case, line) for case, line in zip(cases, lines)]


def parse_response(case, line):
    parts = line.split(" ")
    if parts[0] == "ok":
        values = parts[1:]
        if len(values) != case.cap or any(len(v) != 2 * case.out_width for v in values):
            raise HarnessError(f"{case.name}: malformed ok response")
        try:
            return ("ok", [int(v, 16) for v in values])
        except ValueError as exc:
            raise HarnessError(f"{case.name}: non-hex output") from exc
    if (parts[0] == "err" and len(parts) == 3
            and parts[1] in ("invalid_argument", "overflow_error")
            and parts[2] in ("changed=0", "changed=1")):
        return ("err", parts[1], parts[2] == "changed=1")
    raise HarnessError(f"{case.name}: unparseable response {line[:80]!r}")


def check(case, response):
    """Return a failure description, or None when the response matches."""
    if case.expect[0] == "err":
        if response[0] != "err":
            return f"{case.name}: expected {case.expect[1]}, got ok"
        if response[1] != case.expect[1]:
            return f"{case.name}: expected {case.expect[1]}, got {response[1]}"
        if response[2]:
            return f"{case.name}: output changed on failure"
        return None
    if response[0] != "ok":
        return f"{case.name}: expected ok, got {response[1]}"
    want, got = case.expect[1], response[1]
    for i, (w, g) in enumerate(zip(want, got)):
        if w != g and not (is_nan(w, case.kind) and is_nan(g, case.kind)):
            return f"{case.name}[{i}]: expected {w:#x}, got {g:#x}"
    fill = fill_value(case.out_width)
    if any(g != fill for g in got[len(want):]):
        return f"{case.name}: wrote past the result prefix"
    return None


def elementwise(name, op, shape, tokens, expected, out_width, kind, slack=3):
    return Case(name, op, shape, list(tokens), len(expected) + slack, out_width,
                ("ok", list(expected)), kind)


def decoded(fn, codes):
    return [f.fp32_bits(fn(c)) for c in codes]


def bf16_rounding_inputs():
    """Midpoints of every adjacent finite BF16 pair and their FP32 neighbors, plus specials."""
    values = []
    for sign in (0, 0x80000000):
        for b in range(0x7F80):
            mid = sign | (b << 16) | 0x8000
            values += [mid - 1, mid, mid + 1]
    values += [
        0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x007FFFFF, 0x807FFFFF,
        0x00800000, 0x0080FFFF, 0x7F7FFFFF, 0xFF7FFFFF, 0x7F7F7FFF, 0x7F800000,
        0xFF800000, 0x7F800001, 0x7FC00000, 0x7FFFFFFF, 0xFF800001, 0xFFFFFFFF,
    ]
    rng = random.Random(10)
    values += [rng.getrandbits(32) for _ in range(20000)]
    return values


def parity_cases():
    nibbles = range(16)
    bytes_ = range(256)
    bf16 = range(1 << 16)
    rounding = bf16_rounding_inputs()
    byte_nibbles = [n for b in bytes_ for n in (b & 15, b >> 4)]
    rng = random.Random(11)
    rows = [rng.randrange(16) for _ in range(3 * 2 * 6)]
    grid_points = sorted({f.e2m1_to_grid(c) for c in nibbles})
    return [
        elementwise("e2m1.all", "decode_e2m1", (4, 4), nibbles,
                    decoded(f.decode_e2m1, nibbles), 4, FP32),
        elementwise("e4m3fn.all", "decode_e4m3fn", (16, 16), bytes_,
                    decoded(f.decode_e4m3fn, bytes_), 4, FP32),
        elementwise("e8m0.all", "decode_e8m0", (256,), bytes_,
                    decoded(f.decode_e8m0, bytes_), 4, FP32),
        elementwise("bf16.all", "decode_bf16", (256, 256), bf16,
                    decoded(f.decode_bf16, bf16), 4, FP32),
        elementwise("fp32_to_bf16.rounding", "fp32_to_bf16", (len(rounding),), rounding,
                    [f.fp32_bits_to_bf16(b) for b in rounding], 2, BF16),
        elementwise("pack.all_bytes", "pack_e2m1", (4, 128), byte_nibbles,
                    f.pack_e2m1(byte_nibbles, (4, 128)), 1, RAW),
        elementwise("unpack.all_bytes", "unpack_e2m1", (4, 128), bytes_,
                    f.unpack_e2m1(bytes(bytes_), (4, 128)), 1, RAW),
        elementwise("pack.rows", "pack_e2m1", (3, 2, 6), rows,
                    f.pack_e2m1(rows, (3, 2, 6)), 1, RAW),
        elementwise("grid.from_e2m1", "e2m1_to_grid", (16,), nibbles,
                    [f.e2m1_to_grid(c) & 0xFF for c in nibbles], 1, RAW),
        elementwise("grid.to_e2m1", "grid_to_e2m1", (len(grid_points),),
                    [v & 0xFF for v in grid_points],
                    [f.grid_to_e2m1(v) for v in grid_points], 1, RAW),
    ]


# op -> (input width, output width, packing direction or None)
OPS = {
    "decode_e2m1": (1, 4, None),
    "decode_e4m3fn": (1, 4, None),
    "decode_e8m0": (1, 4, None),
    "decode_bf16": (2, 4, None),
    "fp32_to_bf16": (4, 2, None),
    "e2m1_to_grid": (1, 1, None),
    "grid_to_e2m1": (1, 1, None),
    "pack_e2m1": (1, 1, "pack"),
    "unpack_e2m1": (1, 1, "unpack"),
}
NON_GRID = (5, -5, 7, -7, 9, -9, 10, -10, 11, -11, 13, -13, 127, -127, -128)
FORMAT_CONTRACT_VERSION = 1


def lengths(op, count):
    """Input length and output need for a valid request with `count` logical elements."""
    packing = OPS[op][2]
    if packing == "pack":
        return count, count // 2
    if packing == "unpack":
        return count // 2, count
    return count, count


def rejected(name, op, shape, tokens, cap, kind="invalid_argument", alias=False):
    return Case(name, op, shape, list(tokens), cap, OPS[op][1], ("err", kind), alias=alias)


def contract_cases():
    """Every rejection class per op; metadata errors use buffers shorter than declared."""
    cases = []
    for op in OPS:
        n_in, n_out = lengths(op, 8)
        cases += [
            rejected(f"{op}.rank0", op, (), [], 4),
            rejected(f"{op}.short_input", op, (2, 4), [0] * (n_in - 1), n_out),
            rejected(f"{op}.long_input", op, (2, 4), [0] * (n_in + 1), n_out),
            rejected(f"{op}.short_output", op, (2, 4), [0] * n_in, n_out - 1),
            rejected(f"{op}.overflow", op, (1 << 32, 1 << 32), [], 0, "overflow_error"),
            rejected(f"{op}.overflow_3d", op, (1 << 21, 1 << 21, 1 << 22), [], 0, "overflow_error"),
            rejected(f"{op}.alias", op, (2, 4), [0] * n_in, n_out, alias=True),
        ]
        for shape in ((0,), (3, 0), (0, 4), (1 << 63, 1 << 63, 0)):
            cases.append(Case(f"{op}.empty{shape}", op, shape, [], 2, OPS[op][1], ("ok", [])))
        if OPS[op][2]:
            n_in, n_out = lengths(op, 6)
            cases.append(rejected(f"{op}.odd_last_dim", op, (2, 3), [0] * (n_in - 1), n_out))
    for op in ("decode_e2m1", "e2m1_to_grid", "pack_e2m1"):
        for bad in (0x10, 0xFF):
            cases.append(rejected(f"{op}.bad_nibble_last.{bad:#x}", op, (2, 4), [1] * 7 + [bad], 8))
    for value in NON_GRID:
        cases.append(rejected(f"grid_to_e2m1.non_grid_last.{value}", "grid_to_e2m1", (2, 4),
                              [2] * 7 + [value & 0xFF], 8))
    cases.append(Case("version", "version", (1,), [], 1, 4, ("ok", [FORMAT_CONTRACT_VERSION])))
    return cases


def failures(exe, cases, args=()):
    responses = run_executor(exe, cases, args)
    return [msg for case, resp in zip(cases, responses) if (msg := check(case, resp))]


class Parity(unittest.TestCase):
    def test_default_environment(self):
        cases = parity_cases()
        start = time.perf_counter()
        self.assertEqual(failures(EXE, cases), [])
        print(f"parity: {len(cases)} requests in {time.perf_counter() - start:.2f}s",
              file=sys.stderr)


class Contract(unittest.TestCase):
    def test_rejections_and_version(self):
        self.assertEqual(failures(EXE, contract_cases()), [])

    def test_oracle_agrees_on_non_grid_and_bad_nibbles(self):
        for value in NON_GRID:
            with self.assertRaises(ValueError):
                f.grid_to_e2m1(value)
        for bad in (0x10, 0xFF):
            with self.assertRaises(ValueError):
                f.decode_e2m1(bad)
            with self.assertRaises(ValueError):
                f.pack_e2m1([0, bad], (2,))
        with self.assertRaises(ValueError):
            f.pack_e2m1([0] * 6, (2, 3))


SUITES = {"parity": Parity, "contract": Contract}


def main():
    global EXE
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True, type=Path)
    ap.add_argument("--suite", required=True, choices=sorted(SUITES))
    args = ap.parse_args()
    EXE = args.exe
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SUITES[args.suite])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() and result.testsRun else 1


if __name__ == "__main__":
    sys.exit(main())
