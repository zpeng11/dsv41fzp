#!/usr/bin/env python3
"""Generate the P0021 private BF16-to-SiLU table in the build tree."""

import argparse
import hashlib
import os
import struct
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ref.expert import _clamp_bits, _silu_fp32_bits


ENTRY_COUNT = 1 << 16
INVALID = 0xFFFFFFFF
EXPECTED_SHA256 = "73dbb58780862db5cdc935c5334a29734f957422e071bf7710bd78efdadf4baa"


def values():
    result = []
    for code in range(ENTRY_COUNT):
        bits = code << 16
        if _clamp_bits(bits, gate=True) != bits:
            result.append(INVALID)
        else:
            result.append(_silu_fp32_bits(bits))
    return result


def render(table):
    rows = []
    for start in range(0, len(table), 8):
        rows.append("    " + ", ".join(f"0x{value:08X}u" for value in table[start:start + 8]) + ",")
    return "\n".join([
        "#pragma once",
        "",
        "#include <array>",
        "#include <cstdint>",
        "",
        "namespace df::quant::generated {",
        "",
        f"inline constexpr std::uint32_t kInvalidSilu = 0x{INVALID:08X}u;",
        f"inline constexpr std::array<std::uint32_t, {ENTRY_COUNT}> kSiluBf16ToFp32 = {{{{",
        *rows,
        "}};",
        "",
        "}  // namespace df::quant::generated",
        "",
    ])


def write_if_changed(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    table = values()
    payload = b"".join(struct.pack("<I", value) for value in table)
    digest = hashlib.sha256(payload).hexdigest()
    if digest != EXPECTED_SHA256:
        raise SystemExit(f"SiLU table digest mismatch: expected {EXPECTED_SHA256}, got {digest}")
    write_if_changed(args.output, render(table))


if __name__ == "__main__":
    main()
