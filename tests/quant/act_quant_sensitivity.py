#!/usr/bin/env python3
"""P0017 throwaway production-source mutations; never edits the worktree.

Run: python3 tests/quant/act_quant_sensitivity.py [--cxx c++]
Each pristine/mutated executor links copied production sources, not an oracle
implementation or a test substitute. Named stage/bit mismatches alone count.
Temporary sources and executables are automatically removed after the run.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DRIVER = ROOT / 'tests/quant/test_act_quant_parity.py'

ROUND_HELPER = '''
std::uint64_t mutation_round(std::uint64_t value, int shift) noexcept {
  if (shift <= 0) return value << -shift;
  if (shift >= 64) return 0;
  return (value >> shift) + ((value & ((std::uint64_t{1} << shift) - 1)) >=
                           (std::uint64_t{1} << (shift - 1)));
}
'''
TRUNCATE_HELPER = '''
std::uint64_t mutation_round(std::uint64_t value, int shift) noexcept {
  if (shift <= 0) return value << -shift;
  return shift >= 64 ? 0 : value >> shift;
}
'''
EXPONENT = '''const int exponent = static_cast<int>((product >> 23) & 0xFFu) - 127 +
                         ((product & 0x7FFFFFu) != 0);'''
QUOTIENT = 'return detail::binary_to_fp32(bits & kSign, significand, binary_exponent - exponent);'


def mutations():
    return [
        ('scale-floor', 'scale-ceil', 'act_quant.cpp', [(EXPONENT,
         'const int exponent = static_cast<int>((product >> 23) & 0xFFu) - 127;')]),
        ('scale-round', 'scale-ceil', 'act_quant.cpp', [(EXPONENT,
         '''const std::uint64_t mantissa = 0x800000u | (product & 0x7FFFFFu);
    const int exponent = static_cast<int>((product >> 23) & 0xFFu) - 127 +
                         (mantissa * mantissa > (std::uint64_t{1} << 47));''')]),
        ('four-group-scale', 'four-group-scale', 'act_quant.cpp', [(
         '''for (std::size_t r = 0; r < kGroup; ++r) {
      amax = std::max(amax, static_cast<std::uint32_t>(input[offset + r] & 0x7FFFu) << 16);
    }''',
         '''const std::size_t shared_begin = (offset / 128) * 128;
    const std::size_t shared_end = std::min(shared_begin + 128, input.size());
    for (std::size_t r = shared_begin; r < shared_end; ++r) {
      amax = std::max(amax, static_cast<std::uint32_t>(input[r] & 0x7FFFu) << 16);
    }''')]),
        ('fp8-truncate', 'fp8-midpoints', 'codec.cpp', [
         ('std::uint8_t encode_e4m3fn(std::uint32_t bits) noexcept {',
          TRUNCATE_HELPER + '\nstd::uint8_t encode_e4m3fn(std::uint32_t bits) noexcept {'),
         ('sign | round_shift(significand, -9 - exponent)',
          'sign | mutation_round(significand, -9 - exponent)'),
         ('rounded = round_shift(significand, width - 4)',
          'rounded = mutation_round(significand, width - 4)')]),
        ('fp8-tie-away', 'fp8-midpoints', 'codec.cpp', [
         ('std::uint8_t encode_e4m3fn(std::uint32_t bits) noexcept {',
          ROUND_HELPER + '\nstd::uint8_t encode_e4m3fn(std::uint32_t bits) noexcept {'),
         ('sign | round_shift(significand, -9 - exponent)',
          'sign | mutation_round(significand, -9 - exponent)'),
         ('rounded = round_shift(significand, width - 4)',
          'rounded = mutation_round(significand, width - 4)')]),
        ('negative-zero', 'signed-zero', 'codec.cpp', [(
         'const std::uint32_t sign = (bits >> 24) & 0x80u;',
         'const std::uint32_t sign = (bits & 0x7FFFFFFFu) == 0 ? 0 : (bits >> 24) & 0x80u;')]),
        ('remove-floor', 'amax-floor', 'act_quant.cpp', [(
         'const std::uint32_t bounded = std::max(amax, kAmaxFloor);',
         'const std::uint32_t bounded = amax;'),
         ('constexpr std::uint32_t kAmaxFloor =', '[[maybe_unused]] constexpr std::uint32_t kAmaxFloor =')]),
        ('bf16-daz', 'bf16-subnormal', 'act_quant.cpp', [(
         'const std::uint32_t bits = static_cast<std::uint32_t>(code) << 16;',
         '''const std::uint32_t bits = (code & 0x7F80u) == 0 ?
      static_cast<std::uint32_t>(code & 0x8000u) << 16 : static_cast<std::uint32_t>(code) << 16;''')]),
        ('quotient-ftz', 'quotient-subnormal', 'act_quant.cpp', [(QUOTIENT,
         '''const auto result = detail::binary_to_fp32(bits & kSign, significand, binary_exponent - exponent);
  return (result & kMagnitude) < 0x00800000u ? result & kSign : result;''')]),
        ('quotient-tie-away', 'quotient-halfway', 'act_quant.cpp', [(
         'std::uint32_t quotient_bits(std::uint16_t code, int exponent) noexcept {',
         ROUND_HELPER + '\nstd::uint32_t quotient_bits(std::uint16_t code, int exponent) noexcept {'),
         (QUOTIENT,
         '''const int shift = -149 - (binary_exponent - exponent);
  if (shift > 0) {
    return (bits & kSign) | static_cast<std::uint32_t>(mutation_round(significand, shift));
  }
  return detail::binary_to_fp32(bits & kSign, significand, binary_exponent - exponent);''')]),
    ]


def checked_run(command, **kwargs):
    proc = subprocess.run(command, capture_output=True, text=True, timeout=300, **kwargs)
    if proc.returncode:
        raise RuntimeError(f'command failed {command!r}\n{proc.stdout}\n{proc.stderr}')
    if 'warning:' in proc.stderr or 'warning:' in proc.stdout:
        raise RuntimeError('compilation warning is not acceptance')
    return proc


def build(directory, cxx):
    executable = directory / 'quant_exec'
    checked_run([cxx, '-std=c++20', '-O1', '-g', '-Wall', '-Wextra', '-Wpedantic', '-Werror',
                 '-ffp-contract=off', '-fno-fast-math', '-I' + str(directory / 'include'),
                 '-I' + str(directory / 'src/quant'),
                 str(directory / 'quant_exec.cpp'),
                 *[str(directory / 'src/quant' / name) for name in
                   ('codec.cpp', 'dequant.cpp', 'compute.cpp', 'act_quant.cpp', 'linear.cpp')],
                 '-o', str(executable)])
    return executable


def report(executable, selector=None, fault=None):
    command = [sys.executable, str(DRIVER), '--exe', str(executable), '--suite', 'parity',
               '--numeric-report']
    if selector:
        command += ['--case', selector]
    if fault:
        command += [f'--exe-arg=--fault={fault}']
    proc = subprocess.run(command, capture_output=True, text=True, timeout=300)
    lines = proc.stdout.splitlines()
    if len(lines) != 1 or not lines[0].startswith('P0017_NUMERIC '):
        raise RuntimeError('missing, duplicate or malformed numeric report')
    payload = json.loads(lines[0][len('P0017_NUMERIC '):])
    if set(payload) != {'cases', 'errors', 'skipped'} or payload['skipped'] != 0:
        raise RuntimeError('numeric report schema or skip failure')
    if not isinstance(payload['errors'], list) or not isinstance(payload['cases'], list):
        raise RuntimeError('numeric report field types')
    for case in payload['cases']:
        if set(case) != {'case', 'mismatches'} or not isinstance(case['case'], str) or not isinstance(case['mismatches'], list):
            raise RuntimeError('malformed numeric case')
    failed = bool(payload['errors'] or any(c['mismatches'] for c in payload['cases']))
    if proc.returncode != int(failed):
        raise RuntimeError('driver exit disagrees with numeric report')
    return payload


def require_numeric(payload, selector):
    if payload['errors'] or payload['skipped']:
        raise RuntimeError('exceptions, protocol or sanitizer failures cannot count as sensitivity')
    expected = {selector + '.ordinary', selector + '.trace'}
    if {c['case'] for c in payload['cases']} != expected:
        raise RuntimeError('named detectors not completed')
    if not any(c['mismatches'] for c in payload['cases']):
        raise RuntimeError('production mutation escaped numeric/stage comparisons')
    return sum(len(c['mismatches']) for c in payload['cases'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cxx', default=os.environ.get('CXX', 'c++'))
    args = parser.parse_args()
    guarded = [ROOT / 'tests/quant/quant_exec.cpp', DRIVER, *list((ROOT / 'src/quant').glob('*')),
               *list((ROOT / 'include/df/quant').glob('*')), *list((ROOT / 'ref').glob('*.py'))]
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in guarded if p.is_file()}
    with tempfile.TemporaryDirectory(prefix='p0017-production-sensitivity-') as temp:
        base = Path(temp) / 'pristine'
        (base / 'src').mkdir(parents=True)
        shutil.copytree(ROOT / 'src/quant', base / 'src/quant')
        shutil.copytree(ROOT / 'include', base / 'include')
        shutil.copy(ROOT / 'tests/quant/quant_exec.cpp', base / 'quant_exec.cpp')
        pristine_exe = build(base, args.cxx)
        baseline = report(pristine_exe)
        if baseline['errors'] or any(c['mismatches'] for c in baseline['cases']) or not baseline['cases']:
            raise RuntimeError('pristine production did not pass directed numeric tests')
        print(f'P0017 pristine: {len(baseline["cases"])} numeric cases passed', flush=True)
        for slug, selector, filename, edits in mutations():
            directory = Path(temp) / slug
            shutil.copytree(base, directory, ignore=shutil.ignore_patterns('quant_exec'))
            source = directory / 'src/quant' / filename
            original = source.read_text()
            text = original
            for anchor, replacement in edits:
                if text.count(anchor) != 1:
                    raise RuntimeError(f'{slug}: mutation anchor is not unique')
                text = text.replace(anchor, replacement)
            source.write_text(text)
            exe = build(directory, args.cxx)
            payload = report(exe, selector)
            count = require_numeric(payload, selector)
            print(f'P0017 mutation {slug}: {selector}; {count} numeric/stage mismatches; '
                  'errors=0 skips=0', flush=True)
            source.write_text(original)
            restored = report(build(directory, args.cxx), selector)
            if restored['errors'] or any(c['mismatches'] for c in restored['cases']):
                raise RuntimeError(slug + ': restored production failed')
        for fault in ('crash', 'exit', 'garbage', 'drop'):
            payload = report(pristine_exe, 'signed-zero', fault)
            if not payload['errors']:
                raise RuntimeError('executor fault not categorized as an error: ' + fault)
            try:
                require_numeric(payload, 'signed-zero')
            except RuntimeError:
                pass
            else:
                raise RuntimeError('executor fault incorrectly counted as numeric evidence')
        if report(pristine_exe) != baseline:
            raise RuntimeError('pristine final retest changed')
    for path, digest in hashes.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError('guarded worktree file changed: ' + str(path))
    print('P0017 sensitivity: 10/10 production variants detected numerically; '
          'restoration and pristine retests passed; worktree hashes unchanged', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
