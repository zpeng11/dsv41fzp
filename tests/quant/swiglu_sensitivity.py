#!/usr/bin/env python3
"""P0021 temporary production mutations; only completed numeric mismatches count."""

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

from act_quant_sensitivity import build
from test_codec_parity import Case, HarnessError, check, run_executor
from test_expert import expected_mul
from test_swiglu_parity import ROOT, SwigluCase, check_swiglu, directed_cases


MUTATIONS = (
    ('no-gate-clamp', 'swiglu.cpp', (
        ('const std::uint16_t gate_code = clamp_bf16(gate[i], true);',
         'const std::uint16_t gate_code = gate[i];'),)),
    ('gate-lower-clamp', 'swiglu.cpp', (
        ('const std::uint16_t gate_code = clamp_bf16(gate[i], true);',
         'const std::uint16_t gate_code = clamp_bf16(gate[i], false);'),)),
    ('no-up-clamp', 'swiglu.cpp', (
        ('const std::uint16_t up_code = clamp_bf16(up[i], false);',
         'const std::uint16_t up_code = up[i];'),)),
    ('swap-branches', 'swiglu.cpp', (
        ('const std::uint16_t gate_code = clamp_bf16(gate[i], true);',
         'const std::uint16_t gate_code = clamp_bf16(up[i], true);'),
        ('const std::uint16_t up_code = clamp_bf16(up[i], false);',
         'const std::uint16_t up_code = clamp_bf16(gate[i], false);'))),
    ('wrong-silu-index', 'swiglu.cpp', (
        ('generated::kSiluBf16ToFp32[gate_code];',
         'generated::kSiluBf16ToFp32[static_cast<std::uint16_t>(gate_code + 1)];'),)),
    ('early-bf16', 'swiglu.cpp', (
        ('const std::uint32_t silu_bits = generated::kSiluBf16ToFp32[gate_code];',
         'const std::uint32_t silu_bits = static_cast<std::uint32_t>(detail::bf16_rne('
         'generated::kSiluBf16ToFp32[gate_code])) << 16;'),)),
    ('omit-weight', 'swiglu.cpp', (
        ('weighted ? detail::multiply_fp32(gated_bits, route_weight_bits) : gated_bits;',
         'weighted ? (static_cast<void>(route_weight_bits), gated_bits) : gated_bits;'),)),
    ('weight-on-up', 'swiglu.cpp', (
        ('weighted ? detail::multiply_fp32(gated_bits, route_weight_bits) : gated_bits;',
         'weighted ? detail::multiply_fp32(silu_bits, detail::multiply_fp32(up_bits, route_weight_bits)) : gated_bits;'),)),
    ('inf-zero', 'dequant.cpp', (
        ('return left_magnitude == 0 || right_magnitude == 0 ? kQuietNan : sign | kInf;',
         'return sign | kInf;'),)),
)


def multiply_case():
    pairs = [(0, 0xFF800000), (0x80000000, 0x7F800000),
             (0x7F800000, 0xBF800000), (0xFF800000, 0xBF800000),
             (1, 0x3F000000)]
    expected = [value if (value := expected_mul(left, right)) >= 0 else 0x7FC00000
                for left, right in pairs]
    case = Case('multiply-specials', 'multiply_fp32', (), [], len(pairs), 4,
                ('ok', expected))
    case.line = lambda: ('multiply_fp32 cap=' + str(len(pairs)) + ' : ' +
                         ' '.join(f'{left:x}' for left, _ in pairs) + ' ; ' +
                         ' '.join(f'{right:x}' for _, right in pairs))
    return case


def cases():
    bridge = [case for case in directed_cases()
              if case.name.startswith(('directed.unweighted', 'directed.weighted'))]
    return [*bridge, multiply_case()]


def numeric(executable, selected, args=()):
    responses = run_executor(executable, selected, args)
    if any(response[0] != 'ok' for response in responses):
        raise RuntimeError('valid input rejection cannot count as numeric sensitivity')
    messages = []
    for case, response in zip(selected, responses):
        message = (check_swiglu(case, response) if isinstance(case, SwigluCase)
                   else check(case, response))
        if message:
            messages.append(message)
    return messages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cxx', default=os.environ.get('CXX', 'c++'))
    args = parser.parse_args()
    guarded = [ROOT / 'tests/quant/quant_exec.cpp', ROOT / 'tests/quant/test_swiglu_parity.py',
               ROOT / 'tools/gen_silu_table.py', *list((ROOT / 'src/quant').glob('*')),
               *list((ROOT / 'include/df/quant').glob('*')), *list((ROOT / 'ref').glob('*.py'))]
    hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in guarded if path.is_file()}
    selected = cases()
    with tempfile.TemporaryDirectory(prefix='p0021-production-sensitivity-') as temp:
        base = Path(temp) / 'pristine'
        (base / 'src').mkdir(parents=True)
        shutil.copytree(ROOT / 'src/quant', base / 'src/quant')
        shutil.copytree(ROOT / 'include', base / 'include')
        shutil.copy(ROOT / 'tests/quant/quant_exec.cpp', base / 'quant_exec.cpp')
        pristine = build(base, args.cxx)
        if numeric(pristine, selected):
            raise RuntimeError('pristine production failed the numeric baseline')
        print(f'P0021 pristine: {len(selected)} completed numeric cases passed', flush=True)
        for slug, filename, edits in MUTATIONS:
            directory = Path(temp) / slug
            shutil.copytree(base, directory, ignore=shutil.ignore_patterns('quant_exec'))
            source = directory / 'src/quant' / filename
            text = source.read_text()
            for anchor, replacement in edits:
                if text.count(anchor) != 1:
                    raise RuntimeError(f'{slug}: mutation anchor is not unique')
                text = text.replace(anchor, replacement)
            source.write_text(text)
            executable = build(directory, args.cxx)
            mismatches = numeric(executable, selected)
            if not mismatches:
                raise RuntimeError(f'{slug}: production mutation escaped numeric comparison')
            print(f'P0021 {slug}: {len(mismatches)} numeric/stage detectors; '
                  + '; '.join(mismatches[:3]), flush=True)
        for fault in ('crash', 'exit', 'garbage', 'drop'):
            try:
                numeric(pristine, selected, [f'--fault={fault}'])
            except HarnessError:
                print(f'P0021 {fault}: protocol/launch failure excluded from numeric evidence',
                      flush=True)
            else:
                raise RuntimeError(f'{fault}: protocol control escaped')
        if numeric(pristine, selected):
            raise RuntimeError('pristine final retest changed')
    for path, digest in hashes.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'worktree changed during sensitivity run: {path}')
    print(f'P0021 sensitivity: {len(MUTATIONS)}/{len(MUTATIONS)} production variants '
          'detected numerically; pristine retest and worktree hashes passed', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
