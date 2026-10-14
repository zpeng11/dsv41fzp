#!/usr/bin/env python3
"""P0019 temporary production mutations, accepted only on completed numeric mismatches.

Copies production sources and the executor into an automatically removed temp
folder. No protected data or oracle is changed. Baseline, all mutant responses,
and protocol-failure controls go through the normal strict response parser.
"""

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
from test_codec_parity import HarnessError, run_executor
from test_linear_parity import ROOT, both, check_linear, layout


def mutations():
    return [
        ('wrong-row', 'w_packed.subspan(row * (k / 2), k / 2)',
         'w_packed.subspan(((row + 1) % n) * (k / 2), k / 2)'),
        ('wrong-scale', 'scale_b.subspan(row * groups, groups)',
         'scale_b.subspan(((row + 1) % n) * groups, groups)'),
        ('nibble-reversed', '  act_quant(input, codes, scales);',
         '''  act_quant(input, codes, scales);
  std::vector<std::uint8_t> swapped(w_packed.begin(), w_packed.end());
  for (auto& byte : swapped) byte = (byte >> 4) | ((byte & 15) << 4);
  w_packed = swapped;'''),
        ('missing-last-row', 'row < n; ++row', 'row + 1 < n; ++row'),
        ('result-tamper', 'out.subspan(row, 1));', 'out.subspan(row, 1));\n    out[row] ^= 1;'),
    ]


def numeric(exe, cases, args=()):
    responses = run_executor(exe, cases, args)
    if any(response[0] != 'ok' for response in responses):
        raise RuntimeError('valid input rejection cannot count as numeric sensitivity')
    return [message for case, response in zip(cases, responses)
            if (message := check_linear(case, response))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cxx', default=os.environ.get('CXX', 'c++'))
    args = parser.parse_args()
    guarded = [ROOT / 'tests/quant/quant_exec.cpp', ROOT / 'tests/quant/test_linear_parity.py',
               *list((ROOT / 'src/quant').glob('*')), *list((ROOT / 'include/df/quant').glob('*')),
               *list((ROOT / 'ref').glob('*.py'))]
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in guarded if p.is_file()}
    cases = both('layout.hand', *layout(), workspace=True)
    with tempfile.TemporaryDirectory(prefix='p0019-production-sensitivity-') as temp:
        base = Path(temp) / 'pristine'
        (base / 'src').mkdir(parents=True)
        shutil.copytree(ROOT / 'src/quant', base / 'src/quant')
        shutil.copytree(ROOT / 'include', base / 'include')
        shutil.copy(ROOT / 'tests/quant/quant_exec.cpp', base / 'quant_exec.cpp')
        pristine = build(base, args.cxx)
        if numeric(pristine, cases):
            raise RuntimeError('pristine production failed the numeric baseline')
        print('P0019 pristine: both complete vectors and workspaces passed', flush=True)
        for slug, anchor, replacement in mutations():
            directory = Path(temp) / slug
            shutil.copytree(base, directory, ignore=shutil.ignore_patterns('quant_exec'))
            source = directory / 'src/quant/linear.cpp'
            text = source.read_text()
            if text.count(anchor) != 1:
                raise RuntimeError(f'{slug}: mutation anchor is not unique')
            source.write_text(text.replace(anchor, replacement).replace(
                '#include <array>', '#include <array>\n#include <vector>'))
            exe = build(directory, args.cxx)
            mismatches = numeric(exe, cases)
            if len(mismatches) != len(cases):
                raise RuntimeError(f'{slug}: both output overloads must detect the numeric mutation')
            print(f'P0019 {slug}: {len(mismatches)} numeric detectors; '
                  + '; '.join(mismatches), flush=True)
        for fault in ('crash', 'exit', 'garbage', 'drop'):
            try:
                numeric(pristine, cases, [f'--fault={fault}'])
            except HarnessError:
                print(f'P0019 {fault}: protocol/launch failure excluded from numeric evidence', flush=True)
            else:
                raise RuntimeError(f'{fault}: protocol control escaped')
    for path, digest in hashes.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'worktree changed during sensitivity run: {path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
