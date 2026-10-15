#!/usr/bin/env python3
"""P0022 complete-expert production-source sensitivity; only completed mismatches count."""

import argparse
import hashlib
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

from act_quant_sensitivity import build
from ref.act_quant import act_quant
from ref.compute import trace_expert_grouped_dot
from test_codec_parity import HarnessError, run_executor
from test_expert_parity import ExpertCase, ROOT, check_expert, impl


LINEAR = '''void linear(std::span<const std::uint8_t> codes, std::span<const std::uint8_t> scales,
            ExpertMatrix matrix, std::span<std::uint16_t> out) {
  const std::size_t weight_stride = matrix.k / 2;
  const std::size_t scale_stride = matrix.k / kGroup;
  for (std::size_t row = 0; row < matrix.n; ++row) {
    expert_grouped_dot(codes, matrix.w_packed.subspan(row * weight_stride, weight_stride), scales,
                       matrix.scale_b.subspan(row * scale_stride, scale_stride),
                       out.subspan(row, 1));
  }
}'''

PACKED_ROW_LINEAR = '''void linear(std::span<const std::uint8_t> codes, std::span<const std::uint8_t> scales,
            ExpertMatrix matrix, std::span<std::uint16_t> out) {
  const std::size_t weight_stride = matrix.k / 2;
  const std::size_t scale_stride = matrix.k / kGroup;
  for (std::size_t row = 0; row < matrix.n; ++row) {
    const std::size_t weight_row = (row + 1) % matrix.n;
    expert_grouped_dot(codes, matrix.w_packed.subspan(weight_row * weight_stride, weight_stride), scales,
                       matrix.scale_b.subspan(row * scale_stride, scale_stride),
                       out.subspan(row, 1));
  }
}'''

SCALE_ROW_LINEAR = '''void linear(std::span<const std::uint8_t> codes, std::span<const std::uint8_t> scales,
            ExpertMatrix matrix, std::span<std::uint16_t> out) {
  const std::size_t weight_stride = matrix.k / 2;
  const std::size_t scale_stride = matrix.k / kGroup;
  for (std::size_t row = 0; row < matrix.n; ++row) {
    const std::size_t scale_row = (row + 1) % matrix.n;
    expert_grouped_dot(codes, matrix.w_packed.subspan(row * weight_stride, weight_stride), scales,
                       matrix.scale_b.subspan(scale_row * scale_stride, scale_stride),
                       out.subspan(row, 1));
  }
}'''

TRUNCATE_LINEAR = '''void linear(std::span<const std::uint8_t> codes, std::span<const std::uint8_t> scales,
            ExpertMatrix matrix, std::span<std::uint16_t> out) {
  const std::size_t weight_stride = matrix.k / 2;
  const std::size_t scale_stride = matrix.k / kGroup;
  for (std::size_t row = 0; row < matrix.n; ++row) {
    std::uint32_t fp32_bits = 0;
    expert_grouped_dot(codes, matrix.w_packed.subspan(row * weight_stride, weight_stride), scales,
                       matrix.scale_b.subspan(row * scale_stride, scale_stride),
                       std::span<std::uint32_t>(&fp32_bits, 1));
    out[row] = static_cast<std::uint16_t>(fp32_bits >> 16);
  }
}'''

ROUTE_BEFORE = '''  if (trace == nullptr) {
    if (weighted) {
      swiglu(gate, up, route_weight_bits, hidden);
    } else {
      swiglu(gate, up, hidden);
    }
  } else if (weighted) {
    detail::trace_swiglu(gate, up, route_weight_bits, hidden, *trace);
  } else {
    detail::trace_swiglu(gate, up, hidden, *trace);
  }
  act_quant(hidden, hidden_codes, hidden_scales);
  linear(hidden_codes, hidden_scales, w2, out.first(w2.n));'''

ROUTE_AFTER = '''  if (trace == nullptr) {
    swiglu(gate, up, hidden);
  } else {
    detail::trace_swiglu(gate, up, hidden, *trace);
  }
  act_quant(hidden, hidden_codes, hidden_scales);
  linear(hidden_codes, hidden_scales, w2, out.first(w2.n));
  if (weighted) {
    for (std::size_t row = 0; row < w2.n; ++row) {
      const std::uint32_t output_bits = static_cast<std::uint32_t>(out[row]) << 16;
      out[row] = detail::bf16_rne(detail::multiply_fp32(output_bits, route_weight_bits));
    }
  }'''

MUTATIONS = (
    ('swap-w1-w3', (('linear(input_codes, input_scales, w1, gate);\n'
                     '  linear(input_codes, input_scales, w3, up);',
                    'linear(input_codes, input_scales, w3, gate);\n'
                    '  linear(input_codes, input_scales, w1, up);'),)),
    ('packed-row-mismatch', ((LINEAR, PACKED_ROW_LINEAR),)),
    ('scale-row-mismatch', ((LINEAR, SCALE_ROW_LINEAR),)),
    ('skip-hidden-quant', (('act_quant(hidden, hidden_codes, hidden_scales);',
                           'for (std::size_t i = 0; i < hidden_codes.size(); ++i) {\n'
                           '    hidden_codes[i] = input_codes[i];\n'
                           '  }\n'
                           '  for (std::size_t i = 0; i < hidden_scales.size(); ++i) {\n'
                           '    hidden_scales[i] = input_scales[i];\n'
                           '  }'),)),
    ('route-after-w2', ((ROUTE_BEFORE, ROUTE_AFTER),)),
    ('fp32-dot-truncate', ((LINEAR, TRUNCATE_LINEAR),)),
)


def random_matrix(rng, n=32, k=32):
    nibbles = (0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14)
    packed = [rng.choice(nibbles) | (rng.choice(nibbles) << 4) for _ in range(n * k // 2)]
    scales = [rng.randrange(121, 130) for _ in range(n * k // 32)]
    return impl.FP4Matrix(n, k, packed, scales)


def shifted_rows(matrix, *, packed):
    packed_stride, scale_stride = matrix.k // 2, matrix.k // 32
    weights = matrix.w_packed
    scales = matrix.scale_b
    if packed:
        weights = [value for row in range(matrix.n)
                   for value in matrix.w_packed[((row + 1) % matrix.n) * packed_stride:
                                                ((row + 1) % matrix.n + 1) * packed_stride]]
    else:
        scales = [value for row in range(matrix.n)
                  for value in matrix.scale_b[((row + 1) % matrix.n) * scale_stride:
                                              ((row + 1) % matrix.n + 1) * scale_stride]]
    return impl.FP4Matrix(matrix.n, matrix.k, weights, scales)


def oracle_linear(codes, scales, matrix, *, truncate=False):
    packed_stride, scale_stride = matrix.k // 2, matrix.k // 32
    values = []
    for row in range(matrix.n):
        trace = trace_expert_grouped_dot(
            codes, matrix.w_packed[row * packed_stride:(row + 1) * packed_stride], scales,
            matrix.scale_b[row * scale_stride:(row + 1) * scale_stride])
        values.append(trace.fp32_bits >> 16 if truncate else trace.bf16_bits)
    return tuple(values)


def detectable_by_all(inputs, matrices, route):
    trace = impl.trace_expert(inputs, *matrices, route_weight_bits=route)
    swapped = impl.trace_expert(inputs, matrices[2], matrices[1], matrices[0],
                                route_weight_bits=route)
    packed = impl.trace_expert(inputs, *(shifted_rows(matrix, packed=True)
                                         for matrix in matrices), route_weight_bits=route)
    scales = impl.trace_expert(inputs, *(shifted_rows(matrix, packed=False)
                                         for matrix in matrices), route_weight_bits=route)
    skipped_output = oracle_linear(trace.input_codes, trace.input_scales, matrices[1])
    unweighted = impl.trace_expert(inputs, *matrices, route_weight_bits=None)
    truncated_gate = oracle_linear(trace.input_codes, trace.input_scales, matrices[0],
                                   truncate=True)
    return (swapped != trace and packed != trace and scales != trace and
            skipped_output != trace.output_bf16 and
            unweighted.bridge.weighted_bits != trace.bridge.weighted_bits and
            truncated_gate != trace.gate_bf16)


def sensitivity_case():
    """Use existing ref stages to find a seeded legal counterexample for every mutation."""
    rng = random.Random(220022)
    bf16 = (0, 0x8000, 0x3E80, 0xBE80, 0x3F00, 0xBF00, 0x3F80, 0xBF80,
            0x4000, 0xC000)
    route = 0x3F000000
    for attempt in range(100):
        inputs = [rng.choice(bf16) for _ in range(32)]
        matrices = tuple(random_matrix(rng) for _ in range(3))
        try:
            detected = detectable_by_all(inputs, matrices, route)
        except (ArithmeticError, ValueError):
            continue
        if detected:
            return ExpertCase(f'sensitivity.seed-220022.attempt-{attempt}', inputs, *matrices,
                              route=route, traced=True)
    raise RuntimeError('oracle failed to find the seeded legal six-variant counterexample')


def numeric(executable, selected, args=()):
    responses = run_executor(executable, selected, args)
    if any(response[0] != 'ok' for response in responses):
        raise RuntimeError('valid input rejection cannot count as numeric sensitivity')
    mismatches = [message for case, response in zip(selected, responses)
                  if (message := check_expert(case, response))]
    if any(': expected 0x' not in message or ', got 0x' not in message
           for message in mismatches):
        raise RuntimeError('only completed named-stage value mismatches count as numeric evidence')
    return mismatches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cxx', default=os.environ.get('CXX', 'c++'))
    args = parser.parse_args()
    selected = [sensitivity_case()]
    if not selected:
        raise RuntimeError('empty selected cases are not acceptance')
    guarded = [ROOT / 'tests/quant/quant_exec.cpp', ROOT / 'tests/quant/test_expert_parity.py',
               *list((ROOT / 'src/quant').glob('*')),
               *list((ROOT / 'include/df/quant').glob('*')), *list((ROOT / 'ref').glob('*.py'))]
    hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in guarded if path.is_file()}
    with tempfile.TemporaryDirectory(prefix='p0022-production-sensitivity-') as temp:
        base = Path(temp) / 'pristine'
        (base / 'src').mkdir(parents=True)
        shutil.copytree(ROOT / 'src/quant', base / 'src/quant')
        shutil.copytree(ROOT / 'include', base / 'include')
        shutil.copy(ROOT / 'tests/quant/quant_exec.cpp', base / 'quant_exec.cpp')
        pristine = build(base, args.cxx)
        if numeric(pristine, selected):
            raise RuntimeError('pristine production failed the complete numeric baseline')
        print(f'P0022 pristine: {len(selected)} completed oracle case passed', flush=True)
        for slug, edits in MUTATIONS:
            directory = Path(temp) / slug
            shutil.copytree(base, directory, ignore=shutil.ignore_patterns('quant_exec'))
            source = directory / 'src/quant/expert.cpp'
            original = source.read_text()
            text = original
            for anchor, replacement in edits:
                if text.count(anchor) != 1:
                    raise RuntimeError(f'{slug}: mutation anchor is not unique')
                text = text.replace(anchor, replacement)
            source.write_text(text)
            executable = build(directory, args.cxx)
            mismatches = numeric(executable, selected)
            if not mismatches:
                raise RuntimeError(f'{slug}: completed production mutation escaped stage comparison')
            print(f'P0022 {slug}: {len(mismatches)} completed numeric/stage detector; '
                  + '; '.join(mismatches[:3]), flush=True)
            source.write_text(original)
            restored = build(directory, args.cxx)
            if numeric(restored, selected):
                raise RuntimeError(f'{slug}: pristine source failed the same-case restoration retest')
        for fault in ('crash', 'exit', 'garbage', 'drop'):
            try:
                numeric(pristine, selected, [f'--fault={fault}'])
            except HarnessError:
                print(f'P0022 {fault}: protocol failure excluded from numeric evidence', flush=True)
            else:
                raise RuntimeError(f'{fault}: protocol control escaped')
        if numeric(pristine, selected):
            raise RuntimeError('pristine final same-case retest changed')
    for path, digest in hashes.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'guarded worktree file changed during sensitivity run: {path}')
    print(f'P0022 sensitivity: {len(MUTATIONS)}/{len(MUTATIONS)} complete production variants '
          'detected numerically; each same-case pristine restoration, final retest and guarded '
          'worktree hashes passed', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
