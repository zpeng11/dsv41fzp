#!/usr/bin/env python3
"""P0017 production activation parity, contracts, composition and harness.

Expected outputs and all six stage arrays come directly from ref.act_quant.
P0014 supplies only frozen coverage construction, never numeric expectations.
The executor reports output bytes as uint32 words, then amax, bounded_amax,
product, exponent (two's complement), quotient and clamped stage arrays.
--numeric-report is a strict sensitivity protocol: only completed bit comparisons
count; rejected valid inputs, malformed responses, crashes and ASan are errors.
"""

import argparse
import itertools
import json
import random
import subprocess
import sys
import time
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ref.act_quant import act_quant, trace_act_quant
from ref.compute import trace_expert_grouped_dot
from test_act_quant import (FINITE_BF16_CODES, below_floor_bits, bf16_code,
                            five_group_bits, midpoint_bits, power2,
                            quotient_boundary_bits, signed_zero_bits, subnormal_amax_bits)
from test_codec_parity import (BF16, FP32, RAW, Case, HarnessError, check,
                               parse_response, run_executor)
from test_compute_parity import ComputeCase, pack
from test_dequant_parity import ENVIRONMENTS, MXCSR_STATUS, RawCase, mxcsr_probe

EXE = None
EXE_ARGS = []
SELECT = []
STAGES = ('amax_bits', 'bounded_amax_bits', 'product_bits', 'exponent',
          'quotient_bits', 'clamped_bits')
FILL = 0xA5
BUFFERS = ('input', 'a', 'scale', *STAGES)
OVERLAPS = {'input_a': (0, 1), 'input_scale': (0, 2), 'outputs': (1, 2)}


def shared_expected(bits, trace, traced, a_cap, scale_cap, shape, pair, offsets):
    """Model executor initialization and oracle prefix writes on shared bytes."""
    lengths = [2 * len(bits), a_cap, scale_cap] + ([4 * n for n in shape] if traced else [])
    owners = [bytearray([FILL]) * n for n in lengths]
    starts = [0] * len(lengths)
    if pair is not None:
        backing = bytearray([FILL]) * max(offset + lengths[i] for i, offset in zip(pair, offsets))
        for i, offset in zip(pair, offsets):
            owners[i], starts[i] = backing, offset

    def write(index, offset, value):
        start = starts[index] + offset
        owners[index][start:start + len(value)] = value

    for i in range(1, len(lengths)):
        write(i, 0, bytes([FILL]) * lengths[i])
    for i, value in enumerate(bits):
        write(0, 2 * i, value.to_bytes(2, sys.byteorder))
    for j, group in enumerate(trace.groups):
        write(2, j, bytes([group.scale_code]))
        if traced:
            for stage in range(4):
                value = getattr(group, STAGES[stage]) & 0xFFFFFFFF
                write(stage + 3, 4 * j, value.to_bytes(4, sys.byteorder))
        for r, code in enumerate(group.a_codes):
            index = 32 * j + r
            write(1, index, bytes([code]))
            if traced:
                for stage in (4, 5):
                    value = getattr(group, STAGES[stage])[r]
                    write(stage + 3, 4 * index, value.to_bytes(4, sys.byteorder))
    words, labels = [], []
    for i in range(1, len(lengths)):
        width = 1 if i < 3 else 4
        data = owners[i][starts[i]:starts[i] + lengths[i]]
        words += [int.from_bytes(data[n:n + width], sys.byteorder) for n in range(0, len(data), width)]
        labels += [('a_codes', 'scale_a', *STAGES)[i - 1]] * (len(data) // width)
    return words, labels


class ActCase(Case):
    """Independent oracle byte simulation and library rejection snapshots."""

    def __init__(self, name, bits, traced=False, a_cap=None, scale_cap=None,
                 rejected=False, trace_shape=None, overlap=None, in_offset=None,
                 out_offset=None, poison=False, trace_alias=None, trace_object_alias=False):
        self.a_cap = len(bits) if a_cap is None else a_cap
        self.scale_cap = len(bits) // 32 if scale_cap is None else scale_cap
        self.traced = traced
        self.trace_shape = trace_shape
        self.overlap = overlap
        self.trace_alias = trace_alias
        self.trace_object_alias = trace_object_alias
        self.in_offset = in_offset
        self.out_offset = out_offset
        self.poison = poison
        self.labels = []
        shape = trace_shape if trace_shape is not None else [len(bits) // 32] * 4 + [len(bits)] * 2
        count = self.a_cap + self.scale_cap + (sum(shape) if traced else 0)
        if rejected:
            expected = ('err', 'invalid_argument')
        else:
            trace = trace_act_quant(bits)
            direct = act_quant(bits)
            if direct != (trace.a_codes, trace.scale_a):
                raise HarnessError('oracle ordinary/trace inconsistency')
            pair = (tuple(BUFFERS.index(name) for name in trace_alias.split(','))
                    if trace_alias is not None else OVERLAPS.get(overlap))
            words, self.labels = shared_expected(
                bits, trace, traced, self.a_cap, self.scale_cap, shape, pair,
                (in_offset or 0, out_offset or 0))
            expected = ('ok', words)
        super().__init__(name, 'act_quant_trace' if traced else 'act_quant', (), list(bits),
                         count, 4, expected, RAW)

    def line(self):
        head = [self.op, f'cap={self.a_cap}', f'scale_cap={self.scale_cap}']
        if self.trace_shape is not None:
            head.append('trace_shape=' + ','.join(map(str, self.trace_shape)))
        if self.overlap is not None:
            head.append('overlap=' + self.overlap)
        if self.trace_alias is not None:
            head.append('trace_alias=' + self.trace_alias)
        if self.trace_object_alias:
            head.append('trace_object_alias')
        if self.in_offset is not None:
            head.append(f'in_offset={self.in_offset}')
        if self.out_offset is not None:
            head.append(f'out_offset={self.out_offset}')
        if self.poison:
            head.append('poison')
        return ' '.join(head) + ' : ' + ' '.join(f'{v:x}' for v in self.tokens)


def vector(name, bits, slack=0):
    return [ActCase(name + suffix, bits, traced, len(bits) + slack,
                    len(bits) // 32 + slack)
            for traced, suffix in ((False, '.ordinary'), (True, '.trace'))]


def directed_vectors():
    exponents = (-21, -7, 0, 1, 23, 24, 119)
    values = [448 * power2(e) for e in exponents]
    values += [v * power2(e) for e in (-21, 0, 23, 119) for v in (440, 231)]
    scales = []
    for v in values:
        a, q = bf16_code(v), bf16_code(v / 4)
        scales += [a, q, q | 0x8000, 0, 0x8000] + [0] * 27
    return [
        ('scale-ceil', scales), ('four-group-scale', five_group_bits()),
        ('amax-floor', below_floor_bits()), ('fp8-midpoints', midpoint_bits()),
        ('signed-zero', signed_zero_bits()), ('bf16-subnormal', subnormal_amax_bits()),
        ('quotient-subnormal', quotient_boundary_bits()[:32]),
        ('quotient-halfway', quotient_boundary_bits()[32:]),
        ('product-reciprocal', [0x38D2] + [0] * 31),
    ]


def parity_cases():
    for name, bits in directed_vectors():
        yield from vector(name, bits, slack=3)
    for start in range(0, 0x7F80, 256):
        bits = [v for code in range(start, min(start + 256, 0x7F80))
                for v in [code] + [0] * 31]
        yield from vector(f'exhaustive.amax.{start}', bits)
    ordered = sorted(FINITE_BF16_CODES, key=lambda c: c & 0x7FFF)
    assert len(ordered) == 65280
    for start in range(0, len(ordered), 2048):
        yield from vector(f'exhaustive.binade.{start}', ordered[start:start + 2048])
    anchored = []
    for start in range(0, len(ordered), 31):
        group = [0x7F7F] + ordered[start:start + 31]
        anchored += group + [0] * (32 - len(group))
    for start in range(0, len(anchored), 2048):
        yield from vector(f'exhaustive.max-anchor.{start}', anchored[start:start + 2048])
    rng = random.Random(1717)
    for i in range(24):
        bits = [rng.choice(FINITE_BF16_CODES) for _ in range(32 * rng.choice((1, 3, 5, 8, 17)))]
        yield from vector(f'random.{i}', bits)


def contract_cases():
    bits = list(five_group_bits())
    k, g = len(bits), len(bits) // 32
    for traced in (False, True):
        prefix = 'trace' if traced else 'ordinary'
        for length in (0, 1, 16, 31, 33, 48, 63, 97, 159, 161):
            data = [0x3F80] * length
            for poisoned in (False, True):
                yield ActCase(f'{prefix}.length.{length}.poison={poisoned}', data, traced,
                              length + 3, length // 32 + 3, rejected=True, poison=poisoned)
        for a, s in ((0, g), (k - 1, g), (k, 0), (k, g - 1), (k - 1, g - 1)):
            for poisoned in (False, True):
                yield ActCase(f'{prefix}.capacity.{a}.{s}.poison={poisoned}', bits, traced,
                              a, s, rejected=True, poison=poisoned)
        for target, first, second in (('input_a', 2 * k, k),
                                      ('input_scale', 2 * k, g), ('outputs', k, g)):
            reverse = second if target == 'outputs' else second + second % 2
            placements = [(0, 0), (0, 1), (0, first - 1), (2, 0), (0, first), (reverse, 0)]
            for offset_in, offset_out in placements:
                overlap = offset_in < offset_out + second and offset_out < offset_in + first
                yield ActCase(f'{prefix}.{target}.{offset_in}.{offset_out}', bits, traced,
                              rejected=overlap, overlap=target,
                              in_offset=offset_in, out_offset=offset_out)
                if overlap:
                    yield ActCase(f'{prefix}.{target}.{offset_in}.{offset_out}.poison', bits, traced,
                                  rejected=True, overlap=target,
                                  in_offset=offset_in, out_offset=offset_out, poison=True)
        tail_bits = [0x3F80, 0xBF80, 0x8000, 0x0001] * 8
        for target, a_cap, s_cap, first, second in (
                ('input_a', 96, 1, 32, 0),
                ('input_scale', 32, 66, 2, 0),
                ('outputs', 33, 1, 0, 32),
                ('outputs', 32, 33, 1, 0)):
            yield ActCase(f'{prefix}.unused-tail.{target}.{first}.{second}', tail_bits, traced,
                          a_cap, s_cap, overlap=target, in_offset=first, out_offset=second)
        yield ActCase(f'{prefix}.shared-gap.outputs', tail_bits, traced,
                      overlap='outputs', in_offset=0, out_offset=40)
        for a, s in ((k, g), (k + 3, g), (k, g + 3), (k + 3, g + 3)):
            yield ActCase(f'{prefix}.sentinel.{a}.{s}', bits, traced, a, s)
        nonfinite = list(range(0x7F80, 0x8000)) + list(range(0xFF80, 0x10000))
        assert len(nonfinite) == 256
        for code in nonfinite:
            for position in (0, 31, 32, k - 1):
                bad = bits.copy()
                bad[position] = code
                yield ActCase(f'{prefix}.nonfinite.{code:x}.{position}', bad, traced,
                              k + 3, g + 3, rejected=True)
        for position in range(1, 31):
            bad = bits.copy()
            bad[position] = 0x7F80 if position % 2 else 0xFFFF
            yield ActCase(f'{prefix}.nonfinite.middle.{position}', bad, traced,
                          k + 3, g + 3, rejected=True)
    descriptor_bits = list(five_group_bits()[:64])
    yield ActCase('trace.descriptor.independent', descriptor_bits, True)
    yield ActCase('trace.descriptor.output-alias', descriptor_bits, True, trace_object_alias=True)
    shape = [g] * 4 + [k] * 2
    for index in range(6):
        for delta in (-1, 1):
            bad = shape.copy()
            bad[index] += delta
            for poisoned in (False, True):
                yield ActCase(f'trace.shape.{index}.{delta}.poison={poisoned}', bits, True,
                              k + 3, g + 3, rejected=True, trace_shape=bad, poison=poisoned)


def trace_alias_cases():
    bits = list(five_group_bits())
    k, g = len(bits), len(bits) // 32
    sizes = [2 * k, k, g] + [4 * g] * 4 + [4 * k] * 2
    pairs = [(i, j) for i in range(3, 9) for j in range(i)]
    assert len(pairs) == 33
    for i, j in pairs:
        pair = BUFFERS[i] + ',' + BUFFERS[j]
        second_alignment = 2 if j == 0 else 1 if j < 3 else 4
        reverse = (sizes[j] + 3) // 4 * 4
        placements = [(0, 0), (4, 0),
                      (0, (sizes[i] - 1) // second_alignment * second_alignment),
                      ((sizes[j] - 1) // 4 * 4, 0),
                      (0, sizes[i]), (reverse, reverse - sizes[j])]
        for first, second in dict.fromkeys(placements):
            overlaps = first < second + sizes[j] and second < first + sizes[i]
            for poisoned in ((False, True) if overlaps else (False,)):
                yield ActCase(f'trace.alias.{pair}.{first}.{second}.poison={poisoned}', bits,
                              True, rejected=overlaps, trace_alias=pair,
                              in_offset=first, out_offset=second, poison=poisoned)
    for i in range(3, 9):
        for output in ('a', 'scale'):
            start = ((k if output == 'a' else g) + 3) // 4 * 4
            capacity = start + sizes[i]
            yield ActCase(f'trace.unused-tail.{BUFFERS[i]}.{output}', bits, True,
                          a_cap=capacity if output == 'a' else k,
                          scale_cap=capacity if output == 'scale' else g,
                          trace_alias=BUFFERS[i] + ',' + output,
                          in_offset=start, out_offset=0)


def all_contract_cases():
    yield from contract_cases()
    yield from trace_alias_cases()


def combination_cases():
    rng = random.Random(0x1713)
    vectors = directed_vectors() + [('all-negative-zero', [0x8000] * 64)]
    for name, bits in vectors:
        a, sa = act_quant(bits)
        w = pack([rng.randrange(16) for _ in bits])
        sb = [127 + j % 5 for j in range(len(sa))]
        trace = trace_expert_grouped_dot(a, w, sa, sb)
        for out, value, width in ((FP32, trace.fp32_bits, 4), (BF16, trace.bf16_bits, 2)):
            yield ComputeCase(f'combination.{name}.{out}', 'act_quant_dot_' + out,
                              (bits, w, sb), 3, width, ('ok', [value]), out)


def selected(cases):
    return (c for c in cases if not SELECT or any(s in c.name for s in SELECT))


def failures(exe, cases, args=()):
    responses = run_executor(exe, cases, args)
    return [m for c, r in zip(cases, responses) if (m := check(c, r))]


class EnvironmentProbed(unittest.TestCase):
    def run_matrix(self, builder):
        source = iter(selected(builder()))
        count = 0
        start = time.perf_counter()
        while batch := list(itertools.islice(source, 8)):
            count += len(batch)
            for env, csr in ENVIRONMENTS.items():
                for status in ('clear', 'set'):
                    with self.subTest(env=env, status=status, first=batch[0].name):
                        probe = mxcsr_probe()
                        first, *middle, last = run_executor(
                            EXE, [probe, *batch, probe],
                            [f'--mxcsr={env}', f'--status={status}', *EXE_ARGS])
                        for response in (first, last):
                            self.assertEqual(response[1][0], csr | (MXCSR_STATUS if status == 'set' else 0))
                        bad = [m for c, r in zip(batch, middle) if (m := check(c, r))]
                        self.assertEqual(bad, [])
        self.assertGreater(count, 0, 'empty selection is not acceptance')
        print(f'P0017 {builder.__name__}: {count} cases x 16 MXCSR states; '
              f'{time.perf_counter() - start:.2f}s', file=sys.stderr)


class Parity(EnvironmentProbed):
    def test_all_stages(self):
        self.run_matrix(parity_cases)


class Contract(EnvironmentProbed):
    def test_metadata_contents_outputs_and_aliases(self):
        self.run_matrix(all_contract_cases)

    def test_alias_coverage(self):
        cases = list(trace_alias_cases())
        same = [c for c in cases if c.in_offset == c.out_offset == 0 and not c.poison]
        pairs = {tuple(c.trace_alias.split(',')) for c in same}
        self.assertEqual(len(same), 33)
        self.assertEqual(len(pairs), 33)
        self.assertEqual(sum(second in ('input', 'a', 'scale') for _, second in pairs), 18)
        self.assertEqual(sum(second in STAGES for _, second in pairs), 15)
        self.assertTrue(all(c.expect == ('err', 'invalid_argument') for c in same))
        for case in same:
            family = [c for c in cases if c.trace_alias == case.trace_alias and '.alias.' in c.name]
            self.assertTrue(any(c.poison and c.in_offset == c.out_offset == 0 for c in family))
            self.assertGreaterEqual(sum(c.expect[0] == 'ok' for c in family), 2)
        outputs = [c for c in contract_cases() if c.overlap == 'outputs']
        self.assertEqual(sum(c.in_offset == 5 and c.out_offset == 0 and c.expect[0] == 'ok'
                             for c in outputs), 2)

    def test_oracle_rejection(self):
        for bits in ([], [0] * 31, [0] * 33, [0] * 159 + [0x7F80], [0] * 159 + [0xFFFF]):
            for entry in (act_quant, trace_act_quant):
                with self.assertRaises(ValueError):
                    entry(bits)


class Combination(EnvironmentProbed):
    def test_production_buffers_feed_production_dot(self):
        self.run_matrix(combination_cases)


class Harness(unittest.TestCase):
    def sample(self):
        return vector('sample', five_group_bits())

    def test_faults(self):
        self.assertEqual(failures(EXE, self.sample()), [])
        for fault, evidence in (('crash', 'executor exited'), ('exit', 'executor exited 3'),
                                ('garbage', 'unparseable response'),
                                ('drop', 'expected 2 responses, got 1')):
            with self.subTest(fault=fault), self.assertRaisesRegex(HarnessError, evidence):
                failures(EXE, self.sample(), [f'--fault={fault}'])
        mismatches = failures(EXE, self.sample(), ['--fault=mismatch'])
        self.assertEqual(len(mismatches), 2)
        self.assertTrue(all('expected' in message and 'got' in message for message in mismatches))
        with self.assertRaises(HarnessError):
            failures(EXE.with_name('df_quant_exec_missing'), self.sample())

    def test_success_storage_detectors(self):
        cases = vector('input-snapshot', five_group_bits(), slack=3)
        cases += [c for c in all_contract_cases() if c.expect[0] == 'ok' and
                  (c.overlap is not None or c.trace_alias is not None or c.trace_object_alias)]
        self.assertEqual(failures(EXE, cases), [])
        for case in cases:
            with self.subTest(case=case.name), self.assertRaisesRegex(
                    HarnessError, 'activation input changed on success'):
                failures(EXE, [case], ['--fault=input-write'])
        gaps = [c for c in contract_cases() if '.shared-gap.' in c.name]
        self.assertEqual(len(gaps), 2)
        for case in gaps:
            with self.subTest(case=case.name), self.assertRaisesRegex(
                    HarnessError, 'activation bytes outside write ranges changed on success'):
                failures(EXE, [case], ['--fault=shared-gap-write'])

    def test_protocol(self):
        valid = self.sample()[0].line()
        head, payload = valid.split(' : ')
        malformed = [head, valid + ' ; 0', valid.replace(' cap=160', ''),
                     valid.replace(' scale_cap=5', ''), head + ' cap=160 : ' + payload,
                     head + ' scale_cap=5 : ' + payload, head + ' shape=160 : ' + payload,
                     head + ' scale_shape=5 : ' + payload, head + ' alias : ' + payload,
                     head + ' scale_alias : ' + payload, head + ' in_offset=2 : ' + payload,
                     head + ' out_offset=1 : ' + payload,
                     head + ' overlap=input_a in_offset=1 : ' + payload,
                     head + ' poison poison : ' + payload,
                     head + ' overlap=outputs overlap=input_a : ' + payload,
                     head + ' overlap=nothing : ' + payload,
                     head + ' trace_shape=5,5,5,5,160,160 : ' + payload,
                     valid.replace('act_quant ', 'act_quant_unknown ', 1),
                     head + ' : 10000 ' + payload,
                     'act_quant_trace cap=160 scale_cap=5 trace_shape=5,5 : ' + payload,
                     'act_quant_trace cap=160 scale_cap=5 trace_shape=5,5,5,5,160,160 '
                     'trace_shape=5,5,5,5,160,160 : ' + payload,
                     'decode_bf16 shape=1 cap=1 scale_cap=1 : 0',
                     'act_quant_dot_fp32 cap=1 scale_cap=1 : 0 ; 0 ; 7f',
                     'act_quant_dot_fp32 cap=1 : 0 ; 0',
                     'act_quant_dot_fp32 cap=1 poison : 0 ; 0 ; 7f']
        trace_head = head.replace('act_quant ', 'act_quant_trace ', 1)
        for pair in ('amax_bits,input', 'amax_bits,a', 'amax_bits,exponent'):
            malformed += [head + f' trace_alias={pair} : ' + payload,
                          trace_head + f' trace_alias={pair} trace_alias={pair} : ' + payload,
                          trace_head + f' trace_alias={pair} overlap=outputs : ' + payload,
                          trace_head + f' overlap=outputs trace_alias={pair} : ' + payload,
                          trace_head + f' trace_alias={pair} in_offset=2 : ' + payload]
        for pair in ('input,a', 'amax_bits,amax_bits', 'amax_bits', 'amax_bits,',
                     ',amax_bits', 'amax_bits,nothing', 'amax_bits,input,a'):
            malformed.append(trace_head + f' trace_alias={pair} : ' + payload)
        malformed += [trace_head + ' trace_alias=amax_bits,input out_offset=1 : ' + payload,
                      trace_head + ' trace_alias=amax_bits,exponent out_offset=2 : ' + payload,
                      trace_head + ' overlap=amax_bits,input : ' + payload,
                      'decode_bf16 shape=1 cap=1 trace_alias=amax_bits,input : 0',
                      'act_quant_dot_fp32 cap=1 trace_alias=amax_bits,input : 0 ; 0 ; 7f',
                      head + ' trace_object_alias : ' + payload,
                      trace_head + ' trace_object_alias : ' + payload,
                      trace_head + ' trace_object_alias trace_object_alias : ' + payload,
                      'decode_bf16 shape=1 cap=1 trace_object_alias : 0',
                      'act_quant_trace cap=64 scale_cap=2 trace_object_alias overlap=outputs : '
                      + ' '.join(['0'] * 64)]
        for text in malformed:
            with self.subTest(text=text[:120]), self.assertRaises(HarnessError):
                failures(EXE, [RawCase('bad', text)])
        for arg in ('--status=maybe', '--mxcsr=maybe', '--fault=maybe'):
            with self.assertRaises(HarnessError):
                failures(EXE, self.sample(), [arg])

    def test_trace_response_shape(self):
        trace = self.sample()[1]
        for line in ('ok ' + ' '.join(['00000000'] * (trace.cap - 1)),
                     'ok ' + ' '.join(['0000'] * trace.cap),
                     'ok ' + ' '.join(['00000000'] * (trace.cap + 1))):
            with self.assertRaises(HarnessError):
                parse_response(trace, line)
        trace.expect = ('ok', trace.expect[1][:-1])
        self.assertTrue(failures(EXE, [trace]))

    def test_poison_probe(self):
        asan = run_executor(EXE, [Case('asan', 'asan', (), [], 1, 4, ('ok', []))])[0][1][0]
        self.assertIn(asan, (0, 1))
        for traced in (False, True):
            case = ActCase('poison-valid', five_group_bits(), traced, poison=True)
            if asan:
                with self.assertRaises(HarnessError):
                    failures(EXE, [case])
            else:
                self.assertEqual(failures(EXE, [case]), [])
        for target, first, second in (('amax_bits,input', 0, 20),
                                      ('amax_bits,a', 0, 20),
                                      ('amax_bits,exponent', 0, 20)):
            case = ActCase('poison-valid-alias', five_group_bits(), True, trace_alias=target,
                           in_offset=first, out_offset=second, poison=True)
            if asan:
                with self.assertRaises(HarnessError):
                    failures(EXE, [case])
            else:
                self.assertEqual(failures(EXE, [case]), [])
        print(f'P0017 harness ASan={asan}', file=sys.stderr)

    def test_failure_propagates_to_driver_exit(self):
        clean_runs = {}
        for suite, fault, selector, evidence in (
                ('parity', 'mismatch', 'signed-zero', 'signed-zero.ordinary[0]: expected'),
                ('parity', 'garbage', 'signed-zero', 'unparseable response'),
                ('parity', 'drop', 'signed-zero', 'expected 4 responses, got 3'),
                ('contract', 'crash', 'capacity', 'executor exited'),
                ('combination', 'exit', 'signed-zero', 'executor exited 3'),
                ('parity', 'input-write', 'signed-zero', 'activation input changed on success'),
                ('contract', 'shared-gap-write', 'shared-gap',
                 'activation bytes outside write ranges changed on success')):
            command = [sys.executable, __file__, '--exe', str(EXE), '--suite', suite,
                       '--case', selector]
            with self.subTest(suite=suite, fault=fault, selector=selector):
                if (suite, selector) not in clean_runs:
                    clean_runs[suite, selector] = subprocess.run(
                        command, capture_output=True, text=True, timeout=900)
                clean = clean_runs[suite, selector]
                self.assertEqual(clean.returncode, 0, clean.stdout + clean.stderr)
                proc = subprocess.run([*command, f'--exe-arg=--fault={fault}'],
                                      capture_output=True, text=True, timeout=900)
                self.assertNotEqual(proc.returncode, 0)
                self.assertIn(evidence, proc.stderr)
                self.assertNotIn('empty selection', proc.stderr)
                self.assertNotIn('skipped', proc.stderr.lower())
        for suite, selector, succeeds in (('contract', 'trace.alias.', True),
                                           ('contract', 'descriptor', True),
                                           ('contract', 'unknown-p0017-selection', False)):
            proc = subprocess.run([sys.executable, __file__, '--exe', str(EXE), '--suite', suite,
                                   '--case', selector], capture_output=True, text=True, timeout=900)
            self.assertEqual(proc.returncode == 0, succeeds, proc.stdout + proc.stderr)
            self.assertNotIn('skipped', proc.stderr.lower())
            if not succeeds:
                self.assertIn('empty selection is not acceptance', proc.stderr)
        malformed = self.sample()[1]
        malformed.trace_shape = [5, 5]
        with self.assertRaises(HarnessError):
            failures(EXE, [malformed])


def numeric_report():
    records, errors = [], []
    try:
        cases = list(selected(c for name, bits in directed_vectors() for c in vector(name, bits)))
        if not cases:
            raise HarnessError('empty numeric selection')
        responses = run_executor(EXE, cases, EXE_ARGS)
        for case, response in zip(cases, responses):
            if response[0] != 'ok':
                errors.append(case.name + ': valid request rejected')
                continue
            mismatches = [f'{case.labels[i]}[{i}]' for i, (a, b) in
                          enumerate(zip(case.expect[1], response[1])) if a != b]
            records.append({'case': case.name, 'mismatches': mismatches})
    except Exception as error:
        errors.append(type(error).__name__ + ': ' + str(error))
    payload = {'cases': records, 'errors': errors, 'skipped': 0}
    print('P0017_NUMERIC ' + json.dumps(payload), flush=True)
    return int(bool(errors or any(c['mismatches'] for c in records)))


SUITES = {'parity': Parity, 'contract': Contract, 'combination': Combination, 'harness': Harness}


def main():
    global EXE
    parser = argparse.ArgumentParser()
    parser.add_argument('--exe', required=True, type=Path)
    parser.add_argument('--suite', required=True, choices=SUITES)
    parser.add_argument('--exe-arg', action='append', default=[])
    parser.add_argument('--case', action='append', default=[])
    parser.add_argument('--numeric-report', action='store_true')
    args = parser.parse_args()
    EXE = args.exe
    EXE_ARGS.extend(args.exe_arg)
    SELECT.extend(args.case)
    if args.numeric_report:
        return numeric_report()
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(SUITES[args.suite]))
    return 0 if result.wasSuccessful() and result.testsRun and not result.skipped else 1


if __name__ == '__main__':
    sys.exit(main())
