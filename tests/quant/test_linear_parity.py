#!/usr/bin/env python3
"""P0019 public expert_linear parity, storage contracts and driver fail-closed tests.

shape=N is the row count; three sections are BF16 input, packed row-major W,
and row-major scale_b. K is the input length. Optional workspace responses append
all a_codes and scale_a capacity after all output capacity, as uint32 words.
Expected vectors compose the unchanged activation and grouped-dot references.
"""

import argparse
import itertools
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

from ref import formats as f
from ref.act_quant import act_quant
from ref.compute import trace_expert_grouped_dot
from test_act_quant_parity import directed_vectors
from test_codec_parity import (BF16, FP32, Case, HarnessError, check, is_nan,
                               parse_response, run_executor)
from test_dequant_parity import ENVIRONMENTS, MXCSR_STATUS, RawCase, mxcsr_probe

EXE = None
EXE_ARGS = []
SELECT = []
BUFFERS = ('input', 'w', 'sb', 'a', 'sa', 'out')
FILL = 0xA5


def oracle(bits, n, weights, sb):
    a, sa = act_quant(bits)
    k, g = len(bits), len(bits) // 32
    traces = [trace_expert_grouped_dot(a, weights[r * (k // 2):(r + 1) * (k // 2)],
                                     sa, sb[r * g:(r + 1) * g]) for r in range(n)]
    return list(a), list(sa), traces


class LinearCase(Case):
    """Shared-byte initialization plus independent reference prefix writes."""

    def __init__(self, name, bits, n, weights, sb, kind=FP32, cap=None, a_cap=None,
                 scale_cap=None, workspace=False, overlap=None, in_offset=0,
                 out_offset=0, rejected=False, poison=False, poison_writes=False):
        self.n = n
        self.sections = [list(bits), list(weights), list(sb)]
        self.result_cap = n if cap is None else cap
        self.a_cap = len(bits) if a_cap is None else a_cap
        self.scale_cap = len(bits) // 32 if scale_cap is None else scale_cap
        self.workspace = workspace
        self.overlap = overlap
        self.in_offset, self.out_offset = in_offset, out_offset
        self.poison, self.poison_writes = poison, poison_writes
        self.result_width = 4 if kind == FP32 else 2
        count = self.result_cap + (self.a_cap + self.scale_cap if workspace else 0)
        expected = ('err', 'invalid_argument') if rejected else ('ok', self.expected(kind))
        super().__init__(name + '.' + kind, 'expert_linear_' + kind, (n,), [], count,
                         4 if workspace else self.result_width, expected, kind)

    def expected(self, kind):
        sizes = [2 * len(self.sections[0]), len(self.sections[1]), len(self.sections[2]),
                 self.a_cap, self.scale_cap, self.result_cap * self.result_width]
        owners = [bytearray([FILL]) * s for s in sizes]
        offsets = [0] * 6
        if self.overlap:
            pair = [BUFFERS.index(x) for x in self.overlap.split(',')]
            backing = bytearray([FILL]) * max(sizes[i] + o for i, o in
                                             zip(pair, (self.in_offset, self.out_offset)))
            for i, o in zip(pair, (self.in_offset, self.out_offset)):
                owners[i], offsets[i] = backing, o

        def write(i, data):
            owners[i][offsets[i]:offsets[i] + len(data)] = data

        def read(i):
            return owners[i][offsets[i]:offsets[i] + sizes[i]]

        write(0, b''.join(v.to_bytes(2, sys.byteorder) for v in self.sections[0]))
        write(1, bytes(self.sections[1]))
        write(2, bytes(self.sections[2]))
        data = read(0)
        actual_bits = [int.from_bytes(data[i:i + 2], sys.byteorder) for i in range(0, len(data), 2)]
        a, sa, traces = oracle(actual_bits, self.n, list(read(1)), list(read(2)))
        write(3, bytes(a))
        write(4, bytes(sa))
        results = [t.fp32_bits if kind == FP32 else t.bf16_bits for t in traces]
        write(5, b''.join(v.to_bytes(self.result_width, sys.byteorder) for v in results))
        result = read(5)
        values = [int.from_bytes(result[i:i + self.result_width], sys.byteorder)
                  for i in range(0, len(result), self.result_width)]
        return values + (list(read(3)) + list(read(4)) if self.workspace else [])

    def line(self):
        head = [self.op, f'shape={self.n}', f'cap={self.result_cap}',
                f'a_cap={self.a_cap}', f'scale_cap={self.scale_cap}']
        if self.workspace:
            head.append('workspace')
        if self.overlap:
            head += [f'overlap={self.overlap}', f'in_offset={self.in_offset}',
                     f'out_offset={self.out_offset}']
        if self.poison:
            head.append('poison')
        if self.poison_writes:
            head.append('poison_writes')
        return ' '.join(head) + ' : ' + ' ; '.join(' '.join(f'{v:x}' for v in s)
                                                  for s in self.sections)


def check_linear(case, response):
    if not isinstance(case, LinearCase) or response[0] != 'ok' or case.expect[0] != 'ok':
        return check(case, response)
    for i, (want, got) in enumerate(zip(case.expect[1], response[1])):
        if i < case.result_cap and case.kind == BF16 and got > 0xFFFF:
            return f'{case.name}[{i}]: BF16 output exceeds 16 bits: {got:#x}'
        if want == got:
            continue
        if i < case.n and is_nan(want, case.kind) and is_nan(got, case.kind):
            continue
        return f'{case.name}[{i}]: expected {want:#x}, got {got:#x}'
    return None


def failures(exe, cases, args=()):
    return [m for c, r in zip(cases, run_executor(exe, cases, args)) if (m := check_linear(c, r))]


def both(name, bits, n, weights, sb, **kwargs):
    return [LinearCase(name, bits, n, weights, sb, kind, **kwargs) for kind in (FP32, BF16)]


def layout():
    bits = [0x3F80, 0x4000] + [0] * 30 + [0x4080] + [0] * 31
    # Low nibble first: rows give (1+8)+4*2, (2+2)*4-4, (-1+4)/2+4*3*4.
    weights = ([0x62] + [0] * 15 + [0x02] + [0] * 15
               + [0x24] + [0] * 15 + [0x0A] + [0] * 15
               + [0x4A] + [0] * 15 + [0x05] + [0] * 15)
    return bits, 3, weights, [127, 128, 129, 127, 126, 129]


def parity_cases():
    args = layout()
    for case in both('layout.hand', *args, workspace=True):
        expected = [17.0, 12.0, 49.5]
        manual = [f.fp32_bits(v) if case.kind == FP32 else f.fp32_bits_to_bf16(f.fp32_bits(v))
                  for v in expected]
        assert case.expect[1][:3] == manual
        yield case
    rng = random.Random(1919)
    for name, bits in directed_vectors() + [('negative-zero', [0x8000] * 64)]:
        n, k = 5, len(bits)
        weights = [rng.randrange(256) for _ in range(n * k // 2)]
        sb = [120 + (7 * r + j) % 15 for r in range(n) for j in range(k // 32)]
        yield from both(name, bits, n, weights, sb, cap=n + 3,
                        a_cap=k + 5, scale_cap=k // 32 + 3, workspace=True)
    for n, k in ((1, 32), (2, 512), (65, 32), (33, 64), (7, 160), (3, 256)):
        bits = [rng.choice((0x3F80, 0xBF80, 0x4000, 0x4100, 0x3C80, 0x8000, 1)) for _ in range(k)]
        weights = [rng.randrange(256) for _ in range(n * k // 2)]
        sb = [rng.randrange(110, 145) for _ in range(n * k // 32)]
        yield from both(f'rectangular.{n}x{k}', bits, n, weights, sb, workspace=True)
        yield from both(f'ordinary.{n}x{k}', bits, n, weights, sb)
    for code in (0, 1, 0x7E, 0x7F, 0x80, 0xFD, 0xFE):
        yield from both(f'scale-edge.{code}', [0x7F7F, 0xFF7F] + [1] * 30, 3,
                        [0x22] * 16 + [0xAA] * 16 + [0x2A] * 16, [code] * 3, workspace=True)
    yield from both('nan.opposite-inf', [0x7F7F] * 64, 2,
                    [0x22] * 16 + [0xAA] * 16 + [0xAA] * 32, [0xFE] * 4, workspace=True)
    # Exact FP32 totals 1+2^-8 and 1+3*2^-8 straddle alternating BF16 ties.
    bits = [0x3F80] + [0] * 31 + [0x3F80] + [0] * 31
    weights = [0x02] + [0] * 15 + [0x02] + [0] * 15
    weights += [0x02] + [0] * 15 + [0x05] + [0] * 15
    for case in both('bf16.tie', bits, 2, weights, [127, 119, 127, 119], workspace=True):
        assert case.expect[1][:2] == ([0x3F808000, 0x3F818000] if case.kind == FP32 else [0x3F80, 0x3F82])
        yield case


def contract_cases():
    bits, n, w, sb = layout()
    k, g = len(bits), len(bits) // 32
    for name, b, rows, weights, scales, opts in (
            ('zero-n', bits, 0, [], [], {}),
            ('zero-k', [], 3, [], [], {}),
            ('short-k', bits[:-1], n, w, sb, {}),
            ('long-k', bits + [0], n, w, sb, {}),
            ('short-w', bits, n, w[:-1], sb, {}),
            ('long-w', bits, n, w + [0], sb, {}),
            ('short-sb', bits, n, w, sb[:-1], {}),
            ('long-sb', bits, n, w, sb + [127], {}),
            ('short-a', bits, n, w, sb, {'a_cap': k - 1}),
            ('empty-a', bits, n, w, sb, {'a_cap': 0}),
            ('short-sa', bits, n, w, sb, {'scale_cap': g - 1}),
            ('empty-sa', bits, n, w, sb, {'scale_cap': 0}),
            ('short-out', bits, n, w, sb, {'cap': n - 1}),
            ('empty-out', bits, n, w, sb, {'cap': 0}),
            ('overflow-nk', bits, (1 << 64) - 1, w, sb, {'cap': n}),
            ('overflow-nk-half', bits, (1 << 64) // k + 1, w, sb, {'cap': n})):
        for poison in (False, True):
            yield from both(f'metadata.{name}.poison={poison}', b, rows, weights, scales,
                            rejected=True, poison=poison, workspace=True, **opts)
    for pos in range(len(sb)):
        bad = sb.copy()
        bad[pos] = 255
        for poison_writes in (False, True):
            yield from both(f'contents.sb.{pos}.{poison_writes}', bits, n, w, bad,
                            rejected=True, workspace=True, poison_writes=poison_writes)
    for code in (0x7F80, 0xFF80, 0x7F81, 0x7FC0, 0xFFFF):
        for pos in (0, 31, 32, k - 1):
            bad = bits.copy()
            bad[pos] = code
            yield from both(f'contents.bf16.{code}.{pos}', bad, n, w, sb,
                            rejected=True, workspace=True, poison_writes=True)
    for kind in (FP32, BF16):
        width = 4 if kind == FP32 else 2
        sizes = [2 * k, len(w), len(sb), k, g, n * width]
        alignment = [2, 1, 1, 1, 1, width]
        pairs = [(i, j) for i in range(3, 6) for j in range(i)]
        assert len(pairs) == 12
        for i, j in pairs:
            pair = BUFFERS[i] + ',' + BUFFERS[j]
            placements = [(0, 0), (alignment[i], 0),
                          (0, (sizes[i] - 1) // alignment[j] * alignment[j]),
                          ((sizes[j] - 1) // alignment[i] * alignment[i], 0),
                          (0, (sizes[i] + alignment[j] - 1) // alignment[j] * alignment[j]),
                          ((sizes[j] + alignment[i] - 1) // alignment[i] * alignment[i], 0)]
            for first, second in dict.fromkeys(placements):
                overlaps = first < second + sizes[j] and second < first + sizes[i]
                if kind == FP32 and (i, j) == (5, 0) and overlaps:
                    continue
                for poisoned in ((False, True) if overlaps else (False,)):
                    yield LinearCase(f'alias.{pair}.{first}.{second}.{poisoned}', bits, n, w, sb,
                                     kind, workspace=True, overlap=pair, in_offset=first,
                                     out_offset=second, rejected=overlaps, poison=poisoned)
            if kind == FP32 and (i, j) == (5, 0):
                continue
            # Both full spans overlap, but the later buffer begins beyond the first write prefix.
            start = (sizes[i] + alignment[j] - 1) // alignment[j] * alignment[j]
            opts = {'a_cap': k, 'scale_cap': g, 'cap': n}
            key = {3: 'a_cap', 4: 'scale_cap', 5: 'cap'}[i]
            opts[key] = (start + sizes[j] + alignment[i] - 1) // alignment[i]
            yield LinearCase(f'tail.{pair}', bits, n, w, sb, kind, workspace=True,
                             overlap=pair, in_offset=0, out_offset=start, **opts)
            if j >= 3:
                start = (sizes[j] + alignment[i] - 1) // alignment[i] * alignment[i]
                opts = {'a_cap': k, 'scale_cap': g, 'cap': n}
                opts[{3: 'a_cap', 4: 'scale_cap', 5: 'cap'}[j]] = (
                    start + sizes[i] + alignment[j] - 1) // alignment[j]
                yield LinearCase(f'tail.reverse.{pair}', bits, n, w, sb, kind, workspace=True,
                                 overlap=pair, in_offset=start, out_offset=0, **opts)
        for pair in ('input,w', 'input,sb', 'w,sb'):
            yield LinearCase('readonly.' + pair, bits, n, [0x22] * len(w), [120] * len(sb), kind,
                             overlap=pair, workspace=True)
        yield LinearCase('shared-gap', bits, n, w, sb, kind, overlap='a,sa',
                         out_offset=k + 8, workspace=True)
        yield LinearCase('slack', bits, n, w, sb, kind, cap=n + 3, a_cap=k + 3,
                         scale_cap=g + 3, workspace=True)


def selected(builder):
    return [c for c in builder() if not SELECT or any(s in c.name for s in SELECT)]


class EnvironmentProbed(unittest.TestCase):
    def run_matrix(self, builder):
        cases = selected(builder)
        self.assertTrue(cases, 'empty selection is not acceptance')
        start = time.perf_counter()
        for env, csr in ENVIRONMENTS.items():
            for status in ('clear', 'set'):
                with self.subTest(env=env, status=status):
                    probe = mxcsr_probe()
                    first, *middle, last = run_executor(
                        EXE, [probe, *cases, probe],
                        [f'--mxcsr={env}', f'--status={status}', *EXE_ARGS])
                    for response in (first, last):
                        self.assertEqual(response[1][0], csr | (MXCSR_STATUS if status == 'set' else 0))
                    self.assertEqual([m for c, r in zip(cases, middle) if (m := check_linear(c, r))], [])
        print(f'P0019 {builder.__name__}: {len(cases)} cases x 16 MXCSR states; '
              f'{time.perf_counter() - start:.2f}s', file=sys.stderr)


class Parity(EnvironmentProbed):
    def test_full_vectors_and_workspace(self):
        self.run_matrix(parity_cases)


class Contract(EnvironmentProbed):
    def test_metadata_contents_storage(self):
        self.run_matrix(contract_cases)

    def test_alias_coverage(self):
        cases = list(contract_cases())
        for kind in (FP32, BF16):
            same = [c for c in cases if c.kind == kind and c.name.startswith('alias.')
                    and c.in_offset == c.out_offset == 0 and not c.poison]
            self.assertEqual(len(same), 11 if kind == FP32 else 12)
            adjacent = [c for c in cases if c.kind == kind and c.name.startswith('alias.out,input.')
                        and c.expect[0] == 'ok']
            self.assertEqual(len(adjacent), 2)
            self.assertTrue(all(c.expect[0] == 'err' for c in same))
            for case in same:
                family = [c for c in cases if c.kind == kind and c.overlap == case.overlap]
                self.assertGreaterEqual(sum(c.expect[0] == 'ok' for c in family), 3)
                self.assertTrue(any(c.poison for c in family))
            self.assertEqual(sum(c.kind == kind and c.name.startswith('readonly.') for c in cases), 3)


class Harness(unittest.TestCase):
    def sample(self):
        return both('layout.hand', *layout(), workspace=True)

    def test_faults(self):
        self.assertEqual(failures(EXE, self.sample()), [])
        for fault, evidence in (('crash', 'executor exited'), ('exit', 'executor exited 3'),
                                ('garbage', 'unparseable response'),
                                ('drop', 'expected 2 responses, got 1')):
            with self.subTest(fault=fault), self.assertRaisesRegex(HarnessError, evidence):
                failures(EXE, self.sample(), [f'--fault={fault}'])
        self.assertEqual(len(failures(EXE, self.sample(), ['--fault=mismatch'])), 2)
        with self.assertRaises(HarnessError):
            failures(EXE.with_name('missing_p0019_executor'), self.sample())

    def test_snapshots_and_poison(self):
        for case in self.sample():
            with self.assertRaisesRegex(HarnessError, 'linear input changed on success'):
                failures(EXE, [case], ['--fault=input-write'])
        gaps = [c for c in contract_cases() if c.name.startswith('shared-gap')]
        self.assertEqual(failures(EXE, gaps), [])
        with self.assertRaisesRegex(HarnessError, 'outside write ranges'):
            failures(EXE, gaps, ['--fault=shared-gap-write'])
        probe = Case('asan', 'asan', (), [], 1, 4, ('ok', []))
        asan = run_executor(EXE, [probe])[0][1][0]
        for field in ('poison', 'poison_writes'):
            for case in self.sample():
                setattr(case, field, True)
                if asan:
                    with self.assertRaisesRegex(HarnessError, 'AddressSanitizer'):
                        failures(EXE, [case])
                else:
                    self.assertEqual(failures(EXE, [case]), [])
        print(f'P0019 harness ASan={asan}', file=sys.stderr)

    def test_protocol(self):
        case = self.sample()[0]
        valid = case.line()
        head, payload = valid.split(' : ')
        malformed = [head, valid + ' ;', valid.rsplit(' ; ', 1)[0],
                     valid.replace('shape=3', 'shape=3,64'), valid.replace('shape=3', 'shape='),
                     valid.replace('shape=3 ', ''), valid.replace('cap=3 ', '', 1),
                     head + ' cap=3 : ' + payload, head + ' a_cap=64 : ' + payload,
                     head + ' scale_cap=2 : ' + payload, head + ' workspace : ' + payload,
                     head + ' alias : ' + payload, head + ' scale_alias : ' + payload,
                     head + ' scale_shape=3,2 : ' + payload, head + ' trace_shape=1 : ' + payload,
                     head + ' in_offset=2 : ' + payload, head + ' out_offset=2 : ' + payload,
                     head + ' overlap=a,out out_offset=1 : ' + payload,
                     head + ' overlap=input,a in_offset=1 : ' + payload,
                     head + ' poison poison : ' + payload,
                     head + ' poison_writes poison_writes : ' + payload,
                     head + ' : 10000 ' + payload,
                     valid.replace(' ; 62 ', ' ; 100 ', 1), valid.rsplit(' ; ', 1)[0] + ' ; 100',
                     valid.replace('expert_linear_fp32', 'expert_linear_unknown'),
                     'decode_bf16 cap=1 shape=1 workspace : 0',
                     'act_quant cap=32 scale_cap=1 a_cap=32 : ' + '0 ' * 32]
        for pair in ('a,a', 'a', 'a,', ',a', 'a,foo', 'a,sa,out'):
            malformed.append(head + f' overlap={pair} : ' + payload)
        for text in malformed:
            with self.subTest(text=text[:100]), self.assertRaises(HarnessError):
                failures(EXE, [RawCase('bad', text)])
        for arg in ('--mxcsr=unknown', '--status=unknown', '--fault=unknown'):
            with self.assertRaises(HarnessError):
                failures(EXE, self.sample(), [arg])
        for line in ('ok ' + ' '.join(['00000000'] * (case.cap - 1)),
                     'ok ' + ' '.join(['00000000'] * (case.cap + 1)),
                     'ok ' + ' '.join(['0000'] * case.cap), 'err invalid_argument changed=2'):
            with self.assertRaises(HarnessError):
                parse_response(case, line)
        for response in (('err', 'invalid_argument', False), ('err', 'overflow_error', False)):
            self.assertIsNotNone(check_linear(case, response))
        rejected = next(c for c in contract_cases() if c.expect[0] == 'err')
        self.assertIsNotNone(check_linear(rejected, ('err', 'invalid_argument', True)))
        self.assertIsNotNone(check_linear(rejected, ('err', 'overflow_error', False)))

    def test_bf16_workspace_nan_width(self):
        case = next(c for c in parity_cases() if c.name == 'nan.opposite-inf.bf16')
        self.assertTrue(case.workspace)
        self.assertTrue(is_nan(case.expect[1][0], BF16))
        for value in (0x7FC1, 0xFFC2, 0xABCD7FC0, 0x00017FC1, 0xABCDFFC2):
            words = case.expect[1].copy()
            words[0] = value
            response = parse_response(case, 'ok ' + ' '.join(f'{word:08x}' for word in words))
            with self.subTest(value=hex(value)):
                message = check_linear(case, response)
                if value <= 0xFFFF:
                    self.assertIsNone(message)
                else:
                    self.assertIn('BF16 output exceeds 16 bits', message)

    def test_fp32_input_output_span_construction(self):
        bits, n, w, sb = layout()
        for pair in ('out,input', 'input,out'):
            for tail in (False, True):
                first, second = ((0, 12) if pair == 'out,input' else (12, 0)) if tail else (0, 0)
                case = LinearCase('invalid-typed-storage', bits, n, w, sb, FP32,
                                  cap=35 if tail else n, overlap=pair,
                                  in_offset=first, out_offset=second, rejected=True)
                with self.subTest(pair=pair, tail=tail), self.assertRaisesRegex(
                        HarnessError, 'FP32 output and BF16 input full spans cannot intersect'):
                    failures(EXE, [case])

    def test_shared_output_tail_serialization(self):
        cases = []
        for source in contract_cases():
            target = 'tail.out,w.' if source.kind == FP32 else 'tail.out,input.'
            if not source.name.startswith(target):
                continue
            for workspace in (False, True):
                cases.append(LinearCase('serialize-shared-input-tail', source.sections[0], source.n,
                                        source.sections[1], source.sections[2], source.kind,
                                        cap=source.result_cap, a_cap=source.a_cap,
                                        scale_cap=source.scale_cap, workspace=workspace,
                                        overlap=source.overlap, in_offset=source.in_offset,
                                        out_offset=source.out_offset))
        self.assertEqual(len(cases), 4)
        self.assertEqual(failures(EXE, cases), [])

    def test_numeric_fault_sensitivity(self):
        cases = self.sample()
        for case in cases:
            bits, weights, sb = case.sections
            k = len(bits)
            variants = {
                'wrong-row': (weights[k // 2:] + weights[:k // 2], sb),
                'wrong-scale': (weights, sb[1:] + sb[:1]),
                'nibble-reversed': ([(b >> 4) | ((b & 15) << 4) for b in weights], sb),
            }
            for name, (w, s) in variants.items():
                mutant = LinearCase(name, bits, case.n, w, s, case.kind, workspace=True)
                response = run_executor(EXE, [mutant])[0]
                self.assertIsNotNone(check_linear(case, response), name)
            original = run_executor(EXE, [case])[0][1]
            for pos in (0, case.n - 1, case.result_cap, case.cap - 1):
                changed = original.copy()
                changed[pos] ^= 1
                self.assertIsNotNone(check_linear(case, ('ok', changed)))
            changed = original.copy()
            changed[case.n - 1] = int.from_bytes(bytes([FILL]) * case.result_width, 'little')
            self.assertIsNotNone(check_linear(case, ('ok', changed)), 'last output not written')

    def test_driver_exit(self):
        for suite, selector, fault, evidence in (
                ('parity', 'layout.hand', 'mismatch', 'expected'),
                ('parity', 'layout.hand', 'garbage', 'unparseable response'),
                ('parity', 'layout.hand', 'drop', 'expected 4 responses, got 3'),
                ('contract', 'metadata.short-a', 'exit', 'executor exited 3'),
                ('parity', 'layout.hand', 'input-write', 'linear input changed'),
                ('contract', 'shared-gap', 'shared-gap-write', 'outside write ranges')):
            command = [sys.executable, __file__, '--exe', str(EXE), '--suite', suite, '--case', selector]
            clean = subprocess.run(command, capture_output=True, text=True, timeout=900)
            self.assertEqual(clean.returncode, 0, clean.stdout + clean.stderr)
            proc = subprocess.run([*command, f'--exe-arg=--fault={fault}'],
                                  capture_output=True, text=True, timeout=900)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn(evidence, proc.stderr)
        proc = subprocess.run([sys.executable, __file__, '--exe', str(EXE), '--suite', 'parity',
                               '--case', 'nonexistent-p0019-case'], capture_output=True,
                              text=True, timeout=900)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn('empty selection is not acceptance', proc.stderr)


SUITES = {'parity': Parity, 'contract': Contract, 'harness': Harness}


def main():
    global EXE
    parser = argparse.ArgumentParser()
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--suite', choices=SUITES, required=True)
    parser.add_argument('--exe-arg', action='append', default=[])
    parser.add_argument('--case', action='append', default=[])
    args = parser.parse_args()
    EXE = args.exe
    EXE_ARGS.extend(args.exe_arg)
    SELECT.extend(args.case)
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(SUITES[args.suite]))
    return 0 if result.wasSuccessful() and result.testsRun and not result.skipped else 1


if __name__ == '__main__':
    sys.exit(main())
