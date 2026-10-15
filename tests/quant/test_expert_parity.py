#!/usr/bin/env python3
"""P0022 complete FP4 expert parity, storage contracts and fail-closed harness."""

import argparse
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

from ref import expert as impl
from test_codec_parity import Case, HarnessError, fill_value, parse_response, run_executor
from test_dequant_parity import ENVIRONMENTS, MXCSR_STATUS, mxcsr_probe

EXE = None
EXE_ARGS = []
SELECT = []
OMIT = object()
STAGES = ('input_codes', 'input_scales', 'gate_bf16', 'up_bf16', 'gate_bits',
          'up_bits', 'silu_bits', 'gated_bits', 'weighted_bits', 'hidden_bf16',
          'hidden_codes', 'hidden_scales', 'output_bf16')


def nan16(value):
    return (value & 0x7FFF) > 0x7F80


def nan32(value):
    return (value & 0x7FFFFFFF) > 0x7F800000


def matrix(n, k, code=0x22, scale=127):
    return impl.FP4Matrix(n, k, [code] * (n * k // 2), [scale] * (n * k // 32))


def flatten_trace(trace):
    """Semantic expert trace order shared by parity and the offline weight tool."""
    return [*trace.input_codes, *trace.input_scales, *trace.gate_bf16, *trace.up_bf16,
            *trace.bridge.gate_bits, *trace.bridge.up_bits, *trace.bridge.silu_bits,
            *trace.bridge.gated_bits, *trace.bridge.weighted_bits, *trace.bridge.bf16_bits,
            *trace.hidden_codes, *trace.hidden_scales, *trace.output_bf16]


def execute(exe, cases, args=()):
    """Run complete expert cases and return strict parsed responses."""
    return run_executor(exe, cases, args)


class ExpertCase(Case):
    """Seven-payload complete-expert request with complete workspace serialization."""

    def __init__(self, name, input_bf16, w1, w2, w3, route=OMIT, traced=False,
                 cap=None, work_shape=None, trace_shape=None, rejected=False,
                 runtime_failure=False, poison=False, poison_writes=False,
                 overlap=None, in_offset=0, out_offset=0, expected_trace=None):
        self.input = list(input_bf16)
        self.matrices = [w1, w2, w3]
        self.route = route
        self.traced = traced
        self.result_cap = w2.n if cap is None else cap
        d, h = w1.k, w1.n
        self.work_shape = list(work_shape or (d, d // 32, h, h, h, h, h // 32))
        self.trace_shape = list(trace_shape) if trace_shape is not None else None
        self.runtime_failure = runtime_failure
        self.expected_trace = expected_trace
        self.poison, self.poison_writes = poison, poison_writes
        self.overlap = overlap
        self.in_offset, self.out_offset = in_offset, out_offset
        self.sections = [self.input, list(w1.w_packed), list(w1.scale_b),
                         list(w2.w_packed), list(w2.scale_b),
                         list(w3.w_packed), list(w3.scale_b)]
        trace_caps = self.trace_shape if self.trace_shape is not None else [h] * 5
        self.labels = (['input_codes'] * self.work_shape[0] +
                       ['input_scales'] * self.work_shape[1] +
                       ['gate_bf16'] * self.work_shape[2] +
                       ['up_bf16'] * self.work_shape[3])
        if traced:
            self.labels += [label for label, count in zip(STAGES[4:9], trace_caps)
                            for _ in range(count)]
        self.labels += (['hidden_bf16'] * self.work_shape[4] +
                        ['hidden_codes'] * self.work_shape[5] +
                        ['hidden_scales'] * self.work_shape[6] +
                        ['output_bf16'] * self.result_cap)
        expected = ('err', 'invalid_argument') if rejected else ('ok', self.expected())
        suffix = '.trace' if traced else '.ordinary'
        super().__init__(name + suffix, 'expert_trace' if traced else 'expert_bf16', (), [],
                         len(self.labels), 4, expected)

    def expected(self):
        trace = self.expected_trace
        if trace is None:
            trace = impl.trace_expert(self.input, *self.matrices,
                                      route_weight_bits=None if self.route is OMIT else self.route)
        d, h = self.matrices[0].k, self.matrices[0].n
        sources = {7: trace.input_codes, 8: trace.input_scales, 9: trace.gate_bf16,
                   10: trace.up_bf16, 11: trace.bridge.bf16_bits,
                   12: trace.hidden_codes, 13: trace.hidden_scales, 14: trace.output_bf16}
        if self.traced:
            for index, values in enumerate((trace.bridge.gate_bits, trace.bridge.up_bits,
                                            trace.bridge.silu_bits, trace.bridge.gated_bits,
                                            trace.bridge.weighted_bits), 15):
                sources[index] = values
        widths = [2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 2] + [4] * 5
        counts = [len(self.input), len(self.matrices[0].w_packed), len(self.matrices[0].scale_b),
                  len(self.matrices[1].w_packed), len(self.matrices[1].scale_b),
                  len(self.matrices[2].w_packed), len(self.matrices[2].scale_b),
                  *self.work_shape, self.result_cap]
        if self.traced:
            counts += self.trace_shape or [h] * 5
        sizes = [count * widths[index] for index, count in enumerate(counts)]
        owners = [bytearray([0xA5]) * size for size in sizes]
        offsets = [0] * len(sizes)
        if self.overlap:
            names = ('input', 'w1packed', 'w1scale', 'w2packed', 'w2scale', 'w3packed',
                     'w3scale', 'input_codes', 'input_scales', 'gate', 'up', 'hidden',
                     'hidden_codes', 'hidden_scales', 'out', 'gate_bits', 'up_bits',
                     'silu_bits', 'gated_bits', 'weighted_bits')
            pair = [names.index(name) for name in self.overlap.split(',')]
            backing = bytearray([0xA5]) * max(sizes[index] + offset for index, offset in
                                               zip(pair, (self.in_offset, self.out_offset)))
            for index, offset in zip(pair, (self.in_offset, self.out_offset)):
                owners[index], offsets[index] = backing, offset
        for index in [7, 8, 9, 10, *range(15, len(sizes)), 11, 12, 13, 14]:
            data = b''.join(value.to_bytes(widths[index], sys.byteorder)
                            for value in sources.get(index, ()))
            start = offsets[index]
            owners[index][start:start + len(data)] = data
        order = [7, 8, 9, 10, *range(15, len(sizes)), 11, 12, 13, 14]
        result = []
        for index in order:
            start = offsets[index]
            data = owners[index][start:start + sizes[index]]
            result += [int.from_bytes(data[pos:pos + widths[index]], sys.byteorder)
                       for pos in range(0, len(data), widths[index])]
        return result

    def line(self):
        shape = ','.join(str(v) for m in self.matrices for v in (m.n, m.k))
        head = [self.op, 'shape=' + shape, f'cap={self.result_cap}',
                'work_shape=' + ','.join(map(str, self.work_shape))]
        if self.route is not OMIT:
            head.append(f'route={self.route:x}')
        if self.trace_shape is not None:
            head.append('trace_shape=' + ','.join(map(str, self.trace_shape)))
        if self.runtime_failure:
            head.append('runtime_failure')
        if self.poison:
            head.append('poison')
        if self.poison_writes:
            head.append('poison_writes')
        if self.overlap:
            head += ['overlap=' + self.overlap, f'in_offset={self.in_offset}',
                     f'out_offset={self.out_offset}']
        payload = ' ; '.join(' '.join(f'{value:x}' for value in section)
                             for section in self.sections)
        return ' '.join(head) + ' : ' + payload


def check_expert(case, response):
    if case.expect[0] == 'err':
        if response[0] != 'err':
            return f'{case.name}: expected invalid_argument, got ok'
        if response[1] != 'invalid_argument':
            return f'{case.name}: expected invalid_argument, got {response[1]}'
        if response[2]:
            return f'{case.name}: forbidden storage changed on rejection'
        return None
    if response[0] != 'ok':
        return f'{case.name}: expected ok, got {response[1]}'
    if len(response[1]) != len(case.expect[1]):
        return f'{case.name}: response length mismatch'
    for index, (want, got, label) in enumerate(zip(case.expect[1], response[1], case.labels)):
        if label.endswith('bf16') and got > 0xFFFF:
            return f'{case.name}.{label}[{index}]: BF16 exceeds 16 bits'
        if label in ('input_codes', 'input_scales', 'hidden_codes', 'hidden_scales') and got > 0xFF:
            return f'{case.name}.{label}[{index}]: byte stage exceeds 8 bits'
        equal_nan = ((label.endswith('bf16') and nan16(want) and nan16(got)) or
                     (label in STAGES[4:9] and nan32(want) and nan32(got)))
        if want != got and not equal_nan:
            return f'{case.name}.{label}[{index}]: expected {want:#x}, got {got:#x}'
    return None


def failures(cases, args=()):
    return [message for case, response in zip(cases, execute(EXE, cases, args))
            if (message := check_expert(case, response))]


def random_matrix(rng, n, k):
    return impl.FP4Matrix(n, k, [rng.randrange(256) for _ in range(n * k // 2)],
                          [rng.randrange(120, 132) for _ in range(n * k // 32)])


def parity_cases():
    rng = random.Random(2222)
    for d, h in ((32, 64), (64, 32), (64, 96)):
        values = [rng.choice((0, 0x8000, 1, 0x8001, 0x3F80, 0xBF80, 0x4000,
                              0xC000, 0x0080, 0x8080, 0x3D80, 0xBD80)) for _ in range(d)]
        matrices = (random_matrix(rng, h, d), random_matrix(rng, d, h),
                    random_matrix(rng, h, d))
        for route in (OMIT, 0x3F000001):
            for traced in (False, True):
                yield ExpertCase(f'rectangular.{d}x{h}.route={route is not OMIT}', values,
                                 *matrices, route=route, traced=traced, cap=d + 3,
                                 work_shape=(d + 3, d // 32 + 2, h + 3, h + 2, h + 1,
                                             h + 4, h // 32 + 2))
    d = h = 32
    inputs = [0x3F80, 0xBF80, 0x0000, 0x8000, 0x0001, 0x8001] + [0x3F00] * 26
    w1 = matrix(h, d, 0x76, 254)
    w3 = matrix(h, d, 0x66, 126)
    w2 = matrix(d, h, 0x22, 127)
    for traced in (False, True):
        yield ExpertCase('inf-clamp-recovery', inputs, w1, w2, w3, traced=traced)
    # Exact W2 totals around BF16 ties and signed/underflow paths are preserved by final conversion.
    inputs = [0x3F80] + [0] * 31
    w1 = matrix(32, 32, 0x02, 127)
    w3 = matrix(32, 32, 0x02, 127)
    w2 = impl.FP4Matrix(32, 32, [0x02] * 16 + [0x05] * 16 + [0x8A] * (30 * 16),
                        [119, 119] + [0] * 30)
    yield ExpertCase('final-zero-sign-underflow-bf16round', inputs, w1, w2, w3, traced=True)
    w2_nan = matrix(32, 32, 0xA2, 254)
    yield ExpertCase('final-inf-nan', [0x7F7F] * 32, w1, w2_nan, w3, traced=True)


def contract_cases():
    d, h = 32, 64
    inputs = [0x3F80, 0xBF80] + [0] * 30
    w1, w2, w3 = matrix(h, d), matrix(d, h), matrix(h, d, 0x24, 128)
    base = (inputs, w1, w2, w3)
    for traced in (False, True):
        yield ExpertCase('metadata.short-out', *base, traced=traced, cap=d - 1, rejected=True,
                         poison=True)
        for pos in range(7):
            work = [d, d // 32, h, h, h, h, h // 32]
            work[pos] -= 1
            yield ExpertCase(f'metadata.short-work.{pos}', *base, traced=traced,
                             work_shape=work, rejected=True, poison=True)
        bad_w3 = impl.FP4Matrix(h, d + 32, w3.w_packed, w3.scale_b)
        yield ExpertCase('shape.w3-k', inputs, w1, w2, bad_w3, traced=traced,
                         rejected=True, poison=True)
        bad_w2 = impl.FP4Matrix(d, h + 32, w2.w_packed, w2.scale_b)
        yield ExpertCase('shape.w2-k', inputs, w1, bad_w2, w3, traced=traced,
                         rejected=True, poison=True)
        for route in (0x7F800000, 0xFF800000, 0x7FC00001):
            yield ExpertCase(f'route.{route:x}', *base, route=route, traced=traced,
                             rejected=True, poison_writes=True)
        bad_input = inputs.copy(); bad_input[-1] = 0x7F80
        yield ExpertCase('contents.input-last', bad_input, w1, w2, w3, traced=traced,
                         rejected=True, poison_writes=True)
        scales = list(w2.scale_b); scales[-1] = 0xFF
        bad_scale = impl.FP4Matrix(w2.n, w2.k, w2.w_packed, scales)
        yield ExpertCase('contents.w2-last-scale', inputs, w1, bad_scale, w3, traced=traced,
                         rejected=True, poison_writes=True)
        if traced:
            for pos in range(5):
                shape = [h] * 5; shape[pos] -= 1
                yield ExpertCase(f'trace-shape.{pos}', *base, traced=True, trace_shape=shape,
                                 rejected=True, poison=True)
    # Negative-infinite gate survives clamp, so bridge becomes nonfinite and rejects at hidden boundary.
    hidden_bad_w1 = matrix(h, d, 0xEE, 254)
    hidden_bad_w3 = matrix(h, d, 0x22, 127)
    for traced in (False, True):
        yield ExpertCase('runtime.hidden-nonfinite', [0x7F7F] * d, hidden_bad_w1, w2,
                         hidden_bad_w3, traced=traced, rejected=True, runtime_failure=True)
    # Public same-type alias cases: read/read legal; every write collision and out-tail/scratch collide reject.
    same_w = matrix(h, d, 0x22, 127)
    yield ExpertCase('alias.readonly-w1-w3', inputs, same_w, w2, same_w,
                     overlap='w1packed,w3packed')
    yield ExpertCase('alias.out-input', *base, overlap='out,input', rejected=True)
    yield ExpertCase('alias.gate-input', *base, overlap='gate,input', rejected=True)
    yield ExpertCase('alias.scratch-scratch', *base, overlap='gate,up', rejected=True)
    yield ExpertCase('alias.scratch-tail-neighbor', *base,
                     work_shape=(d, d // 32, 2 * h + 4, h, h, h, h // 32),
                     overlap='gate,up', out_offset=2 * h + 8)
    yield ExpertCase('alias.out-tail-scratch', *base, cap=d + h + 8,
                     overlap='out,gate', out_offset=2 * (d + 4), rejected=True)
    yield ExpertCase('alias.out-scratch-neighbor', *base, cap=d + 4,
                     overlap='out,gate', out_offset=2 * (d + 4))


def storage_cases():
    cases = []
    def add(name, pair, first, second, read_size, write_size, out_size, out_write, rejected):
        case = Case('storage.' + name, 'expert_storage', (), [], 0, 4,
                    ('err', 'invalid_argument') if rejected else ('ok', []))
        case.line = lambda pair=pair, first=first, second=second, read_size=read_size, \
                write_size=write_size, out_size=out_size, out_write=out_write: (
            f'expert_storage cap={out_size} '
            f'work_shape={read_size},{write_size},{out_size},{out_write} '
            f'overlap={pair} in_offset={first} out_offset={second} :')
        cases.append(case)
    # Every probe has sufficient capacities; only byte-range geometry changes.
    add('read-scratch-overlap', 'read,scratch', 0, 0, 64, 64, 96, 64, True)
    add('read-scratch-adjacent', 'read,scratch', 0, 64, 64, 64, 96, 64, False)
    add('scratch-out-full-tail', 'scratch,out', 64, 0, 64, 64, 160, 64, True)
    add('scratch-out-adjacent', 'scratch,out', 160, 0, 64, 64, 160, 64, False)
    add('read-out-prefix-overlap', 'read,out', 0, 0, 64, 64, 160, 64, True)
    add('read-out-tail-overlap', 'read,out', 96, 0, 64, 64, 160, 64, False)
    add('read-out-adjacent', 'read,out', 64, 0, 64, 64, 160, 64, False)
    # Mixed uint16 out / uint32 trace geometry is tested only through production byte checks.
    add('mixed-out-trace-tail-overlap', 'scratch,out', 72, 0, 256, 256, 336, 64, True)
    add('mixed-out-trace-adjacent', 'scratch,out', 336, 0, 256, 256, 336, 64, False)
    return cases


def count_cases():
    maximum = (1 << 64) - 1
    cases = []
    def add(name, op, input_count, shape, out_count, expected):
        case = Case(name, op, (), [], 8 if expected[0] == 'ok' else 0, 8, expected)
        case.line = lambda op=op, input_count=input_count, shape=shape, out_count=out_count: (
            f'{op} input_count={input_count} shape=' + ','.join(map(str, shape)) +
            f' cap={out_count} :')
        cases.append(case)
    add('counts.valid', 'expert_counts', 32, (64, 32, 32, 64, 64, 32), 32,
        ('ok', [1, 2, 1024, 64, 1024, 64, 1024, 64]))
    for name, input_count, shape, out_count in (
            ('input-shape', 64, (64, 32, 32, 64, 64, 32), 32),
            ('matrix-shape', 32, (64, 32, 32, 64, 64, 64), 32),
            ('zero-out-overflow', 32, (64, 32, 32, 64, 64, 32), maximum),
            ('weight-overflow', 32, (maximum // 16 + 1, 32, 32, 64, 64, 32), 32),
            ('trace-overflow', 32, (maximum // 4 + 1, 32, 32, maximum // 4 + 1,
                                    maximum // 4 + 1, 32), 32)):
        add('counts.' + name, 'expert_counts_trace' if name == 'trace-overflow' else 'expert_counts',
            input_count, shape, out_count, ('err', 'invalid_argument'))
    return cases


def selected(cases):
    return [case for case in cases if not SELECT or any(token in case.name for token in SELECT)]


class EnvironmentProbed(unittest.TestCase):
    def run_matrix(self, builder):
        cases = selected(list(builder()))
        self.assertTrue(cases, 'empty selection is not acceptance')
        start = time.perf_counter()
        for environment, csr in ENVIRONMENTS.items():
            for status in ('clear', 'set'):
                probe = mxcsr_probe()
                first, *middle, last = execute(
                    EXE, [probe, *cases, probe],
                    [f'--mxcsr={environment}', f'--status={status}', *EXE_ARGS])
                expected = csr | (MXCSR_STATUS if status == 'set' else 0)
                self.assertEqual(first[1][0], expected)
                self.assertEqual(last[1][0], expected)
                self.assertEqual([message for case, response in zip(cases, middle)
                                  if (message := check_expert(case, response))], [])
        print(f'P0022 {builder.__name__}: {len(cases)} cases x 16 MXCSR states; '
              f'{time.perf_counter() - start:.2f}s', file=sys.stderr)


class Parity(EnvironmentProbed):
    def test_complete_stages(self):
        self.run_matrix(parity_cases)


class Contract(EnvironmentProbed):
    def test_metadata_contents_storage_runtime(self):
        self.run_matrix(contract_cases)

    def run_probe(self, cases):
        responses = execute(EXE, cases, EXE_ARGS)
        messages = []
        for case, response in zip(cases, responses):
            if case.expect[0] == 'err':
                if response[:2] != case.expect or response[2]:
                    messages.append(f'{case.name}: bad rejection {response}')
            elif response[0] != 'ok' or response[1] != case.expect[1]:
                messages.append(f'{case.name}: expected {case.expect}, got {response}')
        self.assertEqual(messages, [])

    def test_count_probe(self):
        if SELECT and not any(token in 'counts' for token in SELECT):
            return
        self.run_probe(count_cases())

    def test_mixed_storage_probe(self):
        if SELECT and not any(token in 'storage' for token in SELECT):
            return
        self.run_probe(storage_cases())


class Harness(unittest.TestCase):
    def sample(self):
        return [case for case in parity_cases() if 'rectangular.32x64.route=True' in case.name]

    def test_faults_and_snapshots(self):
        cases = self.sample()
        self.assertEqual(failures(cases), [])
        for fault, evidence in (('crash', 'executor exited'), ('exit', 'executor exited 3'),
                                ('garbage', 'unparseable response'), ('drop', 'responses')):
            with self.assertRaisesRegex(HarnessError, evidence):
                failures(cases, [f'--fault={fault}'])
        self.assertEqual(len(failures(cases, ['--fault=mismatch'])), len(cases))
        with self.assertRaisesRegex(HarnessError, 'expert input changed'):
            failures([cases[0]], ['--fault=input-write'])
        rejected = next(case for case in contract_cases()
                        if case.name.startswith('metadata.short-out.ordinary'))
        response = execute(EXE, [rejected], ['--fault=reject-input-write'])[0]
        self.assertEqual(response, ('err', 'invalid_argument', True))
        self.assertIsNotNone(check_expert(rejected, response))
        runtime = next(case for case in contract_cases()
                       if case.name.startswith('runtime.hidden-nonfinite.ordinary'))
        response = execute(EXE, [runtime], ['--fault=reject-input-write'])[0]
        self.assertEqual(response, ('err', 'invalid_argument', True))
        self.assertIsNotNone(check_expert(runtime, response))
        with self.assertRaisesRegex(HarnessError, 'did not reach every bridge boundary'):
            execute(EXE, [runtime], ['--fault=expert-early-reject'])

    def test_shared_gap_detection(self):
        success = next(case for case in contract_cases()
                       if case.name.startswith('alias.scratch-tail-neighbor'))
        self.assertEqual(failures([success]), [])
        with self.assertRaisesRegex(HarnessError, 'outside write ranges'):
            failures([success], ['--fault=shared-gap-write'])
        runtime = next(case for case in contract_cases()
                       if case.name.startswith('runtime.hidden-nonfinite.ordinary'))
        runtime.overlap = 'gate,up'
        runtime.in_offset = 0
        runtime.out_offset = 2 * runtime.matrices[0].n + 8
        runtime.work_shape[2] = 2 * runtime.matrices[0].n + 4
        response = execute(EXE, [runtime])[0]
        self.assertEqual(response, ('err', 'invalid_argument', False))
        response = execute(EXE, [runtime], ['--fault=shared-gap-write'])[0]
        self.assertEqual(response, ('err', 'invalid_argument', True))

    def test_protocol_parser_and_driver(self):
        case = self.sample()[0]
        valid = case.line()
        malformed = [valid.split(' : ')[0], valid + ' ;', valid.rsplit(' ; ', 1)[0],
                     valid.replace('work_shape=', 'work_shape=1,'),
                     valid.replace('shape=', 'shape=1,'), valid.replace('cap=', 'cap=1 cap=', 1),
                     valid.replace('expert_', 'expert_unknown_', 1),
                     valid.replace('route=3f000001', 'route=3f000001 route=3f000001')]
        for text in malformed:
            proc = subprocess.run([str(EXE)], input=text + '\n', capture_output=True,
                                  text=True, timeout=900)
            self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        command = [sys.executable, __file__, '--exe', str(EXE), '--suite', 'parity',
                   '--case', 'nonexistent-p0022-case']
        proc = subprocess.run(command, capture_output=True, text=True, timeout=900)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn('empty selection is not acceptance', proc.stderr)

    def test_numeric_stage_sensitivity(self):
        case = next(case for case in self.sample() if case.traced)
        response = execute(EXE, [case])[0]
        self.assertIsNone(check_expert(case, response))
        for label in ('input_codes', 'gate_bf16', 'up_bf16', 'silu_bits', 'hidden_bf16',
                      'hidden_codes', 'output_bf16'):
            index = case.labels.index(label)
            changed = response[1].copy(); changed[index] ^= 1
            self.assertIsNotNone(check_expert(case, ('ok', changed)), label)


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
