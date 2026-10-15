#!/usr/bin/env python3
"""P0021 production SwiGLU parity, storage contracts and fail-closed harness."""

import argparse
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
from test_codec_parity import Case, HarnessError, check, parse_response, run_executor
from test_dequant_parity import ENVIRONMENTS, MXCSR_STATUS, mxcsr_probe
from test_expert import expected_mul, expected_silu

EXE = None
EXE_ARGS = []
SELECT = []
FILL = 0xA5
BUFFERS = ('gate', 'up', 'out', 'gate_bits', 'up_bits', 'silu_bits',
           'gated_bits', 'weighted_bits')
STAGES = BUFFERS[3:]
OMIT = object()


def nan32(bits):
    return (bits & 0x7FFFFFFF) > 0x7F800000


def nan16(bits):
    return (bits & 0x7FFF) > 0x7F80


def words(data, width):
    return [int.from_bytes(data[i:i + width], sys.byteorder)
            for i in range(0, len(data), width)]


class SwigluCase(Case):
    def __init__(self, name, gate, up, weight=OMIT, traced=False, cap=None,
                 trace_shape=None, overlap=None, trace_alias=None, in_offset=0,
                 out_offset=0, rejected=False, poison=False):
        self.gate = list(gate)
        self.up = list(up)
        self.weight = weight
        self.traced = traced
        self.result_cap = len(gate) if cap is None else cap
        self.trace_shape = list(trace_shape) if trace_shape is not None else None
        self.overlap = overlap
        self.trace_alias = trace_alias
        self.in_offset = in_offset
        self.out_offset = out_offset
        self.poison = poison
        shape = self.trace_shape if self.trace_shape is not None else [len(gate)] * 5
        self.labels = ['out'] * self.result_cap
        if traced:
            self.labels += [stage for stage, count in zip(STAGES, shape) for _ in range(count)]
        count = self.result_cap + (sum(shape) if traced else 0)
        expected = ('err', 'invalid_argument') if rejected else ('ok', self.expected(shape))
        super().__init__(name + ('.trace' if traced else '.ordinary'),
                         'swiglu_trace' if traced else 'swiglu', (), [], count, 4, expected)

    def expected(self, shape):
        sizes = [2 * len(self.gate), 2 * len(self.up), 2 * self.result_cap]
        sizes += [4 * count for count in shape] if self.traced else []
        owners = [bytearray([FILL]) * size for size in sizes]
        offsets = [0] * len(sizes)
        pair_name = self.trace_alias or self.overlap
        if pair_name:
            pair = [BUFFERS.index(name) for name in pair_name.split(',')]
            backing = bytearray([FILL]) * max(
                sizes[index] + offset for index, offset in
                zip(pair, (self.in_offset, self.out_offset)))
            for index, offset in zip(pair, (self.in_offset, self.out_offset)):
                owners[index], offsets[index] = backing, offset

        def write(index, data):
            start = offsets[index]
            owners[index][start:start + len(data)] = data

        def read(index):
            start = offsets[index]
            return owners[index][start:start + sizes[index]]

        write(0, b''.join(value.to_bytes(2, sys.byteorder) for value in self.gate))
        write(1, b''.join(value.to_bytes(2, sys.byteorder) for value in self.up))
        actual_gate, actual_up = words(read(0), 2), words(read(1), 2)
        route = None if self.weight is OMIT else self.weight
        trace = impl.trace_swiglu(actual_gate, actual_up, route_weight_bits=route)
        write(2, b''.join(value.to_bytes(2, sys.byteorder) for value in trace.bf16_bits))
        if self.traced:
            for index, stage in enumerate(STAGES, 3):
                write(index, b''.join(value.to_bytes(4, sys.byteorder)
                                      for value in getattr(trace, stage)))
        result = words(read(2), 2)
        if self.traced:
            result += [value for index in range(3, 8) for value in words(read(index), 4)]
        return result

    def line(self):
        head = [self.op, f'cap={self.result_cap}']
        if self.weight is not OMIT:
            head.append(f'route={self.weight:x}')
        if self.trace_shape is not None:
            head.append('trace_shape=' + ','.join(map(str, self.trace_shape)))
        if self.overlap:
            head.append('overlap=' + self.overlap)
        if self.trace_alias:
            head.append('trace_alias=' + self.trace_alias)
        if self.overlap or self.trace_alias:
            head += [f'in_offset={self.in_offset}', f'out_offset={self.out_offset}']
        if self.poison:
            head.append('poison')
        payload = (' '.join(f'{value:x}' for value in self.gate) + ' ; ' +
                   ' '.join(f'{value:x}' for value in self.up))
        return ' '.join(head) + ' : ' + payload


def check_swiglu(case, response):
    if case.expect[0] == 'err':
        if response[0] != 'err':
            return f'{case.name}: expected invalid_argument, got ok'
        if response[1] != 'invalid_argument':
            return f'{case.name}: expected invalid_argument, got {response[1]}'
        if response[2]:
            return f'{case.name}: destination changed on rejection'
        return None
    if response[0] != 'ok':
        return f'{case.name}: expected ok, got {response[1]}'
    if len(response[1]) != len(case.expect[1]):
        return f'{case.name}: response length mismatch'
    for index, (want, got, label) in enumerate(zip(case.expect[1], response[1], case.labels)):
        if label == 'out' and got > 0xFFFF:
            return f'{case.name}[{index}]: BF16 output exceeds 16 bits: {got:#x}'
        equal_nan = ((label == 'out' and nan16(want) and nan16(got)) or
                     (label in ('silu_bits', 'gated_bits', 'weighted_bits') and
                      nan32(want) and nan32(got)))
        if want != got and not equal_nan:
            return f'{case.name}.{label}[{index}]: expected {want:#x}, got {got:#x}'
    return None


def failures(cases, args=()):
    return [message for case, response in zip(cases, run_executor(EXE, cases, args))
            if (message := check_swiglu(case, response))]


def selected(cases):
    return [case for case in cases if not SELECT or any(token in case.name for token in SELECT)]


def directed_cases():
    gate = [0x0000, 0x8000, 0x0001, 0x8001, 0x3F80, 0xBF80, 0x411F, 0x4120,
            0x4121, 0xC11F, 0xC120, 0xC121, 0xC2FF, 0xC300, 0x7F80, 0xFF80,
            0x7FC1, 0xFFC1, 0x7F7F, 0xFF7F]
    up = [0x3F80, 0x3F80, 0x3F00, 0x3F00, 0x4000, 0x4000, 0x4121, 0xC121,
          0x7F80, 0xFF80, 0x7FC1, 0xFFC1, 0x0000, 0x8000, 0x3F80, 0x3F80,
          0x3F80, 0x3F80, 0x0001, 0x8001]
    for traced in (False, True):
        yield SwigluCase('directed.unweighted', gate, up, traced=traced, cap=len(gate) + 3)
        yield SwigluCase('directed.weighted', gate, up, 0x3F000001, traced=traced,
                         cap=len(gate) + 3)
        yield SwigluCase('directed.zero-weight', gate, up, 0, traced=traced,
                         cap=len(gate) + 1)


def exhaustive_cases():
    batch = 4096
    for start in range(0, 1 << 16, batch):
        gate = list(range(start, start + batch))
        up = [0x3F80] * batch
        yield SwigluCase(f'exhaustive.{start:04x}', gate, up, traced=True)


def contract_cases():
    gate = [0x3F80, 0xBF80, 0x4121, 0xC121]
    up = [0x4000, 0x4000, 0x4121, 0xC121]
    for traced in (False, True):
        yield SwigluCase('empty', [], [], traced=traced, cap=3)
        yield SwigluCase('empty.weighted', [], [], 0x80000000, traced=traced, cap=2)
        yield SwigluCase('metadata.short-up', gate, up[:-1], traced=traced, rejected=True)
        yield SwigluCase('metadata.long-up', gate, up + [0], traced=traced, rejected=True)
        yield SwigluCase('metadata.short-out', gate, up, traced=traced, cap=3, rejected=True)
        for weight in (0x7F800000, 0xFF800000, 0x7FC00001, 0xFFC00001):
            yield SwigluCase(f'weight.{weight:08x}', gate, up, weight, traced=traced,
                             rejected=True)
        if traced:
            for pos in range(5):
                shape = [len(gate)] * 5
                shape[pos] -= 1
                yield SwigluCase(f'trace-shape.{pos}', gate, up, traced=True,
                                 trace_shape=shape, rejected=True)
    yield SwigluCase('weight.empty-invalid', [], [], 0x7F800000, rejected=True)
    yield SwigluCase('readonly-overlap', gate, gate, overlap='gate,up',
                     in_offset=0, out_offset=0)
    yield SwigluCase('readonly-partial', gate, gate, overlap='gate,up',
                     in_offset=0, out_offset=2)
    yield SwigluCase('output-adjacent', gate, up, cap=8, overlap='out,gate',
                     in_offset=0, out_offset=2 * len(gate))
    yield SwigluCase('output-tail-overlap', gate, up, cap=12, overlap='out,gate',
                     in_offset=0, out_offset=16)
    yield SwigluCase('shared-gap', gate, up, cap=12, overlap='out,gate',
                     in_offset=0, out_offset=24)
    yield SwigluCase('trace-input-adjacent', gate, up, traced=True,
                     trace_alias='gate_bits,gate', in_offset=0,
                     out_offset=4 * len(gate))
    yield SwigluCase('trace-write-adjacent', gate, up, traced=True,
                     trace_alias='gate_bits,out', in_offset=0,
                     out_offset=4 * len(gate))
    for write in range(2, len(BUFFERS)):
        for other in range(write):
            pair = BUFFERS[write] + ',' + BUFFERS[other]
            traced = write >= 3
            for poison in ((False, True) if write == 2 and other < 2 else (False,)):
                yield SwigluCase(f'alias.{pair}.{poison}', gate, up, traced=traced,
                                 overlap=pair if write == 2 else None,
                                 trace_alias=pair if write >= 3 else None,
                                 rejected=True, poison=poison)


class Parity(unittest.TestCase):
    def test_exhaustive_gate_codes_default(self):
        cases = selected(list(exhaustive_cases()))
        if not cases and SELECT:
            return
        self.assertTrue(cases, 'empty selection is not acceptance')
        start = time.perf_counter()
        self.assertEqual(failures(cases, EXE_ARGS), [])
        print(f'P0021 exhaustive: {sum(len(case.gate) for case in cases)} BF16 gate codes; '
              f'{time.perf_counter() - start:.2f}s', file=sys.stderr)

    def test_directed_stages_all_mxcsr(self):
        cases = selected(list(directed_cases()))
        if not cases and SELECT:
            return
        self.assertTrue(cases, 'empty selection is not acceptance')
        for environment, csr in ENVIRONMENTS.items():
            for status in ('clear', 'set'):
                with self.subTest(environment=environment, status=status):
                    probe = mxcsr_probe()
                    first, *middle, last = run_executor(
                        EXE, [probe, *cases, probe],
                        [f'--mxcsr={environment}', f'--status={status}', *EXE_ARGS])
                    expected_csr = csr | (MXCSR_STATUS if status == 'set' else 0)
                    self.assertEqual(first[1][0], expected_csr)
                    self.assertEqual(last[1][0], expected_csr)
                    self.assertEqual([message for case, response in zip(cases, middle)
                                      if (message := check_swiglu(case, response))], [])

    def test_independent_silu_boundaries(self):
        if SELECT and not any(token in 'independent-boundaries' for token in SELECT):
            return
        codes = [0x0000, 0x8000, 0x0001, 0x8001, 0x007F, 0x0080, 0x3F80,
                 0xBF80, 0x411F, 0x4120, 0x4121, 0xC120, 0xC2C8, 0xC2FF,
                 0xC300, 0xC301, 0xFF7F, 0x7F80, 0xFF80, 0x7FC1, 0xFFC1]
        case = SwigluCase('independent-boundaries', codes, [0x3F80] * len(codes), traced=True)
        response = run_executor(EXE, [case], EXE_ARGS)[0]
        self.assertIsNone(check_swiglu(case, response))
        start = case.result_cap + 2 * len(codes)
        actual = response[1][start:start + len(codes)]
        for code, got in zip(codes, actual):
            clamped = impl._clamp_bits(code << 16, gate=True)
            want = expected_silu(clamped)
            with self.subTest(code=hex(code)):
                self.assertTrue((nan32(want) and nan32(got)) or want == got)

    def test_special_value_multiply_helper(self):
        if SELECT and not any(token in 'multiply-specials' for token in SELECT):
            return
        pairs = [(0, 0xFF800000), (0x80000000, 0x7F800000),
                 (0x7F800000, 0xBF800000), (0xFF800000, 0xBF800000),
                 (0x7FC00001, 0x3F800000), (0x3F800000, 0xFFC00001),
                 (1, 0x3F000000), (0x007FFFFF, 0x3F800001),
                 (0x7F7FFFFF, 0x3F800001)]
        expected = [value if (value := expected_mul(left, right)) >= 0 else 0x7FC00000
                    for left, right in pairs]
        case = Case('multiply-specials', 'multiply_fp32', (), [], len(pairs), 4,
                    ('ok', expected))
        case.line = lambda: ('multiply_fp32 cap=' + str(len(pairs)) + ' : ' +
                             ' '.join(f'{left:x}' for left, _ in pairs) + ' ; ' +
                             ' '.join(f'{right:x}' for _, right in pairs))
        for environment in ENVIRONMENTS:
            for status in ('clear', 'set'):
                response = run_executor(EXE, [case],
                                        [f'--mxcsr={environment}', f'--status={status}',
                                         *EXE_ARGS])[0]
                self.assertEqual(response[0], 'ok')
                for want, got in zip(case.expect[1], response[1]):
                    self.assertTrue((nan32(want) and nan32(got)) or want == got)


class Contract(unittest.TestCase):
    def test_count_overflow_probe(self):
        if SELECT and not any(token in 'counts' for token in SELECT):
            return
        maximum = (1 << 64) - 1
        cases = []
        for op, traced in (('swiglu_counts', False), ('swiglu_counts_trace', True)):
            half, quarter = maximum // 2, maximum // 4
            for count, rejected in ((half, traced), (half + 1, True),
                                    (quarter, False), (quarter + 1, traced)):
                case = Case(f'counts.{op}.{count}', op, (), [], 0, 4,
                            ('err', 'invalid_argument') if rejected else ('ok', []))
                case.line = lambda op=op, count=count: f'{op} cap={count} :'
                cases.append(case)
        responses = run_executor(EXE, cases)
        self.assertEqual([message for case, response in zip(cases, responses)
                          if (message := check(case, response))], [])

    def test_metadata_storage_and_specials(self):
        cases = selected(list(contract_cases()))
        self.assertTrue(cases, 'empty selection is not acceptance')
        for environment, csr in ENVIRONMENTS.items():
            for status in ('clear', 'set'):
                with self.subTest(environment=environment, status=status):
                    probe = mxcsr_probe()
                    first, *middle, last = run_executor(
                        EXE, [probe, *cases, probe],
                        [f'--mxcsr={environment}', f'--status={status}', *EXE_ARGS])
                    expected_csr = csr | (MXCSR_STATUS if status == 'set' else 0)
                    self.assertEqual(first[1][0], expected_csr)
                    self.assertEqual(last[1][0], expected_csr)
                    self.assertEqual([message for case, response in zip(cases, middle)
                                      if (message := check_swiglu(case, response))], [])


class Harness(unittest.TestCase):
    def sample(self):
        return [case for case in directed_cases() if case.name.startswith('directed.weighted')]

    def test_faults(self):
        cases = self.sample()
        self.assertEqual(failures(cases), [])
        for fault, evidence in (('crash', 'executor exited'), ('exit', 'executor exited 3'),
                                ('garbage', 'unparseable response'),
                                ('drop', f'expected {len(cases)} responses, got {len(cases) - 1}')):
            with self.subTest(fault=fault), self.assertRaisesRegex(HarnessError, evidence):
                failures(cases, [f'--fault={fault}'])
        self.assertEqual(len(failures(cases, ['--fault=mismatch'])), len(cases))
        with self.assertRaises(HarnessError):
            run_executor(EXE.with_name('missing_p0021_executor'), cases)

    def test_snapshot_gap_and_poison(self):
        case = self.sample()[0]
        with self.assertRaisesRegex(HarnessError, 'SwiGLU input changed'):
            failures([case], ['--fault=input-write'])
        rejected = next(case for case in contract_cases()
                        if case.name.startswith('metadata.short-up.ordinary'))
        response = run_executor(EXE, [rejected], ['--fault=reject-input-write'])[0]
        self.assertEqual(response, ('err', 'invalid_argument', True))
        self.assertIsNotNone(check_swiglu(rejected, response))
        gap = next(case for case in contract_cases() if case.name.startswith('shared-gap'))
        self.assertEqual(failures([gap]), [])
        with self.assertRaisesRegex(HarnessError, 'outside write ranges'):
            failures([gap], ['--fault=shared-gap-write'])
        probe = Case('asan', 'asan', (), [], 1, 4, ('ok', []))
        asan = run_executor(EXE, [probe])[0][1][0]
        poisoned = SwigluCase('poison-valid', [0x3F80], [0x4000], poison=True)
        if asan:
            with self.assertRaisesRegex(HarnessError, 'AddressSanitizer'):
                failures([poisoned])
        else:
            self.assertEqual(failures([poisoned]), [])

    def test_protocol_and_parser(self):
        case = self.sample()[0]
        valid = case.line()
        cap = f'cap={case.result_cap}'
        malformed = [valid.split(' : ')[0], valid + ' ;', valid.rsplit(' ; ', 1)[0],
                     valid.replace(cap + ' ', ''),
                     valid.replace('route=3f000001', 'route=3f000001 route=3f000001'),
                     valid.replace('route=3f000001', 'route=100000000'),
                     valid.replace('swiglu', 'swiglu_unknown', 1),
                     valid.replace(cap, cap + ' shape=1'),
                     valid.replace(cap, cap + ' scale_cap=1'),
                     valid.replace(cap, cap + ' trace_shape=1,1'),
                     valid.replace(cap, cap + ' overlap=gate'),
                     valid.replace(cap, cap + ' trace_alias=gate,up'),
                     'decode_bf16 cap=1 route=0 shape=1 : 0']
        for text in malformed:
            with self.subTest(text=text[:100]):
                self.assertNotEqual(text, valid, 'protocol mutation did not change the request')
                proc = subprocess.run([str(EXE)], input=text + '\n', capture_output=True,
                                      text=True, timeout=900)
                self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
                self.assertIn('quant_exec:', proc.stderr)
        for argument in ('--mxcsr=unknown', '--status=unknown', '--fault=unknown'):
            with self.assertRaises(HarnessError):
                failures([case], [argument])
        for line in ('ok ' + ' '.join(['00000000'] * (case.cap - 1)),
                     'ok ' + ' '.join(['00000000'] * (case.cap + 1)),
                     'ok ' + ' '.join(['0000'] * case.cap),
                     'err invalid_argument changed=2'):
            with self.assertRaises(HarnessError):
                parse_response(case, line)

    def test_mixed_alias_probe_rejects_other_invalid_metadata(self):
        base = 'swiglu_trace cap=2 trace_alias=gate_bits,gate : 3f80 bf80 ; 4000 4000'
        cases = (
            base.replace('cap=2', 'cap=1'),
            base.replace('4000 4000', '4000'),
            base.replace('cap=2', 'cap=2 trace_shape=1,2,2,2,2'),
            base.replace('cap=2', 'cap=2 route=7f800000'),
        )
        for text in cases:
            with self.subTest(text=text):
                proc = subprocess.run([str(EXE)], input=text + '\n', capture_output=True,
                                      text=True, timeout=900)
                self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
                self.assertIn('mixed SwiGLU alias probe requires otherwise valid metadata',
                              proc.stderr)
                self.assertEqual(proc.stdout, '')

    def test_driver_exit(self):
        matrix = (
            ('parity', 'directed.weighted', 'mismatch', 'expected'),
            ('parity', 'directed.weighted', 'garbage', 'unparseable response'),
            ('contract', 'metadata.short-up', 'exit', 'executor exited 3'),
            ('parity', 'directed.weighted', 'input-write', 'SwiGLU input changed'),
            ('parity', 'exhaustive.0000', 'mismatch', 'expected'),
            ('contract', 'shared-gap', 'shared-gap-write', 'outside write ranges'),
        )
        for suite, selector, fault, evidence in matrix:
            command = [sys.executable, __file__, '--exe', str(EXE), '--suite', suite,
                       '--case', selector]
            clean = subprocess.run(command, capture_output=True, text=True, timeout=900)
            self.assertEqual(clean.returncode, 0, clean.stdout + clean.stderr)
            broken = subprocess.run([*command, f'--exe-arg=--fault={fault}'],
                                    capture_output=True, text=True, timeout=900)
            self.assertNotEqual(broken.returncode, 0)
            self.assertIn(evidence, broken.stderr)
        missing = subprocess.run([sys.executable, __file__, '--exe', str(EXE), '--suite',
                                  'parity', '--case', 'nonexistent-p0021-case'],
                                 capture_output=True, text=True, timeout=900)
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn('empty selection is not acceptance', missing.stderr)


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
    if SELECT:
        if args.suite == 'parity':
            names = ([case.name for case in directed_cases()] +
                     [f'exhaustive.{start:04x}.trace' for start in range(0, 1 << 16, 4096)] +
                     ['independent-boundaries.trace', 'multiply-specials'])
        elif args.suite == 'contract':
            names = [case.name for case in contract_cases()] + [
                'counts.swiglu_counts', 'counts.swiglu_counts_trace']
        else:
            names = [case.name for case in Harness().sample()]
        if not any(any(selector in name for selector in SELECT) for name in names):
            print('empty selection is not acceptance', file=sys.stderr)
            return 1
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(SUITES[args.suite]))
    return 0 if result.wasSuccessful() and result.testsRun and not result.skipped else 1


if __name__ == '__main__':
    sys.exit(main())
