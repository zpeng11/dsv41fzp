#!/usr/bin/env python3
"""P0028 production FP8 grouped-dot parity, contract, environment and harness.

The production executor is compared both with ref.fp8_compute (the validation-chain
parent) and test_fp8_compute.expected_trace (the independent Fraction expectation).
The P0027 suite itself is not rerun.
"""

import argparse
import itertools
import random
import subprocess
import sys
import time
import unittest
from fractions import Fraction
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ref import formats as f
from ref.act_quant import act_quant
from ref.fp8_compute import fp8_grouped_dot, trace_fp8_grouped_dot
from test_codec_parity import BF16, FP32, Case, HarnessError, check, failures, parse_response, run_executor
from test_dequant_parity import ENVIRONMENTS, MXCSR_STATUS, RawCase, mxcsr_probe
from test_fp8_compute import (NAN_SENTINEL, SENSITIVITY_VECTORS, VALID_CODES,
                              build_case, expected_trace, is_nan_bf16_bits,
                              is_nan_fp32_bits, power2)

EXE = None
EXE_ARGS = []
STATUS_STATES = ('clear', 'set')
FILL32 = 0xA5A5A5A5
FILL16 = 0xA5A5


class DotCase(Case):
    def __init__(self, name, op, sections, cap, out_width, expect, kind,
                 overlap=None, in_offset=None, out_offset=None, poison=False,
                 out_cap=None, trace_shape=None, trace_alias=None):
        super().__init__(name, op, (), [], cap, out_width, expect, kind)
        self.sections = [list(values) for values in sections]
        self.overlap = overlap
        self.in_offset = in_offset
        self.out_offset = out_offset
        self.poison = poison
        self.out_cap = out_cap
        self.trace_shape = trace_shape
        self.trace_alias = trace_alias

    def line(self):
        fields = [self.op, f'cap={self.cap}']
        if self.out_cap is not None:
            fields.append(f'out_cap={self.out_cap}')
        if self.trace_shape is not None:
            fields.append('trace_shape=' + ','.join(map(str, self.trace_shape)))
        if self.overlap is not None:
            fields.append('overlap=' + self.overlap)
        if self.trace_alias is not None:
            fields.append('trace_alias=' + self.trace_alias)
        if self.in_offset is not None:
            fields.append(f'in_offset={self.in_offset}')
        if self.out_offset is not None:
            fields.append(f'out_offset={self.out_offset}')
        if self.poison:
            fields.append('poison')
        return ' '.join(fields) + ' : ' + ' ; '.join(
            ' '.join(f'{value:x}' for value in section) for section in self.sections)


def stage_words(trace):
    blocks = trace.blocks
    return ([word for block in blocks for word in block.products]
            + [word for block in blocks for word in block.partial_sums]
            + [block.dot_bits for block in blocks]
            + [block.after_a_bits for block in blocks]
            + [block.block_bits for block in blocks]
            + [block.acc_bits for block in blocks])


def independent_words(vector):
    blocks, fp32, bf16 = expected_trace(*vector)
    words = ([word for block in blocks for word in block['products']]
             + [word for block in blocks for word in block['partial_sums']]
             + [block['dot_bits'] for block in blocks]
             + [block['after_a_bits'] for block in blocks]
             + [block['block_bits'] for block in blocks]
             + [block['acc_bits'] for block in blocks])
    return words, fp32, bf16


def equal_word(actual, expected, kind=FP32):
    is_nan = is_nan_fp32_bits if kind == FP32 else is_nan_bf16_bits
    if expected == NAN_SENTINEL or is_nan(expected):
        return is_nan(actual)
    return actual == expected


def check_dot(case, response):
    if case.expect[0] != 'ok' or response[0] != 'ok':
        return check(case, response)
    want, got = case.expect[1], response[1]
    traced = case.op.startswith('fp8_grouped_dot_trace_')
    bf16 = case.op.endswith('_bf16')
    result_index = len(want) - 1
    if traced:
        k = len(case.sections[0])
        result_index = 2 * k + 4 * (k // 32)
    for index, (expected, actual) in enumerate(zip(want, got)):
        kind = BF16 if bf16 and (not traced or index == result_index) else FP32
        if not equal_word(actual, expected, kind):
            return f'{case.name}[{index}]: expected {expected:#x}, got {actual:#x}'
    fill = FILL32 if case.out_width == 4 else FILL16
    if any(value != fill for value in got[len(want):]):
        return f'{case.name}: wrote past the result prefix'
    return None


def assert_independent(vector, oracle):
    words, fp32, bf16 = independent_words(vector)
    oracle_words = stage_words(oracle)
    if len(words) != len(oracle_words):
        raise HarnessError('independent/oracle trace length mismatch')
    if any(not equal_word(actual, expected) for actual, expected in zip(oracle_words, words)):
        raise HarnessError('independent/oracle stage mismatch')
    if not equal_word(oracle.fp32_bits, fp32) or not equal_word(oracle.bf16_bits, bf16, BF16):
        raise HarnessError('independent/oracle result mismatch')
    return words, fp32, bf16


def vector(name, values, slack=2):
    oracle32 = trace_fp8_grouped_dot(*values)
    oracle16 = trace_fp8_grouped_dot(*values, output=BF16)
    if stage_words(oracle32) != stage_words(oracle16):
        raise HarnessError('oracle output modes changed trace stages')
    words, independent32, independent16 = assert_independent(values, oracle32)
    if not equal_word(oracle32.fp32_bits, independent32) or not equal_word(oracle16.bf16_bits, independent16, BF16):
        raise HarnessError('independent result mismatch')
    direct32 = f.fp32_bits(fp8_grouped_dot(*values))
    direct16 = fp8_grouped_dot(*values, output=BF16)
    if not equal_word(direct32, independent32) or not equal_word(direct16, independent16, BF16):
        raise HarnessError('oracle direct/independent mismatch')
    k = len(values[0])
    shape = [k, k] + [k // 32] * 4
    trace_cap = sum(shape) + 1
    return [
        DotCase(name + '.fp32', 'fp8_grouped_dot_fp32', values, 1 + slack, 4,
                ('ok', [oracle32.fp32_bits]), FP32),
        DotCase(name + '.bf16', 'fp8_grouped_dot_bf16', values, 1 + slack, 2,
                ('ok', [oracle16.bf16_bits]), BF16),
        DotCase(name + '.trace-fp32', 'fp8_grouped_dot_trace_fp32', values, trace_cap, 4,
                ('ok', words + [oracle32.fp32_bits]), FP32),
        DotCase(name + '.trace-bf16', 'fp8_grouped_dot_trace_bf16', values, trace_cap, 4,
                ('ok', words + [oracle16.bf16_bits]), FP32),
    ]


def directed_vectors():
    maximum = [Fraction(200704)] * 32
    return [
        ('weight-decode', SENSITIVITY_VECTORS['wrong-weight-e2m1']()),
        ('move-scale', SENSITIVITY_VECTORS['move-scale-per-product']()),
        ('swap-scale', SENSITIVITY_VECTORS['swap-scales']()),
        ('in-block-order', SENSITIVITY_VECTORS['in-block-order']()),
        ('cross-block-order', SENSITIVITY_VECTORS['cross-block-order']()),
        ('late-bf16', SENSITIVITY_VECTORS['early-bf16']()),
        ('underflow', build_case([[power2(-10)]], (0x00,), (0x73,))),
        ('negative-zero', build_case([[-power2(-10)]], (0x00,), (0x72,))),
        ('opposite-inf', build_case([maximum, [-value for value in maximum]],
                                    (0xFE, 0xFE), (0x7F, 0x7F))),
        ('scale-boundary', build_case([[Fraction(1)]] * 5,
                                      (0x00, 0x01, 0x7F, 0xFD, 0xFE),
                                      (0xFE, 0xFD, 0x7F, 0x01, 0x00))),
    ]


def parity_cases():
    for name, values in directed_vectors():
        yield from vector(name, values)
    pairs = [(a, w) for a in VALID_CODES for w in VALID_CODES]
    pairs.extend([(0, 0)] * (-len(pairs) % 32))
    boundaries = (0x00, 0x01, 0x7E, 0x7F, 0x80, 0xFD, 0xFE)
    a = [left for left, _ in pairs]
    w = [right for _, right in pairs]
    groups = len(a) // 32
    sa = [boundaries[index % len(boundaries)] for index in range(groups)]
    sb = [boundaries[(index * 5 + 3) % len(boundaries)] for index in range(groups)]
    yield from vector('all-finite-code-pairs', (a, w, sa, sb), slack=0)
    rows, k = 35, 96
    weights = tuple(VALID_CODES[(row * 19 + col * 7) % len(VALID_CODES)]
                    for row in range(rows) for col in range(k))
    scales = tuple((0x70 + scale_row * 11 + col * 3) % 0xFF
                   for scale_row in range((rows + 31) // 32) for col in range(k // 32))
    a = tuple(VALID_CODES[(col * 13 + 5) % len(VALID_CODES)] for col in range(k))
    for row in (0, 31, 32, rows - 1):
        w = weights[row * k:(row + 1) * k]
        sb = scales[(row // 32) * 3:(row // 32 + 1) * 3]
        yield from vector(f'2d-row-{row}', (a, w, (0x7D, 0x7F, 0x81), sb))
    rng = random.Random(0xF828)
    for index in range(16):
        groups = rng.choice((1, 2, 3, 5, 7))
        values = (tuple(rng.choice(VALID_CODES) for _ in range(32 * groups)),
                  tuple(rng.choice(VALID_CODES) for _ in range(32 * groups)),
                  tuple(rng.randrange(255) for _ in range(groups)),
                  tuple(rng.randrange(255) for _ in range(groups)))
        yield from vector(f'random-{index}', values)


def baseline(k=64):
    return ([0x38] * k, [0x40] * k, [0x7F] * (k // 32), [0x7E] * (k // 32))


def rejected(name, op, values, cap=2, width=4, kind=FP32, **kwargs):
    return DotCase(name, op, values, cap, width, ('err', 'invalid_argument'), kind, **kwargs)


def accepted(name, out, values, **kwargs):
    oracle = trace_fp8_grouped_dot(*values, output=out)
    width = 4 if out == FP32 else 2
    result = oracle.fp32_bits if out == FP32 else oracle.bf16_bits
    cap = 1 if kwargs.get('overlap') is not None else 3
    return DotCase(name, 'fp8_grouped_dot_' + out, values, cap, width, ('ok', [result]), out, **kwargs)


def ordinary_contract_cases():
    a, w, sa, sb = baseline()
    for out in (FP32, BF16):
        op = 'fp8_grouped_dot_' + out
        width = 4 if out == FP32 else 2
        metadata = [
            ([], [], [], []), (a[:16], w[:16], [], []), (a[:48], w[:48], sa[:1], sb[:1]),
            (a, w[:-1], sa, sb), (a, w + [0], sa, sb), (a, w, sa[:-1], sb),
            (a, w, sa + [0x7F], sb), (a, w, sa, sb[:-1]), (a, w, sa, sb + [0x7F]),
        ]
        for index, values in enumerate(metadata):
            yield rejected(f'{out}.metadata.{index}', op, values, width=width, kind=out)
            yield rejected(f'{out}.metadata.{index}.poison', op, values, width=width, kind=out,
                           poison=True)
        yield rejected(f'{out}.empty-output', op, (a, w, sa, sb), cap=0, width=width, kind=out)
        yield rejected(f'{out}.empty-output.poison', op, (a, w, sa, sb), cap=0,
                       width=width, kind=out, poison=True)
        for section in range(4):
            for bad in ((0x7F, 0xFF) if section < 2 else (0xFF,)):
                values = [list(a), list(w), list(sa), list(sb)]
                values[section][-1] = bad
                yield rejected(f'{out}.nan.{section}.{bad:x}', op, values, width=width, kind=out)
        sizes = [len(a), len(w), len(sa), len(sb)]
        for section, name in enumerate(('a', 'w', 'sa', 'sb')):
            yield rejected(f'{out}.overlap.{name}', op, (a, w, sa, sb), width=width, kind=out,
                           overlap=name)
            yield rejected(f'{out}.overlap.{name}.poison', op, (a, w, sa, sb),
                           width=width, kind=out, overlap=name, poison=True)
            after = ((sizes[section] + width - 1) // width) * width
            yield accepted(f'{out}.adjacent-after.{name}', out, (a, w, sa, sb), overlap=name,
                           out_offset=after)
            yield accepted(f'{out}.adjacent-before.{name}', out, (a, w, sa, sb), overlap=name,
                           in_offset=width)
            partial_offset = width if sizes[section] > width else 0
            yield rejected(f'{out}.partial.{name}', op, (a, w, sa, sb), width=width, kind=out,
                           overlap=name, out_offset=partial_offset)
        yield accepted(f'{out}.valid', out, (a, w, sa, sb))


def trace_case(name, out, values, *, rejected_case=False, trace_shape=None,
               trace_alias=None, in_offset=None, out_offset=None, poison=False, out_cap=1):
    k = len(values[0])
    shape = trace_shape if trace_shape is not None else [k, k] + [k // 32] * 4
    cap = sum(shape) + out_cap
    if rejected_case:
        expect = ('err', 'invalid_argument')
    else:
        oracle = trace_fp8_grouped_dot(*values, output=out)
        result = oracle.fp32_bits if out == FP32 else oracle.bf16_bits
        expect = ('ok', stage_words(oracle) + [result])
    return DotCase(name, 'fp8_grouped_dot_trace_' + out, values, cap, 4, expect, FP32,
                   out_cap=out_cap, trace_shape=trace_shape, trace_alias=trace_alias,
                   in_offset=in_offset, out_offset=out_offset, poison=poison)


def trace_contract_cases():
    values = baseline()
    k, groups = len(values[0]), len(values[0]) // 32
    shape = [k, k] + [groups] * 4
    for out in (FP32, BF16):
        for index in range(6):
            for delta in (-1, 1):
                bad = shape.copy()
                bad[index] += delta
                yield trace_case(f'trace.{out}.shape.{index}.{delta}', out, values,
                                 rejected_case=True, trace_shape=bad)
                yield trace_case(f'trace.{out}.shape.{index}.{delta}.poison', out, values,
                                 rejected_case=True, trace_shape=bad, poison=True)
        yield trace_case(f'trace.{out}.empty-output', out, values, rejected_case=True, out_cap=0)
        yield trace_case(f'trace.{out}.empty-output.poison', out, values,
                         rejected_case=True, out_cap=0, poison=True)
        writes = ('out', 'products', 'partial_sums', 'dot', 'after_a', 'block', 'acc')
        for first, second in itertools.combinations(writes, 2):
            yield trace_case(f'trace.{out}.write-alias.{first}.{second}', out, values,
                             rejected_case=True, trace_alias=first + ',' + second)
            yield trace_case(f'trace.{out}.write-alias.{first}.{second}.poison', out, values,
                             rejected_case=True, trace_alias=first + ',' + second, poison=True)
        for read in ('a', 'w', 'sa', 'sb'):
            for write in writes:
                yield trace_case(f'trace.{out}.read-write-alias.{read}.{write}', out, values,
                                 rejected_case=True, trace_alias=read + ',' + write)
                yield trace_case(f'trace.{out}.read-write-alias.{read}.{write}.poison', out, values,
                                 rejected_case=True, trace_alias=read + ',' + write, poison=True)
        yield trace_case(f'trace.{out}.adjacent.out-products', out, values,
                         trace_alias='out,products', out_offset=4)
        if out == FP32:
            oracle = trace_fp8_grouped_dot(*values)
            words = stage_words(oracle)
            shape = [k, k] + [groups] * 4
            yield DotCase('trace.fp32.unused-output-tail-overlaps-products',
                          'fp8_grouped_dot_trace_fp32', values, sum(shape) + 3, 4,
                          ('ok', words + [oracle.fp32_bits, words[0], words[1]]), FP32,
                          out_cap=3, trace_alias='out,products', in_offset=0, out_offset=4)
        bad = [list(part) for part in values]
        bad[1][-1] = 0xFF
        yield trace_case(f'trace.{out}.nan-weight', out, bad, rejected_case=True)


def combination_cases():
    bits = tuple((0x3F80 + (index * 17) % 0x400) | (0x8000 if index % 5 == 0 else 0)
                 for index in range(96))
    a, sa = act_quant(bits)
    w = tuple(VALID_CODES[(index * 29 + 3) % len(VALID_CODES)] for index in range(96))
    sb = (0x7E, 0x80, 0x7C)
    oracle = trace_fp8_grouped_dot(a, w, sa, sb)
    for out, expected, width in ((FP32, oracle.fp32_bits, 4), (BF16, oracle.bf16_bits, 2)):
        yield DotCase('act-quant-combination.' + out, 'act_quant_fp8_dot_' + out,
                      (bits, w, sb), 3, width, ('ok', [expected]), out)


def contract_cases():
    yield from ordinary_contract_cases()
    yield from trace_contract_cases()
    yield from combination_cases()


def environment_cases():
    values = dict(directed_vectors())
    cases = []
    for name in ('late-bf16', 'underflow', 'negative-zero', 'opposite-inf'):
        cases += vector('environment.' + name, values[name], slack=0)
    a, w, sa, sb = baseline()
    w[-1] = 0xFF
    cases += [rejected('environment.reject.fp32', 'fp8_grouped_dot_fp32', (a, w, sa, sb)),
              rejected('environment.reject.bf16', 'fp8_grouped_dot_bf16', (a, w, sa, sb),
                       width=2, kind=BF16),
              trace_case('environment.reject.trace-fp32', FP32, (a, w, sa, sb), rejected_case=True),
              trace_case('environment.reject.trace-bf16', BF16, (a, w, sa, sb), rejected_case=True)]
    return cases


def case_failures(cases, args=()):
    responses = run_executor(EXE, cases, args)
    return [message for case, response in zip(cases, responses) if (message := check_dot(case, response))]


class Parity(unittest.TestCase):
    def test_oracle_and_independent_expectation(self):
        cases = list(parity_cases())
        start = time.perf_counter()
        self.assertEqual(case_failures(cases, EXE_ARGS), [])
        print(f'P0028 parity: {len(cases)} requests; {time.perf_counter() - start:.2f}s', file=sys.stderr)


class Contract(unittest.TestCase):
    def test_atomic_validation_storage_and_combination(self):
        cases = list(contract_cases())
        self.assertEqual(case_failures(cases, EXE_ARGS), [])
        print(f'P0028 contract: {len(cases)} requests', file=sys.stderr)

    def test_oracle_rejects_encoded_content_and_lengths(self):
        a, w, sa, sb = baseline()
        invalid = [([], [], [], []), (a, w[:-1], sa, sb),
                   (a[:-1] + [0x7F], w, sa, sb), (a, w[:-1] + [0xFF], sa, sb),
                   (a, w, sa[:-1] + [0xFF], sb), (a, w, sa, sb[:-1] + [0xFF])]
        for values in invalid:
            with self.assertRaises(ValueError):
                fp8_grouped_dot(*values)
            with self.assertRaises(ValueError):
                trace_fp8_grouped_dot(*values)


class Environment(unittest.TestCase):
    def test_eight_modes_and_status_states(self):
        cases = environment_cases()
        for env, csr in ENVIRONMENTS.items():
            for status in STATUS_STATES:
                with self.subTest(env=env, status=status):
                    probe = mxcsr_probe()
                    first, *middle, last = run_executor(
                        EXE, [probe, *cases, probe],
                        [f'--mxcsr={env}', f'--status={status}', *EXE_ARGS])
                    expected = csr | (MXCSR_STATUS if status == 'set' else 0)
                    self.assertEqual(first[1][0], expected)
                    self.assertEqual(last[1][0], expected)
                    self.assertEqual([m for c, r in zip(cases, middle) if (m := check_dot(c, r))], [])
        print(f'P0028 environment: {len(cases)} requests x {len(ENVIRONMENTS) * 2} states',
              file=sys.stderr)


class Harness(unittest.TestCase):
    def sample(self):
        return vector('harness.sample', dict(directed_vectors())['late-bf16'], slack=0)

    def test_faults_missing_executor_and_result_corruption(self):
        cases = self.sample()
        self.assertEqual(case_failures(cases), [])
        for fault in ('crash', 'exit', 'garbage', 'drop'):
            with self.subTest(fault=fault), self.assertRaises(HarnessError):
                case_failures(cases, [f'--fault={fault}'])
        self.assertTrue(case_failures(cases, ['--fault=mismatch']))
        with self.assertRaises(HarnessError):
            case_failures(cases, ['--fault=input-write'])
        with self.assertRaises(HarnessError):
            run_executor(EXE.with_name('df_quant_exec_missing'), cases)

    def test_protocol_trace_shape_and_too_wide_tokens(self):
        valid = self.sample()[0].line()
        head, payload = valid.split(' : ')
        malformed = [head, valid + ' ; 0', valid.replace(' cap=1', ''),
                     head + ' cap=1 : ' + payload, head + ' shape=64 : ' + payload,
                     head + ' scale_shape=2 : ' + payload, head + ' alias : ' + payload,
                     head + ' out_cap=1 : ' + payload, head + ' trace_shape=64,64,2,2,2,2 : ' + payload,
                     valid.replace('fp8_grouped_dot_fp32', 'fp8_grouped_dot_fp16'),
                     valid.replace(payload.split()[0], '100')]
        trace = self.sample()[2]
        explicit = trace_case('protocol.baseline', FP32,
                              dict(directed_vectors())['late-bf16'],
                              trace_shape=[64, 64, 2, 2, 2, 2])
        source = explicit.line()
        malformed += [
            source.replace(' out_cap=1', ' out_cap=1 out_cap=1'),
            source.replace(f'cap={explicit.cap}', f'cap={explicit.cap - 1}', 1),
            source.replace('trace_shape=', 'trace_shape=1,'),
            source + ' ; 0',
        ]
        for baseline_text in (valid, source):
            proc = subprocess.run([str(EXE)], input=baseline_text + '\n',
                                  capture_output=True, text=True, timeout=60)
            self.assertEqual(proc.returncode, 0, proc.stderr)
        for text in malformed:
            self.assertNotIn(text, (valid, source), 'protocol mutation is a no-op')
            with self.subTest(text=text[:100]):
                proc = subprocess.run([str(EXE)], input=text + '\n',
                                      capture_output=True, text=True, timeout=60)
                self.assertNotEqual(proc.returncode, 0, proc.stdout)
        for line in ('ok ' + ' '.join(['00000000'] * (trace.cap - 1)),
                     'ok ' + ' '.join(['0000'] * trace.cap)):
            with self.assertRaises(HarnessError):
                parse_response(trace, line)

    def test_rejected_input_snapshots_and_nan_classification(self):
        values = baseline()
        bad = [list(part) for part in values]
        bad[1][-1] = 0xFF
        for out in (FP32, BF16):
            width = 4 if out == FP32 else 2
            cases = [rejected('snapshot.' + out, 'fp8_grouped_dot_' + out, bad,
                              width=width, kind=out),
                     trace_case('snapshot.trace.' + out, out, bad, rejected_case=True)]
            self.assertEqual(case_failures(cases), [])
            self.assertTrue(case_failures(cases, ['--fault=reject-input-write']))
        for out, payload in ((FP32, 0xFFC12345), (BF16, 0xFFC1)):
            case = vector('nan-classification', dict(directed_vectors())['opposite-inf'], slack=0)[
                2 if out == FP32 else 3]
            words = list(case.expect[1])
            words[-1] = payload
            self.assertIsNone(check_dot(case, ('ok', words)))
            words[-1] = 0
            self.assertIsNotNone(check_dot(case, ('ok', words)))

    def test_poison_contract_under_current_build(self):
        asan = run_executor(EXE, [Case('asan', 'asan', (), [], 1, 4, ('ok', []))])[0][1][0]
        a, w, sa, sb = baseline()
        invalid = rejected('poison.invalid', 'fp8_grouped_dot_fp32',
                           (a, w[:-1], sa, sb), poison=True)
        self.assertEqual(case_failures([invalid]), [])
        valid = accepted('poison.valid', FP32, (a, w, sa, sb), poison=True)
        if asan:
            with self.assertRaises(HarnessError):
                case_failures([valid])
        else:
            self.assertEqual(case_failures([valid]), [])
        print(f'P0028 harness ASan={asan}', file=sys.stderr)

    def test_suite_failure_propagates_and_skip_is_not_acceptance(self):
        controls = [('parity', 'mismatch'), ('contract', 'crash'), ('environment', 'exit')]
        for suite, fault in controls:
            proc = subprocess.run([sys.executable, __file__, '--exe', str(EXE), '--suite', suite,
                                   f'--exe-arg=--fault={fault}'], capture_output=True, text=True,
                                  timeout=900)
            self.assertNotEqual(proc.returncode, 0, (suite, fault))
        proc = subprocess.run([sys.executable, __file__, '--exe', str(EXE), '--control', 'skip'],
                              capture_output=True, text=True, timeout=60)
        self.assertNotEqual(proc.returncode, 0)


SUITES = {'parity': Parity, 'contract': Contract, 'environment': Environment, 'harness': Harness}


def main():
    global EXE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', required=True, type=Path)
    parser.add_argument('--suite', choices=SUITES)
    parser.add_argument('--exe-arg', action='append', default=[])
    parser.add_argument('--control', choices=('skip', 'error'))
    args = parser.parse_args()
    EXE = args.exe
    EXE_ARGS.extend(args.exe_arg)
    if args.control:
        class Control(unittest.TestCase):
            def runTest(self):
                if args.control == 'skip':
                    self.skipTest('P0028 skip control')
                raise RuntimeError('P0028 error control')
        suite = unittest.TestSuite([Control()])
    elif args.suite:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(SUITES[args.suite])
    else:
        parser.error('--suite is required unless --control is used')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() and result.testsRun and not result.skipped else 1


if __name__ == '__main__':
    sys.exit(main())
