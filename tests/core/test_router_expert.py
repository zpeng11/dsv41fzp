#!/usr/bin/env python3
"""P0023 router-to-P0022 expert integration acceptance."""

import argparse
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
QUANT_TESTS = ROOT / "tests" / "quant"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(QUANT_TESTS))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_expert as independent_expert
import test_expert_parity as expert_parity
import test_router as independent_router
from ref import expert as expert_impl
from ref import router as router_impl
from test_codec_parity import HarnessError

EXE = None
EXE_ARGS = []
SELECT = []
D = 32
H = 32
N = 9


def gate_matrix():
    """Build nine distinct gate rows whose projections are exactly one through nine."""
    row_codes = (0x3D00, 0x3D80, 0x3DC0, 0x3E00, 0x3E20,
                 0x3E40, 0x3E60, 0x3E80, 0x3E90)
    return router_impl.GateMatrix(N, D, tuple(code for code in row_codes for _ in range(D)))


def expert_matrices():
    """Build ID-distinct FP4 triples so wrong expert mapping survives BF16 boundaries."""
    return tuple(
        (independent_expert.matrix(H, D, 1 + 3 * index, scale=119 + index),
         independent_expert.matrix(D, H, 2 + 5 * index, scale=120 + index),
         independent_expert.matrix(H, D, 7 + 7 * index, scale=121 + index))
        for index in range(N)
    )


def fixture():
    inputs = (0x3F80,) * D
    text_bias = (0x41800000,) * 6 + (0xC1800000,) * 3
    image_bias = (0xC1800000,) * 3 + (0x41800000,) * 6
    return inputs, gate_matrix(), text_bias, image_bias, expert_matrices()


def executable_error(path):
    if not path.exists():
        return f"executor does not exist: {path}"
    if not path.is_file():
        return f"executor is not a file: {path}"
    if not os.access(path, os.X_OK):
        return f"executor is not executable: {path}"
    return None


def expert_trace(expected):
    bridge = expected["bridge"]
    return expert_impl.ExpertTrace(
        expected["input_codes"], expected["input_scales"],
        expected["gate_bf16"], expected["up_bf16"],
        expert_impl.BridgeTrace(
            bridge["gate_bits"], bridge["up_bits"], bridge["silu_bits"],
            bridge["gated_bits"], bridge["weighted_bits"], bridge["bf16_bits"]),
        expected["hidden_codes"], expected["hidden_scales"],
        expected["output_bf16"])


def iter_tests(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from iter_tests(item)
        else:
            yield item


class RouterExpertIntegration(unittest.TestCase):
    def verified_route(self, inputs, gate, bias, *, image_token=False, bias_vl_bits=None):
        want = independent_router.expected_route(
            inputs, gate, bias, image_token=image_token, bias_vl_bits=bias_vl_bits)
        trace = router_impl.trace_route(
            inputs, gate, bias, image_token=image_token, bias_vl_bits=bias_vl_bits)
        for field in independent_router.TRACE_FIELDS:
            self.assertEqual(getattr(trace, field), want[field], field)
        actual = router_impl.route(
            inputs, gate, bias, image_token=image_token, bias_vl_bits=bias_vl_bits)
        self.assertEqual(actual, (want["indices"], want["weights_bits"]))
        self.assertEqual(actual, (trace.indices, trace.weights_bits))
        return actual, want

    def test_text_image_and_image_fallback_routes(self):
        inputs, gate, text_bias, image_bias, _ = fixture()
        text, text_expected = self.verified_route(
            inputs, gate, text_bias, bias_vl_bits=image_bias)
        image, image_expected = self.verified_route(
            inputs, gate, text_bias, image_token=True, bias_vl_bits=image_bias)
        fallback, fallback_expected = self.verified_route(
            inputs, gate, text_bias, image_token=True)
        self.assertEqual(text_expected["indices"], (5, 4, 3, 2, 1, 0))
        self.assertEqual(image_expected["indices"], (8, 7, 6, 5, 4, 3))
        self.assertNotEqual(text, image)
        self.assertEqual(fallback, text)
        self.assertEqual(fallback_expected, independent_router.expected_route(
            inputs, gate, text_bias))

    def test_router_outputs_drive_ordinary_and_trace_experts(self):
        inputs, gate, text_bias, image_bias, experts = fixture()
        cases = []
        branches = (
            ("text", dict(image_token=False, bias_vl_bits=image_bias)),
            ("image", dict(image_token=True, bias_vl_bits=image_bias)),
        )
        for branch, options in branches:
            actual, want = self.verified_route(inputs, gate, text_bias, **options)
            actual_indices, actual_weights = actual
            for position, (actual_index, actual_weight, expected_index, expected_weight) in enumerate(
                    zip(actual_indices, actual_weights, want["indices"], want["weights_bits"])):
                self.assertEqual(actual_index, expected_index)
                self.assertEqual(actual_weight, expected_weight)
                actual_matrices = experts[actual_index]
                expected_matrices = experts[expected_index]
                expected = independent_expert.check_expert(
                    self, (inputs, *expected_matrices), expected_weight)
                expected_trace = expert_trace(expected)
                for traced in (False, True):
                    cases.append(expert_parity.ExpertCase(
                        f"router-{branch}.slot={position}.expert={actual_index}",
                        inputs, *actual_matrices, route=actual_weight, traced=traced,
                        expected_trace=expected_trace))
        self.assertEqual(len(cases), 24)
        responses = expert_parity.execute(EXE, cases, EXE_ARGS)
        failures = [message for case, response in zip(cases, responses)
                    if (message := expert_parity.check_expert(case, response))]
        self.assertEqual(failures, [])

    def test_wrong_expert_and_wrong_weight_complete_with_numeric_mismatches(self):
        inputs, gate, text_bias, image_bias, experts = fixture()
        _, want = self.verified_route(inputs, gate, text_bias, bias_vl_bits=image_bias)
        self.assertEqual(len(set(want["weights_bits"])), 6)
        for position, (index, weight) in enumerate(
                zip(want["indices"], want["weights_bits"])):
            wrong_index = want["indices"][(position + 1) % 6]
            wrong_weight = want["weights_bits"][(position + 1) % 6]
            correct = independent_expert.check_expert(
                self, (inputs, *experts[index]), weight)
            wrong_id = independent_expert.expected_expert(
                inputs, *experts[wrong_index], weight)
            wrong_route = independent_expert.expected_expert(
                inputs, *experts[index], wrong_weight)
            self.assertNotEqual(correct["gate_bf16"], wrong_id["gate_bf16"])
            self.assertNotEqual(correct["output_bf16"], wrong_id["output_bf16"])
            self.assertNotEqual(correct["bridge"]["weighted_bits"],
                                wrong_route["bridge"]["weighted_bits"])
            self.assertNotEqual(correct["output_bf16"], wrong_route["output_bf16"])


class EntryFailClosed(unittest.TestCase):
    def test_missing_and_nonexecutable_cli_targets_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing-executor"
            self.assertIn("does not exist", executable_error(missing))
            nonexecutable = Path(directory) / "not-executable"
            nonexecutable.write_text("not an executor\n", encoding="utf-8")
            self.assertIn("not executable", executable_error(nonexecutable))

    def test_executable_that_is_not_the_protocol_executor_fails_closed(self):
        fake = Path("/bin/true")
        self.assertIsNone(executable_error(fake))
        inputs, _, _, _, experts = fixture()
        expected = independent_expert.expected_expert(
            inputs, *experts[0], 0x3F000000)
        case = expert_parity.ExpertCase(
            "fake-executor", inputs, *experts[0], route=0x3F000000,
            expected_trace=expert_trace(expected))
        with self.assertRaisesRegex(HarnessError, "expected 1 responses, got 0"):
            expert_parity.execute(fake, [case])


TEST_CLASSES = (RouterExpertIntegration, EntryFailClosed)


def main(argv=None):
    global EXE
    parser = argparse.ArgumentParser(description="P0023 router-to-expert integration")
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--exe-arg", action="append", default=[])
    parser.add_argument("--case", action="append", default=[])
    args = parser.parse_args(argv)
    error = executable_error(args.exe)
    if error:
        parser.error(error)
    EXE = args.exe.resolve()
    EXE_ARGS.extend(args.exe_arg)
    SELECT.extend(args.case)
    loader = unittest.TestLoader()
    suite = unittest.TestSuite(loader.loadTestsFromTestCase(cls) for cls in TEST_CLASSES)
    if SELECT:
        suite = unittest.TestSuite(test for test in iter_tests(suite)
                                   if any(token in test.id() for token in SELECT))
    if suite.countTestCases() == 0:
        print("no tests selected", file=sys.stderr)
        return 2
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() and result.testsRun and not result.skipped else 1


if __name__ == "__main__":
    print("P0023 router-to-expert integration; standard library only", flush=True)
    sys.exit(main())
