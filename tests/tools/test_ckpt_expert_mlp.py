#!/usr/bin/env python3
"""P0022 synthetic full-expert checkpoint reads, parity, reports and failures."""

import argparse
import contextlib
import hashlib
import io
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tools import ckpt_expert_mlp as mlp
from test_ckpt_inventory import TINY, build_checkpoint, write_shard

bounded = mlp.bounded
EXE = None
TOOL = ROOT / "tools/ckpt_expert_mlp.py"
PREFIX = "layers.0.ffn.experts.0"
LAYOUT = {"w1": (96, 64), "w3": (96, 64), "w2": (64, 96)}


def matrix_row(short, index):
    """Independent row-major bytes with distinct rows, groups and matrices."""
    _, k = LAYOUT[short]
    seed = {"w1": 11, "w2": 37, "w3": 73}[short]
    packed = bytes((seed + 17 * index + 29 * column) % 256 for column in range(k // 2))
    scale = bytes(112 + (seed + 5 * index + 7 * group) % 31 for group in range(k // 32))
    return packed, scale


def checkpoint_data(name, dtype, length):
    for short, (n, _) in LAYOUT.items():
        for part, suffix in enumerate(("weight", "scale")):
            if name == f"{PREFIX}.{short}.{suffix}":
                result = b"".join(matrix_row(short, index)[part] for index in range(n))
                assert len(result) == length
                return result
    return bytes(length)


def suite_exit(result):
    return int(not result.wasSuccessful() or bool(result.skipped))


class FullExpertTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def checkpoint(self):
        return build_checkpoint(self.root / "fixture", cfg={**TINY, "moe_inter_dim": 96},
                                n_shards=4, data=checkpoint_data)

    def assert_error(self, code, function, *args):
        with self.assertRaises(bounded.ConsumptionError) as caught:
            function(*args)
        self.assertEqual(caught.exception.code, code, caught.exception.detail)
        self.assertTrue(caught.exception.detail)
        return caught.exception

    def cli(self, *args):
        return subprocess.run([sys.executable, str(TOOL), *map(str, args)],
                              capture_output=True, text=True, timeout=300)

    def test_independent_all_rows_layout_and_exact_bounded_intervals(self):
        checkpoint, reference = self.checkpoint()
        snapshot = bounded.inspect_checkpoint(checkpoint, reference)
        pread, calls = os.pread, []
        expected_calls = []
        for short in mlp.MATRIX_ORDER:
            matrix = bounded.select_matrix(snapshot, short)
            for index in range(LAYOUT[short][0]):
                for shard, entry in (matrix.weight, matrix.scale):
                    stride = entry.shape[1]
                    expected_calls.append((str(Path(shard.path).absolute()), stride,
                                           shard.data_start + entry.begin + index * stride))
        def capture(fd, length, offset):
            calls.append((os.readlink(f"/proc/self/fd/{fd}"), length, offset))
            return pread(fd, length, offset)
        with mock.patch.object(bounded.os, "pread", side_effect=capture):
            payloads = mlp.read_expert(snapshot)
        self.assertEqual(list(payloads), list(mlp.MATRIX_ORDER))
        for short in mlp.MATRIX_ORDER:
            payload = payloads[short]
            self.assertEqual((payload.name, payload.n, payload.k),
                             (f"{PREFIX}.{short}", *LAYOUT[short]))
            self.assertEqual(payload.packed,
                             b"".join(matrix_row(short, i)[0] for i in range(payload.n)))
            self.assertEqual(payload.scale,
                             b"".join(matrix_row(short, i)[1] for i in range(payload.n)))
            self.assertEqual([row["index"] for row in payload.rows], list(range(payload.n)))
            for row in payload.rows:
                for read, raw in zip(row["reads"], matrix_row(short, row["index"])):
                    self.assertEqual((read["length"], read["sha256"]),
                                     (len(raw), hashlib.sha256(raw).hexdigest()))
                    self.assertEqual(read["tensor"], f"{PREFIX}.{short}." +
                                     ("weight" if len(raw) == payload.k // 2 else "scale"))
        self.assertEqual(calls, expected_calls)
        self.assertEqual(len(calls), 2 * sum(n for n, _ in LAYOUT.values()))

    def test_independent_fixture_detects_row_modulo_mapping_fault(self):
        checkpoint, reference = self.checkpoint()
        snapshot = bounded.inspect_checkpoint(checkpoint, reference)
        matrix = bounded.select_matrix(snapshot, "w1")
        real = bounded.read_row
        def modulo(candidate, index):
            return real(candidate, index % 3)
        with mock.patch.object(bounded, "read_row", side_effect=modulo):
            payload = mlp.read_matrix(matrix)
        expected = b"".join(matrix_row("w1", index)[1] for index in range(LAYOUT["w1"][0]))
        self.assertNotEqual(payload.scale, expected)
        self.assertNotEqual(payload.packed,
                            b"".join(matrix_row("w1", index)[0]
                                     for index in range(LAYOUT["w1"][0])))

    def test_read_matrix_rejects_wrong_descriptor_before_payload_read(self):
        checkpoint, reference = self.checkpoint()
        snapshot = bounded.inspect_checkpoint(checkpoint, reference)
        matrix = bounded.select_matrix(snapshot, "w1")
        with mock.patch.object(bounded, "read_row") as read:
            for bad in (replace(matrix, n=0), replace(matrix, k=63),
                        replace(matrix, scale=matrix.weight)):
                self.assert_error("row_layout", mlp.read_matrix, bad)
            other = replace(matrix, name=PREFIX + ".shared_experts.w1")
            self.assert_error("row_layout", mlp.read_matrix, other)
            read.assert_not_called()

    def test_shape_relation_rejected_after_only_selected_payloads(self):
        checkpoint, reference = self.checkpoint()
        snapshot = bounded.inspect_checkpoint(checkpoint, reference)
        original = mlp.read_matrix
        def wrong(matrix):
            payload = original(matrix)
            return replace(payload, k=64) if payload.name.endswith(".w2") else payload
        with mock.patch.object(mlp, "read_matrix", side_effect=wrong):
            self.assert_error("matrix", mlp.read_expert, snapshot)

    def test_late_short_read_and_io_are_failures(self):
        checkpoint, reference = self.checkpoint()
        snapshot = bounded.inspect_checkpoint(checkpoint, reference)
        matrix = bounded.select_matrix(snapshot, "w2")
        final_offset = None
        final_length = None
        final_row = bounded.read_row(matrix, matrix.n - 1)
        final_offset, final_length = final_row.reads[-1]["offset"], final_row.reads[-1]["length"]
        real = os.pread
        for exception in (False, True):
            def fail(fd, length, offset):
                if offset == final_offset:
                    if exception:
                        raise OSError("injected late read")
                    return b"\x7f" * max(0, final_length - 1)
                return real(fd, length, offset)
            with mock.patch.object(bounded.os, "pread", side_effect=fail):
                error = self.assert_error("io" if exception else "short_read",
                                          mlp.read_matrix, matrix)
            self.assertIn(f"row={matrix.n - 1}", error.detail)

    def test_input_route_matrix_uses_one_trace_per_case_and_reuses_output(self):
        checkpoint, reference = self.checkpoint()
        payloads = mlp.read_expert(bounded.inspect_checkpoint(checkpoint, reference))
        inputs = bounded.input_cases(64)
        calls = []
        def traced(values, w1, w2, w3, route_weight_bits=None):
            calls.append(route_weight_bits)
            return SimpleNamespace(output_bf16=(1, 2))
        with mock.patch.object(mlp, "trace_expert", side_effect=traced):
            cases = mlp.oracle_cases(payloads, inputs)
        self.assertEqual([(case.name, case.route_weight_bits) for case in cases],
                         [("mixed.unweighted", None), ("mixed.route_half", 0x3F000000),
                          ("group_ramp.unweighted", None),
                          ("group_ramp.route_half", 0x3F000000)])
        self.assertEqual(calls, [None, 0x3F000000, None, 0x3F000000])
        self.assertTrue(all(case.output == (1, 2) for case in cases))

    def test_oracle_error_rejected_with_case_and_stage(self):
        checkpoint, reference = self.checkpoint()
        payloads = mlp.read_expert(bounded.inspect_checkpoint(checkpoint, reference))
        inputs = {"mixed": bounded.input_cases(64)["mixed"]}
        with mock.patch.object(mlp, "trace_expert", side_effect=ValueError("bad scale")):
            error = self.assert_error("oracle", mlp.oracle_cases, payloads, inputs)
        self.assertIn("input=mixed.unweighted", error.detail)
        self.assertIn("stage=full_expert", error.detail)

    def test_adapter_wire_protocol_has_fixed_shapes_and_payload_order(self):
        checkpoint, reference = self.checkpoint()
        payloads = mlp.read_expert(bounded.inspect_checkpoint(checkpoint, reference))
        oracle = mlp.oracle_cases(payloads, {"mixed": bounded.input_cases(64)["mixed"]})[0]
        ExpertCase, omit, _, _ = mlp._load_protocol()
        for traced, op in ((False, "expert_bf16"), (True, "expert_trace")):
            case, _, _, _ = mlp._construct_case(ExpertCase, omit, name="mixed", traced=traced,
                                                 payloads=payloads, oracle=oracle)
            head, body = case.line().split(" : ")
            self.assertTrue(head.startswith(op + " "), head)
            self.assertIn("shape=96,64,64,96,96,64", head)
            self.assertIn("cap=64", head)
            self.assertIn("work_shape=64,2,96,96,96,96,3", head)
            self.assertNotIn("route=", head)
            self.assertEqual("trace_shape=96,96,96,96,96" in head, traced)
            sections = [[int(token, 16) for token in part.split()]
                        for part in body.split(" ; ")]
            self.assertEqual(len(sections), 7)
            self.assertEqual(sections, [list(oracle.input_bf16), list(payloads["w1"].packed),
                                       list(payloads["w1"].scale), list(payloads["w2"].packed),
                                       list(payloads["w2"].scale), list(payloads["w3"].packed),
                                       list(payloads["w3"].scale)])

    def test_real_executor_ordinary_and_trace_all_stages(self):
        checkpoint, reference = self.checkpoint()
        payloads = mlp.read_expert(bounded.inspect_checkpoint(checkpoint, reference))
        inputs = bounded.input_cases(64)
        cases = mlp.oracle_cases(payloads, inputs)
        records, comparisons = mlp.verify(payloads, cases, [EXE])
        self.assertEqual(len(records), 1)
        self.assertEqual([case["case"] for case in records[0]["cases"]],
                         ["mixed.unweighted", "mixed.route_half",
                          "group_ramp.unweighted", "group_ramp.route_half"])
        for case in records[0]["cases"]:
            self.assertEqual(set(case["entries"]), {"expert_bf16", "expert_trace"})
            self.assertTrue(all(entry["pass"] for entry in case["entries"].values()))
        expected_ordinary = len(mlp._expected_values(cases[0].trace, False)[0])
        expected_trace = len(mlp._expected_values(cases[0].trace, True)[0])
        self.assertEqual(comparisons["ordinary"], 4 * expected_ordinary)
        self.assertEqual(comparisons["trace"], 4 * expected_trace)
        self.assertEqual(comparisons["total"], comparisons["ordinary"] + comparisons["trace"])

    def test_stage_compare_nan_class_width_and_mismatch_location(self):
        self.assertEqual(mlp._compare([0x7FC00000, 0x7FC0], [0x7FC12345, 0x7FC1],
                                     ["silu_bits[0]", "output_bf16[0]"],
                                     ["fp32", "bf16"], "ctx"), 2)
        for expected, actual, kind in ((0x7FC0, 0x17FC0, "bf16"),
                                       (0x7F, 0x17F, "raw8"),
                                       (0x7FC00000, 0x17FC00000, "fp32")):
            self.assert_error("protocol", mlp._compare, [expected], [actual],
                              ["stage[0]"], [kind], "ctx")
        error = self.assert_error("mismatch", mlp._compare,
                                  [0x3F800000], [0x3F800001],
                                  ["bridge.gated_bits[17]"], ["fp32"], "input=mixed")
        self.assertIn("stage=bridge.gated_bits[17]", error.detail)
        self.assert_error("protocol", mlp._compare, [1], [], ["x"], ["raw8"], "ctx")

    def test_executor_and_protocol_fail_closed(self):
        checkpoint, reference = self.checkpoint()
        payloads = mlp.read_expert(bounded.inspect_checkpoint(checkpoint, reference))
        cases = mlp.oracle_cases(payloads, {"mixed": bounded.input_cases(64)["mixed"]})
        self.assert_error("executor", mlp.verify, payloads, cases, [self.root / "missing"])
        with mock.patch.object(mlp, "_load_protocol",
                               side_effect=bounded.ConsumptionError("protocol", "missing adapter")):
            self.assert_error("protocol", mlp.verify, payloads, cases, [EXE])

    def test_full_report_identity_fingerprints_intervals_and_two_executors(self):
        checkpoint, reference = self.checkpoint()
        import test_expert_parity as parity
        with mock.patch.object(mlp, "trace_expert", wraps=mlp.trace_expert) as oracle, \
                mock.patch.object(parity.impl, "trace_expert",
                                  side_effect=AssertionError("adapter recomputed oracle")):
            report = mlp.run(checkpoint, [EXE, EXE], reference)
        self.assertEqual(oracle.call_count, 4)
        self.assertEqual((report["plan"], report["decision"], report["pass"]),
                         ("P0022", "D0017", True))
        self.assertEqual(report["payload_order"], list(mlp.PAYLOAD_ORDER))
        self.assertEqual([m["name"].rsplit(".", 1)[1] for m in report["matrices"]],
                         list(mlp.MATRIX_ORDER))
        self.assertEqual(len(report["executors"]), 2)
        digest = hashlib.sha256(Path(EXE).read_bytes()).hexdigest()
        self.assertTrue(all(exe["sha256"] == digest for exe in report["executors"]))
        self.assertEqual(report["version"], (ROOT / "VERSION").read_text().strip())
        self.assertEqual(len(report["git_sha"]), 40)
        self.assertIsInstance(report["dirty"], bool)
        for matrix in report["matrices"]:
            short = matrix["name"].rsplit(".", 1)[1]
            for key, part in (("packed", 0), ("scale_b", 1)):
                raw = b"".join(matrix_row(short, i)[part] for i in range(matrix["n"]))
                self.assertEqual(matrix["payloads"][key],
                                 {"length": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
            self.assertEqual([row["index"] for row in matrix["rows"]], list(range(matrix["n"])))
        self.assertEqual([(item["case"], item["route_weight_bits"]) for item in report["inputs"]],
                         [("mixed.unweighted", None), ("mixed.route_half", 0x3F000000),
                          ("group_ramp.unweighted", None),
                          ("group_ramp.route_half", 0x3F000000)])
        self.assertEqual(report["comparisons"]["total"],
                         sum(report["comparisons"][key] for key in ("ordinary", "trace")))
        json.dumps(report)

    def test_cli_report_is_exclusive_outside_checkpoint_and_checkpoint_unchanged(self):
        checkpoint, reference = self.checkpoint()
        before = {path: hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in Path(checkpoint).iterdir() if path.is_file()}
        output = self.root / "report.json"
        proc = self.cli(checkpoint, "--exe", EXE, "--reference", reference, "--json", output)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        report = json.loads(output.read_text())
        self.assertTrue(report["pass"])
        self.assertEqual(report["command"][-2:], ["--json", str(output)])
        self.assertEqual(before, {path: hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in before})
        self.assertIn("RESULT PASS", proc.stdout)

    def test_cli_invalid_args_resources_and_report_protection(self):
        checkpoint, reference = self.checkpoint()
        existing = self.root / "existing.json"
        existing.write_text("keep")
        checkpoint_link = self.root / "checkpoint-link"
        checkpoint_link.symlink_to(checkpoint, target_is_directory=True)
        dangling = self.root / "dangling.json"
        dangling.symlink_to(self.root / "missing-target")
        invocations = [(self.root / "missing", "--exe", EXE), (checkpoint,),
                       (checkpoint, "--exe", EXE, "--unknown"),
                       (checkpoint, "--exe", self.root / "missing-exe", "--reference", reference)]
        invocations += [(checkpoint, "--exe", EXE, "--reference", reference, "--json", target)
                        for target in (existing, Path(checkpoint) / "report.json",
                                       checkpoint_link / "report.json", dangling)]
        for args in invocations:
            proc = self.cli(*args)
            self.assertNotEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertNotIn("RESULT PASS", proc.stdout)
        self.assertEqual(existing.read_text(), "keep")
        self.assertFalse((Path(checkpoint) / "report.json").exists())
        self.assertFalse((self.root / "missing-target").exists())

    def test_cli_failures_and_test_skips_never_pass(self):
        for code in ("short_read", "io", "executor", "oracle", "mismatch", "protocol", "argument"):
            with mock.patch.object(mlp, "run", side_effect=bounded.ConsumptionError(code, "injected")):
                with contextlib.redirect_stderr(io.StringIO()):
                    self.assertNotEqual(mlp.main([str(self.root), "--exe", str(EXE)]), 0)
        for successful, skipped in ((True, []), (True, ["skip"]), (False, [])):
            result = SimpleNamespace(wasSuccessful=lambda: successful, skipped=skipped)
            self.assertEqual(suite_exit(result), int(not successful or bool(skipped)))


def main(argv=None):
    global EXE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", required=True, type=Path)
    args = parser.parse_args(argv)
    EXE = args.exe.absolute()
    if not EXE.is_file():
        parser.error(f"executor missing: {EXE}")
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(FullExpertTests)
    return suite_exit(unittest.TextTestRunner(verbosity=2).run(suite))


if __name__ == "__main__":
    sys.exit(main())
