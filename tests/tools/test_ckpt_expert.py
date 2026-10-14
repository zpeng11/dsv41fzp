#!/usr/bin/env python3
"""P0018 independent row-layout, CPU consumption, failure and sensitivity tests.

Run with --exe <df_quant_exec>; no checkpoint, executor or skipped-test fallback.
Only the P0016 safetensors container writer is reused for hand-built row data.
"""

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
from tools import ckpt_expert as expert
from test_ckpt_inventory import TINY, build_checkpoint, write_shard
from ref.act_quant import act_quant
from ref.compute import trace_expert_grouped_dot
from ref.formats import unpack_e2m1

inv = expert.inv
PREFIX = "layers.0.ffn.experts.0"
EXE = None
TOOL = ROOT / "tools/ckpt_expert.py"
HAND_ROWS = {
    "w1": (64, (b"\x21" * 16 + b"\x43" * 16,
                 b"\x65" * 16 + b"\xa9" * 16,
                 b"\xcb" * 16 + b"\xed" * 16),
           (b"\x7f\x80", b"\x7e\x81", b"\x82\x7d")),
    "w2": (96, tuple(bytes([v]) * 16 + b"\x32" * 16 + b"\xa4" * 16
                     for v in (0x21, 0x43, 0x65, 0xA9, 0xCB)),
           (b"\x7c\x7f\x82", b"\x7d\x80\x83", b"\x7e\x81\x84",
            b"\x7f\x82\x85", b"\x80\x83\x86")),
    "w3": (32, tuple(bytes([v]) * 16 for v in (0x99, 0x21, 0x43, 0x65, 0xA9)),
           (b"\x7e", b"\x7f", b"\x80", b"\x81", b"\x82")),
}
TINY_LAYOUT = {"w1": (96, 64), "w3": (96, 64), "w2": (64, 96)}


def tiny_rows(short, index):
    """Independent fixture payloads from fixed tiny dimensions, not tool descriptors."""
    _, k = TINY_LAYOUT[short]
    weight = b"".join(bytes([(0x21, 0x43, 0x65)[(index + g) % 3]]) * 16
                      for g in range(k // 32))
    scale = bytes(125 + (index + g) % 5 for g in range(k // 32))
    return weight, scale


def tiny_data(name, dtype, nbytes):
    for short, (n, _) in TINY_LAYOUT.items():
        for part, suffix in enumerate(("weight", "scale")):
            if name == f"{PREFIX}.{short}.{suffix}":
                payload = b"".join(tiny_rows(short, i)[part] for i in range(n))
                assert len(payload) == nbytes
                return payload
    return bytes(nbytes)


def suite_exit(result):
    return int(not result.wasSuccessful() or bool(result.skipped))


class ExpertTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.matrices, self.intervals = {}, {}
        for short, (k, weights, scales) in HAND_ROWS.items():
            name = f"{PREFIX}.{short}"
            path = self.root / f"{short}.safetensors"
            tensors = [("before", "U8", (7,), b"SENTINL"),
                       (name + ".weight", "I8", (len(weights), k // 2), b"".join(weights)),
                       ("between", "U8", (11,), b"not-a-scale"),
                       (name + ".scale", "F8_E8M0", (len(weights), k // 32), b"".join(scales)),
                       ("after", "U8", (4,), b"TAIL")]
            write_shard(path, tensors, metadata={"format": "pt"}, pad=37)
            with path.open("rb") as fh:
                start = 8 + struct.unpack("<Q", fh.read(8))[0]
            shard = inv.read_header(path)
            entries = {e.name: e for e in shard.entries}
            self.matrices[short] = expert.Matrix(name, len(weights), k,
                                                 (shard, entries[name + ".weight"]),
                                                 (shard, entries[name + ".scale"]))
            self.intervals[short] = (start + 7, start + 7 + sum(map(len, weights)) + 11)

    def assert_error(self, code, function, *args):
        with self.assertRaises(expert.ConsumptionError) as ctx:
            function(*args)
        self.assertEqual(ctx.exception.code, code, ctx.exception.detail)
        self.assertTrue(ctx.exception.detail)
        return ctx.exception

    def assert_row(self, short, index, row):
        k, weights, scales = HAND_ROWS[short]
        matrix = self.matrices[short]
        self.assertEqual((row.matrix, row.index, row.k), (matrix.name, index, k))
        self.assertEqual((row.weight, row.scale), (weights[index], scales[index]))
        expected = []
        for suffix, data, base in zip(("weight", "scale"), (weights[index], scales[index]),
                                      self.intervals[short]):
            expected.append({"tensor": matrix.name + "." + suffix,
                             "shard": str(self.root / f"{short}.safetensors"),
                             "offset": base + index * len(data), "length": len(data),
                             "sha256": hashlib.sha256(data).hexdigest()})
        self.assertEqual(row.reads, expected)

    def checkpoint(self):
        return build_checkpoint(self.root / "full", cfg={**TINY, "moe_inter_dim": 96},
                                n_shards=3, data=tiny_data)

    def cli(self, *args):
        return subprocess.run([sys.executable, str(TOOL), *map(str, args)],
                              capture_output=True, text=True, timeout=120)

    def test_hand_rows_exact_bytes_intervals_and_only_two_preads(self):
        real_pread = os.pread
        for short, matrix in self.matrices.items():
            for index in range(matrix.n):
                with self.subTest(matrix=short, index=index):
                    calls = []
                    def capture(fd, length, offset):
                        calls.append((os.readlink(f"/proc/self/fd/{fd}"), length, offset))
                        return real_pread(fd, length, offset)
                    with mock.patch.object(expert.os, "pread", side_effect=capture):
                        row = expert.read_row(matrix, index)
                    self.assert_row(short, index, row)
                    self.assertEqual(calls, [(r["shard"], r["length"], r["offset"]) for r in row.reads])
        self.assertEqual(len({m.weight[0].path for m in self.matrices.values()}), 3)

    def test_low_nibble_is_even_k_and_high_nibble_is_odd_k(self):
        row = expert.read_row(self.matrices["w1"], 0)
        self.assertEqual(row.weight, b"\x21" * 16 + b"\x43" * 16)
        self.assertEqual(unpack_e2m1(row.weight, (64,)), (1, 2) * 16 + (3, 4) * 16)

    def test_invalid_row_indices_include_bool(self):
        for index in (-1, 3, True, False, 1.0, "1", None):
            with self.subTest(index=index):
                self.assert_error("row_range", expert.read_row, self.matrices["w1"], index)

    def test_matrix_pairing_and_descriptor_metadata_rejected_before_read(self):
        matrix = self.matrices["w1"]
        shard, entry = matrix.weight
        bad = [replace(matrix, scale=self.matrices["w3"].scale),
               replace(matrix, n=True), replace(matrix, k=63), replace(matrix, n=4),
               replace(matrix, weight=(shard, entry._replace(dtype="U8"))),
               replace(matrix, weight=(shard, entry._replace(shape=(3, 16)))),
               replace(matrix, weight=(shard, entry._replace(end=entry.end - 1))),
               replace(matrix, weight=(shard, entry._replace(name=PREFIX + ".w2.weight")))]
        with mock.patch.object(expert.os, "pread") as read:
            for candidate in bad:
                self.assert_error("row_layout", expert.read_row, candidate, 0)
            read.assert_not_called()

    def test_shortread_and_io_errors(self):
        for payload in (b"", b"\x21" * 31):
            with mock.patch.object(expert.os, "pread", return_value=payload):
                self.assert_error("short_read", expert.read_row, self.matrices["w1"], 0)
        with mock.patch.object(expert.os, "pread", side_effect=OSError("injected I/O")):
            self.assert_error("io", expert.read_row, self.matrices["w1"], 0)

    def test_file_truncated_after_header_validation(self):
        matrix = self.matrices["w1"]
        with open(matrix.scale[0].path, "r+b") as fh:
            fh.truncate(self.intervals["w1"][1] + 5)
        self.assert_error("short_read", expert.read_row, matrix, 2)

    def test_snapshot_full_tiny_layout_and_no_payload_pread(self):
        ckpt, reference = self.checkpoint()
        with mock.patch.object(expert.os, "pread", side_effect=AssertionError("payload in inspect")):
            snapshot = expert.inspect_checkpoint(ckpt, reference)
        self.assertEqual((snapshot.root, snapshot.revision), (Path(ckpt), "ckpt"))
        for short, dims in TINY_LAYOUT.items():
            matrix = expert.select_matrix(snapshot, short)
            self.assertEqual((matrix.name, matrix.n, matrix.k), (f"{PREFIX}.{short}", *dims))
            for index in (0, dims[0] // 2, dims[0] - 1):
                row = expert.read_row(matrix, index)
                self.assertEqual((row.weight, row.scale), tiny_rows(short, index))

    def test_invalid_selection(self):
        ckpt, reference = self.checkpoint()
        snapshot = expert.inspect_checkpoint(ckpt, reference)
        for selector in ("w0", "W1", "w1.weight", PREFIX + ".w1", True, None):
            self.assert_error("matrix", expert.select_matrix, snapshot, selector)

    def test_checkpoint_metadata_and_truncation_errors(self):
        ckpt, reference = self.checkpoint()
        path = next(Path(ckpt).glob("*.safetensors"))
        original = path.read_bytes()
        header_len = int.from_bytes(original[:8], "little")
        header = json.loads(original[8:8 + header_len])
        header["__metadata__"] = {"format": 1}
        encoded = json.dumps(header).encode()
        path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + original[8 + header_len:])
        self.assert_error("metadata_type", expert.inspect_checkpoint, ckpt, reference)
        path.write_bytes(original[:-1])
        self.assert_error("incomplete_buffer", expert.inspect_checkpoint, ckpt, reference)

    def test_selected_checkpoint_dtype_and_shape_checked(self):
        ckpt, reference = self.checkpoint()
        snapshot = expert.inspect_checkpoint(ckpt, reference)
        name = PREFIX + ".w1.weight"
        shard, entry = snapshot.tensors[name]
        for code, bad in (("dtype", entry._replace(dtype="U8")),
                          ("shape", entry._replace(shape=(48, 64)))):
            changed = replace(snapshot, tensors={**snapshot.tensors, name: (shard, bad)})
            self.assert_error(code, expert.select_matrix, changed, "w1")

    def test_input_recipe_finite_signed_and_cross_group_amplitudes(self):
        cases = expert.input_cases(96)
        self.assertEqual(len(cases), 2)
        self.assertEqual(cases, expert.input_cases(96))
        self.assertNotEqual(*cases.values())
        for bits in cases.values():
            self.assertEqual(len(bits), 96)
            self.assertTrue(all(0 < ((b >> 7) & 255) < 255 for b in bits))
            self.assertEqual({b >> 15 for b in bits}, {0, 1})
            self.assertGreater(len({(bits[g * 32] >> 7) & 255 for g in range(3)}), 1)
        for k in (0, True, 31, -32):
            self.assert_error("input", expert.input_cases, k)

    def test_real_executor_nonzero_rows_and_both_outputs(self):
        results = set()
        for short, matrix in self.matrices.items():
            for index in range(matrix.n):
                row = expert.read_row(matrix, index)
                inputs = expert.input_cases(matrix.k)
                records = expert.verify_row(row, inputs, EXE)
                self.assertEqual([r["case"] for r in records], list(inputs))
                weight, scale = HAND_ROWS[short][1][index], HAND_ROWS[short][2][index]
                for record in records:
                    a, sa = act_quant(inputs[record["case"]])
                    expected = trace_expert_grouped_dot(a, weight, sa, scale)
                    self.assertEqual((record["fp32_bits"], record["bf16_bits"]),
                                     (expected.fp32_bits, expected.bf16_bits))
                    self.assertTrue(record["pass"])
                    self.assertNotEqual(record["fp32_bits"] & 0x7fffffff, 0)
                    results.add(record["fp32_bits"])
        self.assertGreater(len(results), 10)

    def test_hand_calculated_read_and_consume_anchor(self):
        """Each group has inputs (1,2,0,...); .5+2 + 2*(1.5+4) = 13.5."""
        row = expert.read_row(self.matrices["w1"], 0)
        bits = (0x3f80, 0x4000) + (0,) * 30
        record = expert.verify_row(row, {"anchor": bits * 2}, EXE)[0]
        self.assertEqual((record["fp32_bits"], record["bf16_bits"]), (0x41580000, 0x4158))

    def test_legal_overflow_nan_payload_class_and_finite_to_nan_rejection(self):
        row = replace(expert.read_row(self.matrices["w1"], 0),
                      weight=b"\x77" * 32, scale=b"\xfe\xfe")
        inputs = {"opposite_infinities": (0x4000,) * 32 + (0xc000,) * 32}
        pristine = expert.verify_row(row, inputs, EXE)[0]
        self.assertGreater(pristine["fp32_bits"] & 0x7fffffff, 0x7f800000)
        self.assertGreater(pristine["bf16_bits"] & 0x7fff, 0x7f80)
        real_executor = expert.run_executor
        mutated = set()
        def change_nan_payload(exe, cases):
            responses = real_executor(exe, cases)
            case = cases[0]
            for i, value in enumerate(responses[0][1]):
                if expert.is_nan(value, case.kind):
                    responses[0][1][i] = 0x7fc12345 if case.kind == "fp32" else 0x7fc1
                    mutated.add(case.op)
            return responses
        with mock.patch.object(expert, "run_executor", side_effect=change_nan_payload):
            self.assertTrue(expert.verify_row(row, inputs, EXE)[0]["pass"])
        self.assertEqual(mutated, {"expert_grouped_dot_trace", "act_quant_dot_fp32", "act_quant_dot_bf16"})
        finite = expert.read_row(self.matrices["w1"], 0)
        for kind, nan in (("fp32", 0x7fc12345), ("bf16", 0x7fc1)):
            def finite_to_nan(exe, cases):
                responses = real_executor(exe, cases)
                if cases[0].op == "act_quant_dot_" + kind:
                    responses[0][1][0] = nan
                return responses
            with mock.patch.object(expert, "run_executor", side_effect=finite_to_nan):
                self.assert_error("mismatch", expert.verify_row, finite, inputs, EXE)

    def test_invalid_scale_input_and_row_rejected(self):
        row = expert.read_row(self.matrices["w1"], 0)
        inputs = expert.input_cases(64)
        error = self.assert_error("oracle", expert.verify_row, replace(row, scale=b"\xff\x80"), inputs, EXE)
        for token in (row.matrix, "row=0", "stage=compute"):
            self.assertIn(token, error.detail)
        self.assert_error("row_layout", expert.verify_row, replace(row, weight=row.weight[:-1]), inputs, EXE)
        for bad in ({}, {"short": (0x3f80,)}, {"nan": (0x7fc0,) * 64}):
            self.assert_error("input", expert.verify_row, row, bad, EXE)

    def test_full_run_report_and_comparison_counts(self):
        ckpt, reference = self.checkpoint()
        report = expert.run(ckpt, EXE, reference)
        self.assertEqual((report["plan"], report["pass"], report["environment_contract"]), ("P0018", True, "ENV-A"))
        self.assertEqual(report["exe"]["sha256"], hashlib.sha256(Path(EXE).read_bytes()).hexdigest())
        self.assertEqual(report["version"], (ROOT / "VERSION").read_text().strip())
        self.assertEqual(len(report["git_sha"]), 40)
        self.assertIsInstance(report["dirty"], bool)
        outputs = 0
        for matrix in report["matrices"]:
            short = matrix["name"].rsplit(".", 1)[1]
            n, k = TINY_LAYOUT[short]
            self.assertEqual((matrix["n"], matrix["k"]), (n, k))
            self.assertEqual([r["index"] for r in matrix["rows"]], [0, n // 2, n - 1])
            for row in matrix["rows"]:
                for read, payload in zip(row["reads"], tiny_rows(short, row["index"])):
                    self.assertEqual((read["length"], read["sha256"]),
                                     (len(payload), hashlib.sha256(payload).hexdigest()))
                outputs += sum(c["output_comparisons"] for c in row["cases"])
        self.assertEqual((outputs, report["comparisons"]["outputs"]), (36, 36))
        self.assertEqual(report["comparisons"]["total"], sum(report["comparisons"][f] for f in ("activation", "compute", "outputs")))
        json.dumps(report)

    def test_executor_missing_crash_malformed_missing_response_and_reject(self):
        row = expert.read_row(self.matrices["w3"], 0)
        inputs = expert.input_cases(32)
        self.assert_error("executor", expert.verify_row, row, inputs, self.root / "missing-exe")
        for mode, body in (("crash", "import os, signal; os.kill(os.getpid(), signal.SIGKILL)"), ("malformed", "print('garbage')"),
                           ("missingresponse", "pass"), ("libraryreject", "print('err invalid_argument changed=0')")):
            path = self.root / mode
            path.write_text(f"#!{sys.executable}\nimport sys\nsys.stdin.read()\n{body}\n")
            path.chmod(0o700)
            self.assert_error("executor", expert.verify_row, row, inputs, path)

    def test_executor_timeout_propagated_and_command_has_timeout(self):
        row = expert.read_row(self.matrices["w3"], 0)
        with mock.patch.object(expert.subprocess, "run", side_effect=subprocess.TimeoutExpired("exec", 1)) as run:
            self.assert_error("executor", expert.verify_row, row, expert.input_cases(32), EXE)
        self.assertGreater(run.call_args.kwargs["timeout"], 0)

    def test_row_offset_and_scale_offset_faults_detected_by_independent_bytes(self):
        matrix = self.matrices["w1"]
        pristine = expert.read_row(matrix, 0)
        self.assert_row("w1", 0, pristine)
        self.assertTrue(all(r["pass"] for r in expert.verify_row(pristine, expert.input_cases(64), EXE)))
        real_pread = os.pread
        for target, displacement in ((self.intervals["w1"][0], 32), (self.intervals["w1"][1], 2)):
            def shifted(fd, length, offset):
                return real_pread(fd, length, offset + displacement if offset == target else offset)
            with mock.patch.object(expert.os, "pread", side_effect=shifted):
                altered = expert.read_row(matrix, 0)
            self.assertTrue(all(r["pass"] for r in expert.verify_row(altered, expert.input_cases(64), EXE)))
            with self.assertRaises(AssertionError):
                self.assert_row("w1", 0, altered)

    def test_fp32_and_bf16_result_mutations_are_numeric_mismatches(self):
        row = expert.read_row(self.matrices["w1"], 0)
        inputs = {"anchor": ((0x3f80, 0x4000) + (0,) * 30) * 2}
        self.assertEqual(expert.verify_row(row, inputs, EXE)[0]["fp32_bits"], 0x41580000)
        real_executor = expert.run_executor
        for kind in ("fp32", "bf16"):
            def mutate(exe, cases):
                response = real_executor(exe, cases)
                if cases[0].op == "act_quant_dot_" + kind:
                    response[0][1][0] ^= 1
                return response
            with mock.patch.object(expert, "run_executor", side_effect=mutate):
                error = self.assert_error("mismatch", expert.verify_row, row, inputs, EXE)
            self.assertIn("stage=combination." + kind, error.detail)
            self.assertIn("expected", error.detail)

    def test_cli_success_failure_unknown_argument_and_required_exe(self):
        ckpt, reference = self.checkpoint()
        out = self.root / "report.json"
        proc = self.cli(ckpt, "--exe", EXE, "--reference", reference, "--json", out)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertTrue(json.loads(out.read_text())["pass"])
        for args in ((self.root / "absent", "--exe", EXE), (ckpt,),
                     (ckpt, "--exe", EXE, "--unknown"),
                     (ckpt, "--exe", self.root / "no-exe", "--reference", reference)):
            proc = self.cli(*args)
            self.assertNotEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertNotIn("RESULT PASS", proc.stdout)

    def test_cli_validation_failure_and_test_runner_skips_are_nonzero(self):
        for code in ("short_read", "executor", "oracle", "mismatch", "protocol"):
            with mock.patch.object(expert, "run", side_effect=expert.ConsumptionError(code, "injected failure")):
                with contextlib.redirect_stderr(io.StringIO()):
                    self.assertNotEqual(expert.main([str(self.root), "--exe", str(EXE)]), 0)
        for successful, skips in ((True, []), (True, ["skip"]), (False, [])):
            result = SimpleNamespace(wasSuccessful=lambda: successful, skipped=skips)
            self.assertEqual(suite_exit(result), int(not successful or bool(skips)))


def main(argv=None):
    global EXE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", required=True, type=Path)
    args = parser.parse_args(argv)
    EXE = args.exe.absolute()
    if not EXE.is_file():
        parser.error(f"executor missing: {EXE}")
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ExpertTests)
    return suite_exit(unittest.TextTestRunner(verbosity=2).run(suite))


if __name__ == "__main__":
    sys.exit(main())
