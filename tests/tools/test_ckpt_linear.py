#!/usr/bin/env python3
"""P0019 independent full-matrix reads, production entry parity and fault detection."""

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
from tools import ckpt_linear as linear
from test_ckpt_inventory import TINY, build_checkpoint, write_shard
from ref.formats import unpack_e2m1

expert = linear.expert
EXE = None
TOOL = ROOT / "tools/ckpt_linear.py"
NAME = "layers.0.ffn.experts.0.w1"
WEIGHTS = (b"\x21" * 16 + b"\x43" * 16,
           b"\x65" * 16 + b"\xa9" * 16,
           b"\xcb" * 16 + b"\xed" * 16)
SCALES = (b"\x7f\x80", b"\x7e\x81", b"\x82\x7d")


def checkpoint_row(index):
    """Independent known N=96, K=64 fixture; every weight row has distinct bytes."""
    return (bytes((index + 17 * j) % 256 for j in range(32)),
            bytes((124 + index % 7, 125 + (3 * index) % 7)))


def checkpoint_data(name, dtype, length):
    for part, suffix in enumerate(("weight", "scale")):
        if name == NAME + "." + suffix:
            result = b"".join(checkpoint_row(i)[part] for i in range(96))
            assert len(result) == length
            return result
    return bytes(length)


def suite_exit(result):
    return int(not result.wasSuccessful() or bool(result.skipped))


class LinearTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / "hand.safetensors"
        tensors = [("before", "U8", (7,), b"SENTINL"),
                   (NAME + ".weight", "I8", (3, 32), b"".join(WEIGHTS)),
                   ("between", "U8", (11,), b"not-a-scale"),
                   (NAME + ".scale", "F8_E8M0", (3, 2), b"".join(SCALES)),
                   ("after", "U8", (4,), b"TAIL")]
        write_shard(self.path, tensors, metadata={"format": "pt"}, pad=37)
        with self.path.open("rb") as fh:
            start = 8 + struct.unpack("<Q", fh.read(8))[0]
        self.bases = start + 7, start + 7 + 96 + 11
        shard = expert.inv.read_header(self.path)
        entries = {e.name: e for e in shard.entries}
        self.matrix = expert.Matrix(NAME, 3, 64, (shard, entries[NAME + ".weight"]),
                                    (shard, entries[NAME + ".scale"]))
        self.inputs = expert.input_cases(64)

    def checkpoint(self):
        return build_checkpoint(self.root / "full", cfg={**TINY, "moe_inter_dim": 96},
                                n_shards=3, data=checkpoint_data)

    def assert_error(self, code, function, *args):
        with self.assertRaises(expert.ConsumptionError) as ctx:
            function(*args)
        self.assertEqual(ctx.exception.code, code, ctx.exception.detail)
        self.assertTrue(ctx.exception.detail)
        return ctx.exception

    def cli(self, *args):
        return subprocess.run([sys.executable, str(TOOL), *map(str, args)],
                              capture_output=True, text=True, timeout=120)

    def assert_payload(self, payload):
        self.assertEqual((payload.name, payload.n, payload.k), (NAME, 3, 64))
        self.assertEqual(payload.weight, b"".join(WEIGHTS))
        self.assertEqual(payload.scale, b"".join(SCALES))
        self.assertEqual([r["index"] for r in payload.rows], [0, 1, 2])
        for row in payload.rows:
            index = row["index"]
            expected = []
            for suffix, data, base in zip(("weight", "scale"), (WEIGHTS[index], SCALES[index]), self.bases):
                expected.append({"tensor": NAME + "." + suffix, "shard": str(self.path),
                                 "offset": base + index * len(data), "length": len(data),
                                 "sha256": hashlib.sha256(data).hexdigest()})
            self.assertEqual(row["reads"], expected)

    def test_all_rows_exact_bytes_intervals_no_other_payload_reads(self):
        calls, pread = [], os.pread
        def capture(fd, length, offset):
            calls.append((os.readlink(f"/proc/self/fd/{fd}"), length, offset))
            return pread(fd, length, offset)
        with mock.patch.object(expert.os, "pread", side_effect=capture):
            payload = linear.read_matrix(self.matrix)
        self.assert_payload(payload)
        self.assertEqual(calls, [(str(self.path), size, base + index * size)
                                for index in range(3) for size, base in zip((32, 2), self.bases)])
        self.assertEqual(unpack_e2m1(payload.weight[:32], (64,)), (1, 2) * 16 + (3, 4) * 16)

    def test_adapter_is_one_full_matrix_with_three_payloads(self):
        payload = linear.read_matrix(self.matrix)
        for kind, width in (("fp32", 4), ("bf16", 2)):
            case = linear.LinearCase("mixed", kind, 3, self.inputs["mixed"], payload.weight, payload.scale, [1, 2, 3])
            head, body = case.line().split(" : ")
            self.assertEqual(head, f"expert_linear_{kind} shape=3 cap=3")
            self.assertEqual([[int(v, 16) for v in section.split()] for section in body.split(";")],
                             [list(self.inputs["mixed"]), list(b"".join(WEIGHTS)), list(b"".join(SCALES))])
            self.assertEqual((case.cap, case.out_width, case.expect), (3, width, ("ok", [1, 2, 3])))

    def test_wrong_selection_descriptor_and_invalid_payload_fail(self):
        with mock.patch.object(expert, "read_row") as read:
            for bad in (replace(self.matrix, n=0), replace(self.matrix, k=63), replace(self.matrix, n=True),
                        replace(self.matrix, scale=self.matrix.weight)):
                self.assert_error("row_layout", linear.read_matrix, bad)
            read.assert_not_called()
        payload = linear.read_matrix(self.matrix)
        for bad in (replace(payload, weight=payload.weight[:-1]), replace(payload, scale=payload.scale[:-1]),
                    replace(payload, name=NAME.replace("w1", "w2")), replace(payload, n=True)):
            self.assert_error("matrix", linear.verify_matrix, bad, self.inputs, EXE)
        for bad in ({}, {"short": (0,)}, {"nan": (0x7fc0,) * 64}, {"inf": (0x7f80,) * 64}):
            self.assert_error("input", linear.verify_matrix, payload, bad, EXE)

    def test_late_short_read_and_io_propagate(self):
        original = os.pread
        for exception in (False, True):
            def fail_last(fd, length, offset):
                if offset == self.bases[1] + 4:
                    if exception:
                        raise OSError("late row failure")
                    return b"\x7f"
                return original(fd, length, offset)
            with mock.patch.object(expert.os, "pread", side_effect=fail_last):
                error = self.assert_error("io" if exception else "short_read", linear.read_matrix, self.matrix)
            self.assertIn("row=2", error.detail)
        with self.path.open("r+b") as fh:
            fh.truncate(self.bases[1] + 5)
        self.assert_error("short_read", linear.read_matrix, self.matrix)

    def test_independent_layout_detects_weight_scale_and_nibble_faults(self):
        pristine = linear.read_matrix(self.matrix)
        self.assert_payload(pristine)
        faults = [replace(pristine, weight=pristine.weight[32:] + pristine.weight[:32]),
                  replace(pristine, scale=pristine.scale[2:] + pristine.scale[:2]),
                  replace(pristine, weight=bytes((b >> 4) | ((b & 15) << 4) for b in pristine.weight))]
        for fault in faults:
            with self.assertRaises(AssertionError):
                self.assert_payload(fault)

    def test_once_per_input_oracle_and_only_full_production_calls(self):
        payload = linear.read_matrix(self.matrix)
        with mock.patch.object(linear, "act_quant", wraps=linear.act_quant) as quantize, \
                mock.patch.object(linear, "trace_expert_grouped_dot", wraps=linear.trace_expert_grouped_dot) as dot, \
                mock.patch.object(expert, "run_executor", wraps=expert.run_executor) as execute:
            records = linear.verify_matrix(payload, self.inputs, EXE)
        self.assertEqual(quantize.call_count, 2)
        self.assertEqual(dot.call_count, 6)
        self.assertEqual(execute.call_count, 4)
        self.assertEqual([call.args[1][0].op for call in execute.call_args_list],
                         ["expert_linear_fp32", "expert_linear_bf16"] * 2)
        self.assertTrue(all(len(call.args[1]) == 1 and call.args[1][0].cap == 3 for call in execute.call_args_list))
        for record in records:
            self.assertTrue(record["pass"])
            self.assertEqual(record["output_comparisons"], 6)
            self.assertEqual(len(record["fp32_bits"]), 3)
            self.assertEqual(len(set(record["fp32_bits"])), 3)
            self.assertEqual(record["fp32_bits"], record["expected_fp32_bits"])
            self.assertEqual(record["bf16_bits"], record["expected_bf16_bits"])

    def test_hand_calculated_first_and_last_outputs(self):
        payload = linear.read_matrix(self.matrix)
        inputs = {"anchor": ((0x3f80, 0x4000) + (0,) * 30) * 2}
        record = linear.verify_matrix(payload, inputs, EXE)[0]
        self.assertEqual(record["fp32_bits"], [0x41580000, 0xc0900000, 0xc23b0000])
        self.assertEqual(record["bf16_bits"], [0x4158, 0xc090, 0xc23b])

    def test_late_invalid_scale_is_oracle_failure_not_parity(self):
        payload = linear.read_matrix(self.matrix)
        with mock.patch.object(expert, "run_executor") as execute:
            error = self.assert_error("oracle", linear.verify_matrix,
                                      replace(payload, scale=payload.scale[:-1] + b"\xff"), self.inputs, EXE)
            execute.assert_not_called()
        for token in (NAME, "input=mixed", "row=2", "stage=compute"):
            self.assertIn(token, error.detail)

    def test_production_payload_faults_and_last_result_mutations_detected(self):
        payload = linear.read_matrix(self.matrix)
        self.assertTrue(all(r["pass"] for r in linear.verify_matrix(payload, self.inputs, EXE)))
        execute = expert.run_executor
        for mode in ("wrong_row", "wrong_scale", "nibbles", "fp32_result", "bf16_result"):
            def corrupt(exe, cases):
                case = cases[0]
                if mode == "wrong_row":
                    case = replace(case, weight=case.weight[32:] + case.weight[:32])
                elif mode == "wrong_scale":
                    case = replace(case, scale=case.scale[2:] + case.scale[:2])
                elif mode == "nibbles":
                    case = replace(case, weight=bytes((v >> 4) | ((v & 15) << 4) for v in case.weight))
                response = execute(exe, [case])
                if mode == case.kind + "_result":
                    response[0][1][-1] ^= 1
                return response
            with mock.patch.object(expert, "run_executor", side_effect=corrupt):
                error = self.assert_error("mismatch", linear.verify_matrix, payload, self.inputs, EXE)
            self.assertIn("input=mixed", error.detail)
            if mode.endswith("_result"):
                self.assertIn("row=2", error.detail)

    def test_nan_class_signed_zero_and_inf_comparison(self):
        payload = linear.read_matrix(self.matrix)
        payload = replace(payload, weight=b"\x77" * 96, scale=b"\xfe" * 6)
        inputs = {"nan": (0x4000,) * 32 + (0xc000,) * 32}
        pristine = linear.verify_matrix(payload, inputs, EXE)
        self.assertTrue(all(expert.is_nan(v, "fp32") for v in pristine[0]["fp32_bits"]))
        execute = expert.run_executor
        def nan_payload(exe, cases):
            response = execute(exe, cases)
            response[0][1][:] = [0x7fc12345 if cases[0].kind == "fp32" else 0x7fc1] * 3
            return response
        with mock.patch.object(expert, "run_executor", side_effect=nan_payload):
            self.assertTrue(linear.verify_matrix(payload, inputs, EXE)[0]["pass"])
        for name, bits in (("zero", (0,) * 64), ("inf", (0x4000,) * 64)):
            records = linear.verify_matrix(payload, {name: bits}, EXE)
            self.assertEqual(records[0]["fp32_bits"], [0 if name == "zero" else 0x7f800000] * 3)
            def flip_sign(exe, cases):
                response = execute(exe, cases)
                response[0][1][-1] ^= 1 << (31 if cases[0].kind == "fp32" else 15)
                return response
            with mock.patch.object(expert, "run_executor", side_effect=flip_sign):
                self.assert_error("mismatch", linear.verify_matrix, payload, {name: bits}, EXE)

    def test_missing_crashing_malformed_missing_last_and_rejecting_executor(self):
        payload = linear.read_matrix(self.matrix)
        self.assert_error("executor", linear.verify_matrix, payload, self.inputs, self.root / "absent")
        for mode, body in (("crash", "import os, signal; os.kill(os.getpid(), signal.SIGKILL)"),
                           ("malformed", "print('garbage')"), ("absent", "pass"),
                           ("missinglast", "print('ok 00000000 00000000')"),
                           ("reject", "print('err invalid_argument changed=0')")):
            path = self.root / mode
            path.write_text(f"#!{sys.executable}\nimport sys\nsys.stdin.read()\n{body}\n")
            path.chmod(0o700)
            self.assert_error("executor", linear.verify_matrix, payload, self.inputs, path)
        with mock.patch.object(expert.subprocess, "run", side_effect=subprocess.TimeoutExpired("exec", 1)) as run:
            self.assert_error("executor", linear.verify_matrix, payload, self.inputs, EXE)
            self.assertGreater(run.call_args.kwargs["timeout"], 0)
        with mock.patch.object(expert, "run_executor", return_value=[("ok", [0, 0])]):
            self.assert_error("protocol", linear.verify_matrix, payload, self.inputs, EXE)

    def test_full_checkpoint_report_reads_only_w1_all_n_and_full_k(self):
        checkpoint, reference = self.checkpoint()
        pread, calls = os.pread, []
        def capture(fd, length, offset):
            calls.append((os.readlink(f"/proc/self/fd/{fd}"), length, offset))
            return pread(fd, length, offset)
        with mock.patch.object(expert.os, "pread", side_effect=capture):
            report = linear.run(checkpoint, EXE, reference)
        self.assertEqual((report["plan"], report["pass"], report["environment_contract"]), ("P0019", True, "ENV-A"))
        self.assertEqual(len(report["matrices"]), 1)
        matrix = report["matrices"][0]
        self.assertEqual((matrix["name"], matrix["n"], matrix["k"]), (NAME, 96, 64))
        self.assertEqual([r["index"] for r in matrix["rows"]], list(range(96)))
        self.assertEqual(len(calls), 192)
        self.assertEqual(calls, [(r["shard"], r["length"], r["offset"]) for row in matrix["rows"] for r in row["reads"]])
        for row in matrix["rows"]:
            for read, data in zip(row["reads"], checkpoint_row(row["index"])):
                self.assertEqual((read["length"], read["sha256"]), (len(data), hashlib.sha256(data).hexdigest()))
                self.assertTrue(read["tensor"].startswith(NAME + "."))
        for part, label in enumerate(("weight", "scale_b")):
            raw = b"".join(checkpoint_row(i)[part] for i in range(96))
            self.assertEqual(matrix["payloads"][label], {"length": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
        for data in matrix["inputs"]:
            raw = b"".join(b.to_bytes(2, "little") for b in self.inputs[data["case"]])
            self.assertEqual(data["sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual([r["case"] for r in matrix["cases"]], ["mixed", "group_ramp"])
        self.assertTrue(all(len(r["fp32_bits"]) == len(r["bf16_bits"]) == 96 for r in matrix["cases"]))
        self.assertEqual(report["comparisons"], {"outputs": 384, "total": 384})
        self.assertEqual(report["exe"]["sha256"], hashlib.sha256(Path(EXE).read_bytes()).hexdigest())
        self.assertEqual(report["version"], (ROOT / "VERSION").read_text().strip())
        self.assertEqual(len(report["git_sha"]), 40)
        self.assertIsInstance(report["dirty"], bool)
        self.assertEqual(report["revision"], "ckpt")
        json.dumps(report)

    def test_cli_success_and_checkpoint_unchanged(self):
        checkpoint, reference = self.checkpoint()
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(checkpoint).iterdir() if p.is_file()}
        out = self.root / "report.json"
        proc = self.cli(checkpoint, "--exe", EXE, "--reference", reference, "--json", out)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        report = json.loads(out.read_text())
        self.assertTrue(report["pass"])
        self.assertEqual(report["command"][-2:], ["--json", str(out)])
        self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before})

    def test_cli_missing_resources_invalid_args_and_protected_reports(self):
        checkpoint, reference = self.checkpoint()
        out = self.root / "existing.json"
        out.write_text("keep")
        link = self.root / "checkpoint-link"
        link.symlink_to(checkpoint, target_is_directory=True)
        dangling = self.root / "dangling.json"
        dangling.symlink_to(self.root / "nonexistent-target")
        args_list = [(self.root / "absent", "--exe", EXE), (checkpoint,),
                     (checkpoint, "--exe", EXE, "--unknown"),
                     (checkpoint, "--exe", self.root / "no-exe", "--reference", reference)]
        args_list += [(checkpoint, "--exe", EXE, "--reference", reference, "--json", target)
                      for target in (out, Path(checkpoint) / "report.json", link / "report.json", dangling)]
        for args in args_list:
            proc = self.cli(*args)
            self.assertNotEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertNotIn("RESULT PASS", proc.stdout)
        self.assertEqual(out.read_text(), "keep")
        self.assertFalse((Path(checkpoint) / "report.json").exists())
        self.assertFalse((self.root / "nonexistent-target").exists())

    def test_cli_failures_and_skipped_tests_are_never_success(self):
        for code in ("short_read", "io", "executor", "oracle", "mismatch", "protocol", "argument"):
            with mock.patch.object(linear, "run", side_effect=expert.ConsumptionError(code, "injected failure")):
                with contextlib.redirect_stderr(io.StringIO()):
                    self.assertNotEqual(linear.main([str(self.root), "--exe", str(EXE)]), 0)
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
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(LinearTests)
    return suite_exit(unittest.TextTestRunner(verbosity=2).run(suite))


if __name__ == "__main__":
    sys.exit(main())
