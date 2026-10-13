"""ckpt_inventory.py 的合成数据测试（P0016 / D0012）：结构、index、预期比对、配置一致与扫描计数。

真实 checkpoint 的核对不在此处（见 docs/BUILD.md §7）；这里的合成 checkpoint 只验证检出机制，
生成规则的正确性由小配置的手算计数与 shape 抽查独立核对。

运行：python3 tests/tools/test_ckpt_inventory.py
"""

import collections
import importlib.util
import json
import math
import os
import random
import struct
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOOL = os.path.join(REPO_ROOT, "tools", "ckpt_inventory.py")

_spec = importlib.util.spec_from_file_location("ckpt_inventory", TOOL)
inv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(inv)

TINY = {
    "vocab_size": 40, "dim": 64, "moe_inter_dim": 64, "n_layers": 4, "n_mtp_layers": 2,
    "n_heads": 2, "n_routed_experts": 3, "n_shared_experts": 1, "n_activated_experts": 2,
    "q_lora_rank": 48, "head_dim": 32, "rope_head_dim": 16, "o_groups": 2, "o_lora_rank": 16,
    "kv_source_layers": [1, 3], "index_source_layers": [1, 2, 3],
    "compress_ratios": [0, 2, 1, 1, 0, 0], "index_n_heads": 2, "index_head_dim": 16,
    "hc_mult": 2, "engram_layer_ids": [0, 2], "engram_num_embeddings": [5, 7],
    "engram_max_ngram_size": 3, "engram_n_heads": 2, "engram_head_dim": 32,
    "dspark_block_size": 3, "dspark_target_layer_ids": [2, 3], "dspark_markov_rank": 16,
    "dspark_n_routed_experts": 2, "vision_n_layers": 1, "vision_dim": 32, "vision_inter_dim": 48,
    "vision_patch_size": 2, "vision_downsample_ratio": 2, "dtype": "fp8", "expert_dtype": "fp4",
}
TINY_TENSOR_COUNT = 334


def write_raw(path, header_bytes, payload=b"", length=None):
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(header_bytes) if length is None else length))
        fh.write(header_bytes)
        fh.write(payload)


def write_shard(path, tensors, metadata=None, pad=0):
    header, offset = {}, 0
    if metadata is not None:
        header["__metadata__"] = metadata
    for name, dtype, shape, data in tensors:
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
    write_raw(path, json.dumps(header).encode() + b" " * pad, b"".join(t[3] for t in tensors))


def patterned(name, dtype, nbytes):
    """Seeded bytes whose leading elements force the special codes of each dtype."""
    data = bytearray(random.Random(name).randbytes(nbytes))
    specials = {
        "F32": [0x7FC00001, 0x7F800000, 0xFF800000, 0x80000000, 0x00000000, 0x00000001],
        "BF16": [0x7FC1, 0x7F80, 0xFF80, 0x8000, 0x0000, 0x0001],
    }.get(dtype, [])
    width = {"F32": 4, "BF16": 2}.get(dtype, 1)
    for i, value in enumerate(specials):
        if (i + 1) * width <= nbytes:
            data[i * width:(i + 1) * width] = value.to_bytes(width, "little")
    if dtype in ("F8_E4M3", "F8_E8M0", "I8") and nbytes >= 4:
        data[0:4] = bytes([0xFF, 0x7F, 0x80, 0x00])
    return bytes(data)


def build_checkpoint(root, cfg=TINY, n_shards=2, mutate=None, data=patterned):
    ref = os.path.join(root, "reference")
    ckpt = os.path.join(root, "ckpt")
    for base in (ref, ckpt):
        os.makedirs(os.path.join(base, "inference"), exist_ok=True)
        with open(os.path.join(base, "config.json"), "w") as fh:
            json.dump({"tiny": True}, fh)
        with open(os.path.join(base, "inference", "config.json"), "w") as fh:
            json.dump(cfg, fh)
    tensors = []
    for name, exp in sorted(inv.expected_tensors(cfg).items()):
        dtype = exp.storage[0]
        nbytes = math.prod(exp.shape) * inv.DTYPE_BITS[dtype] // 8
        tensors.append([name, dtype, exp.shape, data(name, dtype, nbytes)])
    if mutate is not None:
        tensors = mutate(tensors)
    weight_map, total = {}, 0
    for k in range(n_shards):
        fname = f"model-{k + 1:05d}-of-{n_shards:05d}.safetensors"
        part = tensors[k::n_shards]
        write_shard(os.path.join(ckpt, fname), part)
        for name, _, _, payload in part:
            weight_map[name] = fname
            total += len(payload)
    with open(os.path.join(ckpt, inv.INDEX_NAME), "w") as fh:
        json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, fh)
    return ckpt, ref


def run_tool(ckpt, ref, *extra):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
        out = fh.name
    try:
        proc = subprocess.run(
            [sys.executable, TOOL, ckpt, "--reference", ref, "--json", out, *extra],
            capture_output=True, text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        report = None
        if os.path.getsize(out):
            with open(out) as fh:
                report = json.load(fh)
        return proc.returncode, report, proc
    finally:
        os.unlink(out)


def error_codes(report):
    return collections.Counter(e["code"] for e in report["errors"])


class TempDirCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def path(self, name="x.safetensors"):
        return os.path.join(self.root, name)

    def assert_code(self, code, header_bytes, payload=b"", length=None):
        write_raw(self.path(), header_bytes, payload, length)
        with self.assertRaises(inv.CheckError) as ctx:
            inv.read_header(self.path())
        self.assertEqual(ctx.exception.code, code, ctx.exception.detail)


def entry(dtype, shape, begin, end, **extra):
    return {"dtype": dtype, "shape": shape, "data_offsets": [begin, end], **extra}


class StructureTests(TempDirCase):
    def test_valid_shard_with_padding_metadata_empty_and_scalar(self):
        header = {
            "b": entry("F32", [], 4, 8),
            "__metadata__": {"format": "pt"},
            "a": entry("BF16", [2], 0, 4),
            "empty": entry("F8_E4M3", [0, 3], 8, 8),
        }
        write_raw(self.path(), json.dumps(header).encode() + b"   ", bytes(8))
        shard = inv.read_header(self.path())
        self.assertEqual([e.name for e in shard.entries], ["a", "b", "empty"])
        self.assertEqual(shard.data_start, 8 + len(json.dumps(header)) + 3)

    def test_file_too_small(self):
        with open(self.path(), "wb") as fh:
            fh.write(b"\x01\x00\x00")
        with self.assertRaises(inv.CheckError) as ctx:
            inv.read_header(self.path())
        self.assertEqual(ctx.exception.code, "file_too_small")

    def test_header_too_large(self):
        self.assert_code("header_too_large", b"{}", length=inv.MAX_HEADER_BYTES + 1)

    def test_header_truncated(self):
        self.assert_code("header_truncated", b"{}", length=50)

    def test_header_utf8(self):
        self.assert_code("header_utf8", b'{"\xff": 1}')

    def test_header_start(self):
        self.assert_code("header_start", b' {}')

    def test_header_padding_other_than_space(self):
        self.assert_code("header_padding", b"{}\n")
        self.assert_code("header_padding", b"{} x")

    def test_header_json(self):
        self.assert_code("header_json", b'{"a": }')
        self.assert_code("header_json", b'{"a": NaN}')

    def test_duplicate_key(self):
        one = json.dumps(entry("U8", [1], 0, 1))
        self.assert_code("duplicate_key", f'{{"a": {one}, "a": {one}}}'.encode(), b"\x00")

    def test_metadata_type(self):
        self.assert_code("metadata_type", json.dumps({"__metadata__": {"k": 1}}).encode())
        self.assert_code("metadata_type", json.dumps({"__metadata__": ["k"]}).encode())

    def test_entry_type(self):
        bad = [
            {"dtype": "U8", "shape": [1]},
            entry("U8", [1], 0, 1, extra=1),
            entry("Q4", [1], 0, 1),
            entry("U8", [True], 0, 1),
            entry("U8", [-1], 0, 1),
            entry("U8", [1.0], 0, 1),
            {"dtype": "U8", "shape": [1], "data_offsets": [0]},
            "not an object",
        ]
        for info in bad:
            with self.subTest(info=info):
                self.assert_code("entry_type", json.dumps({"a": info}).encode(), b"\x00")

    def test_offset_gap_hole_overlap_and_reversed(self):
        cases = [
            ({"a": entry("U8", [1], 1, 2)}, 2),
            ({"a": entry("U8", [2], 0, 2), "b": entry("U8", [1], 1, 2)}, 2),
            ({"a": entry("U8", [1], 0, 1), "b": entry("U8", [1], 2, 3)}, 3),
            ({"a": entry("U8", [0], 1, 0)}, 1),
        ]
        for header, size in cases:
            with self.subTest(header=header):
                self.assert_code("offset_gap", json.dumps(header).encode(), bytes(size))

    def test_misaligned_sub_byte(self):
        self.assert_code("misaligned", json.dumps({"a": entry("F4", [3], 0, 2)}).encode(), bytes(2))

    def test_size_mismatch(self):
        self.assert_code("size_mismatch", json.dumps({"a": entry("F32", [2], 0, 4)}).encode(), bytes(4))

    def test_incomplete_buffer(self):
        self.assert_code("incomplete_buffer", json.dumps({"a": entry("U8", [1], 0, 1)}).encode(), bytes(2))


class GeneratorTests(unittest.TestCase):
    """Hand-derived expectations for the tiny config, independent of the generator's code."""

    @classmethod
    def setUpClass(cls):
        cls.x = inv.expected_tensors(TINY)

    def check(self, name, shape, storage, runtime):
        self.assertIn(name, self.x)
        e = self.x[name]
        self.assertEqual((e.shape, e.storage, e.runtime), (shape, storage, runtime), name)

    def test_count(self):
        self.assertEqual(len(self.x), TINY_TENSOR_COUNT)

    def test_shapes_and_dtypes(self):
        self.check("layers.0.attn.wq_a.weight", (48, 64), ("F8_E4M3",), "float8_e4m3fn")
        self.check("layers.0.attn.wq_a.scale", (2, 2), ("F8_E8M0",), "float8_e8m0fnu")
        self.check("layers.0.attn.wo_a.weight", (32, 32), ("F8_E4M3",), "bfloat16")
        self.check("layers.0.attn.wo_a.scale", (1, 1), ("F8_E8M0",), None)
        self.check("layers.1.attn.compressor.wgate.weight", (32, 64), ("BF16",), "float32")
        self.check("layers.3.attn.compressor.wkv.weight", (32, 64), ("BF16",), "bfloat16")
        self.check("layers.2.attn.indexer.wq_b.weight", (32, 48), ("F8_E4M3",), "float8_e4m3fn")
        self.check("layers.1.attn.indexer.wk.weight", (16, 32), ("BF16",), "bfloat16")
        self.check("layers.0.ffn.experts.2.w2.weight", (64, 32), ("I8",), "float4_e2m1fn_x2")
        self.check("layers.0.ffn.experts.2.w2.scale", (64, 2), ("F8_E8M0",), "float8_e8m0fnu")
        self.check("layers.0.attn.attn_sink", (2,), ("F32", "BF16"), "float32")
        self.check("layers.2.engram.wkv.weight", (192, 128), ("F8_E4M3",), "float8_e4m3fn")
        self.check("layers.2.engram.embed.weight", (7, 32), ("F8_E4M3",), "float8_e4m3fn")
        self.check("layers.2.engram.embed.scale", (7, 1), ("F8_E8M0",), "float8_e8m0fnu")
        self.check("layers.0.engram.q_weight", (2, 64), ("BF16",), "bfloat16")
        self.check("layers.0.hc_attn_fn", (8, 128), ("F32", "BF16"), "float32")
        self.check("head.weight", (40, 64), ("BF16",), "float32")
        self.check("mtp.1.ffn.gate.weight", (2, 64), ("BF16",), "bfloat16")
        self.check("mtp.0.main_proj.weight", (64, 128), ("F8_E4M3",), "float8_e4m3fn")
        self.check("mtp.1.confidence_head.proj.weight", (1, 80), ("BF16",), "float32")
        self.check("mtp.1.markov_head.head.weight", (40, 16), ("BF16",), "float32")
        self.check("vision.patch_embed.proj.weight", (32, 12), ("BF16",), "bfloat16")
        self.check("vision.blocks.0.norm1.weight", (32,), ("F32", "BF16"), "float32")
        self.check("vision.blocks.0.mlp.w1.weight", (96, 32), ("BF16",), "bfloat16")
        self.check("aligner.w1.weight", (64, 128), ("BF16",), "bfloat16")

    def test_absent_tensors(self):
        for name in (
            "layers.0.attn.compressor.norm.weight", "layers.3.attn.compressor.wgate.weight",
            "layers.2.attn.indexer.wk.weight", "layers.0.attn.indexer.wq_b.weight",
            "layers.1.engram.wkv.weight", "mtp.0.attn.compressor.norm.weight",
            "mtp.0.attn.indexer.wq_b.weight", "mtp.0.engram.q_weight", "mtp.1.ffn.experts.2.w1.weight",
            "mtp.0.embed.weight", "mtp.0.head.weight", "mtp.0.norm.weight", "mtp.1.main_proj.weight",
            "vision.blocks.0.mlp.w1.bias",
        ):
            self.assertNotIn(name, self.x)

    def test_config_must_carry_every_used_key(self):
        cfg = dict(TINY)
        del cfg["o_groups"]
        with self.assertRaises(inv.UsageError):
            inv.expected_tensors(cfg)


class CheckpointTests(TempDirCase):
    def test_synthetic_checkpoint_passes(self):
        ckpt, ref = build_checkpoint(self.root)
        code, report, proc = run_tool(ckpt, ref)
        self.assertEqual(code, 0, proc.stdout + proc.stderr)
        self.assertEqual((report["tensors"], report["expected"], report["errors"]), (TINY_TENSOR_COUNT,) * 2 + ([],))
        self.assertIn("RESULT PASS", proc.stdout)

    def expect_failure(self, codes, **kwargs):
        ckpt, ref = build_checkpoint(self.root, **kwargs)
        return self.expect_tool_failure(ckpt, ref, codes)

    def expect_tool_failure(self, ckpt, ref, codes):
        code, report, proc = run_tool(ckpt, ref)
        self.assertEqual(code, 1, proc.stdout + proc.stderr)
        self.assertEqual(set(error_codes(report)), set(codes), report["errors"])
        return report

    def test_missing_and_unexpected(self):
        def drop_and_add(ts):
            ts = [t for t in ts if t[0] != "layers.0.attn.q_norm.weight"]
            return ts + [["layers.0.attn.extra.weight", "BF16", (2,), bytes(4)]]
        report = self.expect_failure({"missing", "unexpected"}, mutate=drop_and_add)
        details = " ".join(e["detail"] for e in report["errors"])
        self.assertIn("layers.0.attn.q_norm.weight", details)
        self.assertIn("layers.0.attn.extra.weight", details)

    def retype(self, name, dtype):
        def mutate(ts):
            for t in ts:
                if t[0] == name:
                    t[1] = dtype
                    t[3] = bytes(math.prod(t[2]) * inv.DTYPE_BITS[dtype] // 8)
            return ts
        return mutate

    def test_dtype_mismatch(self):
        for name, dtype in (
            ("layers.0.attn.wq_a.weight", "BF16"),
            ("head.weight", "F32"),
            ("embed.weight", "F32"),
            ("layers.0.ffn.experts.0.w1.weight", "U8"),
        ):
            with self.subTest(name=name):
                report = self.expect_failure({"dtype"}, mutate=self.retype(name, dtype))
                self.assertIn(name, report["errors"][0]["detail"])

    def test_silent_float32_storage_is_reported_not_rejected(self):
        ckpt, ref = build_checkpoint(self.root, mutate=self.retype("vision.norm.weight", "BF16"))
        code, report, proc = run_tool(ckpt, ref)
        self.assertEqual(code, 0, proc.stdout)
        rows = [r for r in report["storage_vs_runtime"] if r["family"] == "vision.norm.weight"]
        self.assertEqual(rows, [{"family": "vision.norm.weight", "runtime": "float32", "storage": "BF16",
                                 "basis": "source silent", "count": 1}])

    def test_shape_mismatch(self):
        def reshape(ts):
            for t in ts:
                if t[0] == "layers.0.attn.kv_norm.weight":
                    t[2] = (16, 2)
            return ts
        self.expect_failure({"shape"}, mutate=reshape)

    def edit_index(self, edit):
        ckpt, ref = build_checkpoint(self.root)
        path = os.path.join(ckpt, inv.INDEX_NAME)
        with open(path) as fh:
            index = json.load(fh)
        edit(index)
        with open(path, "w") as fh:
            json.dump(index, fh)
        return ckpt, ref

    def test_index_errors(self):
        cases = {
            "index_missing_tensor": lambda ix: ix["weight_map"].pop("norm.weight"),
            "index_extra_tensor": lambda ix: ix["weight_map"].update(
                {"ghost": "model-00001-of-00002.safetensors"}),
            "index_wrong_shard": lambda ix: ix["weight_map"].update(
                {"norm.weight": next(f for f in set(ix["weight_map"].values())
                                     if f != ix["weight_map"]["norm.weight"])}),
            "index_unknown_shard": lambda ix: ix["weight_map"].update({"norm.weight": "nowhere.safetensors"}),
            "index_total_size": lambda ix: ix["metadata"].update({"total_size": ix["metadata"]["total_size"] + 1}),
        }
        for code, edit in cases.items():
            with self.subTest(code=code):
                ckpt, ref = self.edit_index(edit)
                self.expect_tool_failure(ckpt, ref, {code})
                self.tearDown()
                self.setUp()

    def test_index_json(self):
        ckpt, ref = build_checkpoint(self.root)
        with open(os.path.join(ckpt, inv.INDEX_NAME), "w") as fh:
            fh.write('{"weight_map": {}}')
        self.expect_tool_failure(ckpt, ref, {"index_json"})

    def test_duplicate_tensor_across_shards(self):
        ckpt, ref = build_checkpoint(self.root, n_shards=1)
        first = inv.read_header(os.path.join(ckpt, "model-00001-of-00001.safetensors")).entries[0]
        write_shard(os.path.join(ckpt, "model-extra.safetensors"),
                    [[first.name, first.dtype, first.shape, bytes(first.end - first.begin)]])
        report = self.expect_tool_failure(ckpt, ref, {"duplicate_tensor", "index_total_size"})
        self.assertIn(first.name, report["errors"][0]["detail"])

    def test_structure_error_is_reported_per_shard_and_blocks_scan(self):
        ckpt, ref = build_checkpoint(self.root)
        with open(os.path.join(ckpt, "model-00002-of-00002.safetensors"), "ab") as fh:
            fh.write(b"\x00")
        code, report, _ = run_tool(ckpt, ref, "--scan")
        self.assertEqual(code, 1)
        self.assertIn("incomplete_buffer", error_codes(report))
        self.assertIn("scan_skipped", error_codes(report))
        self.assertNotIn("scan", report)

    def test_config_identity(self):
        ckpt, ref = build_checkpoint(self.root)
        with open(os.path.join(ckpt, "config.json"), "w") as fh:
            fh.write('{"tiny": false}')
        os.unlink(os.path.join(ckpt, "inference", "config.json"))
        self.expect_tool_failure(ckpt, ref, {"config_mismatch", "config_missing"})

    def test_usage_errors_exit_2(self):
        ckpt, ref = build_checkpoint(self.root)
        cases = [
            (os.path.join(self.root, "absent"), ref, ()),
            (ckpt, os.path.join(self.root, "absent"), ()),
            (ckpt, ref, ("--chunk-bytes", "12")),
            (ckpt, ref, ("--jobs", "0")),
        ]
        for args in cases:
            with self.subTest(args=args):
                code, _, proc = run_tool(args[0], args[1], *args[2])
                self.assertEqual(code, 2, proc.stdout + proc.stderr)
        with open(os.path.join(ref, "inference", "config.json"), "w") as fh:
            json.dump({k: v for k, v in TINY.items() if k != "dim"}, fh)
        code, _, _ = run_tool(ckpt, ref)
        self.assertEqual(code, 2)


def naive_family(name):
    return ".".join("N" if p.isdigit() else p for p in name.split("."))


def naive_scan(ckpt):
    """Element-by-element counts with explicit bit-field definitions, independent of the tool."""
    expected = inv.expected_tensors(TINY)
    fams = collections.defaultdict(collections.Counter)
    for fname in sorted(os.listdir(ckpt)):
        if not fname.endswith(".safetensors"):
            continue
        with open(os.path.join(ckpt, fname), "rb") as fh:
            blob = fh.read()
        n = int.from_bytes(blob[:8], "little")
        header = json.loads(blob[8:8 + n])
        for name, info in header.items():
            begin, end = info["data_offsets"]
            data = blob[8 + n + begin:8 + n + end]
            dtype = info["dtype"]
            c = fams[(naive_family(name), dtype)]
            if dtype == "I8" and expected[name].runtime == "float4_e2m1fn_x2":
                for b in data:
                    c[f"code_{b & 15}"] += 1
                    c[f"code_{b >> 4}"] += 1
            elif dtype == "F8_E4M3":
                for b in data:
                    c["nan" if b & 0x7F == 0x7F else ("-0" if b == 0x80 else "+0") if b & 0x7F == 0 else "finite"] += 1
            elif dtype == "F8_E8M0":
                for b in data:
                    c["nan" if b == 0xFF else f"e8_{b}"] += 1
            elif dtype in ("BF16", "F32"):
                width, mant_bits = (2, 7) if dtype == "BF16" else (4, 23)
                for i in range(0, len(data), width):
                    v = int.from_bytes(data[i:i + width], "little")
                    sign = v >> (8 * width - 1)
                    exp = (v >> mant_bits) & 0xFF
                    mant = v & ((1 << mant_bits) - 1)
                    if exp == 0xFF:
                        c["nan" if mant else ("-inf" if sign else "+inf")] += 1
                    elif exp == 0 and mant == 0:
                        c["-0" if sign else "+0"] += 1
                    else:
                        c["finite"] += 1
    return fams


def tool_counts(report):
    out = {}
    for s in report["scan"]["families"]:
        c = collections.Counter()
        if s["kind"] == "e2m1x2":
            c.update({f"code_{k}": v for k, v in s["codes"].items() if v})
            assert s["neg_zero"] == s["codes"]["8"] and s["pos_zero"] == s["codes"]["0"]
        elif s["kind"] in ("e4m3fn", "bf16", "f32"):
            c.update(s["classes"])
        elif s["kind"] == "e8m0":
            c["nan"] += s["nan"]
            c.update({f"e8_{k}": v for k, v in s["codes"].items() if int(k) != 255})
        out[(s["family"], s["dtype"])] = c
    return out


class ScanTests(TempDirCase):
    def test_counts_match_naive_scan_under_any_split(self):
        ckpt, ref = build_checkpoint(self.root)
        naive = naive_scan(ckpt)
        results = []
        for extra in ((), ("--jobs", "2", "--chunk-bytes", "8", "--task-bytes", "24"), ("--jobs", "1")):
            with self.subTest(extra=extra):
                code, report, proc = run_tool(ckpt, ref, "--scan", *extra)
                self.assertEqual(code, 0, proc.stdout + proc.stderr)
                counts = tool_counts(report)
                self.assertEqual(set(counts), set(naive))
                for family, c in naive.items():
                    self.assertEqual(+counts[family], +c, family)
                results.append(report["scan"])
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[0], results[2])

    def test_specials_are_present_and_aggregated(self):
        ckpt, ref = build_checkpoint(self.root)
        code, report, _ = run_tool(ckpt, ref, "--scan")
        self.assertEqual(code, 0)
        findings = report["scan"]["findings"]
        for kind in ("bf16", "f32"):
            for cls in ("nan", "+inf", "-inf", "-0", "+0", "finite"):
                self.assertGreater(findings[kind].get(cls, 0), 0, (kind, cls))
        self.assertGreater(findings["e8m0"]["nan"], 0)
        self.assertGreater(findings["e4m3fn"]["nan"], 0)
        self.assertGreater(findings["e2m1x2"]["neg_zero"], 0)
        self.assertEqual(findings["e8m0"]["code_max"], 254)


if __name__ == "__main__":
    unittest.main()
