#!/usr/bin/env python3
"""P0016 / D0012: read-only inventory of the official DeepSeek-V4.1-Flash checkpoint.

Checks, in order:
1. config identity: the snapshot's config.json and inference/config.json equal the read-only
   mirror under third_party/reference/ byte for byte;
2. safetensors structure of every shard, per huggingface/safetensors@e246a256 (README "Format";
   safetensors/src/tensor.rs:10,390-425,540,610-647,794-875);
3. index consistency: model.safetensors.index.json against the shard headers;
4. expectation: names, shapes and dtypes generated only from the mirror's inference/config.json
   and the construction rules of the official inference code, never from the checkpoint;
5. with --scan, raw-code histograms of every tensor; codes are classified through ref/formats,
   so no format constant lives in this tool.

Exit status: 0 when every check passes, 1 when a check fails, 2 on usage or I/O errors.

Usage:
    python3 tools/ckpt_inventory.py CHECKPOINT_DIR [--scan] [--jobs N] [--json OUT] [--inventory OUT]
"""

import argparse
import collections
import concurrent.futures
import json
import math
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from ref import formats  # noqa: E402

DEFAULT_REFERENCE = os.path.join(REPO_ROOT, "third_party", "reference")
INDEX_NAME = "model.safetensors.index.json"
CONFIG_FILES = ("config.json", os.path.join("inference", "config.json"))
MAX_HEADER_BYTES = 100_000_000
DTYPE_BITS = {
    "BOOL": 8, "F4": 4, "F6_E2M3": 6, "F6_E3M2": 6, "U8": 8, "I8": 8,
    "F8_E5M2": 8, "F8_E4M3": 8, "F8_E8M0": 8, "F8_E4M3FNUZ": 8, "F8_E5M2FNUZ": 8,
    "I16": 16, "U16": 16, "F16": 16, "BF16": 16, "I32": 32, "U32": 32, "F32": 32,
    "C64": 64, "F64": 64, "I64": 64, "U64": 64,
}
FP8_BLOCK = 32
FP4_BLOCK = 32
STORAGE_OF_RUNTIME = {
    "float8_e4m3fn": "F8_E4M3",
    "float8_e8m0fnu": "F8_E8M0",
    "bfloat16": "BF16",
    "float32": "F32",
}
SCAN_KIND_OF_STORAGE = {"F8_E4M3": "e4m3fn", "F8_E8M0": "e8m0", "BF16": "bf16", "F32": "f32"}
CONFIG_KEYS = (
    "vocab_size", "dim", "moe_inter_dim", "n_layers", "n_mtp_layers", "n_heads",
    "n_routed_experts", "n_shared_experts", "q_lora_rank", "head_dim", "o_groups",
    "o_lora_rank", "kv_source_layers", "index_source_layers", "compress_ratios",
    "index_n_heads", "index_head_dim", "hc_mult", "engram_layer_ids", "engram_num_embeddings",
    "engram_max_ngram_size", "engram_n_heads", "engram_head_dim", "dspark_block_size",
    "dspark_target_layer_ids", "dspark_markov_rank", "dspark_n_routed_experts",
    "vision_n_layers", "vision_dim", "vision_inter_dim", "vision_patch_size",
    "vision_downsample_ratio", "dtype", "expert_dtype",
)


class CheckError(Exception):
    """A failed check with a stable code that tests and reports key on."""

    def __init__(self, code, detail):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


class UsageError(Exception):
    pass


Entry = collections.namedtuple("Entry", "name dtype shape begin end")
Shard = collections.namedtuple("Shard", "path data_start entries")
Expected = collections.namedtuple("Expected", "name shape storage runtime source tag")


def _reject_duplicates(pairs):
    keys = [key for key, _ in pairs]
    if len(keys) != len(set(keys)):
        dup = sorted(key for key, count in collections.Counter(keys).items() if count > 1)
        raise CheckError("duplicate_key", f"duplicate JSON keys {dup[:3]}")
    return dict(pairs)


def _reject_constant(token):
    raise CheckError("header_json", f"non-standard JSON constant {token}")


def _is_count(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def read_header(path):
    """Parses and validates one safetensors shard; raises CheckError on the first violation.

    Rules follow the pinned upstream: little-endian u64 header length bounded by 100 MB, UTF-8
    JSON starting with '{' and padded only with 0x20, no duplicate keys, string-only
    __metadata__, tensors sorted by offset (tensor.rs:540) and contiguous from zero, byte length
    equal to elements times dtype bits with no partial byte (tensor.rs:610-647), and the buffer
    ending exactly at the file end (tensor.rs:420). Every tensor entry must carry exactly
    dtype, shape and data_offsets.
    """
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        head = fh.read(8)
        if len(head) < 8:
            raise CheckError("file_too_small", f"{size} bytes")
        n = int.from_bytes(head, "little")
        if n > MAX_HEADER_BYTES:
            raise CheckError("header_too_large", f"{n} bytes")
        if 8 + n > size:
            raise CheckError("header_truncated", f"header {n} bytes, file {size} bytes")
        raw = fh.read(n)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CheckError("header_utf8", str(exc)) from None
    if not raw.startswith(b"{"):
        raise CheckError("header_start", "header does not begin with '{'")
    if not raw.rstrip(b" ").endswith(b"}"):
        raise CheckError("header_padding", "header is not a JSON object padded only with 0x20")
    try:
        obj = json.loads(text, object_pairs_hook=_reject_duplicates, parse_constant=_reject_constant)
    except json.JSONDecodeError as exc:
        raise CheckError("header_json", str(exc)) from None
    metadata = obj.pop("__metadata__", None)
    if metadata is not None and not (
        isinstance(metadata, dict) and all(isinstance(v, str) for v in metadata.values())
    ):
        raise CheckError("metadata_type", "__metadata__ must map strings to strings")
    entries = []
    for name, info in obj.items():
        if not isinstance(info, dict) or set(info) != {"dtype", "shape", "data_offsets"}:
            raise CheckError("entry_type", f"{name}: entry must hold dtype, shape, data_offsets")
        dtype, shape, offsets = info["dtype"], info["shape"], info["data_offsets"]
        if dtype not in DTYPE_BITS:
            raise CheckError("entry_type", f"{name}: unknown dtype {dtype!r}")
        if not isinstance(shape, list) or not all(_is_count(d) for d in shape):
            raise CheckError("entry_type", f"{name}: shape must be a list of nonnegative integers")
        if not (isinstance(offsets, list) and len(offsets) == 2 and all(_is_count(o) for o in offsets)):
            raise CheckError("entry_type", f"{name}: data_offsets must be two nonnegative integers")
        entries.append(Entry(name, dtype, tuple(shape), offsets[0], offsets[1]))
    entries.sort(key=lambda e: (e.begin, e.end))
    cursor = 0
    for entry in entries:
        if entry.begin != cursor or entry.end < entry.begin:
            raise CheckError("offset_gap", f"{entry.name}: [{entry.begin}, {entry.end}) after {cursor}")
        cursor = entry.end
        bits = math.prod(entry.shape) * DTYPE_BITS[entry.dtype]
        if bits % 8:
            raise CheckError("misaligned", f"{entry.name}: {bits} bits")
        if entry.end - entry.begin != bits // 8:
            raise CheckError("size_mismatch", f"{entry.name}: {entry.end - entry.begin} != {bits // 8}")
    if 8 + n + cursor != size:
        raise CheckError("incomplete_buffer", f"buffer ends at {8 + n + cursor}, file is {size} bytes")
    return Shard(path, 8 + n, entries)


def check_index(index_path, shards):
    """Compares the index weight_map and total_size with the parsed shard headers."""
    errors = []
    try:
        with open(index_path, encoding="utf-8") as fh:
            index = json.load(fh, object_pairs_hook=_reject_duplicates, parse_constant=_reject_constant)
        weight_map = index["weight_map"]
        total_size = index["metadata"]["total_size"]
        if not isinstance(weight_map, dict) or not _is_count(total_size):
            raise TypeError
    except CheckError as exc:
        return [exc]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return [CheckError("index_json", f"{index_path}: {exc!r}")]
    located = {}
    for fname, shard in sorted(shards.items()):
        for entry in shard.entries:
            if entry.name in located:
                errors.append(CheckError("duplicate_tensor", f"{entry.name} in {located[entry.name]} and {fname}"))
            else:
                located[entry.name] = fname
    for name, fname in sorted(weight_map.items()):
        if fname not in shards:
            errors.append(CheckError("index_unknown_shard", f"{name} -> {fname}"))
        elif name not in located:
            errors.append(CheckError("index_extra_tensor", f"{name} -> {fname}"))
        elif located[name] != fname:
            errors.append(CheckError("index_wrong_shard", f"{name} -> {fname}, found in {located[name]}"))
    for name in sorted(set(located) - set(weight_map)):
        errors.append(CheckError("index_missing_tensor", f"{name} in {located[name]}"))
    payload = sum(e.end - e.begin for shard in shards.values() for e in shard.entries)
    if payload != total_size:
        errors.append(CheckError("index_total_size", f"metadata {total_size} != payload {payload}"))
    return errors


class _Expectation:
    """Collects expected checkpoint tensors; storage defaults to the runtime dtype."""

    def __init__(self, cfg):
        self.items = {}
        self.linear_default = "float8_e4m3fn" if cfg["dtype"] == "fp8" else "bfloat16"

    def add(self, name, shape, runtime, source, storage=None, tag=None):
        if storage is None:
            storage = ("F32", "BF16") if runtime == "float32" else (STORAGE_OF_RUNTIME[runtime],)
        elif isinstance(storage, str):
            storage = (storage,)
        if name in self.items:
            raise ValueError(f"expectation lists {name} twice")
        self.items[name] = Expected(name, tuple(shape), storage, runtime, source, tag)

    def linear(self, prefix, in_f, out_f, source, dtype=None, tag=None):
        runtime = dtype or self.linear_default
        if runtime == "float4_e2m1fn_x2":
            self.add(f"{prefix}.weight", (out_f, in_f // 2), runtime,
                     f"{source}; model.py:219-223; convert.py:22,180-181", storage="I8", tag=tag)
            self.add(f"{prefix}.scale", (out_f, in_f // FP4_BLOCK), "float8_e8m0fnu",
                     f"{source}; model.py:222", tag=tag)
        elif runtime == "float8_e4m3fn":
            self.add(f"{prefix}.weight", (out_f, in_f), runtime, f"{source}; model.py:225-226", tag=tag)
            self.add(f"{prefix}.scale", (-(-out_f // FP8_BLOCK), -(-in_f // FP8_BLOCK)),
                     "float8_e8m0fnu", f"{source}; model.py:227-233", tag=tag)
        else:
            self.add(f"{prefix}.weight", (out_f, in_f), runtime, f"{source}; model.py:234-235", tag=tag)

    def rmsnorm(self, prefix, dim, source, runtime="bfloat16"):
        self.add(f"{prefix}.weight", (dim,), runtime, source)

    def nn_linear(self, prefix, in_f, out_f, source, bias=True):
        self.add(f"{prefix}.weight", (out_f, in_f), "bfloat16", source)
        if bias:
            self.add(f"{prefix}.bias", (out_f,), "bfloat16", source)


def _attention(x, cfg, prefix, layer_id):
    dim, heads, head_dim = cfg["dim"], cfg["n_heads"], cfg["head_dim"]
    q_lora, groups, o_lora = cfg["q_lora_rank"], cfg["o_groups"], cfg["o_lora_rank"]
    x.add(f"{prefix}.attn_sink", (heads,), "float32", "model.py:639")
    x.linear(f"{prefix}.wq_a", dim, q_lora, "model.py:640")
    x.rmsnorm(f"{prefix}.q_norm", q_lora, "model.py:641,286")
    x.linear(f"{prefix}.wq_b", q_lora, heads * head_dim, "model.py:642")
    x.linear(f"{prefix}.wkv", dim, head_dim, "model.py:643")
    x.rmsnorm(f"{prefix}.kv_norm", head_dim, "model.py:644,286")
    wo_in, wo_out = heads * head_dim // groups, groups * o_lora
    x.add(f"{prefix}.wo_a.weight", (wo_out, wo_in), "bfloat16", "model.py:645-649; convert.py:157-173",
          storage="F8_E4M3")
    x.add(f"{prefix}.wo_a.scale", (-(-wo_out // FP8_BLOCK), -(-wo_in // FP8_BLOCK)), None,
          "convert.py:157-173", storage="F8_E8M0")
    x.linear(f"{prefix}.wo_b", groups * o_lora, dim, "model.py:650")
    is_backbone = layer_id < cfg["n_layers"]
    ratio = cfg["compress_ratios"][layer_id]
    if is_backbone and layer_id in cfg["kv_source_layers"]:
        x.rmsnorm(f"{prefix}.compressor.norm", head_dim, "model.py:443,286")
        if ratio > 1:
            x.add(f"{prefix}.compressor.wkv.weight", (head_dim, dim), "float32", "model.py:444-446",
                  storage="BF16")
            x.add(f"{prefix}.compressor.wgate.weight", (head_dim, dim), "float32", "model.py:444-448",
                  storage="BF16")
        else:
            x.linear(f"{prefix}.compressor.wkv", dim, head_dim, "model.py:444-446", dtype="bfloat16")
    if is_backbone and layer_id in cfg["index_source_layers"]:
        idx_heads, idx_dim = cfg["index_n_heads"], cfg["index_head_dim"]
        x.linear(f"{prefix}.indexer.wq_b", q_lora, idx_heads * idx_dim, "model.py:514")
        x.linear(f"{prefix}.indexer.weights_proj", dim, idx_heads, "model.py:515", dtype="bfloat16")
        if layer_id in cfg["kv_source_layers"]:
            x.linear(f"{prefix}.indexer.wk", head_dim, idx_dim, "model.py:500,517-518", dtype="bfloat16")
            x.rmsnorm(f"{prefix}.indexer.k_norm", idx_dim, "model.py:519,286")


def _moe(x, cfg, prefix, layer_id):
    if cfg["n_shared_experts"] != 1:
        raise UsageError("model.py:886 requires n_shared_experts == 1")
    dim, inter = cfg["dim"], cfg["moe_inter_dim"]
    n_routed = cfg["n_routed_experts"]
    if layer_id >= cfg["n_layers"]:
        n_routed = cfg["dspark_n_routed_experts"] or n_routed
    x.add(f"{prefix}.gate.weight", (n_routed, dim), "bfloat16", "model.py:143-149,805")
    x.add(f"{prefix}.gate.bias", (n_routed,), "float32", "model.py:806")
    if cfg["vision_n_layers"] > 0:
        x.add(f"{prefix}.gate.bias_vl", (n_routed,), "float32", "model.py:139-140,807")
    expert_dtype = "float4_e2m1fn_x2" if cfg["expert_dtype"] == "fp4" else None
    for e in range(n_routed):
        for name, in_f, out_f in (("w1", dim, inter), ("w2", inter, dim), ("w3", dim, inter)):
            x.linear(f"{prefix}.experts.{e}.{name}", in_f, out_f, "model.py:834-839,872-884",
                     dtype=expert_dtype, tag="routed_expert")
    for name, in_f, out_f in (("w1", dim, inter), ("w2", inter, dim), ("w3", dim, inter)):
        x.linear(f"{prefix}.shared_experts.{name}", in_f, out_f, "model.py:834-839,887")


def _block(x, cfg, prefix, layer_id, with_engram):
    _attention(x, cfg, f"{prefix}.attn", layer_id)
    _moe(x, cfg, f"{prefix}.ffn", layer_id)
    dim, hc = cfg["dim"], cfg["hc_mult"]
    if with_engram and layer_id in cfg["engram_layer_ids"]:
        rows = cfg["engram_num_embeddings"][cfg["engram_layer_ids"].index(layer_id)]
        head_dim = cfg["engram_head_dim"]
        x.add(f"{prefix}.engram.embed.weight", (rows, head_dim), "float8_e4m3fn", "model.py:303,309,343",
              tag="engram_table")
        x.add(f"{prefix}.engram.embed.scale", (rows, head_dim // FP8_BLOCK), "float8_e8m0fnu",
              "model.py:307,310", tag="engram_table")
        hash_cols = (cfg["engram_max_ngram_size"] - 1) * cfg["engram_n_heads"]
        x.linear(f"{prefix}.engram.wkv", hash_cols * head_dim, dim * (hc + 1), "model.py:344-345")
        x.add(f"{prefix}.engram.q_weight", (hc, dim), "bfloat16", "model.py:347; generate.py:118")
        x.add(f"{prefix}.engram.k_weight", (hc, dim), "bfloat16", "model.py:348; generate.py:118")
    x.rmsnorm(f"{prefix}.attn_norm", dim, "model.py:933,286")
    x.rmsnorm(f"{prefix}.ffn_norm", dim, "model.py:934,286")
    mix = (2 + hc) * hc
    for part in ("attn", "ffn"):
        x.add(f"{prefix}.hc_{part}_fn", (mix, hc * dim), "float32", "model.py:938-942")
        x.add(f"{prefix}.hc_{part}_base", (mix,), "float32", "model.py:943-944")
        x.add(f"{prefix}.hc_{part}_scale", (3,), "float32", "model.py:945-946")


def expected_tensors(cfg):
    """Every checkpoint tensor implied by the official construction code at world_size 1.

    Runtime dtypes follow the constructors (Linear storage default model.py:26,1191; torch
    default bfloat16, generate.py:118). Storage equals the runtime dtype unless the official
    source states otherwise: tensors the converter rewrites (convert.py:157-181) and float32
    parameters whose source comment names a bf16 checkpoint copy. Other float32 parameters have
    no stated storage, so float32 and bfloat16 are both accepted (generate.py:133 load_model
    casts on load) and the observed choice is reported. MTP embed/head are tied to the backbone
    and absent from the checkpoint (model.py:1212-1213; convert.py:108-110).
    """
    missing = [k for k in CONFIG_KEYS if k not in cfg]
    if missing:
        raise UsageError(f"inference config lacks {missing}")
    x = _Expectation(cfg)
    dim, vocab = cfg["dim"], cfg["vocab_size"]
    x.add("embed.weight", (vocab, dim), "bfloat16", "model.py:166,1201")
    for layer_id in range(cfg["n_layers"]):
        _block(x, cfg, f"layers.{layer_id}", layer_id, with_engram=True)
    x.rmsnorm("norm", dim, "model.py:1205,286")
    x.add("head.weight", (vocab, dim), "float32", "model.py:1005-1006", storage="BF16")
    if cfg["dspark_block_size"]:
        stages = cfg["n_mtp_layers"]
        rank = cfg["dspark_markov_rank"]
        for stage in range(stages):
            prefix = f"mtp.{stage}"
            _block(x, cfg, prefix, cfg["n_layers"] + stage, with_engram=False)
            if stage == 0:
                x.linear(f"{prefix}.main_proj", dim * len(cfg["dspark_target_layer_ids"]), dim,
                         "model.py:1111-1113")
                x.rmsnorm(f"{prefix}.main_norm", dim, "model.py:1114,286")
            if stage == stages - 1:
                x.rmsnorm(f"{prefix}.norm", dim, "model.py:1115-1116,286")
                x.add(f"{prefix}.markov_head.embed.weight", (vocab, rank), "bfloat16", "model.py:1080,166")
                x.add(f"{prefix}.markov_head.head.weight", (vocab, rank), "float32", "model.py:1081,1005-1006",
                      storage="BF16")
                x.add(f"{prefix}.confidence_head.proj.weight", (1, dim + rank), "float32",
                      "model.py:1092-1093,1118", storage="BF16")
    if cfg["vision_n_layers"] > 0:
        vdim, inter, patch = cfg["vision_dim"], cfg["vision_inter_dim"], cfg["vision_patch_size"]
        x.nn_linear("vision.patch_embed.proj", 3 * patch * patch, vdim, "vision.py:40,94")
        for b in range(cfg["vision_n_layers"]):
            prefix = f"vision.blocks.{b}"
            x.rmsnorm(f"{prefix}.norm1", vdim, "vision.py:28,77", runtime="float32")
            x.nn_linear(f"{prefix}.attn.wqkv", vdim, 3 * vdim, "vision.py:51")
            x.nn_linear(f"{prefix}.attn.wo", vdim, vdim, "vision.py:52")
            x.rmsnorm(f"{prefix}.norm2", vdim, "vision.py:28,79", runtime="float32")
            x.nn_linear(f"{prefix}.mlp.w1", vdim, 2 * inter, "vision.py:66", bias=False)
            x.nn_linear(f"{prefix}.mlp.w2", inter, vdim, "vision.py:67", bias=False)
        x.rmsnorm("vision.norm", vdim, "vision.py:28,96", runtime="float32")
        ratio = cfg["vision_downsample_ratio"]
        x.nn_linear("aligner.w1", vdim * ratio * ratio, dim, "vision.py:110-111")
        x.nn_linear("aligner.w2", dim, dim, "vision.py:112")
        for name in ("image_start", "image_end", "image_newline"):
            x.add(name, (dim,), "bfloat16", "model.py:1220-1222")
    return x.items


def family_of(name):
    return ".".join("N" if part.isdigit() else part for part in name.split("."))


def group_of(entry, expected):
    if expected is not None and expected.tag is not None:
        return expected.tag
    return {"F8_E4M3": "fp8", "F8_E8M0": "fp8", "BF16": "bf16", "F32": "f32"}.get(entry.dtype, entry.dtype)


def compare(expected, observed):
    """Returns (errors, storage report) for expected tensors against observed (shard, entry)."""
    errors = []
    for name in sorted(set(expected) - set(observed)):
        errors.append(CheckError("missing", f"{name} ({expected[name].source})"))
    for name in sorted(set(observed) - set(expected)):
        errors.append(CheckError("unexpected", f"{name} in {observed[name][0]}"))
    chosen = collections.defaultdict(collections.Counter)
    for name in sorted(set(expected) & set(observed)):
        exp, entry = expected[name], observed[name][1]
        if entry.shape != exp.shape:
            errors.append(CheckError("shape", f"{name}: {list(entry.shape)} != {list(exp.shape)} ({exp.source})"))
        if entry.dtype not in exp.storage:
            errors.append(CheckError("dtype", f"{name}: {entry.dtype} not in {list(exp.storage)} ({exp.source})"))
        elif len(exp.storage) > 1 or STORAGE_OF_RUNTIME.get(exp.runtime) != entry.dtype:
            key = (family_of(name), exp.runtime or "consumed by convert.py", entry.dtype,
                   "source silent" if len(exp.storage) > 1 else exp.source)
            chosen[key][entry.dtype] += 1
    report = [
        {"family": fam, "runtime": rt, "storage": st, "basis": basis, "count": sum(c.values())}
        for (fam, rt, st, basis), c in sorted(chosen.items())
    ]
    return errors, report


def summarize_groups(shards, expected):
    groups = collections.OrderedDict()
    shard_groups = collections.defaultdict(set)
    for fname, shard in sorted(shards.items()):
        for entry in shard.entries:
            exp = expected.get(entry.name)
            g = group_of(entry, exp)
            info = groups.setdefault(g, {"tensors": 0, "elements": 0, "bytes": 0, "shards": set()})
            info["tensors"] += 1
            info["elements"] += math.prod(entry.shape)
            info["bytes"] += entry.end - entry.begin
            info["shards"].add(fname)
            shard_groups[fname].add(g)
    for g, info in groups.items():
        info["exclusive_shards"] = sorted(f for f in info["shards"] if shard_groups[f] == {g})
        info["shards"] = len(info["shards"])
    return groups


def _scan_task(path, pieces):
    """Histograms raw codes for (family, kind, offset, length) pieces of one shard."""
    out = {}
    fd = os.open(path, os.O_RDONLY)
    try:
        for family, kind, offset, length in pieces:
            buf = os.pread(fd, length, offset)
            if len(buf) != length:
                raise OSError(f"short read at {path}:{offset}")
            acc = out.setdefault(family, collections.Counter())
            if kind == "u16":
                acc.update(memoryview(buf).cast("H"))
            elif kind == "f32":
                acc.update(_f32_class(formats.fp32_from_bits(v)) for v in memoryview(buf).cast("I"))
            else:
                acc.update(buf)
    finally:
        os.close(fd)
    return out


def _f32_class(value):
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "-inf" if value < 0 else "+inf"
    if value == 0:
        return "-0" if math.copysign(1.0, value) < 0 else "+0"
    return "finite"


def plan_scan(shards, expected, chunk_bytes, task_bytes):
    """Splits every tensor into element-aligned pieces and groups them into per-shard tasks."""
    families = {}
    tasks = []
    for fname, shard in sorted(shards.items()):
        current, current_bytes = [], 0
        for entry in shard.entries:
            exp = expected.get(entry.name)
            if entry.dtype == "I8" and exp is not None and exp.runtime == "float4_e2m1fn_x2":
                scan_kind = "e2m1x2"
            else:
                scan_kind = SCAN_KIND_OF_STORAGE.get(entry.dtype, "raw")
            family = (family_of(entry.name), entry.dtype, scan_kind)
            families.setdefault(family, 0)
            families[family] += 1
            io_kind = {"bf16": "u16", "f32": "f32"}.get(scan_kind, "u8")
            start = shard.data_start + entry.begin
            stop = shard.data_start + entry.end
            while start < stop:
                length = min(chunk_bytes, stop - start)
                current.append((family, io_kind, start, length))
                current_bytes += length
                start += length
                if current_bytes >= task_bytes:
                    tasks.append((shard.path, current))
                    current, current_bytes = [], 0
        if current:
            tasks.append((shard.path, current))
    return families, tasks


def run_scan(tasks, jobs, progress):
    totals = collections.defaultdict(collections.Counter)
    planned = sum(length for _, pieces in tasks for *_, length in pieces)
    done, reported = 0, 0
    if jobs <= 1:
        results = ((task, _scan_task(*task)) for task in tasks)
        for task, result in results:
            done += _merge(totals, task, result)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(_scan_task, *task): task for task in tasks}
            for future in concurrent.futures.as_completed(futures):
                done += _merge(totals, futures[future], future.result())
                if progress and planned and done * 20 // planned > reported:
                    reported = done * 20 // planned
                    print(f"scan {reported * 5}%", file=sys.stderr, flush=True)
    return totals


def _merge(totals, task, result):
    for family, counter in result.items():
        totals[family].update(counter)
    return sum(length for *_, length in task[1])


def classify(totals, families):
    """Turns raw histograms into per-family statistics using ref/formats decoders."""
    nibbles = [formats.unpack_e2m1([b], (2,)) for b in range(256)]
    out = []
    for family in sorted(families):
        pattern, dtype, kind = family
        hist = totals.get(family, collections.Counter())
        stats = {"family": pattern, "dtype": dtype, "kind": kind, "tensors": families[family]}
        if kind == "e2m1x2":
            codes = collections.Counter()
            for byte, count in hist.items():
                for code in nibbles[byte]:
                    codes[code] += count
            stats["codes"] = {str(c): codes[c] for c in range(16)}
            zeros = [c for c in range(16) if formats.decode_e2m1(c) == 0]
            stats["neg_zero"] = sum(codes[c] for c in zeros if math.copysign(1.0, formats.decode_e2m1(c)) < 0)
            stats["pos_zero"] = sum(codes[c] for c in zeros if math.copysign(1.0, formats.decode_e2m1(c)) > 0)
            stats["values"] = sum(codes.values())
        elif kind in ("e4m3fn", "bf16"):
            decode = formats.decode_e4m3fn if kind == "e4m3fn" else formats.decode_bf16
            classes = collections.Counter()
            for code, count in hist.items():
                classes[_f32_class(decode(code))] += count
            stats["classes"] = dict(sorted(classes.items()))
            stats["values"] = sum(classes.values())
        elif kind == "e8m0":
            finite = sorted(code for code in hist if not math.isnan(formats.decode_e8m0(code)))
            stats["nan"] = sum(count for code, count in hist.items() if math.isnan(formats.decode_e8m0(code)))
            stats["values"] = sum(hist.values())
            if finite:
                stats["code_min"], stats["code_max"] = finite[0], finite[-1]
                stats["exponent_min"] = math.frexp(formats.decode_e8m0(finite[0]))[1] - 1
                stats["exponent_max"] = math.frexp(formats.decode_e8m0(finite[-1]))[1] - 1
            stats["codes"] = {str(c): hist[c] for c in sorted(hist)}
        elif kind == "f32":
            stats["classes"] = dict(sorted(hist.items()))
            stats["values"] = sum(hist.values())
        else:
            stats["codes"] = {str(c): hist[c] for c in sorted(hist)}
            stats["values"] = sum(hist.values())
        out.append(stats)
    return out


def scan_findings(stats):
    """Population-level counts across families, keyed by scan kind."""
    agg = collections.defaultdict(collections.Counter)
    ranges = {}
    for s in stats:
        kind, a = s["kind"], agg[s["kind"]]
        a["values"] += s["values"]
        if kind == "e2m1x2":
            for code, count in s["codes"].items():
                a[f"code_{code}"] += count
            a["neg_zero"] += s["neg_zero"]
            a["pos_zero"] += s["pos_zero"]
        elif kind in ("e4m3fn", "bf16", "f32"):
            a.update(s["classes"])
        elif kind == "e8m0":
            a["nan"] += s["nan"]
            if "code_min" in s:
                lo, hi = ranges.get(kind, (s["code_min"], s["code_max"]))
                ranges[kind] = (min(lo, s["code_min"]), max(hi, s["code_max"]))
    out = {kind: dict(sorted(c.items())) for kind, c in sorted(agg.items())}
    for kind, (lo, hi) in ranges.items():
        out[kind]["code_min"], out[kind]["code_max"] = lo, hi
    return out


def check_configs(ckpt_dir, reference):
    errors = []
    for rel in CONFIG_FILES:
        ours, mirror = os.path.join(ckpt_dir, rel), os.path.join(reference, rel)
        if not os.path.isfile(ours):
            errors.append(CheckError("config_missing", ours))
            continue
        with open(ours, "rb") as a, open(mirror, "rb") as b:
            if a.read() != b.read():
                errors.append(CheckError("config_mismatch", f"{ours} differs from {mirror}"))
    return errors


def load_reference_config(reference):
    path = os.path.join(reference, CONFIG_FILES[1])
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError) as exc:
        raise UsageError(f"cannot read reference config {path}: {exc}") from None


def inventory(args):
    ckpt = os.path.abspath(args.checkpoint)
    if not os.path.isdir(ckpt):
        raise UsageError(f"not a directory: {ckpt}")
    if args.chunk_bytes <= 0 or args.chunk_bytes % 8 or args.task_bytes <= 0 or args.jobs <= 0:
        raise UsageError("--chunk-bytes must be a positive multiple of 8; --task-bytes and --jobs positive")
    if sys.byteorder != "little":
        raise UsageError("raw-code histograms assume a little-endian host")
    reference = os.path.abspath(args.reference)
    cfg = load_reference_config(reference)
    expected = expected_tensors(cfg)
    report = {
        "checkpoint": ckpt,
        "revision": os.path.basename(ckpt.rstrip(os.sep)),
        "reference": reference,
        "errors": [],
    }
    errors = check_configs(ckpt, reference)

    index_path = os.path.join(ckpt, INDEX_NAME)
    names = set()
    if os.path.isfile(index_path):
        try:
            with open(index_path, encoding="utf-8") as fh:
                wm = json.load(fh).get("weight_map", {})
            names.update(v for v in wm.values() if isinstance(v, str) and os.sep not in v)
        except (OSError, ValueError, AttributeError):
            pass
    names.update(f for f in os.listdir(ckpt) if f.endswith(".safetensors"))
    shards = {}
    for fname in sorted(names):
        path = os.path.join(ckpt, fname)
        if not os.path.isfile(path):
            continue
        try:
            shards[fname] = read_header(path)
        except CheckError as exc:
            errors.append(CheckError(exc.code, f"{fname}: {exc.detail}"))
    structure_ok = not errors or all(e.code.startswith("config") for e in errors)
    errors.extend(check_index(index_path, shards))

    observed = {}
    for fname, shard in shards.items():
        for entry in shard.entries:
            observed.setdefault(entry.name, (fname, entry))
    diff, storage_report = compare(expected, observed)
    errors.extend(diff)
    groups = summarize_groups(shards, expected)
    report.update({
        "shards": len(shards),
        "tensors": len(observed),
        "expected": len(expected),
        "payload_bytes": sum(e.end - e.begin for s in shards.values() for e in s.entries),
        "groups": groups,
        "storage_vs_runtime": storage_report,
    })
    if args.inventory:
        with open(args.inventory, "w", encoding="utf-8") as fh:
            fh.write("name\tshard\tdtype\tshape\tbegin\tend\tsource\n")
            for name in sorted(observed):
                fname, e = observed[name]
                src = expected[name].source if name in expected else ""
                fh.write(f"{name}\t{fname}\t{e.dtype}\t{'x'.join(map(str, e.shape))}\t{e.begin}\t{e.end}\t{src}\n")

    if args.scan:
        if not structure_ok:
            errors.append(CheckError("scan_skipped", "structure errors prevent a data scan"))
        else:
            families, tasks = plan_scan(shards, expected, args.chunk_bytes, args.task_bytes)
            stats = classify(run_scan(tasks, args.jobs, args.progress), families)
            report["scan"] = {"families": stats, "findings": scan_findings(stats)}

    report["errors"] = [{"code": e.code, "detail": e.detail} for e in errors]
    return report


def print_summary(report):
    print(f"checkpoint {report['checkpoint']}")
    print(f"shards {report['shards']}  tensors {report['tensors']}  expected {report['expected']}  "
          f"payload {report['payload_bytes']} bytes")
    for g, info in report["groups"].items():
        print(f"  group {g:14s} tensors {info['tensors']:6d}  elements {info['elements']:15d}  "
              f"bytes {info['bytes']:15d} ({info['bytes'] / 1e9:.3f} GB)  shards {info['shards']}  "
              f"exclusive {info['exclusive_shards'] if len(info['exclusive_shards']) < 4 else len(info['exclusive_shards'])}")
    for row in report["storage_vs_runtime"]:
        print(f"  storage {row['storage']:8s} runtime {row['runtime']:24s} x{row['count']:<6d} "
              f"{row['family']}  [{row['basis']}]")
    if "scan" in report:
        for kind, counts in report["scan"]["findings"].items():
            print(f"  scan {kind}: {json.dumps(counts, sort_keys=True)}")
    codes = collections.Counter(e["code"] for e in report["errors"])
    for e in report["errors"][:50]:
        print(f"ERROR {e['code']}: {e['detail']}")
    print("RESULT " + ("PASS" if not codes else "FAIL " + json.dumps(dict(sorted(codes.items())))))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoint", help="snapshot directory with the shards and index")
    parser.add_argument("--reference", default=DEFAULT_REFERENCE, help="official mirror (read-only)")
    parser.add_argument("--scan", action="store_true", help="histogram every tensor's raw codes")
    parser.add_argument("--jobs", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--chunk-bytes", type=int, default=64 << 20)
    parser.add_argument("--task-bytes", type=int, default=256 << 20)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--json", dest="json_path", help="write the full report as JSON")
    parser.add_argument("--inventory", help="write one TSV row per tensor")
    args = parser.parse_args(argv)
    try:
        report = inventory(args)
    except UsageError as exc:
        print(f"usage error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"I/O error: {exc}", file=sys.stderr)
        return 2
    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1, sort_keys=True, default=sorted)
    print_summary(report)
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
