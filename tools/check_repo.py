#!/usr/bin/env python3
"""check_repo.py — 仓库纪律机械执法器（P0001 v1；P0002/D0003 扩展 P 生命周期与知识处置）。

校验 AGENTS.md（宪法）工作流与禁区条文中可机械化、零语义判定的子集：
subject/area 格式、编号引用可解析、Bench:/Oracle: trailer 义务与真伪、
VERSION 递增纪律、禁区触碰、plan 条目 front matter 与 supersede 互链、
P 条目知识处置（knowledge/followups 枚举、引用真实、路径存在）、P 状态机迁移边、
条目状态翻转与 STATUS.md 同提交同步、条目文件永不删除。

规则↔条文映射、设计决议与演进规范见 plan/p0001-check-repo.md；
P 生命周期与知识处置扩展见 P0002 与 D0003。
治理起点（epoch）= 首个 plan/decisions/d<NNNN>-*.md 的入库 commit，之前的
提交一律豁免；epoch 未建立时仅执行树级检查并提示。纯标准库 + git。
"""

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass

REPO_AREAS = {"docs", "plan", "bench", "ci", "tools", "serve", "deploy"}
SRC_CODE_SUFFIXES = (".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp")
ORACLE_OBLIGED_PREFIXES = ("src/kernels/", "src/quant/", "include/df/")
FROZEN_PREFIXES = ("bench/prompts/", "tests/data/")
REFERENCE_PREFIX = "third_party/reference/"
PROVENANCE = "third_party/reference/PROVENANCE.md"
STATUS_PAGE = "plan/STATUS.md"
CONSTITUTION = "AGENTS.md"
VERSION_FILE = "VERSION"
P_STATUSES = ("draft", "active", "completed", "abandoned")
D_STATUSES = ("proposed", "accepted", "rejected", "superseded")
P_TERMINAL = ("completed", "abandoned")
P_TRANSITIONS = {
    "draft": {"active", "abandoned"},
    "active": {"completed", "abandoned"},
    "completed": set(),
    "abandoned": set(),
    "sedimented": {"completed"},
}
P_TRANSITIONS_LEGACY = {
    "draft": {"active", "abandoned"},
    "active": {"sedimented", "abandoned"},
    "sedimented": set(),
    "abandoned": set(),
    "completed": set(),
}
P_NEW_KEYS = ("knowledge", "followups")
KNOWLEDGE = ("pending", "updated", "covered", "none", "deferred")
CLOSING_FIELDS = ("- 验收或停止依据：", "- 知识处置：", "- 剩余义务：")
P_FM_KEYS = ("id", "title", "status", "created", "updated", "docs", "bench", "decisions",
             "knowledge", "followups")
D_FM_KEYS = ("id", "title", "status", "created", "updated", "replaces", "superseded-by")

ENTRY_RE = re.compile(r"^(p|d)(\d{4})-([a-z0-9][a-z0-9-]*)\.md$")
SUBJECT_RE = re.compile(r"^([PD])(\d{4}) ([a-z0-9-]+): (.+)$")
MERGE_RE = re.compile(r"^Merge P(\d{4})\b")
META_RE = re.compile(r"^meta: ")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
TRAILER_RE = re.compile(r"^([A-Z][A-Za-z-]+): (.+)$")
SHA_RE = re.compile(r"^[0-9a-f]{40,64}$")
VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


@dataclass
class V:
    """一条违例：sev=E/W，anchor 锚定宪法条文或内部规则，obj 为对象。"""

    sev: str
    anchor: str
    obj: str
    msg: str

    def fmt(self):
        return f"[{self.sev}] {self.anchor} | {self.obj} | {self.msg}"


def _git(repo, *args, check=True):
    r = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r


def git_out(repo, *args):
    return _git(repo, *args).stdout


def git_try(repo, *args):
    r = _git(repo, *args, check=False)
    return r.stdout if r.returncode == 0 else None


def parse_frontmatter(text):
    """解析受控 YAML 子集（键: 标量 或 单行列表；容忍 # 注释）。损坏返回 None。"""
    if not text.startswith("---"):
        return None
    lines = text.splitlines()
    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end is None:
        return None
    fm = {}
    for ln in lines[1:end]:
        m = re.match(r"^([A-Za-z][A-Za-z0-9_-]*):\s*(.*)$", ln)
        if not m:
            continue
        key, val = m.group(1), m.group(2).strip()
        val = re.sub(r"\s+#.*$", "", val).strip()
        if val.startswith("[") and val.endswith("]"):
            inner = val[1:-1].strip()
            val = [x.strip() for x in inner.split(",") if x.strip()]
        fm[key] = val
    return fm


def ref_num(x):
    """归一化互链引用（接受 D0002 / 0002 / P0003 形态）。"""
    m = re.fullmatch(r"([PpDd]?)(\d{4})", str(x).strip())
    return (m.group(1).upper(), m.group(2)) if m else None


def ref_for(x, kind):
    """按字段类型取编号：无字母前缀时采用字段类型，带了则须与字段类型一致。"""
    r = ref_num(x)
    if r is None:
        return None
    letter, num = r
    return num if not letter or letter == kind else None


def rel_under(repo, path, prefix):
    """相对引用须落在 prefix 下：拒绝绝对/.. 逃逸，且 realpath 不越出该实目录。"""
    p = str(path).strip()
    if not p or p.startswith("/") or os.path.isabs(p):
        return None
    norm = os.path.normpath(p)
    if norm.startswith("..") or not norm.startswith(prefix):
        return None
    base = os.path.realpath(os.path.join(repo, prefix.rstrip("/")))
    real = os.path.realpath(os.path.join(repo, norm))
    if real != base and not real.startswith(base + os.sep):
        return None
    return norm


def aslist(v):
    return v if isinstance(v, list) else ([] if v in (None, "") else [v])


class Ctx:
    """一次运行的共享状态：area 白名单与当前树条目索引。"""

    def __init__(self, repo):
        self.repo = repo
        self.areas = set(REPO_AREAS)
        self.entries = {}  # ("P"|"D", "0001") -> (path, fm|None)
        src = os.path.join(repo, "src")
        if os.path.isdir(src):
            self.areas |= {d for d in os.listdir(src) if os.path.isdir(os.path.join(src, d))}

    def exists(self, kind, num):
        return (kind, num) in self.entries


def scan_entry_file(ctx, out, rel, kind, num):
    """校验单个条目文件并登记编号；front matter 细节违规逐条报告。"""
    key = (kind.upper(), num)
    if key in ctx.entries:
        out(V("E", "AGENTS:plan条目", rel,
              f"编号重复：{kind.upper()}{num} 已由 {ctx.entries[key][0]} 占用（永不复用）"))
        return
    path = os.path.join(ctx.repo, rel)
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except OSError:
        text = None
    fm = parse_frontmatter(text) if text is not None else None
    ctx.entries[key] = (rel, fm)
    if fm is None:
        out(V("E", "AGENTS:plan条目", rel, "front matter 缺失或破损"))
        return
    want_keys = P_FM_KEYS if kind == "p" else D_FM_KEYS
    for k in want_keys:
        if k not in fm:
            out(V("E", "AGENTS:plan条目", rel, f"front matter 缺字段：{k}"))
    if fm.get("id") != f"{kind.upper()}{num}":
        out(V("E", "AGENTS:plan条目", rel, f"id 须为 {kind.upper()}{num}（与文件名一致）"))
    statuses = P_STATUSES if kind == "p" else D_STATUSES
    if fm.get("status") not in statuses:
        out(V("E", "AGENTS:plan条目", rel, f"status 非法：{fm.get('status')!r}（合法：{'/'.join(statuses)}）"))
    for k in ("created", "updated"):
        if not isinstance(fm.get(k), str) or not DATE_RE.match(fm.get(k) or ""):
            out(V("E", "AGENTS:plan条目", rel, f"{k} 须为 YYYY-MM-DD：{fm.get(k)!r}"))
    if {fm.get("created"), fm.get("updated")} <= {None} or fm.get("created") == fm.get("updated"):
        pass
    elif isinstance(fm.get("created"), str) and isinstance(fm.get("updated"), str):
        if DATE_RE.match(fm["created"]) and DATE_RE.match(fm["updated"]) and fm["updated"] < fm["created"]:
            out(V("E", "AGENTS:plan条目", rel, "updated 早于 created"))
    if kind == "p" and fm.get("status") in P_TERMINAL:
        check_closing(out, rel, text or "")


def check_closing(out, rel, text):
    """终态条目的收尾记录须落在 `## 收尾` 章节内且非空（仅结构与非空，不判语义）。"""
    lines = text.splitlines()
    start = None
    for i, ln in enumerate(lines):
        if re.match(r"^##\s*收尾\s*$", ln):
            start = i + 1
            break
    if start is None:
        out(V("E", "AGENTS:plan条目", rel, "终态条目须含 `## 收尾` 章节"))
        return
    body = []
    for ln in lines[start:]:
        if re.match(r"^#{1,2}(?!#)\s", ln):
            break
        body.append(ln)
    for marker in CLOSING_FIELDS:
        if not any(ln.strip().startswith(marker) and ln.strip()[len(marker):].strip()
                   for ln in body):
            out(V("E", "AGENTS:plan条目", rel, f"收尾记录缺失或为空：{marker}"))


def scan_tree(ctx, out):
    """树级检查：plan/ 与 plan/decisions/ 下条目命名、front matter、互链。"""
    for sub, kind in (("plan", "p"), ("plan/decisions", "d")):
        d = os.path.join(ctx.repo, sub)
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if not name.endswith(".md") or name == "TEMPLATE.md" or name == "STATUS.md":
                continue
            m = ENTRY_RE.match(name)
            if not m or m.group(1) != kind:
                out(V("E", "AGENTS:plan条目", f"{sub}/{name}",
                      f"命名须为 {kind}<NNNN>-<slug>.md"))
                continue
            scan_entry_file(ctx, out, f"{sub}/{name}", kind, m.group(2))

    for (kind, num), (rel, fm) in sorted(ctx.entries.items()):
        if not fm:
            continue
        if kind == "D":
            check_supersede_links(ctx, out, rel, num, fm)
        else:
            check_p_disposition(ctx, out, rel, num, fm)


def check_p_disposition(ctx, out, rel, num, fm):
    """P 条目的知识处置：引用真实、路径存在、枚举合法与状态互锁。"""
    status = fm.get("status")
    knowledge = fm.get("knowledge")
    docs = aslist(fm.get("docs"))
    bench = aslist(fm.get("bench"))
    decisions = aslist(fm.get("decisions"))
    followups = aslist(fm.get("followups"))
    if knowledge is not None and knowledge not in KNOWLEDGE:
        out(V("E", "AGENTS:plan条目", rel,
              f"knowledge 非法：{knowledge!r}（合法：{'/'.join(KNOWLEDGE)}）"))
    for d in docs:
        rp = rel_under(ctx.repo, str(d).split("#", 1)[0], "docs/")
        if rp is None:
            out(V("E", "AGENTS:plan条目", rel, f"docs 须为 docs/ 下相对路径（禁绝对/逃逸）：{d}"))
        elif not os.path.isfile(os.path.join(ctx.repo, rp)):
            out(V("E", "AGENTS:plan条目", rel, f"docs 指向的文件不存在：{d}"))
    for b in bench:
        rp = rel_under(ctx.repo, str(b), "bench/results/")
        if rp is None:
            out(V("E", "AGENTS:plan条目", rel, f"bench 须为 bench/results/ 下相对路径：{b}"))
        elif not os.path.isdir(os.path.join(ctx.repo, rp)):
            out(V("E", "AGENTS:plan条目", rel, f"bench 指向的目录不存在：{b}"))
    for x in decisions:
        r = ref_for(x, "D")
        if r is None or not ctx.exists("D", r):
            out(V("E", "AGENTS:plan条目", rel, f"decisions 引用悬空或格式非法：{x}"))
    for x in followups:
        r = ref_for(x, "P")
        if r is None or not ctx.exists("P", r):
            out(V("E", "AGENTS:plan条目", rel, f"followups 引用悬空或格式非法：{x}"))
        elif r == num:
            out(V("E", "AGENTS:plan条目", rel, f"followups 不得自指：{x}"))
    if status in P_TERMINAL and knowledge == "pending":
        out(V("E", "AGENTS:plan条目", rel, f"{status} 条目 knowledge 不得为 pending"))
    if knowledge in ("updated", "covered") and not (docs or bench or decisions):
        out(V("E", "AGENTS:plan条目", rel,
              f"knowledge={knowledge} 须至少有 docs/bench/decisions 之一引用"))
    if knowledge == "deferred" and not followups:
        out(V("E", "AGENTS:plan条目", rel, "knowledge=deferred 须回填非空 followups"))


def check_supersede_links(ctx, out, rel, num, fm):
    """supersede 互链：A.superseded-by=B ⟺ B.replaces 含 A，且 A 状态为 superseded。"""
    status = fm.get("status")
    sb = aslist(fm.get("superseded-by"))
    rp = aslist(fm.get("replaces"))
    for x in sb + rp:
        if ref_num(x) is None:
            out(V("E", "AGENTS:plan条目", rel, f"互链引用格式非法：{x}"))
    if status == "superseded" and not sb:
        out(V("E", "AGENTS:plan条目", rel, "superseded 须回填 superseded-by"))
    if sb and status != "superseded":
        out(V("E", "AGENTS:plan条目", rel, "superseded-by 非空但状态非 superseded"))
    for x in sb:
        r = ref_num(x)
        if not r or not ctx.exists(*r):
            out(V("E", "AGENTS:plan条目", rel, f"superseded-by 悬空：{x}"))
            continue
        tfm = ctx.entries[r][1]
        if not tfm:
            continue
        if num not in [ref_num(y)[1] for y in aslist(tfm.get("replaces")) if ref_num(y)]:
            out(V("E", "AGENTS:plan条目", rel, f"互链不一致：{x}.replaces 未回指 D{num}"))
        if tfm.get("status") != "superseded":
            out(V("E", "AGENTS:plan条目", rel, f"互链目标 {x} 状态非 superseded"))
    for x in rp:
        r = ref_num(x)
        if not r or not ctx.exists(*r):
            out(V("E", "AGENTS:plan条目", rel, f"replaces 悬空：{x}"))
            continue
        tfm = ctx.entries[r][1]
        if not tfm:
            continue
        if num not in [ref_num(y)[1] for y in aslist(tfm.get("superseded-by")) if ref_num(y)]:
            out(V("E", "AGENTS:plan条目", rel, f"互链不一致：{x}.superseded-by 未回指 D{num}"))
        if tfm.get("status") != "superseded":
            out(V("E", "AGENTS:plan条目", rel, f"replaces 目标 {x} 状态非 superseded"))


def find_epoch(repo):
    """治理起点：首个把 d<NNNN>-*.md 条目带入历史的 commit。"""
    out = git_try(repo, "log", "--reverse", "--format=%H", "--name-only",
                  "--diff-filter=A", "--", "plan/decisions/")
    if not out:
        return None
    cur = None
    for ln in out.splitlines():
        ln = ln.strip()
        if SHA_RE.match(ln):
            cur = ln
        elif cur and ln.startswith("plan/decisions/"):
            m = ENTRY_RE.match(os.path.basename(ln))
            if m and m.group(1) == "d":
                return cur
    return None


def changed_files(repo, sha, parents):
    """相对第一父的变更文件集（merge 取其带入 main 的内容）。"""
    if parents:
        out = git_try(repo, "diff-tree", "--no-commit-id", "--name-only", "-r",
                      f"{sha}^1", sha)
        if out is not None:
            return [l.strip() for l in out.splitlines() if l.strip()]
    out = git_try(repo, "diff-tree", "--no-commit-id", "--name-only", "-r", "--root", sha) or ""
    return [l.strip() for l in out.splitlines() if l.strip()]


def fm_from_blob(repo, ref):
    text = git_try(repo, "show", ref)
    return parse_frontmatter(text) if text is not None else None


def parse_trailers(body):
    """仅取消息末段的 trailer 块（git interpret-trailers 语义）。"""
    if not body or not body.strip():
        return {}
    last = body.rstrip().split("\n\n")[-1]
    tl = {}
    for ln in last.splitlines():
        m = TRAILER_RE.match(ln)
        if m:
            tl.setdefault(m.group(1), []).append(m.group(2).strip())
    return tl


def read_version(repo, ref):
    text = git_try(repo, "show", ref) if ref else None
    if text is None:
        return None
    m = VERSION_RE.match(text.strip())
    return tuple(int(g) for g in m.groups()) if m else None


def valid_bump(old, new):
    return (new[0] == old[0] and new[1] == old[1] and new[2] == old[2] + 1) or \
           (new[0] == old[0] and new[1] == old[1] + 1 and new[2] == 0)


def check_commit(repo, sha, ctx, out):
    """单个受治 commit 的全部提交级规则。"""
    tag = sha[:10]
    subject = git_out(repo, "show", "-s", "--format=%s", sha).strip()
    body = git_out(repo, "show", "-s", "--format=%b", sha)
    parents = git_out(repo, "show", "-s", "--format=%P", sha).split()
    files = changed_files(repo, sha, parents)
    is_merge = len(parents) >= 2
    ref_kind = ref_num_ = None

    if is_merge:
        m = MERGE_RE.match(subject)
        if not m:
            out(V("E", "AGENTS:git", tag, f"合并提交 subject 须以 `Merge P<NNNN>` 开头：{subject!r}"))
        else:
            ref_kind, ref_num_ = "P", m.group(1)
    else:
        m = SUBJECT_RE.match(subject)
        if m:
            ref_kind, ref_num_, area = m.group(1), m.group(2), m.group(3)
            if area not in ctx.areas:
                out(V("E", "AGENTS:git", tag,
                      f"area {area!r} 不在 src 域名/仓库级域名内：{subject!r}"))
        elif META_RE.match(subject):
            code = [f for f in files
                    if f.endswith(SRC_CODE_SUFFIXES) and f.startswith(("src/", "include/"))]
            if code:
                out(V("W", "AGENTS:git", tag,
                      f"meta: 提交触碰源码（“不改行为”无法机检，须人审）：{code[:3]}"))
        else:
            out(V("E", "AGENTS:git", tag,
                  f"subject 不符 `<P|D 编号> <area>: <论断句>` 或 `meta:`：{subject!r}"))

    if ref_num_ and not ctx.exists(ref_kind, ref_num_):
        out(V("E", "D0001", tag, f"subject 引用的编号 {ref_kind}{ref_num_} 无对应条目（悬空引用）"))

    if CONSTITUTION in files and not (not is_merge and ref_kind == "D"):
        out(V("E", "AGENTS:规则5", tag,
              "触碰宪法文件：宪法修订须以 `D<NNNN> <area>:` subject 直接提交 main（P0001 决议②）"))

    tr = parse_trailers(body)
    oracles = tr.get("Oracle", [])
    benches = tr.get("Bench", [])
    for p in oracles:
        if not os.path.exists(os.path.join(repo, p)):
            out(V("E", "AGENTS:git", tag, f"Oracle: 指向的路径不存在：{p}"))
    for b in benches:
        if not b.startswith("bench/results/") or not os.path.isdir(os.path.join(repo, b)):
            out(V("E", "AGENTS:git", tag, f"Bench: 须为存在的 bench/results/<日期-主题> 目录：{b}"))
    if any(f.startswith(ORACLE_OBLIGED_PREFIXES) for f in files) and not oracles:
        out(V("E", "AGENTS:git", tag, "kernel/量化路径变更缺 `Oracle: <parity 测试路径>` trailer"))

    old_v = read_version(repo, f"{sha}^1:{VERSION_FILE}") if parents else None
    new_v = read_version(repo, f"{sha}:{VERSION_FILE}")
    if VERSION_FILE in files:
        if not is_merge:
            if old_v is not None:
                out(V("E", "AGENTS:版本", tag, "VERSION 仅由合并流程递增，非合并提交不得改动"))
        elif old_v is not None and (new_v is None or not valid_bump(old_v, new_v)):
            out(V("E", "AGENTS:版本", tag,
                  f"合并须按 patch+1（或 minor+1 且 patch=0）递增 VERSION：{old_v} → {new_v}"))
    elif is_merge and old_v is not None:
        out(V("E", "AGENTS:版本", tag, "--no-ff 合入未递增 VERSION（patch+1）"))

    ref_touched = [f for f in files if f.startswith(REFERENCE_PREFIX)]
    if any(f != PROVENANCE for f in ref_touched) and PROVENANCE not in files:
        out(V("E", "AGENTS:禁区", tag, "third_party/reference/ 只读：触碰须同提交更新 PROVENANCE.md"))

    if any(f.startswith("ref/") for f in files):
        out(V("W", "AGENTS:禁区", tag, "ref/（oracle）被触碰：须确认 oracle 自身有误并写明依据"))
    if any(f.startswith(FROZEN_PREFIXES) for f in files):
        out(V("W", "AGENTS:禁区", tag, "冻结区被触碰：视为新工件，须重算 manifest 并说明"))

    flipped = False
    for f in files:
        base = os.path.basename(f)
        m = ENTRY_RE.match(base)
        if not f.startswith("plan/") or not m:
            continue
        old_fm = fm_from_blob(repo, f"{sha}^1:{f}") if parents else None
        new_fm = fm_from_blob(repo, f"{sha}:{f}")
        if not old_fm or not new_fm:
            continue
        if m.group(1) == "p":
            check_p_field_removal(out, tag, f, old_fm, new_fm)
        if old_fm.get("status") != new_fm.get("status"):
            flipped = True
            if m.group(1) == "p":
                check_p_transition(out, tag, f, old_fm, new_fm, is_merge)
    if flipped and STATUS_PAGE not in files:
        out(V("E", "AGENTS:plan条目", tag, "条目状态翻转未与 plan/STATUS.md 同一提交同步"))


def check_p_field_removal(out, tag, rel, old_fm, new_fm):
    """P 的新 schema 字段一经引入不得在后续提交中删除（即使 status 未变）。"""
    for k in P_NEW_KEYS:
        if k in old_fm and k not in new_fm:
            out(V("E", "AGENTS:plan条目", tag, f"P 条目不得删除字段：{rel} 缺 {k}"))


def check_p_transition(out, tag, rel, old_fm, new_fm, is_merge):
    """P 状态机合法迁移；普通提交须单步直接边，merge 净差异为累积状态变化（非单次事件）
    故按对应 schema 图的可达性判定，终态无出边天然不可达。旧 schema（old/new front matter
    均无 knowledge/followups）沿用旧边集，sedimented→completed 仅按新 schema 允许。"""
    old = old_fm.get("status")
    new = new_fm.get("status")
    legacy = not any(k in old_fm for k in P_NEW_KEYS) and not any(k in new_fm for k in P_NEW_KEYS)
    edges = P_TRANSITIONS_LEGACY if legacy else P_TRANSITIONS
    if old not in edges:
        out(V("E", "AGENTS:plan条目", tag, f"P 起始状态非法：{rel} {old!r}"))
        return
    ok = p_reachable(edges, old, new) if is_merge else new in edges[old]
    if not ok:
        out(V("E", "AGENTS:plan条目", tag, f"P 非法状态迁移：{rel} {old} → {new}"))


def p_reachable(edges, old, new):
    """状态图自 old 是否可达 new（含单步；终态无出边故不可再激活）。"""
    seen = {old}
    stack = [old]
    while stack:
        for t in edges.get(stack.pop(), set()):
            if t == new:
                return True
            if t not in seen:
                seen.add(t)
                stack.append(t)
    return False


def audit_deletions(repo, epoch, out):
    """条目文件永不移动删除：历史中不得出现 D 状态的条目路径。"""
    log = git_try(repo, "log", "--no-renames", "--diff-filter=D", "--format=%H",
                  "--name-only", f"{epoch}..HEAD", "--", "plan/") or ""
    seen = set()
    for ln in log.splitlines():
        ln = ln.strip()
        if ln.startswith("plan/") and ENTRY_RE.match(os.path.basename(ln)) and ln not in seen:
            seen.add(ln)
            out(V("E", "AGENTS:plan条目", ln, "条目文件曾被删除/改名（编号必须原地归档）"))


def rev_list(repo, spec):
    out = git_try(repo, "rev-list", "--reverse", spec)
    return out.split() if out else []


def main(argv=None):
    ap = argparse.ArgumentParser(description="仓库纪律机械执法器（P0001）")
    ap.add_argument("--repo", default=".", help="仓库路径（默认当前目录）")
    ap.add_argument("--range", dest="rng", metavar="A..B", help="检查 A..B 内提交")
    ap.add_argument("--audit", action="store_true", help="epoch 起全历史 + 删除扫描")
    ap.add_argument("--strict", action="store_true", help="W 级也计入退出码")
    args = ap.parse_args(argv)

    rr = _git(args.repo, "rev-parse", "--show-toplevel", check=False)
    if rr.returncode != 0:
        print(f"[E] check:internal | - | 不是 git 仓库：{args.repo}", file=sys.stderr)
        return 2
    repo = rr.stdout.strip()

    violations = []

    def out(v):
        violations.append(v)

    ctx = Ctx(repo)
    scan_tree(ctx, out)

    epoch = find_epoch(repo)
    if epoch is None:
        out(V("W", "check:epoch", "-",
              "治理起点未入库（plan/decisions/ 无 d<NNNN> 条目入库 commit），提交级检查跳过"))
        shas = []
    else:
        tip = args.rng.split("..", 1)[1] if args.rng else "HEAD"
        governed = set(rev_list(repo, f"{epoch}..{tip}"))
        governed.add(epoch)
        if args.audit:
            shas = rev_list(repo, f"{epoch}..HEAD") + [epoch]
        elif args.rng:
            if ".." not in args.rng:
                print("[E] check:internal | - | --range 须为 A..B 形式", file=sys.stderr)
                return 2
            shas = [s for s in rev_list(repo, args.rng) if s in governed]
        else:
            head = (git_try(repo, "rev-parse", "HEAD") or "").strip()
            shas = [head] if head in governed else []

    for sha in shas:
        check_commit(repo, sha, ctx, out)

    if args.audit and epoch:
        audit_deletions(repo, epoch, out)

    for v in violations:
        print(v.fmt())
    es = [v for v in violations if v.sev == "E"]
    ws = [v for v in violations if v.sev == "W"]
    print(f"SUMMARY E={len(es)} W={len(ws)}")
    return 1 if es or (args.strict and ws) else 0


if __name__ == "__main__":
    sys.exit(main())
