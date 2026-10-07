"""check_repo.py 的规则级测试（P0001 / P0002 / P0003）：临时仓库验证受治/违规两侧。

运行：python3 tests/tools/test_check_repo.py
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHECKER = os.path.join(REPO_ROOT, "tools", "check_repo.py")

D1 = """---
id: D0001
title: 治理基线
status: accepted
created: 2026-10-07
updated: 2026-10-07
replaces: []
superseded-by: []
---

正文
"""

D2 = D1.replace("D0001", "D0002").replace("治理基线", "修宪")


CLOSING = """
## 收尾

- 验收或停止依据：规则测试通过，或明确停止。
- 知识处置：具体依据见关联材料，无增量时不新增文档。
- 剩余义务：无。
"""


def p_entry(status="draft", docs="[]", knowledge="pending", bench="[]",
            decisions="[]", followups="[]", closing=None):
    if closing is None:
        closing = status in ("completed", "abandoned")
    text = f"""---
id: P0001
title: 工作项
status: {status}
created: 2026-10-07
updated: 2026-10-07
knowledge: {knowledge}
docs: {docs}
bench: {bench}
decisions: {decisions}
followups: {followups}
---

正文
"""
    return text + CLOSING if closing else text


class Fixture:
    """临时 git 仓库：构造提交序列后调 check() 取执法结果。"""

    def __init__(self):
        self.dir = tempfile.mkdtemp(prefix="df-checkrepo-")
        self.git("init", "-q", "-b", "main")
        self.git("config", "user.email", "test@example.com")
        self.git("config", "user.name", "test")
        self.git("config", "commit.gpgsign", "false")

    def git(self, *a):
        r = subprocess.run(["git", "-C", self.dir, *a], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"git {a}: {r.stderr}")
        return r

    def write(self, rel, text):
        p = os.path.join(self.dir, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write(text)

    def remove(self, rel):
        os.remove(os.path.join(self.dir, rel))

    def _commit(self, msg):
        self.git("add", "-A")
        subprocess.run(["git", "-C", self.dir, "commit", "-q", "-F", "-"],
                       input=msg, text=True, capture_output=True, check=True)

    def commit(self, subject, files=None, body=""):
        for k, v in (files or {}).items():
            self.write(k, v)
        self._commit(subject if not body else subject + "\n\n" + body)

    def merge(self, branch, subject, files=None):
        self.git("checkout", "-q", "-b", branch)
        self.commit(f"P0001 tools: work on {branch}", {f"wip-{branch}.txt": "x\n"})
        self.git("checkout", "-q", "main")
        self.git("merge", "--no-ff", "--no-commit", "-q", branch)
        for k, v in (files or {}).items():
            self.write(k, v)
        self._commit(subject)

    def check(self, *extra):
        r = subprocess.run([sys.executable, CHECKER, "--repo", self.dir, *extra],
                           capture_output=True, text=True)
        return r.returncode, r.stdout + r.stderr


def legacy_entry(status):
    return p_entry(status).replace("knowledge: pending\n", "").replace("followups: []\n", "")


def baseline(fx, legacy=False):
    fx.commit("chore: initial import", {"README.md": "x\n", "VERSION": "0.1.0\n"})
    fx.commit("D0001 plan: governance baseline", {
        "plan/decisions/d0001-governance-baseline.md": D1,
        "plan/p0001-checker.md": legacy_entry("draft") if legacy else p_entry(),
        "plan/STATUS.md": "# STATUS\n",
        "AGENTS.md": "# AGENTS\n",
    })


class CheckRepoTests(unittest.TestCase):
    def setUp(self):
        self.fx = Fixture()
        self.addCleanup(shutil.rmtree, self.fx.dir)

    def test_happy_path_audit(self):
        baseline(self.fx)
        self.fx.commit("P0001 tools: add checker", {"tools/check_repo.py": "x\n"})
        rc, out = self.fx.check("--audit")
        self.assertEqual(rc, 0, out)
        self.assertNotIn("[E]", out)

    def test_epoch_exempts_pre_governance(self):
        self.fx.commit("totally bad subject", {"VERSION": "9.9.9\n", "src/quant/a.cu": "x\n"})
        baseline(self.fx)
        rc, out = self.fx.check("--audit")
        self.assertEqual(rc, 0, out)
        self.assertNotIn("[E]", out)

    def test_no_epoch_graceful(self):
        self.fx.commit("chore: seed", {"README.md": "x\n"})
        rc, out = self.fx.check("--audit")
        self.assertEqual(rc, 0, out)
        self.assertIn("[W] check:epoch", out)

    def test_subject_format_area_dangling(self):
        baseline(self.fx)
        self.fx.commit("no format here", {"a.txt": "1\n"})
        rc, out = self.fx.check()
        self.assertEqual(rc, 1)
        self.assertIn("subject", out)
        self.fx.commit("P0001 noarea: bad area", {"a.txt": "2\n"})
        _, out = self.fx.check()
        self.assertIn("area", out)
        self.fx.commit("P9999 tools: dangling ref", {"a.txt": "3\n"})
        _, out = self.fx.check()
        self.assertIn("P9999", out)
        self.assertIn("悬空", out)

    def test_merge_subject(self):
        baseline(self.fx)
        self.fx.merge("b1", "merge stuff in")
        rc, out = self.fx.check()
        self.assertEqual(rc, 1)
        self.assertIn("Merge P<NNNN>", out)

    def test_oracle_trailer(self):
        baseline(self.fx)
        self.fx.commit("P0001 kernels: fast gemm", {"src/kernels/gemm.cu": "x\n"})
        _, out = self.fx.check()
        self.assertIn("Oracle", out)
        self.fx.commit("P0001 kernels: fast gemm ok",
                       {"src/kernels/gemm.cu": "y\n", "tests/cuda/t_gemm.py": "x\n"},
                       body="Oracle: tests/cuda/t_gemm.py")
        rc, out = self.fx.check()
        self.assertEqual(rc, 0, out)
        self.fx.commit("P0001 kernels: dangling oracle",
                       {"src/kernels/gemm.cu": "z\n"},
                       body="Oracle: tests/cuda/missing.py")
        _, out = self.fx.check()
        self.assertIn("不存在", out)

    def test_bench_trailer(self):
        baseline(self.fx)
        self.fx.commit("P0001 tools: bench ref", {"tools/a.py": "x\n"},
                       body="Bench: bench/results/2026-10-07-t1")
        rc, out = self.fx.check()
        self.assertEqual(rc, 1)
        self.assertIn("bench/results/", out)
        self.fx.commit("P0001 tools: bench ok",
                       {"tools/b.py": "x\n", "bench/results/2026-10-07-t1/README.md": "x\n"},
                       body="Bench: bench/results/2026-10-07-t1")
        rc, out = self.fx.check()
        self.assertEqual(rc, 0, out)

    def test_version_rules(self):
        baseline(self.fx)
        self.fx.commit("P0001 tools: manual bump", {"VERSION": "0.1.1\n"})
        _, out = self.fx.check()
        self.assertIn("非合并", out)
        self.fx.merge("bv1", "Merge P0001", {"VERSION": "0.1.2\n"})
        rc, out = self.fx.check()
        self.assertEqual(rc, 0, out)
        self.fx.merge("bv2", "Merge P0001")
        _, out = self.fx.check()
        self.assertIn("未递增", out)
        self.fx.merge("bv3", "Merge P0001", {"VERSION": "0.1.5\n"})
        _, out = self.fx.check()
        self.assertIn("patch+1", out)
        self.fx.merge("bv4", "Merge P0001", {"VERSION": "0.2.0\n"})
        rc, out = self.fx.check()
        self.assertEqual(rc, 0, out)

    def test_reference_readonly(self):
        baseline(self.fx)
        self.fx.commit("P0001 tools: touch ref repo", {"third_party/reference/config.json": "{}\n"})
        rc, out = self.fx.check()
        self.assertEqual(rc, 1)
        self.assertIn("PROVENANCE", out)
        self.fx.commit("P0001 tools: sync mirror",
                       {"third_party/reference/config.json": "{}\n",
                        "third_party/reference/PROVENANCE.md": "| commit | xyz |\n"})
        rc, out = self.fx.check()
        self.assertEqual(rc, 0, out)

    def test_warn_levels(self):
        baseline(self.fx)
        self.fx.commit("P0001 tools: oracle tweak", {"ref/model.py": "x\n"})
        rc, out = self.fx.check()
        self.assertEqual(rc, 0, out)
        self.assertIn("[W]", out)
        self.fx.commit("P0001 tools: prompt tweak", {"bench/prompts/p.txt": "x\n"})
        rc, out = self.fx.check()
        self.assertEqual(rc, 0, out)
        self.assertIn("冻结区", out)
        self.fx.commit("P0001 tools: prompt tweak2", {"bench/prompts/p.txt": "y\n"})
        rc, out = self.fx.check("--strict")
        self.assertEqual(rc, 1, out)

    def test_meta_touching_source(self):
        baseline(self.fx)
        self.fx.commit("meta: reformat", {"src/core/a.cpp": "x\n"})
        rc, out = self.fx.check()
        self.assertEqual(rc, 0, out)
        self.assertIn("meta", out)

    def test_status_flip_sync(self):
        baseline(self.fx)
        self.fx.commit("P0001 tools: start work", {"plan/p0001-checker.md": p_entry(status="active")})
        rc, out = self.fx.check()
        self.assertEqual(rc, 1)
        self.assertIn("STATUS", out)
        self.fx.commit("P0001 tools: start work synced",
                       {"plan/p0001-checker.md": p_entry(status="active"),
                        "plan/STATUS.md": "# STATUS updated\n"})
        rc, out = self.fx.check()
        self.assertEqual(rc, 0, out)

    def test_agents_md_guard(self):
        baseline(self.fx)
        self.fx.commit("P0001 tools: tweak constitution", {"AGENTS.md": "# changed\n"})
        rc, out = self.fx.check()
        self.assertEqual(rc, 1)
        self.assertIn("规则5", out)
        self.fx.commit("D0002 plan: amend constitution",
                       {"AGENTS.md": "# changed2\n",
                        "plan/decisions/d0002-amend.md": D2})
        rc, out = self.fx.check()
        self.assertEqual(rc, 0, out)
        self.fx.merge("bm", "Merge P0001", {"AGENTS.md": "# via merge\n"})
        rc, out = self.fx.check()
        self.assertEqual(rc, 1)
        self.assertIn("规则5", out)

    def test_tree_violations(self):
        baseline(self.fx)
        self.fx.write("plan/p1-bad.md", p_entry())
        self.fx.write("plan/p0002-nostatus.md", "---\nid: P0002\ntitle: x\n---\n")
        self.fx.write("plan/p0001-dup.md", p_entry())
        self.fx.write("plan/p0003-sed.md", p_entry_status_only("P0003", "sedimented"))
        self.fx.write("plan/decisions/d0004-a.md",
                      D1.replace("D0001", "D0004").replace("accepted", "superseded")
                      .replace("replaces: []", "replaces: []")
                      .replace("superseded-by: []", "superseded-by: [D0005]"))
        self.fx.write("plan/decisions/d0005-b.md",
                      D1.replace("D0001", "D0005").replace("治理基线", "新决策"))
        _, out = self.fx.check()
        self.assertIn("命名须为 p<NNNN>", out)
        self.assertIn("缺字段：status", out)
        self.assertIn("编号重复", out)
        self.assertIn("status 非法", out)
        self.assertIn("互链", out)

    def test_audit_deletion(self):
        baseline(self.fx)
        self.fx.commit("P0002 tools: add entry",
                       {"plan/p0002-tmp.md": p_entry_status_only("P0002", "draft")})
        self.fx.remove("plan/p0002-tmp.md")
        self.fx.commit("P0001 tools: remove entry")
        rc, out = self.fx.check("--audit")
        self.assertEqual(rc, 1)
        self.assertIn("删除/改名", out)
        rc, out = self.fx.check()
        self.assertEqual(rc, 0, out)

    def test_decisions_ref_dangling(self):
        baseline(self.fx)
        self.fx.commit("P0001 tools: link decision",
                       {"plan/p0001-checker.md": p_entry().replace("decisions: []",
                                                                   "decisions: [D0009]")})
        rc, out = self.fx.check()
        self.assertEqual(rc, 1)
        self.assertIn("悬空", out)


class SupersedeTests(unittest.TestCase):
    def setUp(self):
        self.fx = Fixture()
        self.addCleanup(shutil.rmtree, self.fx.dir)
        baseline(self.fx)

    def write_pair(self, old_status="superseded", new_status="accepted",
                   successor="D0002", replaces="D0001"):
        old = D1.replace("status: accepted", f"status: {old_status}")
        old = old.replace("superseded-by: []", f"superseded-by: [{successor}]" if successor else "superseded-by: []")
        new = D2.replace("status: accepted", f"status: {new_status}")
        new = new.replace("replaces: []", f"replaces: [{replaces}]" if replaces else "replaces: []")
        self.fx.write("plan/decisions/d0001-governance-baseline.md", old)
        self.fx.write("plan/decisions/d0002-next.md", new)

    def check_error(self, message):
        rc, out = self.fx.check("--audit", "--strict")
        self.assertEqual(rc, 1, out)
        self.assertIn(message, out)

    def test_accepted_successor_passes_audit(self):
        self.write_pair()
        self.fx.commit("D0002 plan: supersede baseline", {
            "plan/STATUS.md": "# STATUS D0001 superseded, D0002 accepted\n",
        })
        rc, out = self.fx.check("--audit", "--strict")
        self.assertEqual(rc, 0, out)
        self.assertIn("SUMMARY E=0 W=0", out)

    def test_supersede_chain_passes_audit(self):
        self.write_pair()
        self.fx.commit("D0002 plan: supersede baseline", {
            "plan/STATUS.md": "# STATUS D0002 accepted\n",
        })
        new = D2.replace("status: accepted", "status: superseded")
        new = new.replace("replaces: []", "replaces: [D0001]")
        new = new.replace("superseded-by: []", "superseded-by: [D0003]")
        third = D1.replace("D0001", "D0003").replace("replaces: []", "replaces: [D0002]")
        self.fx.commit("D0003 plan: supersede next decision", {
            "plan/decisions/d0002-next.md": new,
            "plan/decisions/d0003-latest.md": third,
            "plan/STATUS.md": "# STATUS D0002 superseded, D0003 accepted\n",
        })
        rc, out = self.fx.check("--audit", "--strict")
        self.assertEqual(rc, 0, out)

    def test_missing_successor_rejected(self):
        self.write_pair(successor="")
        self.check_error("superseded 须回填 superseded-by")

    def test_dangling_successor_rejected(self):
        self.write_pair(successor="D9999", replaces="")
        self.check_error("superseded-by 悬空")

    def test_successor_missing_backlink_rejected(self):
        self.write_pair(replaces="")
        self.check_error("replaces 未回指 D0001")

    def test_predecessor_missing_backlink_rejected(self):
        self.write_pair(old_status="accepted", successor="")
        self.check_error("superseded-by 未回指 D0002")

    def test_predecessor_wrong_status_rejected(self):
        self.write_pair(old_status="accepted")
        self.check_error("replaces 目标 D0001 状态非 superseded")

    def test_dangling_predecessor_rejected(self):
        self.write_pair(old_status="accepted", successor="", replaces="D9999")
        self.check_error("replaces 悬空")


class ClosureTests(unittest.TestCase):
    def setUp(self):
        self.fx = Fixture()
        self.addCleanup(shutil.rmtree, self.fx.dir)
        baseline(self.fx)

    def check_ok(self, *args):
        rc, out = self.fx.check(*args)
        self.assertEqual(rc, 0, out)
        return out

    def check_error(self, message, *args):
        rc, out = self.fx.check(*args)
        self.assertEqual(rc, 1, out)
        self.assertIn(message, out)
        return out

    def write_entry(self, **kwargs):
        self.fx.write("plan/p0001-checker.md", p_entry(**kwargs))

    def commit_status(self, status, knowledge="pending", **kwargs):
        self.fx.commit("P0001 tools: update work status", {
            "plan/p0001-checker.md": p_entry(status, knowledge=knowledge, **kwargs),
            "plan/STATUS.md": f"# STATUS {status}\n",
        })

    def test_required_fields(self):
        for field in ("knowledge: pending\n", "followups: []\n"):
            with self.subTest(field=field):
                self.fx.write("plan/p0001-checker.md", p_entry().replace(field, ""))
                self.check_error(f"缺字段：{field.split(':')[0]}")

    def test_knowledge_enum(self):
        self.write_entry(knowledge="unknown")
        self.check_error("knowledge 非法")

    def test_terminal_pending_rejected(self):
        for status in ("completed", "abandoned"):
            with self.subTest(status=status):
                self.write_entry(status=status)
                self.check_error("knowledge 不得为 pending")

    def test_pending_draft_and_active_allowed(self):
        for status in ("draft", "active"):
            with self.subTest(status=status):
                self.write_entry(status=status)
                self.check_ok()

    def test_none_closes_without_docs(self):
        for status in ("completed", "abandoned"):
            with self.subTest(status=status):
                self.write_entry(status=status, knowledge="none")
                self.check_ok()

    def test_updated_and_covered_require_reference(self):
        for knowledge in ("updated", "covered"):
            with self.subTest(knowledge=knowledge):
                self.write_entry(status="completed", knowledge=knowledge)
                self.check_error("须至少有 docs/bench/decisions")

    def test_disposition_reference_alternatives(self):
        self.fx.write("docs/guide.md", "# 使用\n")
        self.fx.write("bench/results/2026-10-08-test/README.md", "证据\n")
        references = (
            {"docs": "[docs/guide.md#使用]"},
            {"bench": "[bench/results/2026-10-08-test]"},
            {"decisions": "[D0001]"},
        )
        for knowledge in ("updated", "covered"):
            for reference in references:
                with self.subTest(knowledge=knowledge, reference=reference):
                    self.write_entry(status="completed", knowledge=knowledge, **reference)
                    self.check_ok()

    def test_docs_paths(self):
        self.fx.write("docs/guide.md", "# 使用\n")
        for path in ("docs/missing.md", "docs/", "README.md", "../docs/guide.md",
                     "docs/../README.md", os.path.join(self.fx.dir, "docs/guide.md")):
            with self.subTest(path=path):
                self.write_entry(docs=f"[{path}]")
                self.check_error("docs")
        self.write_entry(docs="[docs/guide.md#使用]")
        self.check_ok()

    def test_docs_symlink_escape(self):
        self.fx.write("docs/guide.md", "说明\n")
        os.symlink("../README.md", os.path.join(self.fx.dir, "docs/escape.md"))
        self.write_entry(docs="[docs/escape.md]")
        self.check_error("docs")

    def test_bench_paths(self):
        self.fx.write("bench/results/2026-10-08-test/README.md", "证据\n")
        for path in ("bench/results/missing", "bench/results/2026-10-08-test/README.md",
                     "bench/results/../", "../bench/results/2026-10-08-test",
                     os.path.join(self.fx.dir, "bench/results/2026-10-08-test")):
            with self.subTest(path=path):
                self.write_entry(bench=f"[{path}]")
                self.check_error("bench")
        self.write_entry(bench="[bench/results/2026-10-08-test]")
        self.check_ok()

    def test_bench_symlink_escape(self):
        self.fx.write("bench/results/2026-10-08-test/README.md", "证据\n")
        os.symlink("../../plan", os.path.join(self.fx.dir, "bench/results/escape"))
        self.write_entry(bench="[bench/results/escape]")
        self.check_error("bench")

    def test_decision_reference_type(self):
        for ref in ("P0001", "D9999", "garbage"):
            with self.subTest(ref=ref):
                self.write_entry(decisions=f"[{ref}]")
                self.check_error("decisions 引用悬空或格式非法")
        for ref in ("D0001", "d0001", "0001"):
            with self.subTest(ref=ref):
                self.write_entry(decisions=f"[{ref}]")
                self.check_ok()

    def test_deferred_requires_followup(self):
        self.write_entry(status="completed", knowledge="deferred")
        self.check_error("须回填非空 followups")

    def test_followup_reference_type_and_self(self):
        for ref in ("P0001", "0001", "D0001", "P9999", "garbage"):
            with self.subTest(ref=ref):
                self.write_entry(followups=f"[{ref}]")
                self.check_error("followups")

    def test_deferred_with_real_followup(self):
        self.fx.write("plan/p0002-followup.md", p_entry().replace("P0001", "P0002"))
        for ref in ("P0002", "p0002", "0002"):
            with self.subTest(ref=ref):
                self.write_entry(status="completed", knowledge="deferred", followups=f"[{ref}]")
                self.check_ok()
        self.fx.write("plan/p0002-followup.md",
                      p_entry("completed", knowledge="none").replace("P0001", "P0002"))
        self.check_ok()

    def test_terminal_closing_required(self):
        for status in ("completed", "abandoned"):
            with self.subTest(status=status):
                self.write_entry(status=status, knowledge="none", closing=False)
                self.check_error("终态条目须含 `## 收尾`")

    def test_terminal_closing_fields_nonempty(self):
        text = p_entry("completed", knowledge="none")
        for marker in ("- 验收或停止依据：", "- 知识处置：", "- 剩余义务："):
            with self.subTest(marker=marker):
                lines = [marker + "   " if line.startswith(marker) else line
                         for line in text.splitlines()]
                self.fx.write("plan/p0001-checker.md", "\n".join(lines) + "\n")
                self.check_error(f"收尾记录缺失或为空：{marker}")

    def test_closing_fields_must_be_in_closing_section(self):
        for heading in ("## 其他", "# 其他"):
            with self.subTest(heading=heading):
                text = p_entry("completed", knowledge="none", closing=False)
                text += "\n## 收尾\n\n" + heading + "\n" + CLOSING.split("## 收尾", 1)[1]
                self.fx.write("plan/p0001-checker.md", text)
                self.check_error("收尾记录缺失或为空")

    def test_legal_completed_lifecycle_audit(self):
        self.commit_status("active")
        self.commit_status("completed", knowledge="none")
        self.check_ok("--audit", "--strict")

    def test_legal_abandoned_lifecycles(self):
        for initial in ("draft", "active"):
            with self.subTest(initial=initial):
                fx = Fixture()
                try:
                    baseline(fx)
                    if initial == "active":
                        fx.commit("P0001 tools: activate", {
                            "plan/p0001-checker.md": p_entry("active"),
                            "plan/STATUS.md": "# STATUS active\n",
                        })
                    fx.commit("P0001 tools: stop work", {
                        "plan/p0001-checker.md": p_entry("abandoned", knowledge="none"),
                        "plan/STATUS.md": "# STATUS abandoned\n",
                    })
                    rc, out = fx.check("--audit", "--strict")
                    self.assertEqual(rc, 0, out)
                finally:
                    shutil.rmtree(fx.dir)

    def test_draft_cannot_skip_to_completed(self):
        self.commit_status("completed", knowledge="none")
        self.check_error("draft → completed")

    def test_active_cannot_return_to_draft(self):
        self.commit_status("active")
        self.commit_status("draft")
        self.check_error("active → draft")

    def test_terminal_cannot_reactivate(self):
        self.commit_status("active")
        self.commit_status("completed", knowledge="none")
        self.commit_status("active")
        self.check_error("completed → active")
        self.commit_status("abandoned", knowledge="none")
        self.commit_status("active")
        self.check_error("abandoned → active")

    def test_unchanged_terminal_allows_record_update(self):
        self.commit_status("active")
        self.commit_status("completed", knowledge="none")
        self.fx.commit("P0001 tools: correct evidence pointer", {
            "plan/p0001-checker.md": p_entry("completed", knowledge="none") + "\n补充指针\n",
        })
        self.check_ok("--audit", "--strict")

    def test_completed_transition_requires_status_sync(self):
        self.commit_status("active")
        self.fx.commit("P0001 tools: close without status page", {
            "plan/p0001-checker.md": p_entry("completed", knowledge="none"),
        })
        self.check_error("未与 plan/STATUS.md 同一提交同步")

    def test_legacy_history_can_migrate(self):
        fx = Fixture()
        try:
            baseline(fx, legacy=True)
            fx.commit("P0001 tools: old active entry", {
                "plan/p0001-checker.md": legacy_entry("active"),
                "plan/STATUS.md": "# STATUS active legacy\n",
            })
            fx.commit("P0001 tools: old sedimented entry", {
                "plan/p0001-checker.md": legacy_entry("sedimented"),
                "plan/STATUS.md": "# STATUS sedimented legacy\n",
            })
            fx.commit("P0001 tools: migrate terminal entry", {
                "plan/p0001-checker.md": p_entry("completed", knowledge="none"),
                "plan/STATUS.md": "# STATUS completed\n",
            })
            rc, out = fx.check("--audit", "--strict")
            self.assertEqual(rc, 0, out)
        finally:
            shutil.rmtree(fx.dir)

    def test_new_fields_cannot_be_removed_and_restored(self):
        self.commit_status("active")
        self.fx.commit("P0001 tools: remove new fields", {
            "plan/p0001-checker.md": legacy_entry("active"),
        })
        self.fx.commit("P0001 tools: restore new fields", {
            "plan/p0001-checker.md": p_entry("active"),
        })
        self.check_error("不得删除", "--audit")

    def test_schema_downgrade_cannot_enter_sedimented(self):
        self.commit_status("active")
        self.fx.commit("P0001 tools: attempt legacy downgrade", {
            "plan/p0001-checker.md": legacy_entry("sedimented"),
            "plan/STATUS.md": "# STATUS sedimented\n",
        })
        self.commit_status("completed", knowledge="none")
        self.check_error("active → sedimented", "--audit")

    def test_merge_accepts_accumulated_legal_transitions(self):
        self.fx.git("checkout", "-q", "-b", "p0001-complete")
        self.commit_status("active")
        self.commit_status("completed", knowledge="none")
        self.fx.git("checkout", "-q", "main")
        self.fx.git("merge", "--no-ff", "--no-commit", "-q", "p0001-complete")
        self.fx.write("VERSION", "0.1.1\n")
        self.fx._commit("Merge P0001: complete work")
        self.check_ok("--audit", "--strict")

    def test_merge_cannot_hide_illegal_branch_transition(self):
        self.fx.git("checkout", "-q", "-b", "p0001-skip")
        self.commit_status("completed", knowledge="none")
        self.fx.git("checkout", "-q", "main")
        self.fx.git("merge", "--no-ff", "--no-commit", "-q", "p0001-skip")
        self.fx.write("VERSION", "0.1.1\n")
        self.fx._commit("Merge P0001: skip active")
        self.check_error("draft → completed", "--audit", "--strict")

    def test_merge_cannot_reactivate_terminal(self):
        self.commit_status("active")
        self.commit_status("completed", knowledge="none")
        self.fx.git("checkout", "-q", "-b", "p0001-reactivate")
        self.commit_status("active")
        self.fx.git("checkout", "-q", "main")
        self.fx.git("merge", "--no-ff", "--no-commit", "-q", "p0001-reactivate")
        self.fx.write("VERSION", "0.1.1\n")
        self.fx._commit("Merge P0001: reactivate terminal")
        self.check_error("completed → active")

    def test_current_tree_rejects_sedimented(self):
        self.write_entry(status="sedimented")
        self.check_error("status 非法")

    def test_modern_schema_cannot_enter_sedimented(self):
        self.commit_status("active")
        self.commit_status("sedimented")
        self.commit_status("completed", knowledge="none")
        self.check_error("active → sedimented", "--audit")


def p_entry_status_only(num, status):
    return p_entry(status).replace("id: P0001", f"id: {num}")


if __name__ == "__main__":
    unittest.main()
