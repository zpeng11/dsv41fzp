"""check_repo.py 的规则级测试：每个用例在临时 git 仓库里构造受治/违规两侧。

运行：python3 tests/tools/test_check_repo.py
"""

import os
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


def p_entry(status="draft", docs="[]"):
    return f"""---
id: P0001
title: 工作项
status: {status}
created: 2026-10-07
updated: 2026-10-07
docs: {docs}
bench: []
decisions: []
---

正文
"""


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


def baseline(fx):
    fx.commit("chore: initial import", {"README.md": "x\n", "VERSION": "0.1.0\n"})
    fx.commit("D0001 plan: governance baseline", {
        "plan/decisions/d0001-governance-baseline.md": D1,
        "plan/p0001-checker.md": p_entry(),
        "plan/STATUS.md": "# STATUS\n",
        "AGENTS.md": "# AGENTS\n",
    })


class CheckRepoTests(unittest.TestCase):
    def setUp(self):
        self.fx = Fixture()

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
        self.assertIn("sedimented", out)
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


def p_entry_status_only(num, status):
    return p_entry(status).replace("id: P0001", f"id: {num}")


if __name__ == "__main__":
    unittest.main()
