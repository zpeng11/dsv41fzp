---
id: P0001
title: 仓库纪律机械执法器 check_repo.py
status: completed
created: 2026-10-07
updated: 2026-10-08
knowledge: covered
docs: []
bench: []
decisions: [D0001, D0002, D0003]
followups: [P0003]
---

## 意图

把宪法（AGENTS.md）工作流与禁区条文中零语义判定的子集交给工具执法，使 D0001 决策驱动之三（"规则可机械校验，而非停留在 prose"）落地为 exit code。执法范围刻意收窄：需要理解论断内容的检查（性能论断识别、docs 数字测量环境、comment 风格）永远不进工具，属评审域。

## 验收

- [x] `tools/check_repo.py` 纯标准库 + git，覆盖下表 v1 规则集
- [x] `tests/tools/test_check_repo.py` 以临时 git 仓库 fixture 验证每条规则的通过/违例两侧（16 用例）
- [x] 本仓库实跑 E=0（epoch 未入库时优雅降级为 W 提示，符合设计）
- [x] 按宪法 git 工作流入库（D 修宪 → P 分支合入）

## 记录

### 设计决议（2026-10-07 讨论定案）

1. **VERSION bump 挂 merge 本体**：`--no-ff --no-commit` 后改 VERSION 再提交，历史中 bump 与 merge 一一对应；非合并提交改 VERSION 为 E（首建豁免——VERSION 允许由治理基线 commit 首次入库）。
2. **AGENTS.md 触碰须 `D<NNNN> <area>:` subject**：宪法修订立 D 并直接提交 main，merge 携带宪法变更亦为 E——规则 5 的机器投影。
3. **编号引用按当前树判存在**：不校验时点（防绕过优先级低于先跑起来），出现首次真实绕过后再升级。
4. **编号允许跳号**：只查唯一与不回收，不查连续，避免"为凑连续而废号"的反向激励。
5. **工具演进规范**：规则增删/严重级调整 = P 条目（改变执法论断，须验收）；缺陷修复 = `meta:`。
6. **subject 单编号**：跨域改动选主域，不引入多编号语法。

### 规则↔条文映射（v1）

| # | 规则 | 锚点 | 级别 |
| --- | --- | --- | --- |
| 1 | plan 条目命名/front matter 必填项/status 枚举/日期 | AGENTS:plan条目 | E |
| 2 | 编号唯一；条目文件历史中从未删除/改名（audit） | AGENTS:plan条目 | E |
| 3 | subject 三态格式（`<ID> <area>:` / `meta:` / `Merge P<NNNN>`）；area ∈ src 子目录（动态读取）∪ 仓库级域名 | AGENTS:git | E |
| 4 | subject 引用编号存在 | D0001 | E |
| 5 | src/kernels、src/quant、include/df 变更 → `Oracle:` trailer 必带且路径存在 | AGENTS:git 义务表 | E |
| 6 | `Bench:` trailer 须为存在的 `bench/results/<日期-主题>` 目录 | AGENTS:git 义务表 | E |
| 7 | VERSION：非合并不得改（首建豁免）；merge 须 patch+1 或 minor+1&patch=0 | AGENTS:版本 | E |
| 8 | third_party/reference/ 触碰须同提交更新 PROVENANCE.md | AGENTS:禁区 | E |
| 9 | 条目状态翻转 ↔ plan/STATUS.md 同提交（新建 stub 不算翻转） | AGENTS:plan条目 | E |
| 10 | D 的 supersede 互链一致（A.superseded-by=B ⟺ B.replaces 含 A）；P 的 decisions 引用不悬空 | AGENTS:plan条目 | E |
| 11 | meta: 提交触碰源码；ref/ 触碰；bench/prompts、tests/data 触碰（manifest 重算提醒） | AGENTS:git/禁区 | W |

### 边界与已知局限

- 治理起点（epoch）= 首个 d<NNNN> 条目入库 commit，自动发现零硬编码；之前提交豁免。
- trailer 只认消息末段（git interpret-trailers 语义）。
- bench manifest 与字节冻结 hash 校验**未启用**：格式待宪法「bench」工作流定案后随 P 条目开启；冻结区目前仅 W。
- 「src 变更必经 merge 落 main」需要第一父链审计视角，留 v1.5 的 `--audit` 扩展。
- `--range A..B` / exit code 契约面向未来非机柜 CI job（治理检查与机柜解耦）。

## 收尾

- 验收或停止依据：用户于 2026-10-08 明确确认修复合法决策替代误报并完成 P0001。v1 已随实现提交 `81225e3` 与合并提交 `835cc5a` 入库；P0002 / D0003 已补齐结项契约。本次 P0003 补充 accepted 替代者、连续替代链及既有拒绝行为的回归测试，修复前两个合法用例均复现误报，修复后全部 57 项测试通过。本地 macOS arm64，运行 `python3 tests/tools/test_check_repo.py`、`python3 tools/check_repo.py --audit --strict` 和 `git diff --check`，结果分别为通过、E=0 W=0、通过。用户随后确认修复完成并授权结项、提交、合入与推送；本次修复与本条目、STATUS 的结项同步记录于 P0003 修复提交。不涉及目标机柜性能或数值精度验证。
- 知识处置：covered；D0001 定义治理基线，D0002 记录执法范围与演进纪律，D0003 定义结项契约；v1 规则映射与已知局限保留在本条目，后续结项校验扩展见 P0002，不新增重复的 docs 正文。
- 剩余义务：P0003 承担本次修复与结项变更的实际入库跟踪；实现验证已通过。CI 接线、bench manifest / 冻结区 hash、第一父链审计是已声明的后续范围，不是 P0001 v1 验收义务，结项不表示这些功能已实现。
