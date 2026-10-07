---
id: P0001
title: 仓库纪律机械执法器 check_repo.py
status: active
created: 2026-10-07
updated: 2026-10-08
knowledge: pending
docs: []
bench: []
decisions: [D0001, D0003]
followups: []
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
