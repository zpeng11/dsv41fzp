---
id: P0002
title: 工作项交付结项与知识处置解耦
status: completed
created: 2026-10-08
updated: 2026-10-08
knowledge: covered
docs: []
bench: []
decisions: [D0003]
followups: []
---

## 意图

落实 D0003 的结项契约：主状态表达交付与验收是否完成，知识去向独立分类；保留收尾义务，避免为结项制造知识文档。

## 验收

- [x] AGENTS.md 与模板采用 completed 终态、knowledge / followups 字段和收尾结构，并明确簿记不构成验收授权。
- [x] 校验器检查知识枚举、终态处置、收尾结构、引用及 P 状态迁移；合法生命周期与违规两侧均有测试。
- [x] 已有 P0001 仅迁移字段，不自动结项；D0001、D0002 保留原文，历史审计兼容旧 P 状态。
- [x] 规则测试、本仓库严格历史审计、变更空白检查通过；不涉及 CI、manifest 或其他校验器缺陷。

## 记录

实现与测试范围仅限工作项结项契约；确定性的格式和引用检查交给工具，验收充分性、知识增量及移交是否阻塞交付仍由评审判断。

### 规则↔条文映射（D0003 扩展）

收尾义务源于 AGENTS.md「工作项收尾」，迁移兼容源于 D0003；下表锚点与工具输出一致，统一使用 AGENTS:plan条目。

| 规则 | 锚点 | 级别 |
| --- | --- | --- |
| P 状态枚举、knowledge / followups 必填及知识枚举 | AGENTS:plan条目 | E |
| completed / abandoned 禁止 pending，须有收尾章节内三项非空记录 | AGENTS:plan条目 | E |
| updated / covered 至少有 docs / bench / decisions 之一引用；deferred 须 followups | AGENTS:plan条目 | E |
| docs 文件、bench 目录存在且不逃逸；decisions 指向 D，followups 指向 P 且不自指 | AGENTS:plan条目 | E |
| 普通提交检查直接迁移边；merge 检查累积差异可达性，分支逐提交纪律由 range / audit 检查 | AGENTS:plan条目 | E |
| 旧字段结构沿用历史合法边，新字段一经引入不得删除；允许 sedimented → completed 迁移 | AGENTS:plan条目 | E |

### 检查边界

- 收尾只检查章节与非空记录，不以记录内容代替验收授权或证明结论成立。
- docs 可带章节引用，但仅检查文件存在，不校验章节标题。
- 引用继续按当前树解析，不扩展为逐提交时点校验；旧字段结构判定仅用于兼容历史迁移边。
- 移交是否阻塞交付、接收项是否真实承担义务、none / covered 的理由是否充分仍属评审。
- 用户审阅确认后按既有 git 纪律入库：D0003 宪法修订直接提交 main（`390c404`），P0002 实现分支提交（`c6bf346`）以 --no-ff 合入；merge 本体递增 VERSION 至 0.1.2 并同步本条目与 STATUS 的结项，不携带 AGENTS.md 变更。

## 收尾

- 验收或停止依据：2026-10-08，用户明确确认已 review 并授权提交合入。本地 macOS arm64，运行 `python3 tests/tools/test_check_repo.py`，49 项测试通过；`python3 tools/check_repo.py --audit --strict` 为 E=0 W=0；实现分支 `--range main..HEAD --strict` 为 E=0 W=0；`git diff --check` 通过。临时克隆已验证修宪、实现合入及结项流程，实际提交按上述拆分执行。未验证目标机柜；本项不涉及引擎性能与数值精度。
- 知识处置：covered；用户批准的结项契约已由 D0003 与 AGENTS.md 表达，本项只保留规则映射与实现证据，不新增重复的 docs 正文。
- 剩余义务：无。本次约定的修订、验证与入库已完成，没有转移至后续 P 的知识义务；CI、bench manifest 及其他既存缺陷不属于本项范围。
