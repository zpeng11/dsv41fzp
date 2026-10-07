---
id: D0003
title: 工作项以交付验收结项，知识去向独立处置
status: accepted
created: 2026-10-08
updated: 2026-10-08
replaces: []
superseded-by: []
---

## 背景与问题

原 P 状态机以 sedimented 为正常终态，要求结论进入 docs。该设计防止知识留在临时计划中，却将交付验收与一种知识归档方式绑定：无知识增量的修复可能被迫新增文档，已交付工作也可能仅因等待归档而长期 active。P0001 已合入但仍 active，是需要区分交付与收尾的现有样例，不据此自动认定其验收或知识处理已经完成。

## 决策驱动

- 主状态明确表达工作是否完成，不把合并、测试通过或写入文档单独当作验收成立。
- 结项必须交代知识去向，但不得为了结项制造重复或无价值文档。
- 放弃的工作同样可能产生有价值知识，须处理停止原因、已有结果与剩余义务。
- 沿用 docs、D、bench 的职责划分；格式与引用由工具检查，结论充分性由评审判断。
- 免确认簿记不构成验收授权，不允许 agent 自行改变判据或把推测批准成事实。

## 备选方案

- 保留 sedimented 并要求每项工作写入 docs：否决，会混淆开发与归档阻塞，并激励文档膨胀。
- 增加 completed 后不约束知识收尾：否决，会让知识整理长期成为低优先级的无主承诺。
- 引入完整的交付状态机与知识状态机：当前过重，否决。
- 主状态描述工作生命周期，knowledge 分类处置结果，followups 承接移交：采纳。

## 结论

用户在本次对话明确批准该方案，修订 AGENTS.md 与工作项模板，由 P0002 配套实现校验与测试。

P 状态为 draft / active / completed / abandoned：draft 可进入 active 或 abandoned，active 可进入 completed 或 abandoned，终态不得重新激活。completed 要求约定交付到位、验收成立、知识与剩余义务有明确处置。

P 必填 knowledge 与 followups。knowledge 为 pending / updated / covered / none / deferred；两种终态均不得 pending，且须有验收或停止依据、知识处置、剩余义务三项收尾记录。updated / covered 至少有一项 docs / bench / decisions 引用；所有已填写的 docs / bench / decisions / followups 引用均须可解析，不因 draft 或 pending 豁免，尚未产生的目标写在意图或记录中。none 须具体说明无知识增量的理由；deferred 仅适用于不阻塞交付的整理，须接收 P 引用与理由。followups 不得自指；移交记录保留，义务是否完成看接收项。

长期事实、契约与操作知识进入 docs，决策理由保留在 D，实验原始证据保留在 bench，P 只留过程、验收证据与归档指针。工具不判断无增量、既有覆盖、移交或验收是否充分。

## 后果

- 当前 P 条目补齐新字段，不自动结项 P0001，也不为其新增知识文档。
- 当前树不再使用 sedimented；旧历史允许 sedimented → completed 一次迁移，不改写旧提交，不要求旧提交补齐新字段。提交两侧均未引入 knowledge / followups 时沿用旧合法迁移边；任一侧出现新字段即采用新规则，已引入的字段不得删除，以防降级绕过。
- D0001、D0002 的原文保留为历史依据。本决策仅调整 P 结项契约，不整体取代原决策，故不设置 replaces。
- 本次不接 CI、不实现 bench manifest、不修订 D 状态机、不修复其他既存校验器缺陷。
- 内容性写入仍须确认；状态记录免二次确认不授权修改验收条件、接受新决策或批准事实正文。
