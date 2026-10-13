---
id: P0015
title: 按 D0011 将 docs 与 plan 收敛为单一来源并压缩已结项 P 记录
status: active
created: 2026-10-13
updated: 2026-10-13
knowledge: pending
docs: []
bench: []
decisions: [D0011]
followups: []
---

## 意图

落实 D0011：消除 docs、plan 与 STATUS 之间的重复正文，改模板约束后续写法，并压缩已结项 P 的记录。成功标准是每个事实只有一处正文，删除内容均有唯一保留位置，既有引用全部可解析。只改 `docs/` 与 `plan/`，不动代码、`ref/`、冻结区与 AGENTS.md。

## 验收

- [ ] **模板**：`plan/TEMPLATE.md` 的「记录」按 D0011 结论 2 固定结构并给软上限。
- [ ] **覆盖矩阵**：MODEL §4 新增验证链覆盖矩阵；QUANT、COMPUTE、ACTQUANT 的否定式边界清单改为链接。
- [ ] **具名环境**：BUILD §1 定义实测环境名称；docs 与被整理的 P 中重复环境串改为引用。
- [ ] **契约去重**：QUANT §6–§7、COMPUTE §6、ACTQUANT §4–§5 中重复的 MXCSR 段落与验收方法细节收敛到唯一位置；ACTQUANT §5 只留复现入口与方法要点。
- [ ] **STATUS**：按 D0011 结论 5 重写。
- [ ] **已结项 P**：按 D0011 结论 3 压缩，至少覆盖 P0009–P0014；front matter 与 `## 收尾` 语义不变，记录内留压缩前提交指针。
- [ ] **不丢事实**：逐页列出删除段落及其保留位置，供人工核对；带测量的数字仍附环境。
- [ ] **引用可解析**：全部 Markdown 内部链接与锚点经一次性脚本核对无断裂；`check_repo.py --strict` 与 `--range` 均 E=0 W=0 I=0。
- [ ] **人工收尾**：用户评审通过后结项。

## 记录

### 范围与依据

- 依据 [D0011](decisions/d0011-doc-single-source.md)；规模数据来自 2026-10-13 会话审议（`wc -m` 统计，`docs/` 约 4.6 万、`plan/` 约 9.7 万字符）。
- 范围外：已 accepted 的 D、AGENTS.md、`check_repo` 扩展、代码与测试。
- 提交直接进 main，subject 用 `P0015 plan:` 与 `P0015 docs:`；不经合并，不递增 VERSION。
- 2026-10-13：用户审议通过 D0011 与本项范围并授权推进；D0011 accepted，本项 active。

## 收尾

- 验收或停止依据：尚未启动。
- 知识处置：pending；预计 updated，归档目标为 MODEL §4、BUILD §1 与模板。
- 剩余义务：待定。
