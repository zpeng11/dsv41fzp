---
id: D0001
title: 治理基线——引擎版本方案、plan 编号、git 规范
status: accepted
created: 2026-10-07
updated: 2026-10-07
replaces: []
superseded-by: []
---

## 背景与问题

仓库自骨架期进入密集开发期，AGENTS.md 三处 `待填`（编号格式、git 工作流、引擎版本方案）阻塞宪法第 1 条执行；版本方案按宪法要求必须在第一行性能数字落盘前确定。参照系为 Strata（平稳期单模型推理引擎）：其技术溯源链成熟，但过程治理缺失，成熟期积累大量悬空引用（plan 文档删除后 CMakeLists 残留约 15 处 `plan v0.3 P*` 锚点、7 篇被引设计文档缺失、oracle 向量表生成脚本丢失）。

## 决策驱动

- 每行性能数字可 pin 住：版本 + git sha + bench 目录三元组齐全；
- 编号六个月后仍可解析：防止 Strata 式悬空锚点；
- 规则可机械校验：宪法条文能被 check 工具执行，而非停留在 prose；
- 密集开发期仪式成本趋近于零：单人参于 + agent 执行，规则必须一条写清、无解释空间。

## 备选方案

- **版本**：SemVer 全量语义（`0.x` 期兼容语义无从谈起，否决）；CalVer（与逐合并递增冲突，留作 1.0 时刻选项）；llama.cpp 式单调构建号（人类不可读，但其"pin 构建号 + 模型 + 量化才能复现"的纪律被采纳进 bench manifest 要求）；采纳 Strata 的单一真源 + 注入编译定义 + 启动日志打印版本机制，真源从 CMake 改为根目录 `VERSION` 纯文本（Python 工具与 CI 免跑 CMake 即可读）。
- **编号**：层级阶段编号（Strata `P0.1`/`P1.T3`/`E-6` 反例——阶段重排、文档删除后锚点悬空，否决）；ADR/MADR 模板（采纳其不可变与备选方案记录纪律）；Kubernetes KEP（整体过重，采纳其 YAML front matter 机器可读性）。
- **git**：Conventional Commits 全套（feat/fix 类型学对本仓库过粗，发版自动化无对象，只采纳 trailer 机制）；gitflow 与发布集成分支（多端协作形态，否决）；trunk-based 采纳——宪法「优化路径必须带运行时开关」即其前提条件。

## 结论

三件套落入 AGENTS.md 工作流节：根目录 `VERSION` 单一真源，`0.<minor>.<patch>`，逐合并递增 patch，量化/pack 格式版本独立为 `src/quant` 常量；`P<NNNN>` / `D<NNNN>` 扁平编号、ID 进文件名、原地归档永不删除永不复用；trunk-based + 短命分支 + `--no-ff`，subject `<ID> <area>: <论断句>`，义务表落 trailer（`Bench:` / `Oracle:`），`meta:` 仅限不改行为与论断的维护提交。

## 后果

- 本提交起宪法第 1 条可执行；后续 `tools/check_repo.py` 可校验：commit 引用编号存在且状态合法、bench manifest 的 `version` 与 `VERSION` 一致。
- minor 里程碑语义与 1.0 时刻（激活兼容承诺、是否转 CalVer）留待对应 plan 条目决策。
- 本文件自身按新规范以 `D0001` 被引用，作为机制的首个样例。
