---
id: D0002
title: 机械执法器 check_repo.py 入宪——规则子集、六项决议与演进规范
status: accepted
created: 2026-10-07
updated: 2026-10-07
replaces: []
superseded-by: []
---

## 背景与问题

D0001 决策驱动之三要求"宪法条文能被 check 工具执行，而非停留在 prose"，但该工具在其"后果"节中处于无主承诺状态（README tools 清单与 plan 均未追踪）。密集开发期在即，规则若不可机检，纪律将随提交量退化为纯评审。

## 决策驱动

- 零语义边界：只机械校验无需理解论断内容的条文（格式、引用存在性、义务 trailer），需要语义判断的（性能论断识别、docs 数字测量环境、comment 风格）永远留给评审；
- 误报优先修复：规则本身可改，不为绕过误报开逃生门；
- 治理检查与机柜解耦：纯标准库 + git，任何 runner 可执行，为非机柜 CI job 预留契约（exit code + `--range`）。

## 备选方案

- commit-msg hook 快速失败：反馈最快，但 hook 安装是用户侧状态、不可审计，CI 兜底不可省——留 v2 接口；
- 豁免 trailer（`Governance-Exception:`）：否决——豁免机制本身是宪法级变更，遇真实误报应修规则；
- 编号引用逐 commit 时点校验（防"先提交后补条目"绕过）：成本可接受但当前无绕过实证，按测量驱动原则推迟，出现首次绕过后再启用（见 P0001 局限节）。

## 结论

采纳 `tools/check_repo.py` v1（规格、规则↔条文映射与测试见 `plan/p0001-check-repo.md`），宪法工作流节增设「机械执法」条：CI 与每次合入前必跑，治理起点（首个 D 条目入库 commit，即 D0001）之前提交豁免。六项运行决议：① VERSION bump 挂 merge 本体，非合并改动为 E，首建豁免；② AGENTS.md 触碰须 `D<NNNN> <area>:` subject 直接提交 main，merge 携带宪法变更亦为 E（规则 5 的机器投影）；③ 编号引用按当前树判存在；④ 编号允许跳号，只查唯一与不回收；⑤ 工具演进规范——规则增删/严重级调整 = P 条目，缺陷修复 = `meta:`；⑥ subject 单编号，跨域改动选主域。

## 后果

- bench manifest 校验与冻结区 hash 比对暂缓：格式待宪法「bench」工作流定案，届时随 P 条目开启；
- 本决策 + P0001 分支合入构成 git 工作流的首个完整样例（D 修宪直接入 main，P 分支 `--no-ff` 合入并在 merge 本体 bump VERSION）。
