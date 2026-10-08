---
id: P0004
title: 研究 RTX PRO 6000 并将硬件支持条件改为可验证的 GPU P2P
status: completed
created: 2026-10-08
updated: 2026-10-08
knowledge: updated
docs: [docs/HARDWARE.md, docs/MODEL.md]
bench: []
decisions: [D0004]
followups: [P0005]
---

## 意图

按用户本次明确指示，核实 RTX PRO 6000 Blackwell 的互联能力，消除 NVLink 必选与 PRO 6000 桥接对的错误前提，使具备可验证 PCIe 5.0 P2P 的 PRO 6000 机柜纳入支持范围。同步仓库定义、模块描述及当前状态，把 STATUS 恢复为开发阶段初期的简洁入口，不删除已归档治理记录。

本项交付研究依据与支持契约修订，不交付推理引擎、硬件预检程序或目标机性能认证。厂商规格与项目资格门槛必须和实测结果分开；无目标机时不得宣称已经通过部署资质。

## 验收

- [x] HARDWARE 引用可核对的厂商与 CUDA 资料，纠正 PRO 6000 NVLink 描述，区分各版规格与整机 P2P 资格。
- [x] 全局取消 NVLink 必选，保留已有 NVLink 机型；PCIe PRO 6000 的型号范围、拓扑、逐对访问与正确性验证要求明确。
- [x] AGENTS、README 与 MODEL 的传输前提一致；STATUS 表达开发阶段初期，不再复制过时治理实施流水账。
- [x] 治理检查、既有测试与 diff 空白检查通过；未验证部分及后续义务有明确记录。
- [x] 用户评审确认研究结论与文档修订；用户于 2026-10-08 确认审计完成并授权提交合入，本项据此结项，不以工具通过自动结项。

## 记录

- 2026-10-08：用户明确授权新建并推进研究与修改计划、取消 NVLink 必选、纳入 PCIe 5.0 P2P PRO 6000，以及清理 STATUS 的过时治理叙事。AGENTS 的变更仅同步硬件形态，不改变治理规则。
- 只读检查确认引擎、oracle、部署器及硬件资质检查器均未实现；没有需要移除的运行时 NVLink 拒绝逻辑。既有 P0001—P0003 与 D0001—D0003 原地保留。
- 外部研究仅使用公开资料，不上传仓库内容；事实、来源与资格契约归档至 docs/HARDWARE.md，决策理由另立 D，不在本项复制规格表。

## 收尾

- 验收或停止依据：用户于 2026-10-08 确认审计完成并授权提交合入，研究依据、支持契约及 AGENTS/README/MODEL 的传输前提修订已完成。本地 macOS arm64，执行 `python3 tools/check_repo.py --audit --strict`、`PYTHONDONTWRITEBYTECODE=1 python3 tests/tools/test_check_repo.py` 与 `git diff --check`，分别为 E=0 W=0、57 项全部通过、通过；结项状态与 STATUS 同提交同步。该验收仅覆盖文档与治理检查，不代表引擎可运行或目标机已获部署认证。
- 知识处置：updated；厂商规格、来源与整机 GPU P2P 资格契约归档于 docs/HARDWARE.md，模型 K-M1 的传输前提更新于 docs/MODEL.md，支持方向与理由保留于 D0004；不新增无实测证据的 bench 记录。
- 剩余义务：P0005 接收硬件资质检查器、部署预检与首台目标机实测义务，保持 draft。这些执行与认证不属于本项研究及文档交付范围，不阻塞本项结项；在 P0005 验收前不得宣称整机已获部署资质。
