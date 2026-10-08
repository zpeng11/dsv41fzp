---
id: D0004
title: GPU 互联不再强制 NVLink，纳入可验证 PCIe 5.0 P2P 的 RTX PRO 6000
status: accepted
created: 2026-10-08
updated: 2026-10-08
replaces: []
superseded-by: []
---

## 背景与问题

用户明确要求研究 PRO 6000 的支持并取消 NVLink 必选。旧硬件文档将 RTX PRO 6000 Blackwell Server 写成 NVLink 桥接卡，同时排除无 NVLink 的 PCIe 卡，导致型号支持与目标形态相互矛盾。规格核查及部署资格的权威内容归入 docs/HARDWARE.md，由 P0004 推进同步修订。

## 决策驱动

- 保持单机柜、双路 CPU 与同型号多 GPU 的项目边界，不扩展为多机或通用推理引擎。
- 以可验证的 GPU 间访问与数据正确性约束部署，不将互联品牌或 PCIe 接口规格当作资格证明。
- 厂商规格、设计支持范围与实机认证分别表述；互联改变不构成性能等价论断。
- 遵守现有知识归档与生命周期规则，当前状态不再复述已完成治理过程。

## 备选方案

- 保持 NVLink 必选并移除 PRO 6000：不满足用户所需的支持方向。
- 只删 NVLink 要求，允许任意 PCIe 卡或仅凭型号放行：无法保证 CUDA P2P 与整机正确性，不采纳。
- NVLink 可选，允许白名单内的 PCIe 5.0 PRO 6000；部署须通过整机 P2P 资格检查：采纳。

## 结论

依据用户本次明确指示，取消仓库定义中的 NVLink 必选，将 RTX PRO 6000 Blackwell 系列的 PCIe 5.0 P2P 配置纳入硬件支持范围，同时保留既有 A100/A800 SXM 路径。具体型号、拓扑与部署资质门槛见 docs/HARDWARE.md，不在本决策重复规格或断言未经测量的吞吐。

AGENTS 仅修订开篇硬件形态并继续路由 HARDWARE；治理执法条款不变。本决策新增硬件范围约束，不整体取代 D0001—D0003，故不设置 replaces。支持方向的接受不等于 P0004 的文档验收，也不等于任何目标机已获认证。

## 后果

- platform 与 remote_experts 的互联描述以 GPU P2P 为抽象，不再假设 NVLink 桥接对或任意 SXM 机器自带 NVSwitch。
- 放置与搬运必须服从目标机拓扑及实测矩阵，不能把 PCIe 与 NVLink 当作性能等价路径。
- P0004 交付规范修订；后续硬件资质检查与目标机实验须由接收工作项跟踪，当前不新增没有测量证据的 bench 记录。
- STATUS 仅保留当前开发能力、在办工作与阻塞项；已完成治理工作的验收证据保留在原 P/D 条目。
