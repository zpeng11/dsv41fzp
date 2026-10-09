# STATUS — 状态页

状态信息的唯一归宿（状态不进 docs，README 只做路由）。本页保留当前能力、在办工作与阻塞项；过程与验收证据回到原 P/D 条目，不在此复制历史流水账。

## 当前阶段

**开发阶段初期**。目录骨架、模型/硬件事实文档与治理检查工具已有；推理引擎、FP32 oracle、构建、CI、部署器和硬件资质检查器尚未实现。当前没有目标机验证或引擎性能基线，不处于密集实现或可部署阶段。

## 当前工作项

| 工作项 | 状态 | 当前进度或阻塞 |
| --- | --- | --- |
| [P0005 — 部署资质检查与首台验证](p0005-hardware-qualification.md) | draft | 承接 P0004 的硬件预检、部署接线与实机验证义务；缺目标机与 CUDA 探针构建环境，未启动 |
| [P0006 — ref 格式 oracle](p0006-ref-format-oracle.md) | draft | 范围与验收已确认，仅完成计划登记；按用户要求暂不开始实现，尚无数值验收证据 |
| [P0007 — ref/ 触碰提醒收窄](p0007-ref-touch-scope.md) | completed | 用户确认结项；新建 ref/ 文件不再提醒，既有 oracle 文件的修改与删除仍为 W，knowledge: covered |
| [P0008 — 审计 W 计数范围](p0008-audit-history-w.md) | active | 检查器修订与规则测试已完成，待审阅后提交合入 |

## 开发入口与缺口

- 硬件支持范围及资质门槛：[docs/HARDWARE.md](../docs/HARDWARE.md)；支持方向：[D0004](decisions/d0004-gpu-p2p-support.md)。PCIe 5.0 RTX PRO 6000 的支持以逐对双向 P2P 与正确性验收为条件。
- 数值实现须先建立 [ref/](../ref/) 的验证根，再推进 quant 与计算路径；首个格式 oracle 工作项为 [P0006](p0006-ref-format-oracle.md)，当前仅规划，按用户要求暂不开始实现。其 CPU 验收不依赖 P0005 的目标机或 C++ 构建；模型契约见 [docs/MODEL.md](../docs/MODEL.md)。
- 构建与 bench 工作流仍待定，CI 未接线；硬件资格与性能阈值尚无目标机证据。
