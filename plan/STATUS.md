# STATUS — 状态页

状态信息的唯一归宿（状态不进 docs，README 只做路由）。本页保留当前能力、在办工作与阻塞项；过程与验收证据回到原 P/D 条目，不在此复制历史流水账。

## 当前阶段

**开发阶段初期，当前推进 P0006 格式 oracle。** 目录骨架、模型/硬件事实文档与治理检查工具已有；数值代码尚未实现，没有格式验收证据、目标机验证或引擎性能基线，不处于可部署阶段。

## 当前工作项

| 工作项 | 状态 | 下一步或阻塞 |
| --- | --- | --- |
| [P0006 — ref 格式 oracle](p0006-ref-format-oracle.md) | active | 已授权开工；下一步实现 `ref/` 格式慢参照与 `tests/quant/` 独立测试，见下方编码目标；P0007 前置已满足 |
| [P0005 — 部署资质检查与首台验证](p0005-hardware-qualification.md) | draft | 缺目标机与 CUDA 探针构建环境，暂未启动；不阻塞 P0006 的 CPU 验收 |

## 下一步编码目标：P0006

- 实现入口：`p0006-ref-format-oracle` 短命分支；范围与验收以 [P0006](p0006-ref-format-oracle.md#验收) 为准。
- 首先核实格式变体、特殊值与零符号、FP32 运算及 BF16 舍入依据，再实现 E2M1、E4M3FN、E8M0 与 BF16 位模式 codec、E2M1 打包/展开及精确整数格。
- 编写码点穷举与独立预期值测试，再覆盖给定 scale 的逐行分组及二维块反量化、布局边界、错误输入与 BF16 落点；提供一条不依赖 CUDA、TileLang 或模型权重的 CPU 验收命令。
- 必要格式知识归档到 `docs/QUANT.md` 并由 `docs/MODEL.md` 路由，内容性写入仍须用户确认；知识未归档、验收未成立前保持 active，不推进生产 `src/quant` 或宣称 GPU/kernel parity。

## 其他入口与缺口

- 模型与验证链：[docs/MODEL.md](../docs/MODEL.md)；硬件支持与资质：[docs/HARDWARE.md](../docs/HARDWARE.md)、[D0004](decisions/d0004-gpu-p2p-support.md)。
- 构建与 bench 工作流仍待定，CI 未接线；硬件资格与性能阈值仍缺目标机证据，不作为 P0006 格式实现的前置条件。
