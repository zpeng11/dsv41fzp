# STATUS — 状态页

状态信息的唯一归宿（状态不进 docs，README 只做路由）。本页保留当前能力、在办工作与阻塞项；过程与验收证据回到原 P/D 条目，不在此复制历史流水账。

## 当前阶段

**开发阶段初期，P0006 格式 oracle 已通过用户评审与本地 CPU 测试，必要知识已归档，待合入结项。** 目录骨架、模型/硬件事实文档与治理检查工具已有；没有目标机验证或引擎性能基线，不处于可部署阶段。

## 当前工作项

| 工作项 | 状态 | 下一步或阻塞 |
| --- | --- | --- |
| [P0006 — ref 格式 oracle](p0006-ref-format-oracle.md) | active | 用户评审与必选 CPU 测试通过，必要知识已归档到 docs/QUANT.md；待合入提交同步结项 |
| [P0005 — 部署资质检查与首台验证](p0005-hardware-qualification.md) | draft | 缺目标机与 CUDA 探针构建环境，暂未启动；不阻塞 P0006 的 CPU 验收 |

## P0006 交付入口与待办

- 实现分支：`p0006-ref-format-oracle`；代码入口为 `ref/formats.py` 与 `ref/dequant.py`，范围与验收以 [P0006](p0006-ref-format-oracle.md#验收) 为准。
- 复现命令：`python3 tests/quant/test_formats.py`，仅标准库；覆盖码点穷举、E2M1 打包与整数格、FP32/BF16 舍入、给定 scale 的逐行及二维块反量化和错误输入。环境与运行证据见 P0006 记录。
- 数值契约已归档到 [docs/QUANT.md](../docs/QUANT.md)，由 docs/MODEL.md 路由；用户已确认评审通过。当前保持 active，合入时同步结项；不宣称生产 `src/quant`、GPU/kernel parity 或端到端验证完成。

## 其他入口与缺口

- 模型与验证链：[docs/MODEL.md](../docs/MODEL.md)；硬件支持与资质：[docs/HARDWARE.md](../docs/HARDWARE.md)、[D0004](decisions/d0004-gpu-p2p-support.md)。
- 构建与 bench 工作流仍待定，CI 未接线；硬件资格与性能阈值仍缺目标机证据，不作为 P0006 格式实现的前置条件。
