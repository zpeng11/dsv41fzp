# STATUS — 状态页

状态信息的唯一归宿（状态不进 docs，README 只做路由）。本页保留当前能力、在办工作与阻塞项；过程与验收证据回到原 P/D 条目，不在此复制历史流水账。

## 当前阶段

**开发阶段初期，CPU 格式验证根已建立。** `ref/` 已提供低精度格式解码、E2M1 打包与整数格、给定 scale 反量化及 BF16 输出参照，独立 CPU 测试和用户评审通过；数值契约见 [docs/QUANT.md](../docs/QUANT.md)，验收证据见 [P0006](p0006-ref-format-oracle.md#收尾)。推理引擎与完整 FP32 模型 oracle 尚未实现，没有目标机验证或引擎性能基线，不处于可部署阶段。

## 当前工作项

| 工作项 | 状态 | 下一步或阻塞 |
| --- | --- | --- |
| [P0005 — 部署资质检查与首台验证](p0005-hardware-qualification.md) | draft | 缺目标机与 CUDA 探针构建环境，暂未启动 |
| [P0009 — CPU-only C++ 构建与测试基线](p0009-cpu-cpp-build-baseline.md) | draft | 设计已落盘，构建契约待评审；缺 CMake/CTest/Ninja，未开始实现 |

## 数值开发入口与缺口

- 参照入口：`ref/formats.py`、`ref/dequant.py`；复现命令：`python3 tests/quant/test_formats.py`，仅标准库。P0006 已结项，不再列为在办项。
- 生产 `src/quant` 与 K-Q1～K-Q3 尚未实现；后续编码须另建工作项，并对照已验证的 `ref/` 路径验收。当前没有 active 的数值实现工作项，不以格式 oracle 完成替代计算路径验收。
- 动态量化、GPU cast/FTZ、kernel parity 与端到端验证尚无证据；模型用途与验证链见 [docs/MODEL.md](../docs/MODEL.md)。

## 其他入口与缺口

- 硬件支持与资质：[docs/HARDWARE.md](../docs/HARDWARE.md)、[D0004](decisions/d0004-gpu-p2p-support.md)。
- CPU-only C++ 构建与测试基线由 [P0009](p0009-cpu-cpp-build-baseline.md) 拟定，尚未实现或验收，不替代生产 quant 或 CUDA 探针构建；bench 工作流仍待定，CI 未接线。硬件资格与性能阈值仍缺目标机证据。
