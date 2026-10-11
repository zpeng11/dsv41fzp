# STATUS — 状态页

状态信息的唯一归宿（状态不进 docs，README 只做路由）。本页保留当前能力、在办工作与阻塞项；过程与验收证据回到原 P/D 条目，不在此复制历史流水账。

## 当前阶段

**开发阶段初期，CPU 格式验证根、CPU-only C++ 构建基线与生产 quant codec 已建立。** `ref/` 已提供低精度格式解码、E2M1 打包与整数格、给定 scale 反量化及 BF16 输出参照，独立 CPU 测试和用户评审通过；数值契约见 [docs/QUANT.md](../docs/QUANT.md)，验收证据见 [P0006](p0006-ref-format-oracle.md#收尾)。CMake/CTest 构建入口、严格数值 target 与构建时版本元数据已验收，契约见 [docs/BUILD.md](../docs/BUILD.md)，验收证据见 [P0009](p0009-cpu-cpp-build-baseline.md#收尾)。生产 `src/quant` 的格式 codec（解码、FP32→BF16 舍入、E2M1 打包与整数格）已逐位对照 `ref/` 验收，契约见 [QUANT §6](../docs/QUANT.md#6-生产-c-codecsrcquant)，验收证据见 [P0010](p0010-quant-scalar-baseline.md#收尾)。给定 scale 反量化由 P0011 实现并在验收中，尚未结项；推理引擎与完整 FP32 模型 oracle 尚未实现，没有目标机验证或引擎性能基线，不处于可部署阶段。

## 当前工作项

| 工作项 | 状态 | 下一步或阻塞 |
| --- | --- | --- |
| [P0005 — 部署资质检查与首台验证](p0005-hardware-qualification.md) | draft | 缺目标机与 CUDA 探针构建环境，暂未启动 |
| [P0011 — 给定 scale 六类反量化](p0011-quant-dequant-scalar.md) | active | 用户确认代码与知识归档；完成提交态复现与范围审计后同步收尾合入 |

## 数值开发入口与缺口

- 参照入口：`ref/formats.py`、`ref/dequant.py`；复现命令：`python3 tests/quant/test_formats.py`，仅标准库。P0006 已结项，不再列为在办项。
- 生产入口：`df_quant`（`include/df/quant/codec.h`）；复现命令见 [docs/BUILD.md](../docs/BUILD.md)，含 `cpu-asan`。P0010 已结项，不再列为在办项；它覆盖 K-Q1 与 K-Q3 中的 E4M3FN/BF16 解码和 FP32→BF16 舍入。[P0011](p0011-quant-dequant-scalar.md)（六类给定 scale 反量化）为 active，沿用 P0010 的模块、执行器协议与 `cpu-asan`，接口契约见 [D0007](decisions/d0007-quant-integer-dequant.md)。给定 scale 的指数应用与 KV/SWA 布局在 P0011 验收中；点积 scale 折叠、有损编码与动态量化尚未实现，不以 codec 或逐元素反量化完成替代 K-Q1～K-Q3 全部或计算路径验收。
- 计算类 kernel（专家 GEMM、整数点积、scale 折叠）的上一级 scalar 参照放在哪里尚未决定；[MODEL §4](../docs/MODEL.md#4-精确性链条) 链条与 K-Q2 的完整修订留待首个计算类 kernel 工作项开工前一并决定。
- 动态量化、GPU cast/FTZ、kernel parity 与端到端验证尚无证据；模型用途与验证链见 [docs/MODEL.md](../docs/MODEL.md)。

## 其他入口与缺口

- 硬件支持与资质：[docs/HARDWARE.md](../docs/HARDWARE.md)、[D0004](decisions/d0004-gpu-p2p-support.md)。
- 构建：CPU-only 入口见 [docs/BUILD.md](../docs/BUILD.md)（P0009 已结项，不再列为在办项），仅支持实测的 GCC 15.2.0 / Linux x86_64 配置；CUDA 构建尚未接入，不解除 P0005 的探针构建阻塞。bench 工作流仍待定，CI 未接线。硬件资格与性能阈值仍缺目标机证据。
