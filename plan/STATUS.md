# STATUS — 状态页

状态信息的唯一归宿（状态不进 docs，README 只做路由）。本页保留当前能力、在办工作与阻塞项；过程与验收证据回到原 P/D 条目，不在此复制历史流水账。

## 当前阶段

**开发阶段初期，CPU 格式验证根与 CPU-only C++ 构建基线已建立。** `ref/` 已提供低精度格式解码、E2M1 打包与整数格、给定 scale 反量化及 BF16 输出参照，独立 CPU 测试和用户评审通过；数值契约见 [docs/QUANT.md](../docs/QUANT.md)，验收证据见 [P0006](p0006-ref-format-oracle.md#收尾)。CMake/CTest 构建入口、严格数值 target 与构建时版本元数据已验收，契约见 [docs/BUILD.md](../docs/BUILD.md)，验收证据见 [P0009](p0009-cpu-cpp-build-baseline.md#收尾)。推理引擎、生产 quant 与完整 FP32 模型 oracle 尚未实现，没有目标机验证或引擎性能基线，不处于可部署阶段。

## 当前工作项

| 工作项 | 状态 | 下一步或阻塞 |
| --- | --- | --- |
| [P0005 — 部署资质检查与首台验证](p0005-hardware-qualification.md) | draft | 缺目标机与 CUDA 探针构建环境，暂未启动 |
| [P0010 — 生产 quant codec 与 parity 连接](p0010-quant-scalar-baseline.md) | draft | [D0007](decisions/d0007-quant-integer-dequant.md) 已接受（取代 D0006）；待授权开工，尚未实现 |
| [P0011 — 给定 scale 六类反量化](p0011-quant-dequant-scalar.md) | draft | 依赖 P0010 结项，尚未实现 |

## 数值开发入口与缺口

- 参照入口：`ref/formats.py`、`ref/dequant.py`；复现命令：`python3 tests/quant/test_formats.py`，仅标准库。P0006 已结项，不再列为在办项。
- 生产 `src/quant` 与 K-Q1～K-Q3 尚未实现；[P0010](p0010-quant-scalar-baseline.md)（codec、字节布局、整数格与跨语言 parity 连接）与 [P0011](p0011-quant-dequant-scalar.md)（六类给定 scale 反量化）已登记为 draft，沿用 [docs/BUILD.md](../docs/BUILD.md) 的构建入口并对照已验证的 `ref/` 验收；接口契约见 [D0007](decisions/d0007-quant-integer-dequant.md)（codec 与反量化均为整数位运算，不依赖浮点环境）。当前没有 active 的数值实现工作项；两项均不覆盖点积 scale 折叠、有损编码或动态量化，不以其完成替代 K-Q1～K-Q3 全部或计算路径验收。
- 计算类 kernel（专家 GEMM、整数点积、scale 折叠）的上一级 scalar 参照放在哪里尚未决定；[MODEL §4](../docs/MODEL.md#4-精确性链条) 链条与 K-Q2 的完整修订留待首个计算类 kernel 工作项开工前一并决定。
- 动态量化、GPU cast/FTZ、kernel parity 与端到端验证尚无证据；模型用途与验证链见 [docs/MODEL.md](../docs/MODEL.md)。

## 其他入口与缺口

- 硬件支持与资质：[docs/HARDWARE.md](../docs/HARDWARE.md)、[D0004](decisions/d0004-gpu-p2p-support.md)。
- 构建：CPU-only 入口见 [docs/BUILD.md](../docs/BUILD.md)（P0009 已结项，不再列为在办项），仅支持实测的 GCC 15.2.0 / Linux x86_64 配置；CUDA 构建尚未接入，不解除 P0005 的探针构建阻塞。bench 工作流仍待定，CI 未接线。硬件资格与性能阈值仍缺目标机证据。
