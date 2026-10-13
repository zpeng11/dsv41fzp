# STATUS — 状态页

状态信息的唯一归宿（状态不进 docs，README 只做路由）。本页保留当前能力、在办工作与阻塞项；过程与验收证据回到原 P/D 条目，不在此复制历史流水账。

## 当前阶段

**开发阶段初期：格式 oracle、CPU-only C++ 构建、生产 quant codec、给定 scale 反量化基线、专家分组点积 CPU 规范参照及其生产 C++ scalar 基线与动态激活量化 CPU 规范参照已建立。** [P0013](p0013-expert-grouped-dot-scalar.md) 与 [P0014](p0014-act-quant-oracle.md) 的技术验收、干净检出复现、知识归档与用户评审均通过，已结项；同契约计算/激活量化快路径与生产激活量化尚未立项。 推理引擎与完整 FP32 模型 oracle 尚未实现，没有目标机验证或引擎性能基线，不处于可部署阶段。

## 当前工作项

| 工作项 | 状态 | 下一步或阻塞 |
| --- | --- | --- |
| [P0005 — 部署资质检查与首台验证](p0005-hardware-qualification.md) | draft | 缺目标机与 CUDA 探针构建环境，暂未启动 |
| [P0015 — docs 与 plan 单一来源整理](p0015-doc-slimming.md) | active | 按 [D0011](decisions/d0011-doc-single-source.md) 整理 docs、模板、STATUS 与已结项 P |

## 数值入口与边界

- **已验收的参照**：`ref/formats.py`、`ref/dequant.py`；复现命令 `python3 tests/quant/test_formats.py`，仅标准库。格式与给定 scale 反量化契约见 [QUANT §2～§5](../docs/QUANT.md#2-位模式与零符号)，证据见 [P0006](p0006-ref-format-oracle.md#收尾)。
- **已验收的生产入口**：`df_quant`，公共头 `include/df/quant/codec.h` 与 `dequant.h`；接口、布局、失败与环境契约见 [QUANT §6～§7](../docs/QUANT.md#6-生产-c-codecsrcquant)，证据见 [P0010](p0010-quant-scalar-baseline.md#收尾)、[P0011](p0011-quant-dequant-scalar.md#收尾)。它们覆盖格式 codec 与六类给定 scale 的逐元素反量化，不是 GEMM、累加后乘 scale、整数点积或提前 BF16 舍入的参照，不表示 K-Q1～K-Q3 全部完成。
- **已验收的计算参照**：`ref/compute.py`，必选命令 `python3 tests/quant/test_compute.py`，CTest 入口 `quant.compute`。单输出专家 FP8×packed FP4、双方每 32 元素分组的 CPU 契约见 [COMPUTE](../docs/COMPUTE.md)，复现入口见 [BUILD](../docs/BUILD.md#7-测试清单)，证据见 [P0012](p0012-expert-grouped-dot-oracle.md#收尾)。它只供同契约后续实现对照，不直接验收 K-E1/K-E2 或官方 GPU，路由见 [MODEL §4](../docs/MODEL.md#4-精确性链条)。
- **已验收的生产计算**：`df_quant` 公共头 `include/df/quant/compute.h`，入口 `expert_grouped_dot`；接口、失败、环境与验证边界见 [COMPUTE §6](../docs/COMPUTE.md#6-生产-c-scalarsrcquant)，CTest 入口 `quant.compute.{parity,contract,add,harness}` 见 [BUILD §7](../docs/BUILD.md#7-测试清单)，证据见 [P0013](p0013-expert-grouped-dot-scalar.md#收尾)。同契约快路径只对照它，它不直接验收 K-E1/K-E2 或官方 GPU。
- **已验收的动态激活量化参照**：`ref/act_quant.py`，入口 `act_quant` 与 `trace_act_quant`；单行 BF16→E4M3FN+E8M0 CPU 契约见 [ACTQUANT](../docs/ACTQUANT.md)，复现命令 `python3 tests/quant/test_act_quant.py`，CTest 入口 `quant.act_quant` 见 [BUILD §7](../docs/BUILD.md#7-测试清单)，证据见 [P0014](p0014-act-quant-oracle.md#收尾)。它是后续同契约生产 scalar 的上一级，不直接验收 K-X1 kernel、K-E1/K-E2 或官方 GPU，路由见 [MODEL §4](../docs/MODEL.md#4-精确性链条)。
- **仍未覆盖**：生产动态激活量化、其他有损编码/scale 生成、动态 KV 量化、同契约计算快路径、K-E1 同路径参照与跨路径质量偏差、整数改写及 scale 折叠、SIMD/CUDA、GPU cast/FTZ、官方 GPU parity 与双 arch 端到端一致尚未立项。W4A16 decode 默认规划不变。

## 其他入口与阻塞

- **构建**：CPU-only 入口与测试清单见 [BUILD](../docs/BUILD.md)，证据见 [P0009](p0009-cpu-cpp-build-baseline.md#收尾)；当前仅支持实测的 GCC 15.2.0 / Linux x86_64 配置。CUDA 尚未接入，`cpu-asan` 仍限 quant，不解除 P0005 的探针构建阻塞。
- **硬件**：支持与资质契约见 [HARDWARE](../docs/HARDWARE.md)、[D0004](decisions/d0004-gpu-p2p-support.md)；整机资格与性能阈值仍缺目标机证据。
- **工程缺口**：bench 工作流仍待定，CI 未接线。
