# STATUS — 状态页

状态信息的唯一归宿（状态不进 docs，README 只做路由）。本页只写当前能力、在办工作与阻塞项；过程与证据在对应 P/D，覆盖边界在 [MODEL §4](../docs/MODEL.md#4-精确性链条)。

## 当前阶段

**开发阶段初期：已有 CPU 格式、反量化、专家分组点积与动态激活量化参照及生产 scalar，单 token 线性层、SwiGLU 桥与完整 FP4 专家生产 scalar、主干 MoE 路由 CPU 参照及生产 scalar 与合成专家接口验证、routed 分支 CPU 参照及生产 scalar 与多专家聚合验证，以及官方 checkpoint 清点、选定行、完整 w1 和固定专家三矩阵验证入口。** 推理引擎、CUDA 路径、完整 FP32 模型 oracle、目标机验证与引擎性能基线均未开始，不可部署。

## 当前工作项

| 工作项 | 状态 | 下一步或阻塞 |
| --- | --- | --- |
| [P0005 — 部署资质检查与首台验证](p0005-hardware-qualification.md) | draft | 缺目标机与 CUDA 探针构建环境，暂未启动 |
| [P0027 — FP8×FP8 块缩放单输出点积 CPU 参照](p0027-fp8-grouped-dot-oracle.md) | active | 用户已人工审议通过并授权提交合入；待已提交干净复现补证后结项 |

## 已验收入口

| 入口 | 契约 | 证据 |
| --- | --- | --- |
| `ref/formats.py`、`ref/dequant.py` | [QUANT §2～§5](../docs/QUANT.md#2-位模式与零符号) | [P0006](p0006-ref-format-oracle.md) |
| `df_quant` codec 与给定 scale 反量化 | [QUANT §6～§7](../docs/QUANT.md#6-生产-c-codecsrcquant) | [P0010](p0010-quant-scalar-baseline.md)、[P0011](p0011-quant-dequant-scalar.md) |
| `ref/compute.py` | [COMPUTE](../docs/COMPUTE.md) | [P0012](p0012-expert-grouped-dot-oracle.md) |
| `df_quant` 专家分组点积 | [COMPUTE §6](../docs/COMPUTE.md#6-生产-c-scalarsrcquant) | [P0013](p0013-expert-grouped-dot-scalar.md) |
| `ref/act_quant.py` | [ACTQUANT](../docs/ACTQUANT.md) | [P0014](p0014-act-quant-oracle.md) |
| `ref/expert.py` | [EXPERT](../docs/EXPERT.md) | [P0020](p0020-expert-oracle.md) |
| `ref/router.py` 与合成专家接口 | [ROUTER](../docs/ROUTER.md)；[MODEL §4](../docs/MODEL.md#4-精确性链条) 的主干路由与接口边界 | [P0023](p0023-router-oracle.md) |
| `df_core` 主干 router 与生产专家接口 | [ROUTER §6](../docs/ROUTER.md#6-生产-cpu-scalar)；[MODEL §4](../docs/MODEL.md#4-精确性链条) 的生产路由边界 | [P0024](p0024-router-scalar.md) |
| `ref/moe.py` routed 分支与多专家聚合 | [MOE](../docs/MOE.md)；覆盖边界见 [MODEL §4](../docs/MODEL.md#4-精确性链条) | [P0025](p0025-routed-branch-oracle.md) |
| `df_core` routed 分支生产 scalar | [MOE §5](../docs/MOE.md#5-生产-cpu-scalar)；覆盖边界见 [MODEL §4](../docs/MODEL.md#4-精确性链条) | [P0026](p0026-routed-branch-scalar.md) |
| `df_quant` 单行动态激活量化 | [ACTQUANT §7](../docs/ACTQUANT.md#7-生产-c-scalarsrcquant) | [P0017](p0017-act-quant-scalar.md) |
| `tools/ckpt_inventory.py` | [MODEL §2](../docs/MODEL.md#2-结构与格式) 的 [ckpt] 事实 | [P0016](p0016-checkpoint-inventory.md) |
| `tools/ckpt_expert.py` | [MODEL §4](../docs/MODEL.md#4-精确性链条) 的真实权重选定行验证边界；[BUILD §7](../docs/BUILD.md#7-测试清单) 的操作约束 | [P0018](p0018-checkpoint-expert-consumption.md) |
| `df_quant` 专家线性层 | [COMPUTE §7](../docs/COMPUTE.md#7-单-token-专家线性层) | [P0019](p0019-expert-linear-scalar.md) |
| `tools/ckpt_linear.py` | [MODEL §4](../docs/MODEL.md#4-精确性链条) 的完整 w1 验证边界；[BUILD §7](../docs/BUILD.md#7-测试清单) 的操作约束 | [P0019](p0019-expert-linear-scalar.md) |
| `df_quant` SwiGLU bridge | [EXPERT §5](../docs/EXPERT.md#5-生产-swiglu-bridge) | [P0021](p0021-swiglu-bridge-scalar.md) |
| `df_quant` 完整 FP4 专家 | [EXPERT §6](../docs/EXPERT.md#6-生产完整专家) | [P0022](p0022-expert-scalar.md) |
| `tools/ckpt_expert_mlp.py` | [MODEL §4](../docs/MODEL.md#4-精确性链条) 的固定专家完整三矩阵边界；[BUILD §7](../docs/BUILD.md#7-测试清单) 的操作约束 | [P0022](p0022-expert-scalar.md) |

复现命令与 CTest 入口见 [BUILD §7](../docs/BUILD.md#7-测试清单)。尚未立项：动态 KV 量化、同契约计算快路径、K-E1/K-E2 参照、SIMD/CUDA 与官方 GPU parity。

## 其他入口与阻塞

- **构建**：CPU-only，仅 ENV-A（[BUILD §1](../docs/BUILD.md#1-支持环境)）；CUDA 未接入，不解除 P0005 的探针构建阻塞。
- **硬件**：契约见 [HARDWARE](../docs/HARDWARE.md)、[D0004](decisions/d0004-gpu-p2p-support.md)；整机资格与性能阈值缺目标机证据。
- **工程缺口**：bench 工作流待定，CI 未接线。
