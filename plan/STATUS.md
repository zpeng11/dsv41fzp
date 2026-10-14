# STATUS — 状态页

状态信息的唯一归宿（状态不进 docs，README 只做路由）。本页只写当前能力、在办工作与阻塞项；过程与证据在对应 P/D，覆盖边界在 [MODEL §4](../docs/MODEL.md#4-精确性链条)。

## 当前阶段

**开发阶段初期：只有 CPU 上的格式、反量化、专家分组点积与动态激活量化参照及其部分生产 scalar，以及官方 checkpoint 的清点工具。** 推理引擎、CUDA 路径、完整 FP32 模型 oracle、目标机验证与引擎性能基线均未开始，不可部署。

## 当前工作项

| 工作项 | 状态 | 下一步或阻塞 |
| --- | --- | --- |
| [P0005 — 部署资质检查与首台验证](p0005-hardware-qualification.md) | draft | 缺目标机与 CUDA 探针构建环境，暂未启动 |
| [P0017 — 单行动态激活量化生产 scalar 与组合 parity](p0017-act-quant-scalar.md) | active | 用户已验收，证据见工作项；待提交范围审计与已提交干净检出 |

## 已验收入口

| 入口 | 契约 | 证据 |
| --- | --- | --- |
| `ref/formats.py`、`ref/dequant.py` | [QUANT §2～§5](../docs/QUANT.md#2-位模式与零符号) | [P0006](p0006-ref-format-oracle.md) |
| `df_quant` codec 与给定 scale 反量化 | [QUANT §6～§7](../docs/QUANT.md#6-生产-c-codecsrcquant) | [P0010](p0010-quant-scalar-baseline.md)、[P0011](p0011-quant-dequant-scalar.md) |
| `ref/compute.py` | [COMPUTE](../docs/COMPUTE.md) | [P0012](p0012-expert-grouped-dot-oracle.md) |
| `df_quant` 专家分组点积 | [COMPUTE §6](../docs/COMPUTE.md#6-生产-c-scalarsrcquant) | [P0013](p0013-expert-grouped-dot-scalar.md) |
| `ref/act_quant.py` | [ACTQUANT](../docs/ACTQUANT.md) | [P0014](p0014-act-quant-oracle.md) |
| `tools/ckpt_inventory.py` | [MODEL §2](../docs/MODEL.md#2-结构与格式) 的 [ckpt] 事实 | [P0016](p0016-checkpoint-inventory.md) |

复现命令与 CTest 入口见 [BUILD §7](../docs/BUILD.md#7-测试清单)。尚未立项：动态 KV 量化、同契约计算快路径、K-E1/K-E2 参照、SIMD/CUDA 与官方 GPU parity。

## 其他入口与阻塞

- **构建**：CPU-only，仅 ENV-A（[BUILD §1](../docs/BUILD.md#1-支持环境)）；CUDA 未接入，不解除 P0005 的探针构建阻塞。
- **硬件**：契约见 [HARDWARE](../docs/HARDWARE.md)、[D0004](decisions/d0004-gpu-p2p-support.md)；整机资格与性能阈值缺目标机证据。
- **工程缺口**：bench 工作流待定，CI 未接线。
