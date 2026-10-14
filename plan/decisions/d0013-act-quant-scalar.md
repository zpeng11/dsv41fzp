---
id: D0013
title: 单行 BF16 激活量化归入 df_quant，以整数阶段和调用方双输出建立生产 scalar
status: accepted
created: 2026-10-14
updated: 2026-10-14
replaces: []
superseded-by: []
---

## 背景与问题

[P0017](../p0017-act-quant-scalar.md) 接续已验收的 [P0014](../p0014-act-quant-oracle.md) 动态激活量化参照和 [P0013](../p0013-expert-grouped-dot-scalar.md) 生产分组点积。[D0010](d0010-act-quant-reference.md) 要求生产 C++ 的接口、数值路径与模块归属另立决策；本决策落实该前置，不重述 [ACTQUANT §3](../../docs/ACTQUANT.md#3-数值阶段) 数值规格，不推定官方 GPU 逐位等价。

## 决策驱动

- 生产入口须与 Python 上一级逐阶段一致，拒绝不返回部分结果；双输出可直接供既有生产点积消费。
- 普通与观测路径须复用同一核心，宿主浮点环境不能改变数值或拒绝行为。
- 沿用现有格式与整数舍入基础，不建第二个格式配置中心或通用软件浮点 API。
- 范围限单行模型实际 BF16 变体，不增加矩阵、调度、融合或性能义务。

## 备选方案

- 模块归入 `df_quant`，公共头 `include/df/quant/act_quant.h`，实现 `src/quant/act_quant.cpp`：采纳，与已有 codec、反量化、点积共享内部基础。建设 `src/aq` 调度框架：本阶段不采纳，不改变未来模块规划。
- 调用方提供两份输出 span：采纳，容量与实际写入范围明确，不需结果动态分配。返回动态容器或允许输入输出别名：不采纳，前者改变生产接口风格，后者引入额外临时存储或破坏校验与输出原子性。
- 纯整数构造 binary32 阶段，复用有限乘法与舍入：采纳。宿主 float 路径：不采纳，增加 MXCSR/DAZ/FTZ 依赖；以精确除以 448 替代契约乘积：不采纳，阶段位可能不同。
- 可空私有观测器连接同一核心：采纳，延续 [D0009](d0009-compute-scalar-integer-path.md)。另写测试专用量化实现：不采纳，不能证明生产阶段。

## 结论

2026-10-14 用户通过开工方案审议，明确确认本生产设计及对应内容写入，接受 D0013 并授权 P0017 转 active 后实现；本次接受不替代实现、证据和知识处置的最终人工评审。

1. 公共入口为 `void act_quant(std::span<const std::uint16_t> bf16_bits, std::span<std::uint8_t> a_codes, std::span<std::uint8_t> scale_a)`。输入长度 K 为正且整除模型固定组大小；输出容量至少 K 与 K/32，只写结果前缀，额外容量不写。
2. 输入实际读取范围与任一输出实际写入范围不得重叠，两份输出写入范围也不得重叠。精确相邻允许，未使用输出尾部不参与重叠判定。空输入、错误分组、容量不足、非有限 BF16 和禁止的重叠抛 `std::invalid_argument`。
3. 元数据、容量、字节范围与重叠检查先于输入读取，全部内容有限性检查先于任何写出；拒绝时两份输出全部不变。
4. 数值路径严格按 ACTQUANT §3：原始 BF16 位映射与幅值比较，既有 `multiply_finite` 生成乘积；商为精确乘以 `2^-e` 后一次 binary32 RNE，复用 `binary_to_fp32` 的整数舍入与渐进下溢；保留 clamp 与零符号。codec 内新增有限、已 clamp binary32 位的私有 E4M3FN 编码 helper，现有 `round_shift` 提升为内部共享函数。全程不经宿主浮点，不读取或修改调用方浮点环境。
5. 私有 `ActQuantTrace` 与入口放在 `src/quant/detail.h`。每组记录 amax、下限后值、乘积位及指数，每元素记录商与 clamp 后位；scale 与编码由同次调用的普通输出提供。普通及 trace 共用一个核心。trace span 长度精确匹配组数或 K，其实际写入范围与输入、两份输出以及其他 trace 范围互不重叠；全部 trace 元数据检查先于输入读取，拒绝时 trace 也不变。
6. 组合仅在测试执行器中存在，生产量化结果缓冲直接交给既有生产点积；不新增融合或组合公共 API。

## 后果

- 不取代 D0007/D0009/D0010；`ref/act_quant.py` 是本 scalar 的上一级，本 scalar 是未来同契约快路径的上一级。
- P0017 按既定条件完成阶段 parity、失败契约、MXCSR、ASan 边界、生产组合及临时数值变体敏感性；不因决策接受视作技术验收。
- 经本次确认只增补 ACTQUANT 的生产表示与失败入口、BUILD 的测试和 sanitizer 范围、MODEL §4 的验证链路由，不重复数值规格。
- 不验收 SIMD/CUDA、官方 GPU、inplace/KV/FP4、K-E1/K-E2、完整专家 MLP、checkpoint/pack 或端到端质量；不设性能承诺。
