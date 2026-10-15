# EXPERT.md — 单 token FP4 专家与 SwiGLU CPU 契约

本页定义 `ref/expert.py` 的组合与非线性语义，以及同契约生产 SwiGLU bridge；输入量化只引用 [ACTQUANT](ACTQUANT.md)，逐行点积与线性输出转换只引用 [COMPUTE §3](COMPUTE.md#3-逐步计算与舍入)，格式只引用 [QUANT](QUANT.md)。参照选择见 [D0015](../plan/decisions/d0015-expert-oracle.md)，生产选择见 [D0016](../plan/decisions/d0016-swiglu-scalar.md)，实施证据见 [P0020](../plan/p0020-expert-oracle.md) 与 [P0021](../plan/p0021-swiglu-bridge-scalar.md)，验证链覆盖边界只见 [MODEL §4](MODEL.md#4-精确性链条)。

**来源标记**：**[官方]** 只读镜像 `model.py:830–851`、`kernel.py:478–484,550–557`；**[D]** D0015/D0016 接受的 CPU 选择；**[实现]** `ref/expert.py` 与 `src/quant/swiglu.cpp`。下述数字为格式、算法或接口约束，不是性能测量；支持环境为 [BUILD §1](BUILD.md#1-支持环境) 的 ENV-A。

## 1. 接口与输入

- `FP4Matrix(n, k, w_packed, scale_b)` 表达逐行连续 packed E2M1 及 E8M0 scale。每行长度分别为 `K/2`、`K/32`；组大小来自 [MODEL §2](MODEL.md#2-结构与格式)。不广播、转置或隐式 flatten。[D]
- `expert(input_bf16, w1, w2, w3, *, route_weight_bits=None)` 返回 BF16 位模式 tuple。D/H 为正且整除组大小；输入长度 D，w1/w3 为 H×D，w2 为 D×H。输入、权重及 scale 是一维原始整数编码序列；位宽、类型及 scale 合法域沿用上述格式与量化契约，bool 不作整数编码。计算前完整校验并快照，不修改调用方序列。[D][实现]
- `swiglu(gate_bf16, up_bf16, *, route_weight_bits=None)` 单独执行 §2 的桥接，返回 BF16 tuple；两输入等长，允许空序列及非有限 BF16，以便观察特殊值。[D]
- 外部权重为有限 FP32 原始位整数，允许次正规数和正负零，不限制概率范围；`None` 完全省略权重乘法。这里不生成或归一化权重。[D]
- `trace_expert`/`trace_swiglu` 与普通入口共享执行核心。桥接 trace 返回钳位后 gate/up、SiLU、两次乘积的 FP32 位及 BF16 输出；专家 trace 另含两次量化编码/scale、w1/w3 BF16 与最终输出。不增加整矩阵点积内部 trace。[实现]

## 2. 运算与舍入

令 `R32` 为 binary32 RNE，`B16` 为 QUANT 的 FP32→BF16 RNE，`linear_bf16` 为既有量化编码供每行点积并输出 BF16。[D]

```text
(codes, scales) = act_quant(input_bf16)
gate = widen(linear_bf16(codes, scales, w1))
up   = widen(linear_bf16(codes, scales, w3))
gate = min(gate, limit)
up   = clamp(up, -limit, limit)
silu = R32(real(gate / (1 + exp(-gate))))
gated = R32(silu * up)
weighted = gated if route_weight_bits is None else R32(weight * gated)
hidden_bf16 = B16(weighted)
(hidden_codes, hidden_scales) = act_quant(hidden_bf16)
out = linear_bf16(hidden_codes, hidden_scales, w2)
```

`limit` 为 [MODEL §3.5](MODEL.md#35-路由srccore) 的模型钳位参数，非可配置激活函数。非对称钳位、外部权重在 w2 前及转换回输入 dtype 的顺序来自官方；每个线性层均保留 BF16 落点，再精确拓宽。禁止提前 BF16、跨步骤融合、重排乘法或把权重移至 w2 输出后。[官方][D]

SiLU 是实数函数一次正确舍入 FP32，**不是 exp/add/div 各自 FP32 舍入**，也不规定官方 GPU 的 SiLU 内部实现。有限运算渐进下溢，乘积零符号异或；BF16 与 NaN 比较规则沿用 QUANT/COMPUTE。[D]

## 3. 特殊值、失败与环境

- clamp 传播 NaN，保留未越界的零符号；gate 的正 Inf 被钳位、负 Inf 保留，up 的两种 Inf 均被钳位。SiLU 正负零保号，NaN 与负 Inf 得 NaN。乘法传播 NaN，零乘 Inf 得 NaN，溢出得带符号 Inf。[D]
- 完整专家只在原始输入和进入第二次激活量化的 BF16 边界要求有限；不提前拒绝能被钳位恢复的线性 Inf。最终 w2 Inf/NaN 是合法结果。[D]
- 类型错误抛 `TypeError`，维度、长度、位宽、非法 scale/权重及上述有限性错误抛 `ValueError`。不返回部分结果；SiLU 区间无法认证舍入时抛 `ArithmeticError`，不猜测、不饱和或清零兜底。[D]
- 硬件浮点异常须屏蔽。数值独立于调用方舍入/FTZ/DAZ，控制位不变、已有 sticky 异常位不清除；Decimal/Python 内部允许新增 sticky 异常位。**不承诺完整浮点环境无副作用，也不放宽已有生产 scalar 的环境契约。** 显式 Decimal context 不污染或继承调用方的精度、舍入、trap 与 flags。[D][实现]

## 4. 实现与验证方法

参照 SiLU 用显式 Decimal context 的正确舍入 exp 及相邻十进制值包围真实指数，精确有理数传播区间；整数商余数决定两端 FP32 舍入，只有同码才返回。使用稳定负指数；有限 gate≤−128 时由 `128*exp(-128)<2^-150` 及单调性直接得到负零。初始精度 192 位、递增至 3072 位，仍未认证则失败；这是实现资源上限，不是近似容差。[实现]

独立期望使用 Fraction Taylor 正项级数、保守余项界、区间平方及格点搜索；三矩阵期望复用测试侧独立量化和点积，始终消费期望的前阶段结果。生产桥穷举全部 BF16 gate 编码并逐阶段对照参照，另保留独立边界期望与临时生产变体门禁；环境、命令与覆盖范围归 [BUILD §7](BUILD.md#7-测试清单)，最终证据归对应 P。[实现]

## 5. 生产 SwiGLU bridge

- `df::quant::swiglu(gate, up, out)` 与带 `route_weight_bits` 的重载消费等长 BF16 raw-bit span，返回写入 `out` 前缀的 BF16。无权重重载完全省略 §2 的可选乘法；带权重重载要求有限 FP32 raw bits，即使输入为空也先验证。输入可为空，两个只读 span 可重叠；实际输出前缀必须与两个输入读区间分离。[D][实现]
- 输出容量至少为输入长度，只写该前缀。长度、容量、字节计数、重叠或非法权重抛 `std::invalid_argument`；全部元数据先于输入元素读取，全部拒绝先于写入，失败时完整输出 span 不变。非有限 gate/up 不作内容拒绝，按 §3 传播。[D][实现]
- 生产 SiLU 只覆盖钳位后的有限域，不提供通用数学 API；以构建期全域表实现，生成物不入库。运行时钳位、查表、两次可能的 FP32 RNE 乘法与 BF16 RNE 均为整数位运算，不读写宿主浮点控制或状态。选择理由见 D0016，生成与命令见 [BUILD §7](BUILD.md#7-测试清单)。[D][实现]
