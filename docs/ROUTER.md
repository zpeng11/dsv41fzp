# ROUTER.md — 单 token MoE 路由 CPU 契约

本页定义 `ref/router.py` 的慢参照契约。模型参数与阶段来源见 [MODEL §3.5](MODEL.md#35-路由srccore)，选择理由见 [D0018](../plan/decisions/d0018-router-reference.md)，验收证据归 [P0023](../plan/p0023-router-oracle.md)，验证链边界只见 [MODEL §4](MODEL.md#4-精确性链条)。

**来源标记**：**[官方]** 只读镜像 `model.py:809–827` 及 MODEL 引用的 PyTorch softplus 算子说明；**[D]** D0018 接受的 CPU 数值选择；**[实现]** `ref/router.py`。本页数字是编码、算法或接口约束，不是性能测量；支持环境为 [BUILD §1](BUILD.md#1-支持环境) ENV-A。

## 1. 接口与输入

- `GateMatrix(n, k, weight_bf16)` 描述行连续 N×K 的 BF16 原始位序列，不广播、转置或 flatten。K 为正，N 至少覆盖 MODEL 的主干激活专家数；不强制发布矩阵尺寸，以允许合成验证。[D]
- `route(input_bf16, gate, bias_bits, *, image_token=False, bias_vl_bits=None)` 返回 `(indices, weights_bits)`，两者为对应位置配对的 tuple；编号为行下标，权重为 FP32 原始位整数。输入长度 K、权重长度 N×K、bias 长度 N。[D]
- 输入与权重编码为有限 BF16，bias 编码为有限 FP32；宽度分别为 16/32 位，bool 不作整数编码。接收一维整数序列，不接收嵌套张量、生成器或多维 memoryview。[D]
- `image_token` 必须为 bool。为真且提供 `bias_vl_bits` 时选用视觉 bias，否则选普通 bias。无论实际选用哪一个，所有提供的序列都在计算前完整校验并快照；普通和视觉 bias 允许引用同一只读序列，调用方内容不变。[官方][D]
- 固定主干的激活专家数、温度、归一化和缩放设置；不暴露其他 score 函数或路由参数。模型常量的值与依据只见 MODEL §3.5。[D]

## 2. 运算与舍入

令 `R32` 为 binary32 round-to-nearest ties-to-even；有限运算渐进下溢，保留 IEEE 零符号。所有中间编码为 FP32。[D]

```text
for each expert row i:
    accumulator = +0
    for j in increasing K order:
        product = R32(widen(input[j]) * widen(weight[i,j]))
        accumulator = R32(accumulator + product)
    linear[i] = accumulator
    temp[i] = R32(linear[i] / model_temperature)
    softplus[i] = temp[i] if above_softplus_threshold(temp[i])
                  else R32(real(log(1 + exp(temp[i]))))
    score[i] = R32(real(sqrt(softplus[i])))
    selection[i] = R32(score[i] + chosen_bias[i])
indices = first model_topk rows sorted by descending selection, then ascending index
selected = score gathered in indices order
sum = sequential R32 additions of selected, starting at +0
denominator = R32(sum + R32(model_routing_epsilon))
normalized[t] = R32(selected[t] / denominator)
weights[t] = R32(normalized[t] * model_route_scale)
```

投影不得重排归约或融合乘加；有限 BF16 因子仍可能使 FP32 乘积溢出，不能推定所有乘积都有限且精确。温度步骤保留在 trace；固定温度下它是恒等映射。[D]

softplus 保持 MODEL 所引算子的严格阈值分支；非线性正确舍入是 CPU 选择，不规定官方 exp/log 内部算法。sqrt 消费**已经舍入的 softplus**，不能以复合函数一次舍入替代。偏置只参与选择，权重始终来自未加 bias 的 score；防零、除法和缩放各保留独立落点。[官方][D]

top-k 以数值比较，正负零视为同分；同分编号升序。返回顺序也是归一化求和顺序。该确定性规则与顺序归约不表示官方 `torch.topk` 平局、GEMM 或 GPU 归约行为。[D]

## 3. 拒绝与环境

- 类型、序列维数或 bool 伪装编码错误抛 `TypeError`；形状、长度、编码位宽及非有限输入抛 `ValueError`。外部参数完整校验先于数值计算，调用方输入始终不变。[D]
- 每个投影乘积与逐步累加都要求有限，一旦非有限即 `ValueError`，不允许后续抵消挽救。加 bias 后的选择分数同样要求有限。失败不返回部分结果；非线性区间无法认证时抛 `ArithmeticError`，不猜测、饱和或清零兜底。[D]
- 硬件异常须屏蔽。数值不依赖调用方舍入、FTZ/DAZ；控制位不变，已有 sticky 异常位不清除，允许 Decimal 内部新增 sticky。显式 Decimal context 不继承或污染调用方的精度、舍入、traps 与 flags；不承诺完整浮点环境无副作用。[D][实现]

## 4. 观测与实现方法

`trace_route` 与普通入口共享执行核心，返回不可变 `RouteTrace`：[实现]

| 字段 | 内容 |
| --- | --- |
| `linear_bits` / `temp_bits` | 全部行的投影及温度后 FP32 |
| `softplus_bits` / `score_bits` | softplus 及 sqrt 后 FP32 |
| `selection_bits` | 加实际选用 bias 后的 FP32 |
| `indices` / `selected_bits` | 所选编号及相同顺序的未加 bias 分数 |
| `sum_bits` / `denominator_bits` | FP32 逐项和与加防零项后的标量 |
| `normalized_bits` / `weights_bits` | 除法及缩放后的 FP32 |

除两个标量字段外，其余字段均为 tuple。不会暴露每个 K 乘加的全量轨迹。[D]

位宽校验、位级编码、精确乘法、加法及有理数 RNE 复用已有参照。softplus 用 `max(x,0)+ln(1+exp(-abs(x)))` 的稳定形式和显式 Decimal exp/ln 外向区间，将舍入误差向外传播，再以精确有理数判断两端的 FP32 RNE 是否相同。`x≤−128` 时 `softplus(x)<exp(x)≤exp(−128)<(3/8)^128<2^−150`（由 `e>8/3`），故可认证为正零；其他输入从 192 位十进制精度逐次翻倍至 3072 位，仍不能认证则失败。这是实现资源上限，不是数值容差。sqrt 使用整数平方根与精确中点比较，不依赖宿主 sqrt。[实现]

## 5. 验证方法与衔接

独立期望采用 Fraction 精确运算、FP32 格点搜索、Taylor 指数界及对数级数包围；不调用被测 Decimal exp/ln、sqrt 或舍入 helper。期望后续阶段仅消费期望前阶段，核对所有约定阶段。边界、显式错误链和环境证据归 P0023，命令归 [BUILD §7](BUILD.md#7-测试清单)。[实现]

合成接口验证把 `(indices, weights_bits)` 分别映射到不同 FP4 三矩阵与既有专家 `route_weight_bits`，直接复用 P0022 执行器协议；不引入调度或聚合。专家内部权重位置与数值契约只见 [EXPERT](EXPERT.md)。此处验证范围只以 MODEL §4 的覆盖矩阵为准。[实现]
