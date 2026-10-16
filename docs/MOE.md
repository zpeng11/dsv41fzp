# MOE.md — 单 token routed 分支 CPU 契约

本页定义单 token routed 分支的参照与生产契约，不是完整 MoE。上游接口与数值分别只见 [ROUTER](ROUTER.md)、[EXPERT](EXPERT.md)，模型依据与验证链边界只见 [MODEL §3.5/§4](MODEL.md#35-路由srccore)。选择理由见 [D0020](../plan/decisions/d0020-routed-branch-reference.md)，参照与生产实施证据分别归 [P0025](../plan/p0025-routed-branch-oracle.md)、[P0026](../plan/p0026-routed-branch-scalar.md)；决策与验收状态以相应 D/P 为准。

**来源标记**：**[官方]** 只读镜像 `model.py:841–851,889–904`；**[D]** D0020 记录的 CPU 选择；**[实现]** `ref/moe.py` 与 §5 列出的生产文件。下述数字为编码、接口或算法约束，不是性能测量；环境限 [BUILD §1](BUILD.md#1-支持环境) ENV-A。

## 1. 接口与预检

- `FP4Expert(w1, w2, w3)` 按 EXPERT 的 `FP4Matrix` 表达一个专家的三矩阵。[实现]
- `routed(input_bf16, gate, bias_bits, experts, *, image_token=False, bias_vl_bits=None)` 返回 D 长 FP32 原始位 tuple，即 routed 部分和。Gate、bias 与图像选择参数沿用 ROUTER；`experts` 为按全局编号组织的一维 `Sequence[FP4Expert]`，长度须等于 Gate 行数 N，不接收生成器或多维 memoryview。[D]
- D 等于 Gate 的 K；全部专家采用共同 H，w1/w3 为 H×D，w2 为 D×H，维度与矩阵编码合法域沿用 EXPERT。不支持缺失、惰性或按需加载的专家。[D]
- 所有提供的输入、Gate、两种 bias 及全部专家矩阵在任何数值计算前完整校验并快照，包括未选用的 bias 和未选专家；之后只消费快照。调用方序列不变，允许只读内容共享。不保证并发修改或自定义序列主动修改其他对象时的一致性。[D]

## 2. 聚合与精度落点

令 `R32` 为 binary32 round-to-nearest ties-to-even，有限加法渐进下溢；`expert` 与 `route` 严格沿用各自既有契约。[D]

```text
(indices, weights_bits) = route(input_bf16, gate, bias_bits, ...)
pairs = paired indices/weights sorted by increasing expert index
accumulator = D copies of FP32 +0
for (index, weight) in pairs:
    output_bf16 = expert(input_bf16, experts[index], route_weight_bits=weight)
    for j in increasing D order:
        accumulator[j] = R32(accumulator[j] + widen(output_bf16[j]))
return accumulator as FP32 raw bits
```

路由编号与权重排序时始终配对；专家只求值一次，按全局编号递增累加，不能以 top-k 返回的选择分数次序替代。[官方][D]

专家内部已经在第二次激活量化与 W2 之前应用权重。聚合不再乘权重，不移动权重或量化落点；每个专家 BF16 输出精确拓宽后参与逐步 FP32 加法，不使用精确总和一次舍入、宿主浮点 sum 或中间 BF16 累加。[官方][D]

routed 部分和保持 FP32，不转换 BF16。官方在 routed 求和后加入共享专家才转换回输入 dtype；本入口停在共享支路加入之前，不能将其输出视为完整 MoE 输出。编号顺序与逐步 RNE 是确定性的 CPU 规范，不推定官方 GPU kernel 或多进程 all-reduce 的逐位行为。[官方][D]

## 3. 特殊值、失败与环境

- 上游拒绝边界保持不变。专家最终 BF16 Inf/NaN 仍是合法输出；加法传播 NaN，异号 Inf 相加得 NaN，溢出得带符号 Inf，不因后续抵消而回退。NaN 的比较口径沿用 [COMPUTE](COMPUTE.md)，不承诺 payload 或符号。[D]
- 精确抵消及异号零相加得正零，同号负零相加得负零；初始累加器为正零。[D]
- 类型或 bool 伪装编码错误抛 `TypeError`，形状、长度、位宽或非法内容抛 `ValueError`。参数校验先于路由/专家求值；上游数值拒绝或非线性认证失败按原异常传播，不返回部分结果或部分 trace。[D]
- 硬件异常须屏蔽。数值不依赖调用方舍入/FTZ/DAZ；控制位不变、已有 sticky 异常位不清除，允许上游 Decimal/Python 新增 sticky。显式 Decimal context 不继承或污染调用方精度、舍入、traps 与 flags；不承诺完整浮点环境无副作用。[D]

## 4. 观测与验证方法

`trace_routed` 与普通入口共用执行核心，返回不可变 `RoutedTrace`：[实现]

| 字段 | 内容 |
| --- | --- |
| `route` | 既有 `RouteTrace`，保持原路由返回顺序 |
| `expert_indices` | 实际求值与累加的专家编号，递增 |
| `expert_traces` | 同序的既有 `ExpertTrace` |
| `partial_sums_bits` | 每个专家加入后的 D 长 FP32 位 tuple |
| `output_bits` | 最后一个部分和，即正式输出 |

聚合复用既有 FP32 加法；trace 不新增整矩阵点积轨迹，也不改变上游观测字段。[实现]

新增聚合期望用独立精确算术与 FP32 格点判定生成，覆盖逐步舍入与特殊值；完整组合核对实际路由、专家映射、权重落点及部分和。不可控的特殊值用私有聚合节点定向补证，不冒充真实专家组合证据。错误链与环境验证过程归 P0025，复现入口只见 [BUILD §7](BUILD.md#7-测试清单)。[实现]

## 5. 生产 CPU scalar

本节来源为 `include/df/core/moe.h`、`src/core/moe.cpp` 与用户审议通过的 [P0026](../plan/p0026-routed-branch-scalar.md) 设计；数值语义继承 §2～§3 与 D0020，不另立数值选择。支持环境限 BUILD 的 ENV-A，验收证据与状态由 P0026 记录。[实现]

- **接口与存储**：`df::core::routed(input, gate, bias, experts, work, out, image_token, bias_vl)` 接收按全局编号排列的 `RoutedExpert` span，每项复用既有 `ExpertMatrix` 表达 w1/w2/w3。`RoutedWorkspace` 含既有 router/expert 工作区、一份 D 长 BF16 专家临时输出及 D 长 FP32 累加区；工作区允许更大容量，仅有效前缀参与写入。`out` 至少 D 个 FP32 raw-bit 元素，仅成功后提交该前缀，不做最终 BF16 转换。[实现]
- **预检与快照**：全部尺寸、容量与字节数，以及实际读区、有效写前缀和完整正式输出的地址区间检查先于数值载荷读取；全部输入、Gate、两种 bias 和专家矩阵在首次数值写入前检查，包括未选专家。调用方拥有全部存储；入口参数与当前使用的描述符按值快照，数值载荷保持只读，不复制全模型权重，不保证并发修改的一致性。专家数、共同 H 与三矩阵关系沿用 §1；上游校验由私有预检复用。[实现]
- **别名与失败**：有效工作区写前缀互不重叠，并与所有只读输入（含专家描述符表）及完整 `out` span 分离；正式输出有效前缀与只读输入分离，只读内容可共享、精确相邻允许。参数拒绝抛 `std::invalid_argument` 且全部存储不变；上游运行中拒绝或认证失败原样传播，工作区有效前缀可变，完整正式输出与只读输入保持不变。[实现]
- **观测与环境**：普通与私有观测共用核心。`RoutedTrace` 只保留实际路由结果、升序执行编号与每步 FP32 部分和，不复制专家内部观测。私有调用另需至少 `6*D` 个 FP32 位的暂存工作区；其有效前缀允许在运行失败时改变，正式 trace 对象及部分和仅成功后提交，尾部不写。观测相关对象与缓冲同样参与跨组件别名检查。普通与观测均无动态数值存储分配，纯整数数值路径在成功与拒绝下保持完整 MXCSR 不变。[实现]
- **验证路由**：真实生产组合直接对照 `ref/moe.py` 的路由结果、执行编号、部分和及最终输出；聚合边界复用 P0025 的独立精确期望，同一私有聚合节点仅用于定向补证。公共消费者、存储/失败、环境及错误检出证据归 P0026，命令只见 BUILD §7。[实现]
