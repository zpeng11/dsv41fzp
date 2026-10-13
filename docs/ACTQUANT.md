# ACTQUANT.md — 动态激活量化 CPU 参照契约

本页定义模型实际调用的单行 BF16 激活量化规范。选择理由见 [D0010](../plan/decisions/d0010-act-quant-reference.md)，实施与验收证据见 [P0014](../plan/p0014-act-quant-oracle.md)，模型用途与格式只引用 [MODEL §2](MODEL.md#2-结构与格式)，格式定义只引用 [QUANT §2](QUANT.md#2-位模式与零符号)。本契约不宣称官方 GPU 逐位等价。

## 1. 来源与边界

**来源标记**：**[代码]** 官方只读镜像（来源见 [PROVENANCE](../third_party/reference/PROVENANCE.md)）· **[格式]** QUANT · **[D]** D0010 接受的项目 CPU 语义 · **[参照]** [ref/act_quant.py](../ref/act_quant.py) · **[测试]** [tests/quant/test_act_quant.py](../tests/quant/test_act_quant.py)。以下数字是规格或算术约束，不是性能测量。

- `model.py:27–30,181–198` 指定 `act_quant(x, 32, "ue8m0", float8_e8m0fnu)`；FP4 与 FP8 线性层共用该入口，专家 w2 输入在 `model.py:841–851` 转回 BF16。[代码] 组大小取模型实际调用，不取 kernel 默认值。
- `kernel.py:22–38,74–91` 给出 absmax、amax 下限、FP32 乘倒数、位域 ceil(log2)、幂 scale、商的 clamp 与 FP8/scale 写入阶段。[代码]
- 只覆盖单行、固定组、非 inplace 的 BF16→E4M3FN+E8M0。矩阵/批量、FP32 输入、非幂 scale、128 元素组、inplace 量化后反量化、`fp4_act_quant`、权重量化和 checkpoint 转换均不在本契约内。[D]

## 2. 输入、输出与失败

`act_quant(bf16_bits)` 接收一维 BF16 原始整数位序列，返回 `(a_codes, scale_a)` 两个不可变 tuple，分别可直接作 [COMPUTE §2](COMPUTE.md#2-输入与失败契约) 的激活编码与激活 scale。[参照]

- `K = len(bf16_bits)`，要求 `K > 0` 且整除 32；输出 K 个 E4M3FN 字节编码及 K/32 个 E8M0 编码，第 j 组只由输入 `[32*j:32*(j+1)]` 决定。[代码][D]
- 位模式只收 `0..0xFFFF` 的 Python 整数，不接受 bool、浮点值或负 int16 数值；每个序列元素表示一个完整 BF16 位模式，不把相邻字节拼成 BF16。不隐式 flatten、补齐或广播，不修改输入。[格式][D][参照]
- 空输入、分组错误、超宽/负编码及 BF16 NaN/Inf 抛 `ValueError`；非序列、非一维 `memoryview` 及非整数元素抛 `TypeError`。先验证元数据，再验证全部内容，最后计算；错误不返回部分结果。[D][参照] 不宣称官方入口也拒绝非有限输入。

## 3. 数值阶段

令 `R32` 表示一次 binary32 RNE（中点取偶），`R8` 表示 E4M3FN RNE。BF16 位映射为 `code << 16`；下限 `fl32(1e-4)` 位为 `0x38D1B717`，倒数 `fl32(1/448)` 位为 `0x3B124925`。[格式][D][测试]

```text
for each group of 32 elements:
    amax = max(abs(x))
    bounded_amax = max(amax, fl32(1e-4))
    product = R32(bounded_amax * fl32(1/448))
    e = ceil(log2(product))
    scale_a[j] = e + 127
    for each element x:
        quotient = R32(x / 2^e)
        clamped = clamp(quotient, -448, 448)
        a_codes[i] = R8(clamped)
```

- absmax 使用原始 BF16 位的幅值顺序，不经宿主浮点解码。乘积立即落为 binary32 位；其指数规则是源码阶段，不以精确 `amax/448` 偷换该阶段。[D][参照] 以精确 `1/448` 代替 `fl32(1/448)` 在 BF16 域上不改变指数、scale 与最终编码，但不保证乘积阶段位相同：amax `0x38D2` 时源码乘积位为 `0x34700001`，精确除以 448 再 R32 为 `0x34700000`。[测试]
- 商采用渐进下溢，不采用 DAZ/FTZ。原始零与下溢到零都保留输入符号；E4M3FN 零也保号，负零编码为 `0x80`。[D][格式]
- 全零组或 amax 低于下限时，指数为 −22，scale 编码为 `0x69`。下限只改变 scale 计算，不能把低幅度非零元素清零：全组 `2^-14` 的商为 256、FP8 编码为 `0x78`；全组 `2^-15` 的商为 128、编码为 `0x70`。[D][测试]
- 全部有限 BF16 amax 上，源码 FP32 阶段指数与精确 `ceil(log2(max(amax, fl32(1e-4))/448))` 一致，指数范围 −22～120，E8M0 编码范围 `0x69`～`0xF7`；独立期望逐一检查 `amax/2^e ≤ 448`。因此 clamp 在该输入域内不可达，但参照保留其阶段与观测。[测试] 不把此结论外推到 FP32 输入。

## 4. 观测与环境

`trace_act_quant` 与普通入口调用同一个执行核心。`ActQuantTrace` 包含 `groups`、`a_codes` 与 `scale_a`；每个 `GroupTrace` 记录 `amax_bits`、`bounded_amax_bits`、`product_bits`、`exponent`、`scale_code`，以及各含 32 项的 `quotient_bits`、`clamped_bits`、`a_codes`。[D][参照]

BF16 解码、amax、乘积、商、clamp、编码与 trace 均通过整数位域构造，不调用既有值级 `decode_bf16`/`fp32_from_bits`，不经宿主 float 往返，不读取或修改调用方浮点环境。仅依赖 Python 标准库，不初始化 Torch/TileLang/CUDA，不依赖权重、设备 dump 或网络。[D][参照]

阶段位是契约的一部分。BF16 次正规被冲零或 binary32 商下溢规则错误时，最终 FP8 仍可能是同号零；最终编码相同不能替代中间位一致。[D]

在 ENV-A 的 [BUILD §7](BUILD.md#7-测试清单) 八模式 MXCSR 下（含状态位预置），普通及 trace 的定向结果与独立期望一致，接受与拒绝路径的完整 MXCSR 不变；探针实现与平台边界见 BUILD §7。[测试]

## 5. 独立验收方法

仅声明 [BUILD §1](BUILD.md#1-支持环境) 的 ENV-A，复现命令见 [BUILD §7](BUILD.md#7-测试清单)。[测试]

期望使用标准库 `Fraction` 位域定义、精确算术与相邻编码距离比较，不调用被测解码、amax、scale、除法或编码辅助；scale 另由精确除以 448 的指数规则独立复核，每组及每元素的 trace 位与普通输出均检查。必选用例覆盖有限 BF16 amax 穷举、有限 BF16 元素量化、非有限拒绝、§3 所列边界与跨 128 元素布局，并把输出接入计算参照比较。敏感性只在临时源码副本注错，只接受指定用例的数值或阶段位比较失败为证据。用例清单、敏感性变体与门禁分类见 [P0014](../plan/p0014-act-quant-oracle.md)。[D][测试]

## 6. 验证链与待仲裁项

本参照与 [COMPUTE](COMPUTE.md) 组合提供 BF16 激活出发的单输出 CPU 规范验证；组合只存在于测试，不新增组合入口，不修改既有格式、反量化或计算 oracle。覆盖与不覆盖范围见 [MODEL §4](MODEL.md#4-精确性链条) 覆盖矩阵。[D]

官方 `reduce_absmax`/`T.max` 的 NaN 语义、编译后的 fast-math/FTZ、乘法与除法指令、FP8/E8M0 cast 的舍入与饱和、多行分块是否影响单行结果均待设备仲裁。生产 C++ 的接口、数值路径和模块归属另立决策，不从本页推定。[D]
