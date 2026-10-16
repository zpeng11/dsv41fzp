# COMPUTE.md — 专家分组点积与线性层 CPU 契约

本页定义官方 FP8×FP4 路径阶段约束下的单输出 CPU 规范及复用它的完整线性层契约，不宣称官方 GPU 逐位等价。选择理由见 [D0008](../plan/decisions/d0008-expert-grouped-dot-reference.md)，实施与验收见 [P0012](../plan/p0012-expert-grouped-dot-oracle.md)，验证链见 [MODEL §4](MODEL.md#4-精确性链条)。模型事实只引用 MODEL，格式与既有舍入只引用 [QUANT](QUANT.md)。FP8×FP8 的独立单输出契约见 §8，选择理由见 D0021。

## 1. 来源与参照边界

**来源标记**：**[代码]** 官方只读镜像，来源见 [PROVENANCE](../third_party/reference/PROVENANCE.md)；**[格式]** QUANT 的格式与舍入定义；**[D]** D0008 接受的项目计算语义。

- `model.py:27–30,186–194` 明确 UE8M0 scale 及激活、权重沿 K 每 32 元素分组，覆盖 kernel 的激活组默认值。[代码] 模型用途与格式见 [MODEL §2](MODEL.md#2-结构与格式)。
- `kernel.py:478–507,525–557` 给出 FP4→FP32→FP8、块内 GEMM、`(block_dot * scale_a) * scale_b`、跨块累加及最终输出转换的阶段；累加为 FP32，默认输出为 BF16。E2M1 有限值可由 E4M3FN 精确表示，CPU 直接复用解码值。[代码][格式]
- 源码不确定 `T.gemm` 内部归约、设备有效累加精度、编译后融合/重排、cast 舍入及 FTZ；以下顺序 FP32 RNE、无融合和渐进下溢是项目选择。[D]

以下数字是规格与算法约束，不是性能测量；CPU 评审探针的环境、方法和结果见 P0012。

## 2. 输入与失败契约

输入为四份一维原始编码序列：E4M3FN 激活 `a_codes`、沿 K 打包的 E2M1 权重 `w_packed`、E8M0/UE8M0 激活 scale `scale_a` 与权重 scale `scale_b`，仅计算一个输出。[D]

- `K = len(a_codes)`，要求 `K > 0` 且整除模型固定组大小 32；权重字节数为 `K/2`，两份 scale 长度各为 `K/32`。第 j 个块分别使用 `scale_a[j]` 与 `scale_b[j]`。[代码][D] 不支持 128 元素激活组或可配置组大小。
- 每个 packed 字节的低 nibble 对应先出现的 K 元素，后接高 nibble；长度严格匹配，不广播、不补齐、不推断或生成 scale，不接受矩阵、stride 或隐式 flatten。[格式][D]
- 原始编码只收位宽内非负 Python 整数，拒绝 bool、浮点或负的 int8 数值。NaN 激活编码 `0x7F/0xFF` 与 NaN scale 编码 `0xFF` 拒绝；E8M0 有限编码均为严格正值，不能编码零或 Inf。全部 E2M1 nibble 含正负零均合法。[格式][D]
- 输出模式只接受 `"fp32"` 与 `"bf16"`，默认 `"fp32"`。前者返回精确承载 binary32 结果的 Python float，后者返回 BF16 原始整数位模式；模式不改变块内和跨块计算。[D]
- 空输入、长度或分组错误、超宽/负编码、NaN 输入与非法输出模式抛 `ValueError`；非整数编码抛 `TypeError`。元数据先校验，随后全部输入内容校验，最后计算；不会返回坏输入的部分结果。[D] 本入口校验不宣称官方所有入口也如此拒绝。

## 3. 逐步计算与舍入

令 `R32` 表示一次 binary32 RNE（中点取偶）；各步结果立即落到 FP32，不能跨步骤保留额外精度。每个块处理 32 项，块及项均按 K 索引递增。[D]

```text
acc = +0
for j in ascending block order:
    dot = +0
    for r in 0..31:
        p = R32(decode_e4m3fn(a_codes[32*j+r]) * decode_e2m1(weight_code[32*j+r]))
        dot = R32(dot + p)
    after_a = R32(dot * decode_e8m0(scale_a[j]))
    block = R32(after_a * decode_e8m0(scale_b[j]))
    acc = R32(acc + block)
fp32_result = acc
bf16_result = RNE_BF16(fp32_result)
```

禁止树形归约、精确求和后只舍入一次、跨块重排、折叠或交换两份 scale、先逐元素 dequant 后点积、乘加融合以及提前 BF16 舍入。BF16 仅在全部跨块累加完成后进行一次转换，不是 Python binary64 直接转 BF16。[D][格式]

合法输入的中间下溢按渐进下溢处理，上溢产生带符号 Inf；不钳制，不将非有限块贡献当作零或跳过。NaN 传播，异号 Inf 相加产生 NaN，同号 Inf 相加保留 Inf，Inf 与有限数相加保留 Inf。[D] 两份 scale 有限且严格为正，中间 Inf 乘下一份 scale 仍为原符号 Inf。

有限非零乘法的零符号遵守操作数符号异或，下溢到零保留乘积符号；RNE 加法的精确异号相消为正零，同号零相加保留符号、异号零相加为正零。NaN 不承诺 payload、quiet/signaling 状态或符号；有限值（含零）及 Inf 按位检查，NaN 按分类检查。最终 BF16 转换复用 QUANT 的 RNE 与特殊值规则。[D][格式]

## 4. 实现与验证约束

数值语义不依赖宿主浮点加乘、环境舍入或 FTZ，不读取或修改调用方浮点环境，也不改写既有格式 oracle。执行仅依赖 Python 标准库，不初始化官方 GPU 栈或依赖真实权重、设备 dump、在线来源；平台支持只能按实测证据声明。[D]

参照入口为 `ref/compute.py` 的 `expert_grouped_dot` 与 `trace_expert_grouped_dot`。分阶段观测记录同一执行路径的块内结果、两次 scale 乘法、每次跨块累加和最终输出位模式，不另写测试专用点积。独立期望由标准库 `Fraction` 位域定义、精确算术与相邻格点逐步 RNE 构造，不调用被测计算或舍入辅助；敏感性只在临时源码副本注错。[D] 复现命令见 [BUILD §7](BUILD.md#7-测试清单)。

## 5. 验证链路由

独立验收后的 Python 参照是 §6 生产 C++ scalar 的上一级；同契约快路径只对照 §6，不直接对照 Python 参照。该计算分支不是逐元素 dequant 的已验证后继。[D]

覆盖与不覆盖范围见 [MODEL §4](MODEL.md#4-精确性链条) 覆盖矩阵。激活编码与 scale 的上游 CPU 规范见 [ACTQUANT](ACTQUANT.md)，本入口只消费给定编码与 scale，不生成它们。[D] 三矩阵与非线性的单专家参照组合见 [EXPERT](EXPERT.md)。

## 6. 生产 C++ scalar（`src/quant`）

本节来源标记：**[D0009]** [D0009](../plan/decisions/d0009-compute-scalar-integer-path.md) 接受的接口与实现选择 · **[C++计算]** 本仓 `include/df/quant/compute.h` 与 `src/quant/compute.cpp` · **[计算测试]** `tests/quant/test_compute_parity.py` 经测试执行器的实测，环境见 [BUILD §1](BUILD.md#1-支持环境)，方法与结果见 [P0013](../plan/p0013-expert-grouped-dot-scalar.md)。数值语义仍以 §3 为唯一来源。

- **表示**：入口 `df::quant::expert_grouped_dot`，四份 `std::span<const uint8_t>` 只读视图依次为激活、打包权重、`scale_a`、`scale_b`；K 取激活视图长度，长度要求同 §2。输出为调用方提供的 FP32 原始位 `std::span<uint32_t>` 或 BF16 原始位 `std::span<uint16_t>` 重载，只写第一个元素；重载不改变中间计算。[D0009][C++计算]
- **失败**：只抛 `std::invalid_argument`：K 为零或不整除 32、权重或 scale 长度不符、输出为空、NaN 激活编码 `0x7F/0xFF` 或 NaN scale 编码 `0xFF`、任一输入与输出元素有字节重叠（相邻不算）。长度与重叠先于任何输入读取，内容校验先于输出写入，失败时输出不变。本入口不做元数据乘法，没有 `std::overflow_error` 类；K=0 被拒绝，与 QUANT 反量化接受零元素张量的约定有意不同。Python 侧的类型、宽度与输出模式拒绝属于 §2 的 oracle 契约，C++ 由定宽类型与重载表达，两者不一一对应。[D0009][C++计算][计算测试]
- **数值与环境**：全部阶段以定宽整数位运算构造，不执行浮点运算，不读写浮点环境。块内乘积复用 QUANT §7 的有限乘法；乘 scale 前，Inf/NaN 中间值先分流（scale 严格为正，原样保留），只有有限值进入乘法。加法采用定宽对齐与 sticky，复用既有 binary32 RNE 构造；合法输入产生的 Inf/NaN 是结果，不是失败。在 [BUILD §7](BUILD.md#7-测试清单) 的八模式 MXCSR 下（含状态位预置），逐阶段结果均与 `ref/compute.py` 一致，完整 MXCSR 不变。[D0009][C++计算][计算测试]
- **验证边界**：公开两种入口与私有 trace 连接执行同一计算核心，逐阶段对照 §4 的观测；私有加法探针对照 `ref/compute.py` 的加法，只补充公开路径难以构造的边界，不替代公开路径证据。不覆盖项见 [MODEL §4](MODEL.md#4-精确性链条)，测试入口见 [BUILD §7](BUILD.md#7-测试清单)。[D0009][计算测试]

## 7. 单 token 专家线性层

本节为 [D0014](../plan/decisions/d0014-expert-linear-scalar.md) 接受的生产接口契约，实施与验收证据见 [P0019](../plan/p0019-expert-linear-scalar.md)。数值规格不新增：输入量化遵循 [ACTQUANT](ACTQUANT.md)，每个输出行的点积与最终转换遵循本页 §3、§6；维度和分组数字为契约约束，不是性能测量。[D0014]

- **入口与所有权**：`include/df/quant/linear.h` 的 `df::quant::expert_linear(input, n, w_packed, scale_b, a_codes, scale_a, out)`；input 是 BF16 原始位只读 span，权重与 scale_b 是字节只读 span，两个工作缓冲是可写字节 span。out 提供 FP32 原始位 `span<uint32_t>` 与 BF16 原始位 `span<uint16_t>` 两个重载；调用方拥有全部存储，库不动态分配。[D0014]
- **布局与容量**：K 为 input 长度，N 为 n，两者为正且 K 整除组大小 32。权重按输出行连续存储，每行 K/2 字节，scale_b 每行 K/32 字节；两份输入长度须分别精确等于 `N*(K/2)`、`N*(K/32)`。w1/w3 与 w2 仅 N/K 不同，不转置、不广播、不接收任意 stride。a_codes、scale_a、out 容量至少 K、K/32、N，仅写有效前缀。[D0014]
- **别名与拒绝**：三个实际写入前缀互不重叠，并与所有输入的实际读取区间不重叠；只读输入之间允许重叠，未使用尾部不参与判定，精确相邻允许。维度、尺寸及字节计数溢出、长度、容量与别名检查先于内容读取；全矩阵 scale_b 的 `0xFF` 编码及非有限 BF16 输入均在任何写入前拒绝。所有约定参数错误抛 `std::invalid_argument`，输出与两份工作缓冲全部不变，即使错误位于最后一行。合法结果中的 Inf/NaN 不算拒绝。[D0014]
- **组合与环境**：调用一次生产 act_quant，将其实际输出缓冲供全部行的生产 expert_grouped_dot 直接消费；FP32/BF16 重载不改变中间计算。沿用整数数值路径，不读取或修改调用方浮点环境。不提供整矩阵 trace 或新的舍入、归约路线。[D0014]
- **验证路由**：上一级与覆盖边界见 [MODEL §4](MODEL.md#4-精确性链条)，测试及离线工具见 [BUILD §7](BUILD.md#7-测试清单)，验收方法与证据见 P0019。[D0014]

## 8. FP8 块缩放单输出点积

本节是 [D0021](../plan/decisions/d0021-fp8-grouped-dot-reference.md) 接受的独立契约，实施验收见 [P0027](../plan/p0027-fp8-grouped-dot-oracle.md)。数值为格式或算法约束，不是性能测量；环境名称只引用 [BUILD §1](BUILD.md#1-支持环境)。本节 **[代码]** 指官方只读镜像，**[格式]** 指 QUANT，**[D0021]** 指项目 CPU 选择，不将其视为 D0008 已验收范围。

### 8.1 来源与选择

- `model.py:27–30,196–205` 将 FP8 激活量化及 FP8 GEMM 的 K 分组均设为 32；`model.py:225–233` 的权重为 E4M3FN `[N,K]`，E8M0 scale 为 `[ceil(N/32),ceil(K/32)]`，与 FP4 的逐输出行 scale 不同。[代码][格式]
- `kernel.py:227–270,277–307` 给出 E4M3FN×E4M3FN 块内 GEMM、先激活 scale 后权重 scale、FP32 跨块累加及输出转换，入口要求 K 整除分组大小。权重行 n 的第 j 个 K 块使用 `scale_b[n//32,j]`；模型默认输出 dtype 为 BF16。[代码]
- 源码不能确定 `T.gemm` 内部归约及有效精度、编译后的融合/重排、设备 cast/FTZ。本节选择顺序 FP32 RNE、无融合及渐进下溢，建立可独立核验的 CPU 规范，不据此推断 GPU 逐位等价。[D0021]

### 8.2 输入与失败

入口 `ref/fp8_compute.py::fp8_grouped_dot(a_codes, w_codes, scale_a, scale_b, *, output="fp32")` 仅消费四份一维编码序列，计算一个输出；`trace_fp8_grouped_dot` 使用同一输入契约。[D0021]

- 激活与权重均为 E4M3FN 原始字节编码，长度同为 K，K 为正且整除 32。两份 E8M0 scale 长度严格为 K/32，第 j 块分别消费 `scale_a[j]` 和 `scale_b[j]`。不打包 nibble，不广播、不补齐、不生成 scale，不接收矩阵或 stride。[代码][格式][D0021]
- 接受 `collections.abc.Sequence` 的一维序列；`memoryview` 须为一维。元素只接受 `0..255` 的 Python int，拒绝 bool、浮点和负的 int8 表示。激活及权重各自的 NaN 编码 `0x7F/0xFF`、任一 scale 的 NaN 编码 `0xFF` 均拒绝；其余 E4M3FN 编码含正负零均合法，有限 E8M0 为严格正的幂。[格式][D0021]
- `output` 仅为 `"fp32"` 或 `"bf16"`，默认前者；分别返回精确承载 binary32 的 Python float 或 BF16 原始整数位，模式不改变中间计算。[D0021]
- 非序列、非一维 memoryview 或非整数元素抛 `TypeError`；非法模式、长度/分组、越界编码与 NaN 输入抛 `ValueError`。先验证元数据，再取得并验证全部编码快照，最后计算；不改输入，失败不返回部分结果，不宣称官方入口有相同拒绝政策。[D0021]
- 多行权重只在测试侧抽取连续行及对应二维 scale 行后调用上述单输出入口；输出行尾块使用实际行数，不引入 K 补齐。[代码][D0021]

### 8.3 执行与观测

以下是新的完整执行链，不调用 FP8×FP4 点积。`R32`、逐步 FP32 RNE、加法/乘法的零符号、渐进下溢、Inf/NaN 传播及最终 BF16 转换政策，明确共用本页 §3 的算术定义；其中 FP4 解码和 packed 布局不适用于本节。[格式][D0021]

```text
acc = +0
for j in ascending K-block order:
    dot = +0
    for r in 0..31:
        i = 32*j + r
        p = R32(decode_e4m3fn(a_codes[i]) * decode_e4m3fn(w_codes[i]))
        dot = R32(dot + p)
    after_a = R32(dot * decode_e8m0(scale_a[j]))
    block = R32(after_a * decode_e8m0(scale_b[j]))
    acc = R32(acc + block)
fp32_result = acc
bf16_result = RNE_BF16(fp32_result)
```

每个乘积、每次加法和两次 scale 应用分别落到 FP32，不移动、交换或合并 scale，不树形归约、不以精确求和替代逐步归约、不融合乘加、不提前 BF16。[D0021] 合法有限编码产生的中间及最终 Inf/NaN 是计算结果，不作为输入拒绝；有限值、零和 Inf 按位比较，NaN 只按分类比较，payload 与符号不作承诺。[D0021]

普通与 trace 共用执行核心。观测包含每块的 32 个乘积位、32 个块内部分和位、块内点积位、两次 scale 后位和跨块累加位，以及最终 FP32/BF16 位和所选模式的正式结果。独立期望使用标准库精确算术与格点 RNE，不调用被测解码、计算或舍入 helper；最终输出相同不能替代阶段一致。[D0021]

### 8.4 环境与验证路由

数值运算不依赖宿主浮点加乘、环境舍入或 DAZ/FTZ，不读取或修改调用方浮点环境。只依赖 Python 标准库，支持声明限 ENV-A；普通与 trace 的成功及拒绝调用均要求完整 MXCSR 不变，阶段位及正式输出在 [BUILD §7](BUILD.md#7-测试清单) 的八模式与状态位清空/预置下核对。[D0021]

数值阶段以整数位域执行，复用既有 binary32 构造、有限值拆分及 BF16 位转换；FP8 乘积、加法与 scale 应用仅作本计算链私有构件，最终 FP32 结果才构造精确 binary64 载体。不经值级 `fp32_bits` 或 `as_integer_ratio` 往返，不修改旧 oracle 的语义，不因复用而免除本节完整执行链的独立验证。既有 [ACTQUANT](ACTQUANT.md) 的实际编码与 scale 可直接作为输入，组合只在测试侧存在。覆盖矩阵见 [MODEL §4](MODEL.md#4-精确性链条)，增量复现入口见 [BUILD §7](BUILD.md#7-测试清单)。[D0021]

### 8.5 生产 C++ scalar

本节 **[C++FP8]** 指 `include/df/quant/compute.h`、`src/quant/fp8_compute.cpp`；生产方式沿用 D0009，数值语义仍只见 §8.3。实施与验收证据归 [P0028](../plan/p0028-fp8-grouped-dot-scalar.md)，不以新增接口代替验收。

- **表示**：`df::quant::fp8_grouped_dot` 消费四份 `std::span<const uint8_t>`，依次为激活、权重、`scale_a`、`scale_b`，长度遵循 §8.2。输出为调用方提供的 FP32 原始位 `span<uint32_t>` 或 BF16 原始位 `span<uint16_t>`，只写首元素；库不动态分配，不生成 scale。[C++FP8]
- **失败与存储**：约定参数错误均抛 `std::invalid_argument`：K、权重或 scale 长度非法，输出为空，激活或权重的 NaN 编码、scale 的 NaN 编码，以及任一输入与输出首元素字节重叠。元数据与重叠检查先于内容读取，全部内容校验先于任何写入；失败时输出不变。只读输入可互相重叠，精确相邻允许，未使用输出尾部不参与别名检查且不写入。公共入口不做元数据乘法，没有 `std::overflow_error` 类；Python 类型与模式拒绝由 C++ 定宽类型及重载表达。[C++FP8]
- **执行与观测**：全部阶段复用既有整数解码、有限乘法、含特殊值的乘法与加法、BF16 RNE 构造，不执行宿主浮点运算，不读写浮点环境。公开入口与私有 trace 执行同一核心；trace 使用 §8.3 的阶段位，长度精确匹配 K 与 K/32，字节计数不得溢出，所有观测写区及正式输出首元素彼此不重叠且与输入不重叠。私有入口也在写入前完成全部校验，失败时正式输出与观测均不变。[C++FP8]
- **验证路由**：上一级为 `ref/fp8_compute.py`，新增路径另以 P0027 已验收独立期望核对；二维 scale 选行及生产激活量化的实际输出消费仅在测试侧组合。覆盖边界只见 MODEL §4，实际复现入口只见 BUILD §7。[C++FP8]
