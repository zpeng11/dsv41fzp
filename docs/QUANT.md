# QUANT.md — 格式 oracle 数值契约

模型用途、分组大小与官方路径以 [MODEL.md](MODEL.md#2-结构与格式) 为模型事实源。本页归档 P0006 首块 `ref/` 格式 oracle 的位表示、异常契约和验证边界，以及 P0010 生产 C++ codec（§6）与 P0011 给定 scale 反量化（§7）的接口、失败与环境契约，不建立第二个生产格式配置中心；生产量化真源为 `src/quant`。

## 1. 来源与验证方法

- **[MX]** [OCP Microscaling Formats (MX) v1.0，Sep 2023](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)：§5.3、Table 5（E2M1）、Table 7（E8M0）、Table 2（E4M3，复录 OFP8）。原件 SHA256：`d195d6a36dd4a0c89064af0c479bcaad5c0fe29d63f628502ea6d7c4b4279421`；2026-10-10 从 [原件的存档快照](https://web.archive.org/web/20261002180025id_/https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) 取回。
- **[FP8]** [FP8 Formats for Deep Learning，arXiv:2209.05433v2](https://arxiv.org/abs/2209.05433v2)：Table 1 交叉验证 E4M3FN 的特殊编码、bias 与次正规范围。
- **[BF16]** [PyTorch BFloat16.h](https://raw.githubusercontent.com/pytorch/pytorch/main/torch/headeronly/util/BFloat16.h)：2026-10-10 检索快照，`detail::round_to_nearest_even`，RNE 默认；来源文件 SHA256：`302c7ba9e45d361efc5ef6ebce3b67a0cc4083bd09b60851e350273d650ff48b`。该 URL 为浮动 main，不宣称已 pin 对应上游 commit。
- **[PTX]** [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)：§5.2.3 替代浮点格式、§6.5.2 rounding、§9.7.3 subnormal/`.ftz`；仅用作浮点契约依据，不当作本机 GPU parity 证据。
- **[代码]** 官方模型只读镜像的 commit 与来源见 [PROVENANCE](../third_party/reference/PROVENANCE.md)。上述格式定义补足官方实现仅通过 dtype 指定、未自行定义的编码语义。
- **[参照]** 本仓 [ref/formats.py](../ref/formats.py)、[ref/dequant.py](../ref/dequant.py)；**[测试]** [tests/quant/test_formats.py](../tests/quant/test_formats.py)。
- **[C++]** 本仓 [include/df/quant/codec.h](../include/df/quant/codec.h)、[src/quant/](../src/quant/)；**[C++测试]** [tests/quant/test_codec_parity.py](../tests/quant/test_codec_parity.py) 经测试执行器 [quant_exec.cpp](../tests/quant/quant_exec.cpp) 调用生产库。

本页编码数字是规格值，不是吞吐或性能测量。验证环境为 [BUILD §1](BUILD.md#1-支持环境) 的 ENV-A；依赖仅标准库，不依赖 NumPy、Torch、GPU 或模型权重。方法为整数位域解码、整数有效数乘法与显式 RNE；独立测试用 `Fraction` 位域定义和相邻格点距离比较。验收命令与证据保留在 [P0006](../plan/p0006-ref-format-oracle.md#收尾)。

## 2. 位模式与零符号

| 格式 | 本 oracle 契约 | 依据 |
| --- | --- | --- |
| E2M1 | `S EE M`，bias 1；幅值及整数格见 MODEL §2.3；原生解码保留 `0000` 正零和 `1000` 负零；无 Inf/NaN | [MX] Table 5 |
| E4M3FN | `S EEEE MMM`，bias 7；含次正规、正负零；`0x7F/0xFF` 是 NaN，无 Inf；不是 FNUZ 变体 | [MX] Table 2、[FP8] Table 1 |
| E8M0 / UE8M0 | `eeeeeeee`，无符号，bias 127；`0x00..0xFE` 为 `2^(code-127)`；`0xFF` 是 NaN；没有零或 Inf。`0x00` 在 FP32 中是次正规数而非零 | [MX] Table 7 |
| BF16 | `S EEEEEEEE MMMMMMM`，bias 127；解码为相同位段左移到 FP32，含次正规、正负零、Inf、NaN | [BF16]、[PTX] |

官方 `convert.py:13–15` 的 FP4_TABLE 将码点 8 写成正零。[代码] 本 oracle 的原生 E2M1 解码遵守带符号零定义，不宣称该码点与官方 LUT 逐位相同；独立测试显式验证这一差异，其他码点与官方常量按位相等。[MX][测试]

E2M1 整数格为解码值乘二的精确整数。正负零都变成整数零，反向整数格规范编码为码点 0；只接受精确整型格点，拒绝 bool、浮点输入及非格点。原始编码可逆性由 nibble/byte 层保证，不要求数值或整数格恢复负零编码。[参照][测试]

所有原始 nibble/byte/位模式只收对应宽度的非负 Python 整数，不接受负的 int8 数值。打包输入用 `bytes` 或 `0..255` 元素；来自 int8 存储时应在调用方按位重解释为原始字节，不能算术右移。低半字节是最后维度中先出现的元素；逻辑最后维必须偶数，逐行配对，不截断、不填充。显式 shape 与元素数量必须精确匹配。[代码] `convert.py:31–34`；[参照][测试]

## 3. 浮点运算与输出

- `fp32_bits` 将 Python float/int 按 binary32 的 RNE 转换；`fp32_from_bits` 读取位模式。NaN payload/quiet-signaling 状态不承诺保留。[参照]
- `multiply_fp32` 先将两个操作数落到 FP32，精确乘整数有效数，再进行一次 FP32 RNE。次正规结果渐进下溢，零符号遵守乘法符号；上溢为带符号 Inf，不钳制。实现不依赖宿主 FP32 乘法、GPU FTZ 或库 dtype。[参照][测试] 这是本 oracle 选择的运算契约；MX §5.1 对超出 Float32 范围的编码值允许实现自定义，不能从该选择推出所有 MX/GPU 路径相同。
- `fp32_bits_to_bf16` 按 RNE（中点取偶）转换 FP32 位模式；`encode_bf16` 先 FP32 再 BF16，**不是 Python binary64 直接舍入到 BF16**。有限溢出转 Inf，次正规/下溢按 RNE；正负零保留。[BF16][参照][测试] NaN 规范化为 quiet NaN并保留符号，但测试与 parity 只比较分类，不承诺 payload 或与 GPU 的 NaN 位模式相等；也不声称该 NaN 符号策略与 PyTorch CPU 的规范正 NaN 相同。
- 反量化默认返回 FP32 值。`output="bf16"` 返回 **BF16 原始整数位模式**，保留 shape；此选项描述独立输出边界，不强制任何 GEMM 在逐元素乘 scale 后提前落 BF16。[参照]

## 4. 反量化 API 与布局

`ref.dequant.Tensor(data, shape)` 使用 flat row-major 数据及显式 shape，不隐式广播；允许零元素张量，shape 必须非空、各维是非负整数。[参照]

- `dequantize_expert`：二维 packed E2M1 与逐行 E8M0 scale。
- `dequantize_fp8_weights`：二维 E4M3FN 与二维块 E8M0 scale，尾块用向上取整 shape。
- `dequantize_engram`：二维 E4M3FN 行与逐行 E8M0 scale。
- `dequantize_swa`：E4M3FN 与沿最后维度分组 E8M0 scale，支持批维。
- `dequantize_main_kv`：packed E2M1 与沿最后维度分组 E4M3FN scale，支持批维。
- `dequantize_index_k`：packed E2M1 与沿最后维度分组 E8M0 scale，支持批维。

以上入口依据 [MODEL §2.1～§2.3](MODEL.md#2-结构与格式) 的模型用途和分组定义实现。[代码][参照] 逐行路径要求逻辑最后维整除组大小，scale shape 精确为前导维加组数；二维块路径覆盖非整块尾部，不接受转置或仅元素数量相等的错误 scale shape。[参照][测试]

底层 codec 识别全部编码；模型反量化入口拒绝 NaN 数据/scale 和非正或非有限 scale。非法编码/shape/输出模式抛 `ValueError`，非整数位模式/shape 抛 `TypeError`。有效输入的乘积可超出 FP32 或 BF16 范围，遵守上述浮点输出规则，不静默钳制。拒绝坏输入是本 oracle 入口契约，不宣称官方所有入口也如此校验。[参照][测试]

## 5. 验证链边界

测试入口为 `python3 tests/quant/test_formats.py`（从仓库根运行，也支持从其他 cwd 按绝对路径运行）。必选穷举与边界无 skip，无可选 dtype 依赖；来源快照或网络不参与运行。[测试]

独立期望值由有理数位域定义、FP32 有序编码搜索、BF16 相邻格点距离比较构造，不调用被测 LUT/舍入辅助函数；官方 FP4_TABLE 用 AST 读取字面常量，不导入官方 GPU 模块。测试不写入 `tests/data/`，没有新增冻结区工件或 manifest。[测试]

官方落点需按路径区分：Engram `model.py:319–320`、转换器 `convert.py:157–173` 的 wo_a 路径有明确 BF16 落点；KV/SWA 的 inplace 量化回写输入 dtype（`kernel.py:82–86,167–172`）；专家 `kernel.py:538–553` 是 FP4→FP32→FP8 后 GEMM，再对累加值应用 scale。[代码] 不能从逐元素反量化验收推出移动 scale、提前 BF16 舍入或整数点积改写的等价性。

本页各参照的覆盖与不覆盖范围见 [MODEL §4](MODEL.md#4-精确性链条) 覆盖矩阵。模型实际 BF16 激活动态量化见 [ACTQUANT](ACTQUANT.md)，计算分支见 [COMPUTE](COMPUTE.md)。

## 6. 生产 C++ codec（`src/quant`）

选择理由见 [D0007](../plan/decisions/d0007-quant-integer-dequant.md)，验收证据见 [P0010](../plan/p0010-quant-scalar-baseline.md)。

**入口**：`df::quant` 命名空间，公共头 `include/df/quant/codec.h`。`decode_e2m1`、`decode_e4m3fn`、`decode_e8m0`、`decode_bf16` 输出 FP32 位模式（`uint32_t`）；`fp32_to_bf16` 输出 BF16 位模式（`uint16_t`）；`pack_e2m1` / `unpack_e2m1` 在每字节一个 nibble 与打包字节之间转换；`e2m1_to_grid` / `grid_to_e2m1` 在 nibble 与 `int8_t` 整数格之间转换。位模式、零符号、打包次序与 §2～§3 的 `ref/` 契约逐位相同，含码点 8 解码为负零。[C++][C++测试]

**输入输出**：连续 row-major 只读视图、显式 shape（`std::span<const std::size_t>`）与调用方提供的输出缓冲区，不支持 stride、广播或所有权。输入长度须与 shape 精确匹配；输出容量不小于结果长度即可，只写结果前缀。shape 至少一维；任一维为 0 时元素数为 0，其余维不参与溢出判定。打包入口的 shape 是逻辑 nibble shape。[C++]

**失败**：rank 0、长度与 shape 不匹配、输出容量不足、打包入口的奇数最后维、输入输出重叠、非法 nibble（>15）、非格点整数抛 `std::invalid_argument`；元素数超出 `size_t` 抛 `std::overflow_error`。元数据校验在读取任何数据前完成，内容校验在写入任何输出前完成，失败时输出不变。重叠只按实际读写的区间判定，空区间不算重叠。[C++][C++测试]

**静态类型排除**（无运行时检查）：负维度（`size_t`）、bool 或浮点充当整数、超出 `uint8_t`/`uint16_t`/`uint32_t` 宽度的位模式。对应 `ref/` 的 `TypeError` 与位宽 `ValueError` 在 C++ 中不可表达；测试执行器在窄化前检查 token 宽度，超宽按执行器故障处理，不计为库的拒绝。[C++][C++测试]

**环境无关性**：全部入口以定宽整数位运算构造结果，数值路径不执行浮点运算，不读取也不修改浮点环境。parity 在默认、MXCSR 向上舍入、FTZ、DAZ 四种调用环境下均与 `ref/` 逐位一致（ENV-A）。[C++测试]

**格式契约版本**：`format_contract_version()` 返回 `src/quant/contract.h` 中的常量，与引擎 `VERSION` 独立。任一位模式、零符号、打包次序或失败契约变化时递增。当前值：

格式契约版本：1

**验证边界**：parity 由 [C++测试] 覆盖 §5 所列码点穷举、全部 BF16 相邻格点中点及其两侧邻值，以及每类拒绝。它对应 K-Q1（codec、布局与整数格）以及 K-Q3 中的 E4M3FN/BF16 解码与 FP32→BF16 舍入；乘 scale 由 §7 单独验证，其余不覆盖项见 [MODEL §4](MODEL.md#4-精确性链条)。

## 7. 给定 scale 的生产 scalar 反量化

选择理由见 [D0007](../plan/decisions/d0007-quant-integer-dequant.md)，验收证据见 [P0011](../plan/p0011-quant-dequant-scalar.md)。以下 **[C++反量化]** 指 [dequant.h](../include/df/quant/dequant.h) 与 [dequant.cpp](../src/quant/dequant.cpp)，**[反量化测试]** 指 [test_dequant_parity.py](../tests/quant/test_dequant_parity.py) 经同一个测试执行器调用生产库。

**入口与布局**：`df_quant` 的公共头 `include/df/quant/dequant.h` 提供与 §4 同名的六个 `dequantize_*`，各有 FP32 位模式（`std::span<uint32_t>`）与 BF16 位模式（`std::span<uint16_t>`）输出重载。数据与 scale 分别传连续 row-major 只读视图和精确的**存储 shape**，不广播、不转置、不支持 stride、不推断 scale。packed E2M1 输入的最后维为逻辑最后维的一半，与 codec pack/unpack 的逻辑 nibble shape 不同；换算检查溢出。输出缓冲区容量可大于结果，只写结果前缀。[C++反量化]

专家、普通 FP8 权重、Engram 只接受二维数据；SWA、主 KV、索引 K 至少一维并支持批维。逐行分组的 scale shape 必须精确为前导维加逻辑最后维组数，最后维须整除对应组大小；普通 FP8 权重的 scale shape 为二维块数的向上取整，包含行列尾块。格式、模型用途和组大小沿用 §4 与 MODEL §2，不在本节另立定义。[C++反量化][参照]

**失败与校验顺序**：rank/shape/长度/容量不匹配、组不整除、NaN 数据、NaN 或非正/非有限 scale、数据/输出或 scale/输出重叠抛 `std::invalid_argument`；必要维度换算、元素数与输出字节长度等元数据算术溢出抛 `std::overflow_error`。数值上溢产生带符号 Inf，不是调用失败。元数据校验先于全部内容读取，内容校验先于任何输出写入，失败时输出不变。零元素张量允许，但仍须满足 rank、分组与 scale shape；元素计数沿用零维短路，不豁免必要的 packed 最后维换算。校验期间输入被并发修改属于调用方数据竞争，不在契约范围内。[C++反量化][D0007]

packed E2M1 每个字节均是合法数据，所用 scale 格式均不能编码 Inf，因此测试不伪造这些静态表示不可表达的拒绝类；原始编码宽度由 C++ 定宽类型约束，执行器在窄化前检查 token。NaN 在模型反量化入口是拒绝用例，不使用 codec 的 NaN 分类比较代替拒绝。[C++反量化][反量化测试]

**数值与环境**：复用 §6 解码与 BF16 舍入；有限 FP32 操作数拆成有效数和指数，用定宽整数乘法及通用 RNE 构造结果，含渐进下溢、符号零、舍入进位和上溢到 Inf。BF16 是独立输出边界，不表示 GEMM 必须提前舍入。数值路径不执行浮点算术，不读取或修改调用方浮点环境，不依赖“合法有限乘积都精确”简化算法。[C++反量化]

在 [BUILD §1](BUILD.md#1-支持环境) 的支持环境中，六类入口的合法有限数据/scale 编码组合及两种输出分别对照 `ref/dequant.py`，多行、多组、批维和二维双尾块逐位一致；同一个生产有限乘法 helper 通过私有测试连接对照 `ref/formats.py::multiply_fp32`，补充公开合法编码组合不能覆盖的有限不精确乘积和 RNE 边界，不形成公共通用乘法 API。两者在 [BUILD §7](BUILD.md#7-测试清单) 的八模式 MXCSR 下均逐位一致，完整 MXCSR 不变。上溢敏感用例 `-448 × 2^127` 在向上舍入下仍为 `-inf`，次正规乘积与 E8M0 `0x00` scale 不被 FTZ/DAZ 冲刷。[反量化测试]

**版本与验证边界**：本项只新增入口，不改变既有 codec 的位模式、零符号、打包次序或失败契约，§6 的格式契约版本仍为 1，不新增 pack 容器格式。本基线是逐元素 dequant 类 kernel 的上一级参照，不覆盖项见 [MODEL §4](MODEL.md#4-精确性链条)。[C++反量化][反量化测试]
