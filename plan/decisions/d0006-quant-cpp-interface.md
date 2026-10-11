---
id: D0006
title: 生产 quant scalar 接口采用连续视图、标准异常、调用时数值环境检查与 sanitizer 检查配置
status: accepted
created: 2026-10-11
updated: 2026-10-11
replaces: []
superseded-by: []
---

## 背景与问题

P0010（codec 层）与 P0011（给定 scale 反量化）将建立第一个生产 C++ 模块 `src/quant`。输入表示、失败报告、宿主浮点环境的处理方式以及越界证据的取得方式，会被后续 CPU kernel 的 parity 测试和调用方继承，形成事实上的先例；须在实现前确定并限定适用范围。

## 决策驱动

- 技术约束：parity 要求结果确定。产生 FP32 的运算受 MXCSR 的舍入模式、FTZ/DAZ 与异常掩码影响。codec 中涉及 FP32 次正规数的只有 E8M0 `0x00`（2^-127）、BF16 次正规位模式的解码，以及 FP32 次正规输入到 BF16 的舍入；若用浮点运算实现，FTZ/DAZ 会把它们静默冲刷为零。E4M3FN 与 E2M1 的全部有限值都在 FP32 正常数范围内（或为零），解码不受影响。乘 scale 的乘积可落入次正规区，同样受影响。
- 技术约束：坏元数据（shape、长度、容量）须在读取数据前、坏内容（编码、scale）须在写入输出前可判定地拒绝，并能被测试执行器区分于执行器自身故障。
- 技术约束：仅靠功能测试只能证明已测坏输入被拒绝，不能证明没有越界读写或未定义行为。
- 项目约束：宿主 C++20（D0005），不禁用 exceptions；不提前建设全引擎错误框架、通用 Tensor 或稳定 ABI。
- 项目约束：未来 CUDA 与热路径可能不适合使用异常或逐调用检查，本决策不得预先约束它们。

## 实测前提

环境同 [BUILD 支持环境](../../docs/BUILD.md#1-支持环境)。方法：`/tmp` 中以 `g++ -std=c++20 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all` 编译一个对 `std::vector<int>(4)` 做运行时下标越界读取的探针。结果：离线编译链接成功，运行时报告 `heap-buffer-overflow` 并以非零码退出。只证明本机 GCC 15.2.0 自带的 sanitizer 运行库可用，不是 quant 的验收证据。

浮点环境探针：同一环境（glibc 2.43），`g++ -std=c++20 -O2 -ffp-contract=off`，操作数经 `volatile` 读入，防止常量折叠；计算 `-448.0f * 0x1p127f`。

| 调用环境 | `fegetround() == FE_TONEAREST` | 结果 |
| --- | --- | --- |
| 默认（MXCSR `0x1F80`） | 是 | `-inf` |
| 仅用 `_mm_setcsr` 把 MXCSR 舍入位设为向上 | 是 | `-0x1.fffffep+127` |
| 仅用 `_mm_setcsr` 解除溢出掩码 OM（`0x0400`） | 是 | 进程收到 SIGFPE |

结论：glibc 在 x86_64 上的 `fegetround()` 读取 x87 控制字，察觉不到只改 MXCSR 的情况；FP32 运算走 SSE，由 MXCSR 控制字段决定舍入、FTZ/DAZ 与是否陷入异常。

## 备选方案

### 失败报告

- 标准异常：无效编码、非格点、shape/长度/容量不匹配、非法 scale、缓冲区重叠抛 `std::invalid_argument`；元数据算术溢出抛 `std::overflow_error`；调用浮点环境不符合前提抛 `std::runtime_error`。C++20 原生可用、调用方与测试执行器易于区分类别、不需要新类型：采纳，仅限本模块。
- 返回状态码或自定义结果类型：要建立新的错误类型，等于开始全引擎错误体系；`std::expected` 要 C++23，与 D0005 不符：不采纳。
- 断言或 abort：测试无法逐个验证拒绝路径，Release 下可能被移除：否决。

### 输入与输出表示

- 连续 row-major 只读视图 + 显式 shape + 调用方提供的输出缓冲区：与 `ref/` 的 flat 数据 + shape 契约同构，没有所有权语义：采纳。
- 任意 stride 视图：当前没有需求，又多一层索引越界风险：不采纳。
- 自有 Tensor 类型：超出本模块范围，会提前定型引擎张量体系：否决。

### 缓冲区重叠

- bulk 操作拒绝输入与输出重叠，用 `std::less` 或 `uintptr_t` 判断地址区间，不直接比较无关指针：采纳。
- 允许原位：失败时不修改输出这一条在原位时无法成立，没有已知需求：不采纳。

### 宿主浮点环境

- codec（解码、BF16 位舍入、nibble 与整数格）用整数运算直接构造 FP32/BF16 位模式，不依赖浮点环境，无须检查：采纳。
- 产生 FP32 运算结果的入口（乘 scale 的反量化）在每次 bulk 调用前读取当前线程的 MXCSR，要求控制字段等于默认值，即 `(mxcsr & ~0x3F) == 0x1F80`：舍入为 RNE、FTZ 与 DAZ 关闭、六个异常掩码全部置位；低 6 位异常状态标志忽略。不符合就抛异常；不逐元素检查，不缓存进程级“已验证”状态：采纳。检查放在单独的非内联函数中，并在首次读取输入数据前调用，使编译器不能把输入读取与乘法提前到检查之前。
- 以 `fegetround()` 为依据：读取 x87 控制字，察觉不到只改 MXCSR 的舍入（见实测前提），也不覆盖异常掩码：否决；可作为额外的保守检查，不作为依据。
- 只检查舍入与 FTZ/DAZ：解除异常掩码后上溢直接陷入 SIGFPE，违反“上溢到 Inf 不属于失败”：否决。
- 进程启动时检查一次：线程或第三方库可能事后改动环境，证明不了调用时的环境：否决。
- 在调用内设置后恢复环境：会静默改变调用方配置，掩盖环境错误：否决。
- 不检查：FTZ 下会静默得到错误结果：否决。

### 越界与未定义行为的证据

- 新增 `cpu-asan` preset（Debug、`-fsanitize=address,undefined`、`-fno-sanitize-recover=all`、`-fno-omit-frame-pointer`），只构建 quant 生产库与测试执行器并运行 quant 测试；任何 sanitizer 报告都让 CTest 失败：采纳。这是对 D0005 preset 列表的追加，不改 `cpu-debug`/`cpu-release` 或 `df_strict_fp` 的语义。
- Valgrind：本机未安装，开销大：不采纳。
- 不做 sanitizer、收窄验收措辞：用户选择取得越界证据，不采纳。

## 结论

2026-10-11 经用户确认采纳：

1. `src/quant` scalar 入口用连续 row-major 只读视图、显式 shape 和调用方提供的输出缓冲区，不支持 stride、广播或所有权。
2. 失败用上述三类标准异常报告。元数据校验在读取任何数据前完成，之后只读取已确认在界内的数据；内容校验在写入任何输出前完成；失败时输出不变。校验期间输入被并发修改属于调用方的数据竞争，不在契约范围内。数值上溢到 Inf 不属于失败。
3. bulk 操作拒绝输入与输出重叠。
4. codec 用整数位运算，不依赖浮点环境；乘 scale 的入口在每次 bulk 调用前检查当前线程 MXCSR 的完整控制字段（舍入、FTZ、DAZ、异常掩码），不以 `fegetround()` 为依据；不符合就拒绝，不修改环境。
5. 新增 `cpu-asan` preset，作为 quant 越界与未定义行为的检查配置。

## 后果

- 适用范围只有 `src/quant` 的 scalar 正确性基线。CPU SIMD、CUDA kernel 和热路径如要采用不同的失败或环境策略，须另立决策，不援引本条目。
- MXCSR 检查只适用于 x86_64 SSE；平台扩展时须另行确定对应的检查依据。P0009 的 `bb_fp_env` 与 D0005 结论 5 以 `fegetround()` 检查舍入，反例只经 `fesetround()` 演练，存在同类盲区；两者为终态，本决策不改写，quant 也不依赖它们。
- quant target 不能用 `-fno-exceptions` 编译；`bb.compile_flags` 类检查须覆盖新 target。
- `cpu-asan` 依赖 GCC 自带的 libasan/libubsan；换用不带这些库的工具链时，该配置明确失败，不 skip。
- 接口、失败与环境契约归档到 `docs/QUANT.md`，`cpu-asan` 的命令与支持证据归档到 `docs/BUILD.md`；本条目只保留选择理由。
