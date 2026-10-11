---
id: P0011
title: 建立给定 scale 六类反量化的生产 scalar 基线
status: completed
created: 2026-10-11
updated: 2026-10-12
knowledge: updated
docs: [docs/MODEL.md, docs/QUANT.md, docs/BUILD.md]
bench: []
decisions: [D0001, D0005, D0006, D0007]
followups: []
---

## 意图

在 [P0010](p0010-quant-scalar-baseline.md) 的生产 codec 与跨语言 parity 连接之上，交付 `ref/dequant.py` 对应的六类给定 scale 反量化生产入口，并逐位对照 `ref/`。成功标准是独立 C++ 消费者调用生产库，在声明的环境与输入契约下获得与 `ref/` 一致的 FP32 与 BF16 位模式。

本项是逐元素反量化与 dequant 类 kernel 的上一级参照，不是专家 FP4→FP8 GEMM、累加后乘 scale、提前 BF16 舍入或整数点积改写的参照；这些路径须另建参照另项验收。

## 验收

- [x] **六类入口**：专家、普通 FP8 权重、Engram、SWA、主 KV、索引 K 六类反量化，各自支持 FP32 输出和 BF16 原始位模式输出。逐行分组、批维、二维块尾部与 scale shape 遵守 `docs/QUANT.md`，不广播、不转置、不推断 scale。BF16 输出只是独立的输出边界，不表示 GEMM 必须提前舍入。
- [x] **输入与失败契约**：按 D0007 可判定地拒绝：rank、shape、数据长度、scale shape（含同元素数但 shape 错误）、输出容量、最后维不整除组大小、NaN 数据或 scale、非正或非有限 scale、索引与长度计算溢出、输入输出重叠。元数据校验在读取任何数据前完成，之后只读取已确认在界内的数据；内容校验在写入任何输出前完成；失败时输出不变；允许零元素张量。每类拒绝至少一个测试：元数据错误用短于声明的数据或 scale 缓冲区，在 `cpu-asan` 下证明拒绝前无越界读；内容错误把坏数据或坏 scale 放在最后一个元素，证明输出未被提前写入。
- [x] **数值环境无关**：按 D0007，乘 scale 与 BF16 输出全部用整数位运算构造，数值路径不执行浮点运算。乘 scale parity 在默认环境下运行一遍，再在经 `_mm_setcsr` 设置的以下 MXCSR 下各运行一遍：舍入向上、向下、向零，FTZ，DAZ，解除溢出掩码。每种环境下结果均与 `ref/` 逐位一致，进程不陷入，调用前后 MXCSR 不变。敏感性用例：向上舍入时，`-448 × 2^127`（E4M3FN 最大值乘 E8M0 `0xFE`）仍得 `-inf`，不得为负的最大有限数；FTZ 与 DAZ 下，次正规乘积与 E8M0 `0x00` scale 的结果保持次正规位模式。
- [x] **乘 scale parity**：对照 `ref/`，覆盖既有数据编码与合法 scale 编码的全部有限组合、极端 scale、符号零、渐进下溢与上溢到 Inf，FP32 与 BF16 输出逐位比较；NaN 只比较分类。
- [x] **布局边界**：每类入口覆盖多行、多组、批维、相邻不同 scale、行列不对称、二维块双尾块、组边界与零元素张量，确保 scale 错轴不能通过。随机用例只作可复现补充。
- [x] **可复现、越界检查与测试敏感性**：`cpu-debug`、`cpu-release` 运行全部必选测试，`cpu-asan` 运行全部 quant 测试且无 sanitizer 报告；从源码目录外及干净检出验证。临时注入逐行 scale 索引、二维块尾部 scale 索引、批维步长、BF16 输出舍入四类错误，均被识别，恢复后全部通过。记录环境、命令与结果。
- [x] **知识、治理与人工收尾**：经确认后，反量化入口契约、调用时环境要求与验证边界最小增补到 `docs/QUANT.md`，测试清单增补到 `docs/BUILD.md`。数值提交的 `Oracle:` 指向真实的 C++ parity 测试；仓库检查与合入前提交范围审计通过；用户评审后才结项。

## 记录

### 范围与依据

- 前置：P0010 completed，D0007 accepted。沿用 P0010 的模块、测试协议、执行器与 `cpu-asan` 配置，只扩展反量化操作。
- 范围外：有损编码与 scale 生成、动态激活/KV 量化、RoPE 与 cache 管理、点积 scale 折叠、整数点积、GEMM/GEMV、官方转换器的完整重分块、真实权重、CUDA 与 SIMD、性能论断。
- 数值依据：[QUANT §3～§5](../docs/QUANT.md#3-浮点运算与输出)、`ref/dequant.py`、[MODEL §2](../docs/MODEL.md#2-结构与格式)。

### 执行契约

- 乘 scale 按 D0007 用定宽整数构造：拆出有效数与指数，整数相乘，指数相加，按通用 RNE 构造 binary32 位模式，覆盖次正规、带符号零与上溢到 Inf。不依赖“合法输入从不舍入”这一性质，不移植 Python 任意精度实现，不新增通用软件浮点库，不检查或修改浮点环境。解码沿用 P0010 的 codec，BF16 输出沿用 P0010 的整数舍入，不能以解码 parity 代替乘法 parity。

### 完成声明边界

- K-Q2 只覆盖给定 E8M0 的指数应用，不覆盖点积折叠；K-Q3 只覆盖给定 scale 的 KV/SWA 布局，不覆盖动态量化。不笼统勾选 K-Q1～K-Q3。
- 不新增性能论断或优化开关。

### 执行与知识处置

- 分支 `p0011-quant-dequant-scalar`；提交用 `P0011 quant:` 并带 `Oracle:`。顺序：先专家反量化打通，再补其余五类、失败契约、环境无关性与敏感性注入。
- 知识预计为 updated；必要知识缺失阻塞结项。

### 设计记录

- 2026-10-11：P0010 评审后经用户同意，从 P0010 拆出本项，承接六类反量化、乘 scale parity、调用时环境检查与相应失败契约。仅登记 draft，未开始实现。
- 2026-10-11：用户评审指出 glibc 的 `fegetround()` 读取 x87 控制字，只改 MXCSR 的舍入不可见。本机探针确认，并发现解除溢出掩码会使上溢陷入 SIGFPE（见 D0006 实测前提）。经用户同意，环境检查改为以 MXCSR 完整控制字段为准，反例改为只经 `_mm_setcsr` 构造并加入上溢符号的敏感性用例。
- 2026-10-11：开工前评审指出六类乘积对合法输入要么精确要么上溢，`ref/` 穷举探针确认无有限不精确乘积（见 D0007 实测前提）。经用户同意，以 D0007 取代 D0006：乘 scale 改用整数位运算，删去调用时 MXCSR 检查、`std::runtime_error` 与非内联检查函数要求，环境条目改为环境无关性验收并保留上溢敏感用例。仍为 draft，未开始实现。

### 开工与实现记录

- 2026-10-12：用户确认实现拆解并授权开工，沿用现有 P0011 和 D0007；同步 active 与 STATUS，在 `p0011-quant-dequant-scalar` 实现。公共入口采用实际存储 shape、独立 scale shape 与 FP32/BF16 输出重载；私有 helper 只共享必要函数，不暴露 codec 专用符号。
- 2026-10-12：六类入口、整数乘 scale、三阶段校验、测试执行器与四个反量化测试套件已实现。审阅发现的向零 MXCSR 常量、私有探针坏请求分类、harness 递归、溢出用例 scale shape 及协议空分量/重复字段问题已修复；补充有限乘积舍入进位及进位溢出的定向锚点。
- 2026-10-12：用户要求收束两处边界：codec 专用 helper 留在匿名命名空间，必要共享函数显式 `detail::` 调用；`-fno-exceptions` 仅由 quant 库及执行器的 target 检查，不扩为全仓规则。三个配置 nm 与合成编译命令证明边界成立，数值函数体除限定名外不变；全量回归通过。
- 2026-10-12：用户确认验收代码与文档并要求推进；QUANT §7 与 BUILD 测试清单按确认归档，提交态干净检出与非空范围审计在合入前补齐。`.zcode/` 当前只有会话审议稿，忽略根目录下的 `.zcode/plans/`，不删除文件、不屏蔽其他潜在共享配置。

### 验收证据

环境：Intel Xeon Platinum 8350C 双路（KVM），Linux 5.15.0-72-generic x86_64 / glibc 2.43；GCC 15.2.0、CMake/CTest 4.4.4、Ninja 1.13.2、Python 3.12.8（conda-forge）、Git 2.54.0。2026-10-12 最终提交态复现使用分支 tip `a211439`，实现提交 `bdbda5d`；工作仓库与全新本地克隆均无补丁、无源码改动，测试后保持干净。

- 构建与复现：`cpu-debug`、`cpu-release` 各 15/15 通过；`cpu-asan` 8/8 通过，无编译告警、无 sanitizer 报告。三个 preset 均执行 configure/build/ctest；全新本地克隆结果相同，从 `/tmp` 用 `ctest --test-dir <clone>/build/cpu-release` 运行 15/15。普通构建测试离线，仅标准库，不改 ref 或冻结区。
- 入口与数值：六类入口均提供 FP32/BF16 位模式输出，最终公开 parity 为 64 个批量请求。每入口分别覆盖全部有限数据码与合法 scale 码组合：E2M1×E8M0 4,080 对；E4M3FN×E8M0 64,770 对；E2M1×正 E4M3FN 2,016 对。符号零、极端 scale、次正规与上溢到 Inf 均逐位对照 `ref/dequant.py`；不以 codec parity 代替乘法 parity。
- 布局：多行、多组、相邻不同 scale、非对称行列与组边界；批维入口覆盖 rank 1、3、4；普通 FP8 权重覆盖完整块及 35×67 双尾块。scale shape 同元素数但错轴或转置均拒绝；无广播或 scale 推断。
- 失败：最终 324 个 contract 请求逐入口、逐输出覆盖 rank、数据与 scale 长短、scale shape、容量、组不整除、两输入与输出重叠、元素数/packed 维度及数量换算/输出字节溢出、零元素张量。错误元数据配短缓冲区，ASan/UBSan 无越界读；NaN 数据和非法 scale 位于最后元素，失败时全输出缓冲区不变。packed E2M1 无非法字节，所用 scale 格式无 Inf 编码，不伪造不可表达的拒绝类。
- 私有乘法：同一个生产有限 FP32 helper 经执行器直接调用，48 个定向对与 seed 21 的 20,000 个随机对逐位对照 `ref/formats.py::multiply_fp32`；覆盖有限不精确、中点取偶、正常/次正规边界、下溢到符号零、舍入指数进位及进位上溢，不新增公共通用乘法 API。定向 `0x3F900001 × 0x3FE38E37` 舍入到 `0x40000000`，提高左操作数指数后的 `0x7F100001 × 0x3FE38E37` 经进位得到 Inf。
- 环境：公开 parity 与私有 multiply 均在默认、向上、向下、向零、FTZ、DAZ、FTZ+DAZ、解除溢出掩码下运行。probe 核实实际控制字段，每次生产调用前后完整 MXCSR 不变，包含拒绝路径；无陷入。`-448 × 2^127` 在向上舍入下仍为 -Inf，E8M0 `0x00` 与次正规乘积不被 FTZ/DAZ 冲刷。专项守护污染探针在成功与异常路径均被识别。
- 故障传播：永久 `quant.dequant.harness` 覆盖崩溃、非零退出、乱码、少响应、缺执行器、未知操作、超宽数据/scale token、缺字段/分隔符、空 shape 分量、重复字段、私有探针坏请求与错结果。parity/contract/multiply 子进程注入故障均非零，不递归。
- 敏感性：只在 scratch 独立副本逐项注入逐行 scale 行步长错误、仅二维尾块 scale 索引错误、批维步长错误、BF16 截断，均使 `quant.dequant.parity` 失败（CTest 退出码 8）；恢复后三个 preset 全部通过，交付源逐字节未受影响。额外私有中点不取偶、漏舍入进位、冲刷次正规三类注入均使 multiply 失败，干净与恢复对照通过。临时脚本与原始输出不入库。
- 边界收束：三个库的 nm 确认 codec 专用函数/模板不再是外部链接符号，仅必要共享 helper 留在 detail，函数体除限定名外与基线相同。两个编译检查 profile 的合成命令证明 quant 库/执行器带 `-fno-exceptions` 时拒绝，无关 target 不因该选项误伤；原有全仓 fast-math 规则仍生效。
- 测试连接耗时：本机 Debug 下公开 parity 的 ref 期望值生成约 2.10s，每环境执行比较约 0.41s（补充 rank 1/4 布局前的 52 请求测量）；测试连接足够批量运行，不构成引擎性能论断。
- 治理与知识：真实非空 `main..HEAD`（全新 clone 用 `origin/main..HEAD`）提交审计 E=0 W=0 I=0，范围 diff 检查通过。`.zcode/plans/` 精确忽略已验证，不删除本地审议稿，其他潜在共享配置不被屏蔽。用户确认后 QUANT §7 与 BUILD 测试清单归档；数值提交 Oracle 指向真实 C++ parity 驱动。

## 收尾

- 验收或停止依据：2026-10-12 用户确认验收代码与文档并授权推进合入；提交态干净复现、非空范围审计与逐条证据见上方。本合并提交同步 completed、STATUS 与合并流程的 VERSION patch+1，版本为 0.1.10。不证明 GEMM、累加后乘 scale、点积 scale 折叠、提前 BF16 舍入、scale 生成、动态量化、SIMD/CUDA、真实权重或端到端模型一致。
- 知识处置：updated；入口、存储/逻辑 shape、失败、环境无关性与验证边界归档到 [QUANT §7](../docs/QUANT.md#7-给定-scale-的生产-scalar-反量化)，新增测试与执行器/编译要求归档到 [BUILD §7](../docs/BUILD.md#7-测试清单)；选择理由保留在 [D0007](decisions/d0007-quant-integer-dequant.md)，过程证据保留在本项。
- 剩余义务：无；followups 为空。范围外的计算类 scalar 参照与验证链完整修订仍按 STATUS 的既有路由在首个计算类 kernel 工作项开工前决定，不以本项验收推定完成。
