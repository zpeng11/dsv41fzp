---
id: P0010
title: 建立低精度格式 codec 的生产 scalar 基线与跨语言 parity 连接
status: active
created: 2026-10-11
updated: 2026-10-11
knowledge: pending
docs: [docs/MODEL.md, docs/QUANT.md, docs/BUILD.md]
bench: []
decisions: [D0001, D0005, D0006, D0007]
followups: []
---

## 意图

在 P0006 已验证的格式 oracle 和 P0009 已验收的 CPU-only 构建入口之上，建立真实可链接的生产 `src/quant` codec 层，以及 C++ 生产库对照 `ref/` 的跨语言 parity 连接。成功标准是独立 C++ 消费者调用生产库，在声明的环境与输入契约下逐位对上 `ref/`。

本项交付的 codec 是后续 unpack/解码类 kernel 的上一级参照，不是 GEMM、整数点积或乘 scale 路径的参照；给定 scale 的六类反量化由 [P0011](p0011-quant-dequant-scalar.md) 在本项之上完成。本项不重新激活 P0006，也不依赖 P0005。

## 验收

- [ ] **生产模块与链接边界**：`src/quant/` 建立编译后的生产库，公共头位于 `include/df/quant/`，源文件显式列举，数值逻辑放在 `.cpp`。独立测试消费者链接该库，不包含生产 `.cpp`，不用测试侧算法副本代替库调用。生产库只依赖 C++ 标准库。
- [ ] **格式覆盖**：E2M1、E4M3FN、E8M0、BF16 解码到 FP32 位模式；FP32 位模式到 BF16 的 RNE 转换；E2M1 nibble 打包与展开；E2M1 与整数格双向映射。零符号、整数零规范编码与格式变体遵守 `docs/QUANT.md`。
- [ ] **接口与失败契约**：按 D0007 使用连续视图、显式 shape、调用方提供的输出缓冲区和两类标准异常。可判定地拒绝：长度与 shape 不匹配、输出容量不足、奇数逻辑最后维、非法 nibble、非格点整数、shape 乘积与 packed 长度换算溢出、输入输出重叠；元数据校验在读取任何数据前完成，之后只读取已确认在界内的数据；内容校验在写入任何输出前完成；失败时输出不变；允许零元素张量。每类拒绝至少一个测试：元数据错误用短于声明的数据缓冲区，在 `cpu-asan` 下证明拒绝前无越界读；内容错误把坏值放在最后一个元素，证明输出未被提前写入。另列出由 C++ 静态类型排除、无须运行时检查的输入；测试执行器不得先窄化坏输入再冒充库已校验。
- [ ] **数值环境无关**：codec 全部入口用整数位运算构造结果。parity 测试在默认环境，以及经 `_mm_setcsr` 设置的 MXCSR 向上舍入、FTZ、DAZ 四种调用环境下各运行一遍，结果均与 `ref/` 逐位一致，以此证明结果与环境无关。
- [ ] **码点穷举 parity**：对照 `ref/`，覆盖 E2M1 全部 16 个码点、E4M3FN 与 E8M0 各 256 个编码、BF16 全部 65,536 个位模式、全部 256 个 packed byte，以及整数格全部合法格点和代表性非格点。原始编码、整数格、FP32 与 BF16 位模式逐位比较；NaN 只比较分类；零检查符号。
- [ ] **BF16 舍入边界**：覆盖全部有限相邻 BF16 格点的中点及其 FP32 两侧邻值、指数进位、正常/次正规边界、上溢到 Inf、下溢、正负零、Inf 与 NaN，逐位对照 `ref/`。随机用例只作可复现补充。
- [ ] **格式契约版本**：`src/quant` 定义量化格式契约版本常量，由公共接口读取，独立于引擎 VERSION。初值、递增条件（任一位模式、零符号、打包次序或失败契约变化）归档到 `docs/QUANT.md`；测试断言该常量可读且等于归档值。
- [ ] **跨语言连接与失败传播**：测试侧执行器批量接受操作、shape 与编码，调用生产库，以十六进制位模式返回结果或分类后的异常；Python 标准库驱动从 `ref/` 生成期望值。CMake 用 `$<TARGET_FILE:...>` 显式传入执行器与解释器路径。缺执行器、崩溃、非零退出、协议损坏或比较失败均使 CTest 非零，必选用例不 skip；临时失败注入证明失败能传到 CTest。实现初期测量一次全量 parity 耗时并记录。
- [ ] **编译与越界检查**：生产库链接 `df_cxx_baseline` 与 `df_strict_fp`；有效编译选项检查扩展到新 target，Debug/Release 均确认严格选项生效且不污染无关 target，无 fast-math、`-fno-exceptions` 或全局 `-march`。按 D0007 新增 `cpu-asan` preset，运行全部 quant C++ 测试，无 sanitizer 报告；临时注入一个越界读，证明该配置能识别并使 CTest 失败。
- [ ] **可复现与测试敏感性**：`cpu-debug`、`cpu-release` 运行全部必选测试，`cpu-asan` 运行 quant 测试；从源码目录外及干净检出验证。普通构建与测试离线，不引入第三方测试框架，不写冻结区。记录编译器、CMake/CTest、Python、CPU/平台、选项、命令与结果。临时注入 nibble 次序、BF16 中点舍入、负零、E8M0 次正规解码四类错误，均被识别，恢复后全部通过。
- [ ] **知识、治理与人工收尾**：经确认后，C++ 接口/失败契约、环境无关性与格式契约版本最小增补到 `docs/QUANT.md`，target、测试入口与 `cpu-asan` 增补到 `docs/BUILD.md`。数值提交的 `Oracle:` 指向真实的 C++ parity 测试。现有 oracle、治理回归、仓库检查与合入前提交范围审计通过；用户评审后才结项。

## 记录

### 范围与依据

- 范围内：生产 codec、E2M1 字节布局与整数格、最小公共 C++ 接口、测试执行器与 parity 数据协议、`cpu-asan` 检查配置、现有 CMake/CTest 接线和必要知识归档。
- 范围外：给定 scale 的反量化与乘 scale（P0011）；任意浮点到 E2M1/E4M3 的有损编码、amax 与 scale 生成、动态激活/KV 量化；点积 scale 折叠、整数点积、GEMM/GEMV；checkpoint/pack 容器与真实权重加载；CUDA、手写 SIMD 与 ISA 分派；性能优化与吞吐论断；通用 Tensor、稳定 ABI 与安装发布；全引擎错误框架；全套 CI/bench。
- 数值依据：[QUANT](../docs/QUANT.md)、[MODEL 验证链](../docs/MODEL.md#4-精确性链条)、`ref/formats.py` 与 [P0006](p0006-ref-format-oracle.md#收尾)。工程依据：[BUILD](../docs/BUILD.md)、[D0005](decisions/d0005-cpu-build-toolchain.md)、[D0007](decisions/d0007-quant-integer-dequant.md) 与 [P0009](p0009-cpu-cpp-build-baseline.md#收尾)。
- 官方只读镜像仅作仲裁，不参与编译、链接或期望值生成。不因 C++ 实现不便修改 `ref/`；确认 oracle 自身有误时另行确认并按禁区规则单独处理。

### 执行契约

- **模块边界**：格式常量只由生产 quant 定义，测试与其他域不形成第二个生产配置中心。公共头只暴露必要类型与调用，避免公共 inline 数值运算依赖消费者未验证的编译选项。原始编码用无符号定宽类型，整数格用有符号类型，BF16 与 FP32 输出均为原始位模式。
- **测试协议**：批量、定宽十六进制位模式文本协议，shape 与操作显式传入，FP32 不经十进制往返。执行器先检查范围再窄化；生产异常映射为可判定的测试响应，执行器自身故障不能伪装为预期的输入错误。该协议只是测试连接，不是生产序列化或 pack 格式，P0011 沿用并扩展。
- **构建接线**：根 CMake 接入 `src/quant`，测试目录建立消费者；`cpu-asan` 只构建 quant 生产库与执行器，不改变现有 preset 的语义。
- **工件与版本**：不预存期望值，不写 `tests/data/`。引擎 VERSION 由合并流程递增；格式契约版本由 `src/quant` 常量定义，测试协议不代替它，也不提前定义 pack 格式。

### 完成声明边界

- E2M1 码点 8：本仓原生解码为负零，官方 FP4_TABLE 为正零；官方一致性声明保留此例外。原始 byte/nibble 往返保持编码，整数格零反向编码为规范零，两者不混为一项可逆性承诺。
- K-Q1 对应 codec、布局与整数格；K-Q3 只覆盖 E4M3FN 与 BF16 解码，以及 FP32 到 BF16 的舍入，不覆盖 BF16 到 E4M3 的编码。K-Q2 与 KV 布局归 P0011。不笼统勾选 K-Q1～K-Q3。
- 本项是唯一的正确性 scalar 基线，不新增性能论断或优化开关。

### 执行与知识处置

- 前置：D0007 接受后才可开工。顺序：同步 active 与 STATUS；在 `p0010-quant-scalar-baseline` 短命分支先打通一个解码入口的执行器与 parity 连接并测耗时，再补齐其余格式、失败契约、环境无关性、`cpu-asan` 与敏感性注入；最后复现验收、归档知识、人工评审。
- 提交：数值及其测试用 `P0010 quant:`，构建接线用 `ci`，计划与知识用 `plan`/`docs`。触碰 `src/quant` 或公共头的提交带 `Oracle: <C++ parity 测试路径>`，不能只指向旧 Python oracle 测试。不用 `meta:` 隐藏行为改动。
- 验收命令沿用 BUILD 的三步 preset 命令（`cpu-debug`、`cpu-release`、`cpu-asan`），保留 `python3 tests/quant/test_formats.py` 与治理回归；合入前运行 `python3 tools/check_repo.py --audit --range main..HEAD --strict` 与 `git diff --check`。
- 知识预计为 updated；必要接口与数值知识缺失阻塞结项，不以 deferred 代替。P0011 是开工前的范围拆分，不是本项的剩余义务，不填入 followups。

### 设计记录

- 2026-10-11：用户要求评议是否适合新建生产 scalar P；只读核对确认 P0006/P0009 已结项，生产 quant 与跨语言 parity 尚未实现。重跑格式 oracle 19 项测试全部通过，`--audit --strict` 为 E=0 W=0 I=0；这些是前置复核，不是本项证据。
- 2026-10-11：用户同意评议中的范围与设计建议，新增本 draft 并同步 STATUS。
- 2026-10-11：评审指出：codec 解码在 FTZ 下可能静默出错；越界验收缺证据；“kernel parity 上一级参照”说法过宽；接口选择缺 D；格式契约版本未进验收；条目过大。用户选定：拆分为本项（codec 与 parity 连接）和 P0011（反量化布局与失败契约）；起草 D0006（proposed）；本项新增 `cpu-asan`。本机 GCC 15.2.0 的 ASan/UBSan 离线探针可用（见 D0006 实测前提）。未开始实现。
- 2026-10-11：评审继续收紧 D0006：校验分为读数据前的元数据校验与写输出前的内容校验；改正次正规数举例；环境检查改以 MXCSR 完整控制字段为准。用户接受 D0006。MODEL.md 中“与官方 convert.py 逐位一致”未注明码点 8 例外、§4 断言 kernel parity 只对照 quant，经用户同意改为单独的 docs 修订，不再作为本项知识条目；计算类上一级参照的完整链条修订留待首个计算类 kernel 工作项。本项仍为 draft，待授权开工。
- 2026-10-11：经用户同意，以 D0007 取代 D0006：反量化也改用整数位运算，取消调用时 MXCSR 检查，失败类别由三类减为两类（去掉 `std::runtime_error`）。本项的 codec、环境无关性与 `cpu-asan` 要求不变，只回填决策引用与异常类别数。仍为 draft。
- 2026-10-11：用户授权开工；P0010 翻转为 active，实现走 `p0010-quant-scalar-baseline` 分支，接口与环境契约以 D0007 为准。
- 2026-10-11：用户同意实现拆解：`decode_e2m1` 接受每字节一个 nibble；输出容量可大于所需、只写前缀；格式契约版本初值 1；执行器故障传播改为永久的 `quant.codec.harness` 测试，四类数值错误与越界读仍临时注入。根 CMake 接线因须指向已存在的目录，并入首个端到端提交，不单独提交。

### 验收证据（待用户评审）

环境：Intel Xeon Platinum 8350C 双路，Ubuntu 26.04 LTS，Linux 5.15.0-72-generic x86_64；`g++ (Ubuntu 15.2.0-16ubuntu1) 15.2.0`；CMake/CTest 4.4.4、Ninja 1.13.2（pip，`/root/miniforge`）；Python 3.12.8；Git 2.54.0。工作仓库与全新本地克隆的实现提交均为分支上的 `9201b8a`。

- 构建与测试：`cpu-debug`、`cpu-release` 各 11/11 通过，`cpu-asan` 4/4 通过（`quant.compile_flags`、`quant.codec.parity`、`quant.codec.contract`、`quant.codec.harness`），三者构建零告警；从 `/tmp` 用 `ctest --test-dir` 运行 `cpu-release` 11/11；全新本地克隆三个 preset 结果相同。普通构建与测试离线，无第三方测试框架，不写 `tests/data/`。
- 模块边界：`df_quant` 静态库只依赖标准库；`df_quant_exec` 只链接该库，不编译生产 `.cpp`。`quant.codec.*` 经 `$<TARGET_FILE:df_quant_exec>` 与 configure 时的解释器路径运行。
- parity 范围：E2M1 16 码点、E4M3FN 与 E8M0 各 256 编码、BF16 全部 65,536 位模式、全部 256 个 packed byte 的打包与展开、多行打包、整数格 15 个合法格点双向；FP32→BF16 覆盖全部有限相邻 BF16 格点中点及其 FP32 两侧邻值（195,840 个）、18 个特殊值与 20,000 个固定种子随机值。输出缓冲区额外 3 个元素，确认前缀之后未被写入。
- 环境无关性：默认、向上舍入、FTZ、DAZ 四种 MXCSR 下逐位一致；每次运行先由执行器回报 MXCSR，驱动核对舍入、FTZ、DAZ 位已生效，执行器退出前确认控制位未变。
- 失败契约：9 个入口各覆盖 rank 0、输入过短/过长、输出容量不足、元素数溢出（两维与三维）、输入输出重叠、4 种零元素 shape（含 `(2^63, 2^63, 0)`，按乘积为 0 接受）；打包入口另测奇数最后维。元数据错误均给出短于声明的数据缓冲区，在 `cpu-asan` 下无 sanitizer 报告。非法 nibble（`0x10`、`0xFF`）与 15 个代表性非格点放在最后一个元素，执行器比对确认输出缓冲区未变，且 `ref/` 对相同输入同样拒绝。格式契约版本读到 1。
- 编译选项：`bb.compile_flags`（Debug/Release）确认 `df_quant` 带 `-fno-fast-math -ffp-contract=off`，`df_quant_exec` 不带，全部 target 无 `-fsanitize`、fast-math 与 `-march`；`quant.compile_flags`（`cpu-asan`）确认两者均带 D0007 的 sanitizer 选项。
- 耗时：Debug 下 262,144 元素单请求，`ref/` 期望值生成 0.27s、执行与比较 0.22s；全量 parity 每种环境约 0.23s（Debug）、0.6s（`cpu-asan`）。
- 故障传播（永久测试 `quant.codec.harness`）：执行器崩溃、提前退出、输出乱码、少一条响应、缺执行器、token 超出元素宽度、未知操作均使驱动报 `HarnessError`；结果被篡改时比较失败；parity 与 contract 套件在故障注入下进程非零退出。
- 敏感性（临时注入，已恢复，源码逐字节还原）：nibble 次序互换、BF16 中点改为向上取整、E2M1 负零改为正零、E8M0 `0x00` 解码为零，`quant.codec.parity` 均失败（CTest 退出码 8），分别定位到 `pack.all_bytes[1]`、`fp32_to_bf16.rounding[1]`、`e2m1.all[8]`、`e8m0.all[0]`。在 `decode_e4m3fn` 入口注入读取越界一个元素，`cpu-asan` 报 heap-buffer-overflow，parity 与 contract 失败。
- 开发中发现：首次 `cpu-asan` 运行由 UBSan 报出执行器在零字节输出时向 `memcmp` 传空指针，已在 `ci` 提交中修复；生产库无报告。
- 治理：`python3 tests/quant/test_formats.py`、`python3 tests/tools/test_check_repo.py` 通过；`python3 tools/check_repo.py --audit --range main..HEAD --strict` 为 E=0 W=0 I=0；`git diff --check main..HEAD` 无输出。
- 未完成：知识归档（`docs/QUANT.md`、`docs/BUILD.md`）待用户确认正文；用户评审。

## 收尾

- 验收或停止依据：待实现、复现验收并经用户评审后填写。
- 知识处置：pending。
- 剩余义务：待收尾时据实际交付判断。
