# BUILD.md — CPU-only 构建与测试契约

本页归档 C++ 构建契约、操作命令与支持环境。选择理由见 [D0005](../plan/decisions/d0005-cpu-build-toolchain.md) 与 [D0007](../plan/decisions/d0007-quant-integer-dequant.md)（`cpu-asan`），构建基线证据见 [P0009](../plan/p0009-cpu-cpp-build-baseline.md)；数值入口的状态与验收证据由 [STATUS](../plan/STATUS.md) 路由，规格见各契约页。

**来源标记**：**[D]** D0005/D0007/D0013 决策 · **[代码]** 本仓构建文件（`CMakeLists.txt`、`CMakePresets.json`、`cmake/`、`tests/CMakeLists.txt`）· **[实测]** 下述支持环境中的实测结果。

## 1. 支持环境

首批支持声明只覆盖下表实测配置，全仓以 **ENV-A** 指代；其他编译器、工具版本、操作系统与 CUDA translation unit 均未验证，不从本表推定支持。[D][实测]

| 项 | 实测配置 |
| --- | --- |
| 平台 | Linux 5.15.0-72-generic x86_64，Ubuntu 26.04 LTS，glibc 2.43；Intel Xeon Platinum 8350C 双路（含 FMA） |
| 编译器 | `g++ (Ubuntu 15.2.0-16ubuntu1) 15.2.0` |
| CMake / CTest | 4.4.4（pip 包 `cmake`） |
| 生成器 | Ninja 1.13.2（pip 包 `ninja`） |
| Python | 3.12.8（miniforge，conda-forge，GCC 13.3.0 构建） |
| Git | 2.54.0 |

`cmake_minimum_required` 与 preset schema 声明 3.21，是所用命令与 preset 特性的下限，不是已验证的支持版本。[代码]

CMake 只接受 GCC 与 Linux x86_64，其余在 configure 时 FATAL_ERROR；该拒绝分支本机未演练。[代码]

## 2. 工具准备

CMake、CTest 与 Ninja 装入运行测试的同一 conda 环境，经 conda 或 pip 安装，同一工具不混用两种来源；构建本身不联网、不自动安装，工具缺失时 configure 失败。[D] 实测环境的安装命令：

```bash
python3 -m pip install cmake ninja
```

换用其他 conda 环境即换用其中的工具版本，须重新验证后才扩展本页支持声明。[D]

## 3. 命令

在仓库根目录执行；二进制目录为 `build/<preset>`（已 git-ignore）。源码目录内直接构建会在 configure 时被拒绝，但 CMake 在拒绝前已写入 `CMakeCache.txt` 与 `CMakeFiles/`，须手动删除。[代码][实测]

```bash
cmake --preset cpu-debug
cmake --build --preset cpu-debug
ctest --preset cpu-debug
```

Release 将三处 `cpu-debug` 换为 `cpu-release`。从其他目录运行测试用 `ctest --test-dir <repo>/build/<preset>`。[实测] 不启用 CUDA 语言，不需要 GPU、Torch/TileLang 或模型权重。[代码][实测]

越界与未定义行为检查配置把三处换为 `cpu-asan`：Debug，`DF_SANITIZE=ON`、`DF_TEST_SCOPE=quant`，只构建 quant 生产库与测试执行器，只注册 quant C++ 测试与其编译选项检查。测试 preset 设 `ASAN_OPTIONS=detect_leaks=1:halt_on_error=1:exitcode=86` 与 `UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`，任何 sanitizer 报告使对应测试失败。依赖 GCC 自带的 libasan/libubsan，实测于 §1 的 GCC 15.2.0；工具链缺这些库时构建失败，不 skip。注错检出证据见 [P0010](../plan/p0010-quant-scalar-baseline.md)。[D][代码][实测]

## 4. target 契约

选项按 target 施加，不经全局 `CMAKE_CXX_FLAGS`、`include_directories` 或全仓 `-march` 传递；`CMAKE_CXX_EXTENSIONS` 关闭。[D][代码]

| 名称 | 类型 | 作用 |
| --- | --- | --- |
| `df_cxx_baseline` | INTERFACE | C++20（`-std=c++20`，无 GNU 扩展），`-Wall -Wextra -Wpedantic`；不开 `-Werror`，不禁用 exceptions/RTTI |
| `df_strict_fp` | INTERFACE | `-fno-fast-math -ffp-contract=off`；Debug 与 Release 均生效，数值 target 显式链接 |
| `df_use_version_info(<target>)` | 函数 | 使 target 依赖版本刷新步骤并可包含 `df/version_info.h`；该头只应被单个 TU 包含 |
| `df_sanitize` | INTERFACE | `DF_SANITIZE=ON` 时带 `-fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer`（编译与链接），否则为空；只挂到 quant 生产库与执行器 |
| `df_quant` | STATIC | 生产 quant 库（`src/quant`），公共头根为 `include/`，向消费者只传播 C++20；私有链接 `df_cxx_baseline`、`df_strict_fp`、`df_sanitize` |

缓存变量 `DF_TEST_SCOPE` 取 `all`（默认）或 `quant`，其他值在 configure 时 FATAL_ERROR。[代码]

## 5. 数值前提

- 链接 `df_strict_fp` 的 target 在启用 `-mfma` 时仍不发生 FMA 收缩。实测 GCC 15.2.0 在 ISO C++ 模式、`-O2 -mfma` 下默认会收缩，因此必须显式 `-ffp-contract=off`；`-O0` 不收缩，收缩探针固定 `-O2 -mfma` 编译。[D][实测]
- 运行时前提：binary32、`FLT_EVAL_METHOD == 0`、RNE、正负零可区分、次正规数输出与输入不被冲刷（FTZ/DAZ 关闭）。检查程序在进程内切到向上舍入、FTZ、DAZ 后须识别违规，否则测试失败。[代码][实测]
- 检查的输入与期望值必须是编译期常量：运行时计算的期望值同样受被测环境影响，会掩盖违规。[实测]
- `CMAKE_CXX_FLAGS` 或任一配置专属 flags（含经 `CXXFLAGS` 环境变量传入的值）出现 `-ffast-math`、`-Ofast`、`-funsafe-math-optimizations`、`-ffp-contract=fast`、`-ffinite-math-only` 时，configure FATAL_ERROR。[代码][实测]
- 上述只是构建前提，不替代 quant 的格式、乘法与舍入 parity。

## 6. 版本元数据

- 每次 build 执行 `df_version_info` 步骤：读取 `VERSION`、`git rev-parse --verify HEAD`、`git status --porcelain --untracked-files=normal`，生成 `build/<preset>/generated/df/version_info.h`；内容未变时不改写，不触发重编。[D][代码][实测]
- dirty = 存在已跟踪改动（含暂存）或非忽略的未跟踪文件；`.gitignore` 覆盖的 `build/`、`scratch/` 不算。显示版本仅在 dirty 时为 `<VERSION>+dirty`，后缀不写回 `VERSION`。[D][代码]
- 改动工作树后只需 build，不需重跑 configure。[实测]
- 不在 Git 检出中构建、或源码目录不是 Git 顶层时，build 失败，不伪造 SHA 或 clean 状态。[代码][实测]
- 版本信息只反映最近一次 build 时的工作树；build 之后再改动不会反映到二进制，bench 工具须在运行时自行判定 dirty 并比对。[D]

## 7. 测试清单

CTest 统一入口注册以下测试，任一失败使 `ctest` 非零退出；Python 测试经 configure 时确定的解释器绝对路径运行，并设 `PYTHONDONTWRITEBYTECODE=1`，不在源码树留下缓存。[代码][实测]

| 测试 | 内容 |
| --- | --- |
| `bb.smoke` | 静态库与消费者跨 TU 链接 |
| `bb.version` | 二进制报告的版本、SHA、dirty 与当前工作树一致 |
| `bb.fp_env` | §5 运行时前提及其反例 |
| `bb.contract.strict` / `bb.contract.fast_control` | 严格 target 不收缩；`-ffp-contract=fast` 对照版须收缩，证明探针敏感 |
| `bb.compile_flags` | 从 `compile_commands.json` 验证严格选项只出现在应有的 target（含 `df_quant`，不含其执行器），全部 target 无 fast-math、`-march` 与 `-fsanitize` |
| `quant.formats` | `tests/quant/test_formats.py` |
| `quant.compute` | `tests/quant/test_compute.py`：专家 FP8×packed FP4 单输出计算参照的独立逐阶段期望、输入契约、临时副本敏感性与失败传播 |
| `quant.act_quant` | `tests/quant/test_act_quant.py`：BF16 动态激活量化参照的独立逐阶段期望、有限码点穷举、布局与计算组合、临时副本敏感性与失败传播 |
| `quant.codec.parity` | 生产 codec 在默认、向上舍入、FTZ、DAZ 四种 MXCSR 下逐位对照 `ref/` |
| `quant.codec.contract` | 每类输入拒绝、失败时输出不变、格式契约版本 |
| `quant.codec.harness` | 执行器崩溃、退出、乱码、缺响应、缺执行器、协议错误与结果篡改均使驱动失败 |
| `quant.dequant.parity` | 六类给定 scale 反量化、两种输出、合法编码组合和布局边界，在八模式 MXCSR 下对照 `ref/dequant.py` |
| `quant.dequant.contract` | 数据与 scale 的元数据/内容拒绝、两类重叠、失败时输出不变和零元素 |
| `quant.dequant.multiply` | 同一个生产有限 FP32 整数乘法 helper 的私有连接，定向 RNE 边界与固定种子补充，在八模式 MXCSR 下对照 `ref/formats.py` |
| `quant.dequant.harness` | 新协议、私有探针坏请求、执行器及结果故障均传播到进程退出 |
| `quant.compute.parity` | 生产专家分组点积两种输出与私有 trace 的逐阶段结果，在八模式 MXCSR（含状态位预置）下对照 `ref/compute.py`；与 Python oracle 自身的 `quant.compute` 是不同测试 |
| `quant.compute.contract` | 每类长度、内容与容量拒绝，ASan 下读前加毒的元数据拒绝，双向部分重叠拒绝与恰好相邻接受，失败时输出不变 |
| `quant.compute.add` | 同一生产 FP32 加法 helper 的私有连接，定向舍入与 sticky 边界加固定种子补充 |
| `quant.compute.harness` | 计算协议、trace 形状、超宽 token、未知输出模式、加毒有效性与执行器故障均传播到进程退出 |
| `quant.act_quant.parity` | 生产 BF16 动态激活量化输出与全部私有阶段位，在八模式 MXCSR（状态位清空与预置）下对照 `ref/act_quant.py`，含有限码点穷举 |
| `quant.act_quant.contract` | 长度、双输出容量、非有限、实际写入范围重叠、trace 形状与别名拒绝，失败输出不变；精确分配、相邻范围、未使用尾部共享与 ASan 输入加毒 |
| `quant.act_quant.combination` | 执行器内生产量化结果直接供生产分组点积，两种输出对照 Python 量化和计算参照组合 |
| `quant.act_quant.harness` | 量化协议、trace 响应形状、执行器故障与 ASan 加毒有效性均传播到驱动失败 |
| `tools.check_repo` | `tests/tools/test_check_repo.py` |
| `tools.ckpt_inventory` | `tests/tools/test_ckpt_inventory.py`：合成 safetensors 上的每类结构与 index 拒绝、预期比对的缺失/多余/dtype/shape 检出、小配置的手算张量计数与 shape 抽查、扫描计数与逐元素朴素计数一致且不随分块与进程数变化 |
| `tools.ckpt_expert` | `tests/tools/test_ckpt_expert.py`：合成 safetensors 的独立行字节与读取区间期望、权重/scale 配对、错误拒绝、生产组合 parity 和故障传播；通过 `df_quant_exec` 消费原始编码，不依赖真实 checkpoint |

计算参照与动态激活量化参照的全部必选验收也可单独运行；从源码目录外运行时使用脚本绝对路径。两者仅依赖标准库，不初始化 Torch/TileLang/CUDA，不依赖目标机、权重、设备 dump 或网络；错误结果、异常或 skip 使命令非零退出。验证方法与契约分别以 [COMPUTE](COMPUTE.md) 与 [ACTQUANT](ACTQUANT.md) 为唯一来源。[代码][实测]

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tests/quant/test_compute.py
PYTHONDONTWRITEBYTECODE=1 python3 tests/quant/test_act_quant.py
```

生产量化套件由上述三种 CPU preset 的 CTest 运行，也可单独调用驱动；`--suite` 取 `parity`、`contract`、`combination` 或 `harness`。临时生产变体敏感性单独运行，不进入每次 CTest，证据见 [P0017](../plan/p0017-act-quant-scalar.md)。以下命令不需要权重或 GPU；源码目录外运行时将脚本与执行器路径换成绝对路径。[代码]

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tests/quant/test_act_quant_parity.py --exe build/cpu-release/tests/quant/df_quant_exec --suite parity
PYTHONDONTWRITEBYTECODE=1 python3 tests/quant/act_quant_sensitivity.py
```

**真实 checkpoint 核对**不进 CTest（依赖本机快照与约 510 GB 数据），在仓库根目录执行；`<snapshot>` 为 HF 快照目录，本机为 `/mnt/pfs/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/2cba9e42aa026125f3ed06c6d98c1db82f7ca027`。[代码][实测]

```bash
python3 tools/ckpt_inventory.py <snapshot> --json build/ckpt.json --inventory build/ckpt.tsv
python3 tools/ckpt_inventory.py <snapshot> --scan --jobs 96 --progress --json build/ckpt-scan.json
```

第一条核对配置文件与镜像逐字节相同、shard 结构、index 与独立预期；第二条另做全量编码扫描，读完全部载荷，`--jobs` 按可用核心设置。退出码 0 表示全部检查通过，1 表示有检查失败，2 表示用法或 I/O 错误；JSON 与 TSV 是生成物，写入 git-ignore 的 `build/` 或 `scratch/`。结构规则 pin 于 `huggingface/safetensors@e246a256`；预期只由 `third_party/reference/inference/config.json` 与官方构造代码生成，每条规则的源码行号随 TSV 输出。工具仅依赖标准库，要求小端主机，码点分类经 `ref/formats`。结果归档于 [MODEL §2](MODEL.md#2-结构与格式) 的 [ckpt] 事实。[代码][实测]

**真实专家权重消费**由 `tools/ckpt_expert.py` 显式执行；验收过程见 [P0018](../plan/p0018-checkpoint-expert-consumption.md)，验证链边界见 [MODEL §4](MODEL.md#4-精确性链条)。先按清点工具的结构、index、配置与独立预期核对全部头部，再只读取 `layers.0.ffn.experts.0` 的 w1/w3/w2 各首、中（`N // 2`）、末输出行及对应 scale，保留完整 K 维。张量偏移、dtype、shape 与名称配对错误、越界或短读均失败；不扫描其他权重载荷，不修改 checkpoint。[代码]

```bash
python3 tools/ckpt_expert.py <snapshot> --exe build/cpu-release/tests/quant/df_quant_exec --json build/ckpt-expert.json
```

输入为工具内固定整数位构造的 `mixed`、`group_ramp` 两组有限 BF16 模式，各矩阵独立运行。激活普通/trace、点积阶段与直接生产缓冲组合两类输出均对照既有 Python 参照；有限值、零与 Inf 按位比较，NaN 按 COMPUTE 分类比较。工具复用仓库测试协议连接，不是生产加载器或新的数值 oracle；执行器失败与计算拒绝不能算作 parity 通过。[代码]

成功 JSON 记录快照 revision、源码 SHA/VERSION/dirty、实测运行环境、执行器路径与 SHA256、输入构造和原始位指纹、每段载荷的张量名/文件/绝对字节偏移/长度/SHA256、比较计数及两类输出。源码身份与执行器指纹分别记录，不能据前者推定任意外部二进制的构建身份。报告以独占创建写入，拒绝覆盖已有目标或写入 checkpoint 目录；复跑须指定新报告名。退出码 0 表示通过，1 表示验证失败，2 表示参数或 I/O 错误；缺权重不 skip。`--reference` 仅用于指定结构核对的配置来源，正式快照验证使用默认只读镜像。报告属生成物，不入库。[代码]

`tools.ckpt_expert` 只在 CPU Debug/Release 注册，使用临时合成文件与当前构建的执行器；不扩大 quant 专用 sanitizer 范围。源码目录外运行上述入口时，脚本、快照、执行器和报告目标均使用绝对路径。[代码]

`quant.compute` 与 `quant.act_quant` 注册在 quant 专用 scope 返回之后，只进入全量 CPU Debug/Release，不扩大 sanitizer 范围。`quant.act_quant` 的浮点环境探针用标准库 `ctypes` 调用 glibc `fegetenv`/`fesetenv`，按 Linux x86_64 的 `fenv_t` 布局设置并核对 MXCSR；该主机探针不宣称支持其他平台，不 skip 或要求临时编译辅助库。[代码]

`cpu-asan` 只注册 `quant.codec.*`、`quant.dequant.*`、`quant.compute.{parity,contract,add,harness}`、`quant.act_quant.{parity,contract,combination,harness}` 与 `quant.compile_flags`（验证两个 quant target 均带 sanitizer 选项）；Python oracle 的 `quant.compute` 与 `quant.act_quant` 不在其中。四组 quant 套件经 `$<TARGET_FILE:df_quant_exec>` 定位执行器，非共享缓冲区按精确长度分配，使越界访问对 ASan 可见；私有乘法、加法与逐阶段观测探针直接调用库中同一 helper 或计算核心，不将其加入公共 API。该执行器以十六进制逐行承载 shape 与各份原始编码，只是测试连接，不是生产序列化或 pack 格式。[代码][实测]

**八模式 MXCSR**（各契约页以此名引用）：默认、向上、向下、向零舍入、FTZ、DAZ、FTZ+DAZ 与解除溢出掩码。执行器或探针回报设置，并在每次被测调用前后核对完整 MXCSR，包含拒绝与异常路径，stdio 在核对窗口外；声明"状态位预置"的套件另在每种模式下分别预置状态位清空与置满。`quant.dequant.{parity,multiply}`、`quant.compute.parity`、`quant.act_quant` 与生产 `quant.act_quant.{parity,contract,combination}` 使用八模式，`quant.codec.parity` 只用默认、向上舍入、FTZ、DAZ 四种。量化库和捕获其异常的执行器不得带 `-fno-exceptions`，两个编译选项检查 profile 按 target 检查此项，不将其扩为全仓规则或约束未来 SIMD/CUDA 热路径。[D][代码][实测]

`bb.*` 工件只在 `tests/cpu/build_baseline/`，是构建基线的测试件，不是引擎模块。[代码] 仓库检查 `tools/check_repo.py` 仍按 AGENTS.md 工作流独立执行，不由 CTest 替代。
