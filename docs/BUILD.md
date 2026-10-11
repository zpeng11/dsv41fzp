# BUILD.md — CPU-only 构建与测试契约

本页归档当前真实的 C++ 构建契约、操作命令与支持证据。选择理由见 [D0005](../plan/decisions/d0005-cpu-build-toolchain.md) 与 [D0007](../plan/decisions/d0007-quant-integer-dequant.md)（`cpu-asan`），验收过程与证据见 [P0009](../plan/p0009-cpu-cpp-build-baseline.md)、[P0010](../plan/p0010-quant-scalar-baseline.md) 与 [P0011](../plan/p0011-quant-dequant-scalar.md)；状态不在本页记录。

**来源标记**：**[D]** D0005/D0007 决策 · **[代码]** 本仓构建文件（`CMakeLists.txt`、`CMakePresets.json`、`cmake/`、`tests/CMakeLists.txt`）· **[实测]** 下述支持环境中的实测结果。

## 1. 支持环境

首批支持声明只覆盖下表实测配置；其他编译器、工具版本、操作系统与 CUDA translation unit 均未验证，不从本表推定支持。[D][实测]

| 项 | 实测配置 |
| --- | --- |
| 平台 | Linux 5.15.0-72-generic x86_64，Ubuntu 26.04 LTS；Intel Xeon Platinum 8350C 双路（含 FMA） |
| 编译器 | `g++ (Ubuntu 15.2.0-16ubuntu1) 15.2.0` |
| CMake / CTest | 4.4.4（pip 包 `cmake`） |
| 生成器 | Ninja 1.13.2（pip 包 `ninja`） |
| Python | 3.12.8（miniforge，conda-forge） |
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

越界与未定义行为检查配置把三处换为 `cpu-asan`：Debug，`DF_SANITIZE=ON`、`DF_TEST_SCOPE=quant`，只构建 quant 生产库与测试执行器，只注册 quant C++ 测试与其编译选项检查。测试 preset 设 `ASAN_OPTIONS=detect_leaks=1:halt_on_error=1:exitcode=86` 与 `UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`，任何 sanitizer 报告使对应测试失败。依赖 GCC 自带的 libasan/libubsan，实测于 §1 的 GCC 15.2.0；工具链缺这些库时构建失败，不 skip。临时注入的越界读被识别为 heap-buffer-overflow 并使 CTest 失败。[D][代码][实测]

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
| `quant.codec.parity` | 生产 codec 在默认、向上舍入、FTZ、DAZ 四种 MXCSR 下逐位对照 `ref/` |
| `quant.codec.contract` | 每类输入拒绝、失败时输出不变、格式契约版本 |
| `quant.codec.harness` | 执行器崩溃、退出、乱码、缺响应、缺执行器、协议错误与结果篡改均使驱动失败 |
| `quant.dequant.parity` | 六类给定 scale 反量化、两种输出、合法编码组合和布局边界，在八种 MXCSR 下对照 `ref/dequant.py` |
| `quant.dequant.contract` | 数据与 scale 的元数据/内容拒绝、两类重叠、失败时输出不变和零元素 |
| `quant.dequant.multiply` | 同一个生产有限 FP32 整数乘法 helper 的私有连接，定向 RNE 边界与固定种子补充，在八种 MXCSR 下对照 `ref/formats.py` |
| `quant.dequant.harness` | 新协议、私有探针坏请求、执行器及结果故障均传播到进程退出 |
| `tools.check_repo` | `tests/tools/test_check_repo.py` |

`cpu-asan` 只注册 `quant.codec.*`、`quant.dequant.*` 与 `quant.compile_flags`（验证两个 quant target 均带 sanitizer 选项）。两类 quant 套件经 `$<TARGET_FILE:df_quant_exec>` 定位执行器；私有乘法探针直接调用库中同一个 helper，不将通用乘法加入公共 API。该执行器与携带数据/scale shape 及两份原始编码的批量十六进制协议只是测试连接，不是生产序列化或 pack 格式。[代码][实测]

反量化与私有乘法测试覆盖默认、向上、向下、向零、FTZ、DAZ、FTZ+DAZ 与解除溢出掩码；执行器回报设置，并在每次生产调用前后核对完整 MXCSR，包含异常路径，stdio 在核对窗口外。量化格式、数值与验证边界见 [QUANT §7](QUANT.md#7-给定-scale-的生产-scalar-反量化)。量化库和捕获其异常的执行器不得带 `-fno-exceptions`，两个编译选项检查 profile 按 target 检查此项，不将其扩为全仓规则或约束未来 SIMD/CUDA 热路径。[D][代码][实测]

`bb.*` 工件只在 `tests/cpu/build_baseline/`，是构建基线的测试件，不是引擎模块。[代码] 仓库检查 `tools/check_repo.py` 仍按 AGENTS.md 工作流独立执行，不由 CTest 替代。
