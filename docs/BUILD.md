# BUILD.md — CPU-only 构建与测试契约

本页归档当前真实的 C++ 构建契约、操作命令与支持证据。选择理由见 [D0005](../plan/decisions/d0005-cpu-build-toolchain.md)，验收过程与证据见 [P0009](../plan/p0009-cpu-cpp-build-baseline.md)；状态不在本页记录。

**来源标记**：**[D]** D0005 决策 · **[代码]** 本仓构建文件（`CMakeLists.txt`、`CMakePresets.json`、`cmake/`、`tests/CMakeLists.txt`）· **[实测]** 下述支持环境中的实测结果。

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

## 4. target 契约

选项按 target 施加，不经全局 `CMAKE_CXX_FLAGS`、`include_directories` 或全仓 `-march` 传递；`CMAKE_CXX_EXTENSIONS` 关闭。[D][代码]

| 名称 | 类型 | 作用 |
| --- | --- | --- |
| `df_cxx_baseline` | INTERFACE | C++20（`-std=c++20`，无 GNU 扩展），`-Wall -Wextra -Wpedantic`；不开 `-Werror`，不禁用 exceptions/RTTI |
| `df_strict_fp` | INTERFACE | `-fno-fast-math -ffp-contract=off`；Debug 与 Release 均生效，数值 target 显式链接 |
| `df_use_version_info(<target>)` | 函数 | 使 target 依赖版本刷新步骤并可包含 `df/version_info.h`；该头只应被单个 TU 包含 |

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
| `bb.compile_flags` | 从 `compile_commands.json` 验证严格选项只出现在应有的 target，全部 target 无 fast-math 与 `-march` |
| `quant.formats` | `tests/quant/test_formats.py` |
| `tools.check_repo` | `tests/tools/test_check_repo.py` |

`bb.*` 工件只在 `tests/cpu/build_baseline/`，是构建基线的测试件，不是引擎模块。[代码] 仓库检查 `tools/check_repo.py` 仍按 AGENTS.md 工作流独立执行，不由 CTest 替代。
