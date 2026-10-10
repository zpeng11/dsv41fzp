---
id: D0005
title: CPU-only 构建采用 CMake/CTest + Ninja、C++20 与按 target 施加的严格数值选项
status: accepted
created: 2026-10-10
updated: 2026-10-10
replaces: []
superseded-by: []
---

## 背景与问题

P0009 要在生产 `src/quant` 之前建立第一套 C++ 构建与测试入口。AGENTS.md 已规定 CMake 读入 VERSION、dirty 后缀与生成物不入库，但工具来源、语言标准、target 组织、严格数值选项及其验证方式尚无决策；这些选择会被后续 quant、CPU kernel 与 CUDA 工作继承，须在实现前确定。

## 决策驱动

- 技术约束：quant 与 CPU kernel 的 parity 要求宿主浮点行为确定，即 binary32、RNE、保留正负零与渐进下溢、不发生非预期的 FMA 收缩；选项须在运行时可验证，而非只看配置文本。
- 技术约束：未来 AVX-512/FMA 路径会在 target 上启用 `-mfma` 类 ISA 选项，严格数值设置必须在启用 FMA 时仍然生效。
- 项目约束：普通构建离线、不隐式下载；支持声明只覆盖实测环境；选项按 target 施加，不经全局 flags 传播。
- 项目约束：首轮不为 CUDA、SIMD 分派或安装导出预设结构，避免占位模块。

## 实测前提

环境：Intel Xeon Platinum 8350C（双路，支持 FMA/AVX-512F/AVX-512 VNNI），Ubuntu 26.04 LTS，`g++ (Ubuntu 15.2.0-16ubuntu1) 15.2.0`；方法：`scratch/` 中对运行时输入计算 `a*a+c`，其中 `a = 1+2^-12`，`c = -(1+2^-11)`，精确结果为 `2^-24`，分别舍入时为 0。

| 编译选项（均含 `-mfma`） | 结果 |
| --- | --- |
| `-std=c++20 -O2`（不指定 contract） | `0x1p-24`（已收缩） |
| `-std=gnu++20 -O2`（不指定 contract） | `0x1p-24`（已收缩） |
| `-O2 -ffp-contract=fast` | `0x1p-24` |
| `-O2 -ffp-contract=off` | `0x0p+0` |
| `-O0`，`fast` 或 `off` | `0x0p+0` |

结论：该 GCC 在 ISO C++ 模式下默认仍会收缩，必须显式 `-ffp-contract=off`；`-O0` 不收缩，Debug 构建无法证明反例的敏感性。

## 备选方案

### 工具来源

- 系统包管理器（apt：CMake 4.2.3、Ninja 1.13.2）：与 Python 环境解耦，版本随发行版固定；需要 root，不同发行版版本不同，不采纳。
- 装入运行测试的同一 conda 环境（经 conda 或 pip）：无需 root，可锁版本，构建工具与测试用 Python 来自同一环境；代价是切换环境即换工具：用户选定采纳。
- Kitware 官方二进制或自动下载：后者违反离线约束，否决；手动放置的官方二进制与 apt 相比没有明显收益，不采纳。

### 构建系统

- CMake + CTest + Ninja + CMakePresets：采纳。CTest 可同时注册 C++ 与 Python 测试；未来 CUDA 可用 CMake 原生 CUDA 语言支持。
- Meson 或手写 Makefile：生态与 CUDA 集成弱于 CMake，且 AGENTS.md 已写明由 CMake 读入 VERSION，不采纳。

### 语言标准

- C++20，`CXX_EXTENSIONS OFF`：采纳。只约束宿主代码，不预先批准 CUDA translation unit 的标准。
- C++17：缺少 `std::bit_cast`、concepts 等格式与 kernel 代码常用的设施，不采纳。
- C++23：编译器支持仍不均，nvcc 的兼容性未验证，不采纳。

### 严格数值选项的施加方式

- 全局 `CMAKE_CXX_FLAGS`：会污染无关 target，也无法与未来需要不同选项的 target 共存，否决。
- 专用 INTERFACE target（拟名 `df_strict_fp`）携带 `-fno-fast-math -ffp-contract=off`，数值 target 显式链接：采纳。

### 收缩反例的验证方式

- 只检查编译命令中的生效选项：能证明选项到位，但证明不了运行时行为。
- 在 x86-64 基线 ISA 上做运行时反例：基线 ISA 没有 FMA，反例永远不会触发，否决。
- 反例 target 固定 `-O2 -mfma`，与配置无关；用同一源码编译两份：`-ffp-contract=fast` 的一份必须得到收缩结果（证明测试敏感），链接 `df_strict_fp` 的一份必须得到分别舍入的结果：采纳，并辅以编译命令检查。宿主不支持 FMA 时该测试明确失败，不 skip。

### 版本 dirty 判定时机

- configure 时快照：实现最简，但 configure 之后的改动不反映到 dirty，脏构建会自称干净并污染 bench manifest；也偏离 AGENTS.md「构建时工作树脏则加 `+dirty`」的表述，不采纳。
- 构建时刷新：每次 build 运行一个 `cmake -P` 脚本查询 Git 状态，内容变化才改写生成头；符合 AGENTS.md，代价是多一个始终执行的自定义 target 和每次 build 一次 `git status`：用户选定采纳。

## 结论

经用户确认采纳：

1. CMake + CTest + Ninja，提供 `cpu-debug` 与 `cpu-release` 两个 preset，二进制目录 `build/<preset>`。`cmake_minimum_required` 取所用 preset schema 与命令所需的最低版本，支持声明只覆盖实测版本。
2. 宿主代码 C++20，关闭编译器扩展。
3. 通用告警设置以 INTERFACE target 提供：`-Wall -Wextra -Wpedantic`，不全局开 `-Werror`。不禁用 exceptions/RTTI，不设全局 `-march`。
4. 严格数值设置由 `df_strict_fp` INTERFACE target 提供，在 Debug 与 Release 下均生效。
5. 运行时前提检查：`std::numeric_limits<float>::is_iec559`、`FLT_EVAL_METHOD == 0`、`fegetround() == FE_TONEAREST`、正负零可区分、对运行时输入的次正规数运算结果非零（同时覆盖 FTZ 与 DAZ），以及上述 FMA 收缩反例。
6. configure 时若 `CMAKE_CXX_FLAGS` 或任一配置专属 flags 含 `-ffast-math`、`-Ofast`、`-funsafe-math-optimizations` 或 `-ffp-contract=fast`，以 FATAL_ERROR 拒绝，不静默接受矛盾选项。
7. CMake、CTest 与 Ninja 装入运行测试的同一 conda 环境，经 conda 或 pip 安装，同一工具不混用两种来源；构建本身不自动安装，缺失时明确失败。CTest 使用的 Python 解释器仍显式传入。
8. 版本信息在构建时刷新：始终执行的自定义 target 读取 VERSION、`git rev-parse HEAD` 与 `git status --porcelain`（遵守 `.gitignore`），经临时文件 `copy_if_different` 更新生成头并声明为 BYPRODUCTS；生成头只被单个版本 TU 包含，状态未变时不重编。非 Git 检出或缺少 `git` 时构建失败。

## 后果

- P0009 依本决策实现；未来 quant 与 CPU kernel target 通过链接 `df_strict_fp` 获得数值前提，启用 `-mfma`/AVX-512 时收缩仍被禁止。
- 未来确有收缩或 fast-math 需求的路径，须经运行时开关与 bench 证据另立决策，不得修改 `df_strict_fp` 的语义。
- 支持环境、命令与选项归档到 `docs/BUILD.md`；本条目只保留选择理由。
- 构建时刷新只保证二进制反映构建时刻的状态，build 之后、运行之前的改动仍无法反映。未来 bench 工具须在运行时自行判定 dirty 并与二进制报告值比对，不一致拒绝落盘；此义务归 bench 工作流，不属于 P0009。
- 换用其他 conda 环境即换用其中的工具版本；BUILD 记录实测环境，不在其他环境上推定支持。
