---
id: P0009
title: 建立 CPU-only C++ 构建与测试基线
status: completed
created: 2026-10-10
updated: 2026-10-13
knowledge: updated
docs: [docs/BUILD.md]
bench: []
decisions: [D0001, D0005]
followups: []
---

## 意图

在实现生产 `src/quant` 前，建立可独立验收的第一套 C++ 构建与测试入口，明确工具链、target 边界、数值编译前提和版本元数据刷新规则。成功标准是无需 CUDA 或模型权重即可从干净检出完成配置、构建和测试，且后续 quant 能直接接入同一入口；不交付推理引擎、量化实现或完整基础设施。

## 验收

- [x] **CPU-only、离线与独立构建**：干净检出按归档命令完成 configure/build/CTest；不寻找 CUDA、不要求 Torch/TileLang、权重或目标机；不联网、不隐式安装，工具缺失明确失败；生成物只在忽略的 build 目录，官方镜像不参与构建。
- [x] **可判定的首个支持环境**：记录实际工具链、平台、选项与命令，以 C++20 编译链接 smoke 库与消费者；支持声明只覆盖实测配置。
- [x] **target 边界真实可用**：跨 TU 库与消费者链接验证；include、语言、选项按 target 管理，编译命令证明数值选项不污染无关 target；无 fast-math、全局 native ISA 或无依据的 exceptions/RTTI 禁用；无占位业务模块。
- [x] **数值编译前提可检查**：Debug/Release 均禁用 fast-math 与收缩；运行时检查 binary32、RNE、正负零与渐进下溢，反例能识别舍入环境与 FTZ/DAZ；矛盾选项不得无声破坏严格 target。不替代 quant parity。
- [x] **统一测试与失败传播**：CTest 注册 C++ smoke、`test_formats.py` 与 `test_check_repo.py`；解释器路径显式传入，源码目录外可运行；缺依赖不 skip，失败注入传到 CTest。
- [x] **开发配置**：CPU Debug 与 Release 入口均运行全部必选测试。
- [x] **版本信息构建时刷新**：每次 build 读取 VERSION、SHA、dirty，仅 dirty 时加 `+dirty` 且不写回；状态不变不重编；干净、改动后只 build、撤销三态演练成立。
- [x] **知识、治理与人工收尾**：契约归档 docs，理由形成 D；回归、仓库检查与范围审计通过；用户评审后结项，不写成 quant、CUDA 或部署资质已验收。

## 记录

### 范围与依据

- 依据 [D0005](decisions/d0005-cpu-build-toolchain.md)（GCC/Linux x86_64、C++20、conda/pip 工具、`cpu-debug`/`cpu-release` preset、构建时 dirty 判定）；契约见 [BUILD](../docs/BUILD.md)。
- 范围外：生产 quant、CUDA、SIMD/native、pack、安装/ABI、CI/bench 全套、P0005；评审中移出 sanitizer 配置、Python 调用 C++ 探针与版本状态组合矩阵。

### 验收证据

- 环境：ENV-A（[BUILD §1](../docs/BUILD.md#1-支持环境)）。
- `cpu-debug`/`cpu-release` 各 8/8，零告警；`/tmp` 下 `ctest --test-dir` 与全新克隆 `cpu-release` 均 8/8，克隆仅见 `build/`。
- 收缩探针（`-O2 -mfma`）：严格版 `0x0p+0`，`-ffp-contract=fast` 对照版 `0x1p-24`；`bb_fp_env` 在 FE_UPWARD、FTZ、DAZ 下均识别违规。
- 失败路径：`-ffast-math` 与 `CXXFLAGS=-ffp-contract=fast` configure FATAL_ERROR；不存在的 Python 解释器 configure 失败；非 Git 目录 build 失败；注入 `SystemExit` 与错误 smoke 期望时 CTest 返回 8。
- 版本刷新：`0.1.7` → 改动后只 build 得 `0.1.7+dirty` 且仅重编 `version.cpp` → 撤销后恢复。
- 未演练：非 GCC 与非 Linux x86_64 的拒绝分支。

### 评审处置

- 2026-10-10：用户裁剪范围（见上）；选定 conda/pip 工具与构建时 dirty；接受 D0005，转 active。
- 开发中发现：`-O0` 下运行时计算的期望值同样受 FTZ 影响而掩盖反例，改为十六进制字面量。
- 2026-10-11：用户评审通过并授权合入。

### 提交

- `afedb16` 实现；合并提交同步结项、STATUS 与 VERSION 0.1.8。
- 压缩前全文见提交 `b60526d`（依 [D0011](decisions/d0011-doc-single-source.md) 压缩）。

## 收尾

- 验收或停止依据：2026-10-11 用户评审通过并授权合入；验收逐条见上方验收证据。`cpu-debug`、`cpu-release` 与全新克隆的 CTest 均 8/8 通过，构建零告警；数值前提正反例、矛盾选项与缺失前提的失败、失败传播及版本构建时刷新演练均成立，仓库检查与合入前分支审计通过。实现提交 `afedb16`，本合并提交同步结项与 STATUS，VERSION 为 0.1.8。非 GCC 编译器与非 Linux x86_64 的拒绝分支未演练，支持声明不覆盖它们；不证明生产 quant、CUDA 构建或部署资质。
- 知识处置：updated；构建契约、命令、支持环境、target 契约、数值前提、版本刷新规则与测试清单归档到 [docs/BUILD.md](../docs/BUILD.md)，README 增加路由；长期选择理由在 [D0005](decisions/d0005-cpu-build-toolchain.md)，过程证据保留在本项。
- 剩余义务：无；followups 为空。bench 运行时比对 dirty 属于未来 bench 工作流，生产 src/quant、Python 调用 C++ 的 parity 连接、sanitizer 配置与版本状态组合矩阵属于本项明确排除的未来工作，不视为移交义务，也不宣称已实现。AGENTS.md「构建：待填」的填充属宪法修订，须用户另行授权。
