---
id: P0022
title: 建立单 token FP4 完整专家的生产 CPU scalar 与真实三矩阵闭环
status: completed
created: 2026-10-15
updated: 2026-10-15
knowledge: updated
docs: [docs/EXPERT.md, docs/BUILD.md, docs/MODEL.md]
bench: []
decisions: [D0011, D0014, D0015, D0016, D0017]
followups: []
---

## 意图

接续 [P0021](p0021-swiglu-bridge-scalar.md)，将已有生产组件组合为单 token、单个已选 FP4 专家的完整入口，逐阶段对照 [P0020](p0020-expert-oracle.md) oracle，并完成固定真实专家三矩阵闭环。数值及生产契约归 [EXPERT](../docs/EXPERT.md)，验证边界归 [MODEL §4](../docs/MODEL.md#4-精确性链条)。

## 验收

- [x] **开工决策**：[D0017](decisions/d0017-expert-scalar.md) 经用户接受，完整 API、独立阶段工作缓冲、共享量化和分层拒绝保证明确。
- [x] **生产入口**：普通消费者仅依赖公共头与 `df_quant`，按 EXPERT §2/§6 完成完整 BF16 专家；复用既有数值核心，无动态分配，两种权重入口均覆盖。
- [x] **逐阶段 parity**：直接对照 `expert`/`trace_expert`，核对两次量化、双投影、bridge 及完整 W2 输出；生产后续阶段消费生产结果，普通与观测共用核心。覆盖单/多组、非方阵、矩阵与行 scale、首末输出及两种权重入口，不以最终 BF16 偶合替代阶段验证。
- [x] **存储与失败**：按 EXPERT §6 验证元数据前置、溢出、容量、别名及三矩阵内容拒绝，含 W2 末行坏 scale；区分参数错误全存储不变和 hidden 非有限拒绝的完整 out 不变，覆盖钳位恢复及最终合法非有限输出。合法对象/span、sanitizer 与共享写区间并集检查输入、输出尾部和范围外字节。
- [x] **错误检出与环境**：分支、权重/scale 行、第二量化、route 位置及 BF16 落点的生产变体以完成调用后的阶段数值 mismatch 检出，具备同用例正确基线；异常或协议故障不计数值证据。普通、观测、接受与拒绝路径覆盖 BUILD 的 MXCSR 模式及初始状态，完整 MXCSR 不变。
- [x] **真实三矩阵**：固定专家完整 W1/W3/W2 经生产入口逐阶段对照 P0020；独立核对尺寸、布局、读取区间及指纹，记录环境、快照、源码/执行器身份和比较结果。离线命令缺权重或失败不 skip，不引入生产加载器。
- [x] **增量复现**：新增无权重测试、公共消费者及受实际改动影响的回归在 Debug/Release/适用 ASan 通过；已提交干净检出、源码目录外复现，无告警、skip 或非预期报告。既有组件证据继承，不重做 SiLU 全域证明；有效 `Oracle:`、严格检查、范围审计和修改页引用检查通过，不设置性能门槛。
- [x] **知识与人工收尾**：按 D0011 归档契约、操作和覆盖边界；用户审议实现、证据及知识处置通过，授权精简、结项与合入，同步 STATUS。

## 记录

### 范围与依据

- 前置为 [P0017](p0017-act-quant-scalar.md)、[P0019](p0019-expert-linear-scalar.md)、P0020/P0021；环境限 [BUILD §1](../docs/BUILD.md#1-支持环境) ENV-A，生产取舍见 D0017，既有 D 原文保留。
- 固定真实验收为 MODEL 的 [ckpt] 快照中 `layers.0.ffn.experts.0` 完整 w1/w3/w2，人工输入采用既有方法，不裁剪 N/K、不扫描其他专家。范围外项见 MODEL §4；本项不承接 P0005、CI 或 bench 工作流。
- 知识分工：EXPERT 唯一契约正文，BUILD 操作入口，MODEL 覆盖边界，D0017 选择理由；本项只留过程与证据。

### 最终证据

- ENV-A；最终实现 `dde35d99017aa64b9b560514cae26c5a05e053f2`，VERSION 0.1.19。生产/执行器/工具工件 `b0073f9b2861331558218e150005d651cb15e4f0` 经 `git clone --no-local --single-branch --branch p0022-expert-scalar` 检出至 `/tmp/p0022-clean-EuevPg/repo`，dirty=false；后继提交仅修敏感性脚本导入路径，其余相关文件 diff 确认未变，沿用下述证据。
- 构建三个 CPU preset，从源码目录外通过 `ctest --test-dir <checkout>/build/<preset> -R <增量选择> --output-on-failure --no-tests=error -j 4`：Debug 20/20、Release 20/20、ASan 17/17。选择范围为新 `quant.expert.*`、三个 checkpoint 工具合成套件、既有执行器 contract/harness 及编译选项检查；无告警、skip 或非预期报告，sanitizer 环境按 BUILD §3。未重跑旧 oracle、旧全域 parity 或旧真实权重验收。
- 新专家 parity 16 例、contract 44 例各覆盖 16 种 MXCSR 状态，另含计数/字节探针、消费者及四项 harness；新工具合成 15/15。runtime 固定 fixture 核验实际 hidden 全 H 为 BF16 NaN、阶段前缀已写、第二量化缓冲未写与完整 out 不变，提前拒绝负控有效；不宣称独立调用计数证明。
- 按 BUILD §7 的完整专家命令传入 Debug/Release 两个 `--exe`，快照 `2cba9e42aa026125f3ed06c6d98c1db82f7ca027`；四个输入/权重组合各计算一次 oracle trace。完整 w1/w3 为 2304×5120、w2 为 5120×2304（来源 MODEL §2），407168 项比较通过：ordinary 157504、trace 249664，有限/零/Inf 逐位、NaN 按类。独立头部解析及 seek/read 复核 19456 段、18800640 字节、三矩阵指纹和双执行器阶段结果及 SHA256。
- 最终实现另干净检出至 `/tmp/p0022-fixed-Z8JXBR/repo`，从 `/tmp` 运行敏感性脚本绝对路径：6/6 生产变体完成调用后以阶段数值 mismatch 检出，每项恢复基线、最终 pristine 和受保护文件哈希通过。该检出的 `check_repo.py --repo <checkout> --range 9300229..HEAD --audit --strict` E=0 W=0 I=0，范围 diff 与修改页链接/锚点通过；两个检出运行后仍干净。
- 完整命令、日志、身份、真实报告和独立读取复核在 `scratch/p0022-final/`（临时工件）。未改 oracle、只读镜像、冻结数据、AGENTS 或实现提交的 VERSION，不将 CPU 证据扩作性能、官方 GPU、真实模型激活或质量论断。

### 评审处置

- 2026-10-15：用户确认范围并授权起草，登记 draft。
- 2026-10-15：用户批准含 D0017 的开工方案、实现、必要文档、增量验证及分支提交；D0017 accepted，本项 active。
- 2026-10-15：独立评审发现别名探针、oracle 重算、超宽 NaN、读取自证、混型对象、共享间隙及提前拒绝假阳性；修复为合法构造、预计算接口与负控、位宽前置、独立区间、共用字节检查、写区间并集及实际阶段诊断，限定复核无阻塞。
- 2026-10-15：用户审议改动通过，授权依 D0011 精简归档并继续提交合入；据已有验收证据完成结项。

### 提交

- `7b5823a`：接受 D0017，登记 active 并同步 STATUS。
- `b0073f9`：生产专家、验收与真实工具、契约归档；携带 `Oracle: tests/quant/test_expert_parity.py`。
- `dde35d9`：修复目录外导入，生产/执行器/工具未变；携带同一 `Oracle:`。
- `3929cf0`：回填最终证据并等待人工审议；为本次压缩前全文指针。
- 本次结项提交依 D0011 精简、记录人工验收并同步 STATUS；合并及 patch 递增由 `Merge P0022` 提交记录。

## 收尾

- 验收或停止依据：八项验收成立；用户于 2026-10-15 审议通过并授权精简、提交与合入，最终证据见上。覆盖边界以 MODEL §4 为准。
- 知识处置：updated；[EXPERT §6](../docs/EXPERT.md#6-生产完整专家) 归档生产契约，BUILD §7 归档操作入口，MODEL §4 归档覆盖边界，D0017 保留选择理由；用户已审议知识处置。
- 剩余义务：无本项阻塞或移交义务，followups 为空；后续 MoE、快路径和生产加载工作另立 P，不重开本项，P0005 阻塞不解除。
