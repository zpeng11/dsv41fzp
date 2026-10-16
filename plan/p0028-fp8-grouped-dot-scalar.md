---
id: P0028
title: 建立 FP8×FP8 块缩放单输出点积的生产 CPU scalar
status: completed
created: 2026-10-16
updated: 2026-10-16
knowledge: updated
docs: [docs/COMPUTE.md, docs/MODEL.md, docs/BUILD.md]
bench: []
decisions: [D0009, D0011, D0021]
followups: []
---

## 意图

接续 [P0027](p0027-fp8-grouped-dot-oracle.md)，补齐其参照对应的生产 C++ scalar，闭合 FP8×FP8 块缩放单输出点积的验证链，为后续 FP8 线性层与共享专家提供可复用的计算构件。成功交付是对齐既有 CPU 契约的生产入口、逐阶段 parity 及存储和环境保证，不以此宣称完整线性层、共享专家、完整 MoE 或性能提升。

## 验收

- [x] **契约承接与生产接口**：[COMPUTE §8](../docs/COMPUTE.md#8-fp8-块缩放单输出点积) 为唯一数值来源，生产接口、失败与观测见 §8.5；继承 D0021 与 D0009，无独立新数值选择。
- [x] **单输出生产入口**：`df_quant` 公共 C++ 调用消费一维编码视图，交付调用方持有的 FP32/BF16 原始位输出；公共头独立消费、成功无分配，不依赖 Python 或测试侧计算。
- [x] **数值路径与观测**：复用既有整数算术构件，普通与私有 trace 共用执行核心；两种输出逐阶段对照 `ref/fp8_compute.py`，有限值、零与 Inf 按位比较，NaN 按分类比较。
- [x] **数值覆盖与最小组合**：复用 P0027 独立期望覆盖合法码点与数值边界；二维 scale 仅测试侧选行，既有生产激活量化实际输出直接供新入口消费，不扩矩阵 API。
- [x] **存储与失败保证**：两种输出的长度、容量、编码与偏移重叠拒绝、相邻接受及有效写范围通过；全部校验先于写入，拒绝时输入、正式输出与观测不变；trace 尺寸与别名经真实入口及 ASan 定向验证。
- [x] **环境保证**：整数数值路径不读写浮点环境；ENV-A 八模式 MXCSR 与状态位清空/预置下，普通及观测两输出的成功与拒绝保持完整环境，阶段和正式结果一致。
- [x] **错误检出与失败传播**：七类生产变体及二维错映射均有独立正确基线和完成调用后的数值差异；协议、执行器、结果与 trace 故障及缺依赖、skip 非零退出，不混同库拒绝与数值检出。
- [x] **增量复现与治理**：已提交干净检出从源码目录外完成 Debug/Release 与既有 quant ASan+UBSan 新增测试及编译选项检查，无 skip、告警或非预期报告；严格范围审计及引用检查通过，不重跑旧全量或扩大 sanitizer scope，不设性能门槛。
- [x] **知识与人工收尾**：用户人工审议实现、证据及知识处置通过；干净复现补证成立，按 D0011 单一来源归档并同步 STATUS，不以提交或合并替代验收。

## 记录

### 范围与依据

- 前置 [P0027](p0027-fp8-grouped-dot-oracle.md) 已验收；生产方式参考 [P0013](p0013-expert-grouped-dot-scalar.md)/D0009，量化组合复用 [P0017](p0017-act-quant-scalar.md)，独立数值选择见 [D0021](decisions/d0021-fp8-grouped-dot-reference.md)。来源与数值正文只见 COMPUTE §8，生产保证只见 §8.5，覆盖边界只见 [MODEL §4](../docs/MODEL.md#4-精确性链条)，环境限 [BUILD §1](../docs/BUILD.md#1-支持环境) ENV-A。
- 本项止于单输出生产 scalar，不承接 P0005、CI 或 bench 工作流。共享专家参照可独立基于 P0027 建立，本项先行不构成其硬依赖；旧 oracle、冻结工件及量化格式版本不变。

### 最终证据

- 实现 `70c034b6604eb5248714e66fc1e7d093581834a1`，VERSION 0.1.25。`git clone --no-local --single-branch --branch p0028-fp8-grouped-dot-scalar` 至 `/tmp/p0028-clean-hr55a6yt/repo`，起止 dirty=false，生产、测试、接线与文档指纹与人工审议候选一致。全部命令从 `/tmp` 执行，三份空外置 build 仅构建 `df_quant_exec df_fp8_dot_consumer`。
- Debug/Release/quant ASan+UBSan 的新增五项 `quant.fp8_dot` CTest 及对应编译选项检查各 6/6。parity 各 124 请求逐阶段覆盖全部 64,516 对合法码点；contract 各 349 请求，含二维选行与实际生产量化组合；environment 各 20 请求 × 16 控制/状态组合；harness 各 5 项；公共 consumer 两输出成功无分配。无 skip、构建告警或非预期 sanitizer 报告，未重跑旧全量。
- 七个临时生产变体及测试侧二维错映射均完成普通/观测两输出调用并检出正式结果差异。FP32 正确→变体：错权重解码 `41000000→00000000`；scale 移入乘积 `00000000→7fc00000`；交换/合并 scale 各 `7f800000→40000000`；块内倒序 `00000000→3a800000`；跨块倒序 `00000000→3f800000`；提前 BF16 `3b800000→00000000`；二维错映射 `44400000→43400000`。正确生产源指纹未变，阶段与基线证据见同目录 `sensitivity/result.json`，不冒充生产矩阵验证。
- 干净检出 `check_repo.py --repo <checkout> --audit --range 8299078..HEAD --strict` 为 E=0 W=0 I=0；168 个本地引用 broken=0，差异格式检查通过。最终命令、日志及身份归 `scratch/p0028-fp8-grouped-dot-scalar/closure/`；压缩前全文见 `70c034b`，不重复保留候选树证据。

### 评审处置

- 2026-10-16：用户确认范围并授权起草，随后审议通过 P，授权按仓库流程与 D0011 实现及必要归档，登记 active。
- 2026-10-16：生产静态审阅无阻塞；协议负控伪检出、BF16 trace NaN 分类、拒绝输入快照及 consumer 分配/尾部证据缺口修正，定点复核关闭。
- 2026-10-16：用户人工审议通过改动与知识处置，授权提交合入；已提交干净检出三配置增量复现及严格范围审计补证通过，据此记录结项。

### 提交

- `70c034b`：生产实现、增量验证及单一来源归档，携带 `Oracle: tests/quant/test_fp8_compute_parity.py`。
- 本次结项提交精简最终证据并同步 STATUS；版本递增由 `Merge P0028` 合并提交记录。

## 收尾

- 验收或停止依据：九项验收成立，用户人工审议通过；已提交干净检出三配置增量复现与严格范围审计补证通过，不以合并替代验收，覆盖边界以 MODEL §4 为准。
- 知识处置：updated；COMPUTE §8.5、MODEL §4 与 BUILD §7 分别归档生产接口、覆盖及操作入口，数值正文与选择理由沿用 D0021/D0009；用户已审议，依 D0011 不重复规范正文。
- 剩余义务：无本项阻塞或移交义务，followups 为空；完整 FP8 线性层、共享专家与完整 MoE 等方向另立 P，P0005 阻塞不解除。
