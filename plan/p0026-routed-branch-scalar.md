---
id: P0026
title: 建立单 token 主干 routed 分支的生产 CPU scalar
status: completed
created: 2026-10-16
updated: 2026-10-16
knowledge: updated
docs: [docs/MOE.md, docs/MODEL.md, docs/BUILD.md]
bench: []
decisions: [D0011, D0020]
followups: []
---

## 意图

接续 [P0025](p0025-routed-branch-oracle.md)，把 [P0024](p0024-router-scalar.md) 的生产 router 与 [P0022](p0022-expert-scalar.md) 的完整 FP4 专家组合为单次生产调用，闭合单 token 主干 routed 分支。成功交付是对齐既有参照的 FP32 routed 部分和及组合层的存储、失败保证，不以此宣称完整 MoE 或性能提升。

## 验收

- [x] **生产接口与契约**：[MOE §5](../docs/MOE.md#5-生产-cpu-scalar) 明确调用方存储、完整专家表、预检/别名及失败/环境保证，继承 D0020 及用户已审议设计，不新增独立数值选择。
- [x] **生产入口**：`df_core` 单次公共调用复用生产 router、weighted expert 与整数 FP32 加法，交付 FP32 raw-bit 输出；普通与私有观测共用核心，无动态数值存储，不依赖 Python 或测试侧串接。
- [x] **真实组合 parity**：文本/图像及不同专家三矩阵的合成调用直接对照 `ref/moe.py` 和既有独立期望，核对实际编号/权重、执行次序、每步部分和及最终输出。
- [x] **聚合边界与错误检出**：复用 P0025 独立期望覆盖舍入、抵消、零、次正规及合法非有限结果；六类错误均有同例基线与完成调用后的数值差异，定向注入与真实组合证据分开记录。
- [x] **存储与失败**：包含未选专家的完整预检、容量/别名及读写范围通过；参数拒绝全存储不变，router 或后续专家运行失败不提交正式输出与观测。
- [x] **公共消费与环境**：公共头消费者可独立调用且成功无分配，普通/观测成功、参数拒绝及运行失败保持完整 MXCSR，输出与观测尾部不写。
- [x] **增量复现**：ENV-A 已提交干净检出、源码目录外 Debug/Release 与 core ASan+UBSan 新增测试通过，无 skip、告警或非预期报告；严格范围审计及引用检查通过，沿用旧组件证据，不设性能门槛。
- [x] **知识与人工收尾**：用户人工审议实现、证据及知识处置通过；干净复现补证成立，按 D0011 精简归档并同步 STATUS，不以提交或合并代替验收。

## 记录

### 范围与依据

- 前置为 P0025/P0024/P0022；数值契约只见 [MOE](../docs/MOE.md)，上游分别见 [ROUTER §6](../docs/ROUTER.md#6-生产-cpu-scalar)、[EXPERT §6](../docs/EXPERT.md#6-生产完整专家)。模型依据与覆盖边界只见 [MODEL §3.5/§4](../docs/MODEL.md#35-路由srccore)，环境限 [BUILD §1](../docs/BUILD.md#1-支持环境) ENV-A。
- MOE §5 唯一生产契约，MODEL §4 覆盖矩阵，BUILD §7 操作入口；选择继承 D0020 与已审议设计。本项不承接 P0005、CI 或 bench 工作流；旧 oracle 与冻结工件不变。

### 最终证据

- 最终实现 `ab3f34caa4237ef51ccdf61a0f2ad1a89fdd5b77`，VERSION 0.1.23。`git clone --no-local --single-branch --branch p0026-routed-branch-scalar` 至 `/tmp/p0026-clean-nstug_os/repo`，起止 dirty=false，源码指纹与审议版本一致；全部命令从 `/tmp` 执行，仅构建 `df_core_exec/df_router_consumer`。
- `ctest --test-dir <checkout>/build/<config> -R '^core\.(moe\.(consumer|parity|harness)|compile_flags)$' --verbose --output-on-failure --no-tests=error -j 2`：Debug/Release/core ASan+UBSan 各 4/4，Python parity 3/3、harness 4/4，无 skip、构建告警或非预期报告。三配置 `test_moe_scalar.py --suite environment` 各 1/1，覆盖八模式 MXCSR × 两状态初值的普通/观测成功、参数拒绝、路由运行溢出及后续专家失败。
- 参数全快照、未选专家后置矩阵元数据非法的读毒预检、跨组件别名、正式输出/观测提交边界通过；ASan 故意读毒负控退出 86。旧组件全量套件、真实权重与性能验收未执行。
- 六类临时生产变体完成调用后检出：错编号/权重配对/顺序、重复权重及提前 BF16 使用真实合成组合；非逐步求和在真实执行核心注入 BF16 序列，原序逐步基线为 `00000000`，原序 double 累加后一次 FP32 舍入为 `3f800000`。普通/观测同例基线、恢复与原树指纹检查通过；不冒充真实专家组合。
- 干净检出 `check_repo.py --repo <checkout> --audit --range b75595d..HEAD --strict` 为 E=0 W=0 I=0；diff 与引用检查通过。最终命令、日志及身份归 `scratch/p0026-routed-branch-scalar/closure/`，敏感性有效证据见同目录上级 `sensitivity-final.log`。压缩前记录见 `ab3f34c`。

### 评审处置

- 2026-10-16：用户确认范围并授权起草，随后审议通过设计，授权依 D0011 实现与必要归档，登记 active。
- 2026-10-16：生产及验证链独立审阅的必要修正已完成；协议拒绝、失败断言、未选矩阵预检及非逐步变体分类经定点复核关闭。
- 2026-10-16：用户人工审议通过实现、证据及知识处置，要求继续推进；已提交干净复现与严格范围审计补证通过，据此记录结项。

### 提交

- `ab3f34c`：生产组合、增量验证及单一来源归档，携带 `Oracle: tests/core/test_moe_scalar.py`。
- 本次结项提交精简记录、校正文档 scope 描述并同步 STATUS；版本递增由 `Merge P0026` 合并提交记录。

## 收尾

- 验收或停止依据：八项验收成立；用户已人工审议通过，已提交干净检出增量复现及严格范围审计补证通过，不以合并代替验收。覆盖边界以 MODEL §4 为准。
- 知识处置：updated；MOE §5、MODEL §4 与 BUILD §7 分别归档生产契约、覆盖边界及操作，选择依据沿用 D0020，用户已审议；不重复保留规范正文。
- 剩余义务：无本项阻塞或移交义务，followups 为空。完整 MoE 等后续另立 P，不重开本项，P0005 阻塞不解除。
