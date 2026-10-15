---
id: P0023
title: 建立单 token MoE 路由的 CPU 数值参照与独立验证
status: completed
created: 2026-10-15
updated: 2026-10-15
knowledge: updated
docs: [docs/ROUTER.md, docs/MODEL.md, docs/BUILD.md]
bench: []
decisions: [D0011, D0015, D0018]
followups: []
---

## 意图

接续 [P0022](p0022-expert-scalar.md)，补齐完整专家上游的选择与权重生成，为后续生产 router 建立独立验证的上一级。单 token Gate 契约归 [ROUTER](../docs/ROUTER.md)，模型依据与覆盖边界归 [MODEL §3.5/§4](../docs/MODEL.md#35-路由srccore)，不扩为完整 MoE。

## 验收

- [x] **开工决策**：[D0018](decisions/d0018-router-reference.md) 经用户接受，数值阶段、确定性选择、拒绝与环境边界明确。
- [x] **完整 Gate 参照**：`ref/router.py` 满足 ROUTER §1～§4，普通与 trace 共用核心，输入不变，输出专家编号与 FP32 权重编码。
- [x] **独立阶段期望**：投影、非线性、选择与权重逐阶段对照独立精确算术/认证区间；期望后续只消费期望前阶段，不调用被测数值 helper 或以 Torch/libm 作真值。
- [x] **边界与失败**：覆盖投影抵消、累加顺序、溢出与次正规，非线性与双舍入边界，top-k 截止/平局，两种 bias、防零及极小权重；验证完整预检、快照、输入不变、失败传播和约定环境保证。
- [x] **错误检出**：九种显式错误路由模型有同例正确基线，以编号或阶段 mismatch 检出；异常、协议故障和参数拒绝不作数值证据。
- [x] **专家衔接**：合成不同三矩阵验证实际编号/权重进入既有专家普通与 trace 入口；错矩阵/错权重完成后有数值反例，不新增调度或聚合入口。
- [x] **增量复现**：ENV-A 的已提交干净检出、源码目录外命令及 Debug/Release 新 CTest 通过，无 skip、告警或非预期报告；严格检查、范围审计与修改页引用通过，继承既有组件证据，不设性能门槛。
- [x] **知识与人工收尾**：按 D0011 归档契约、理由、模型边界与操作入口；用户审议实现、证据及知识处置通过，授权精简、结项和合入，同步 STATUS。

## 记录

### 范围与依据

- 前置为 P0022 与 [P0020](p0020-expert-oracle.md)；来源为只读镜像 `model.py:792–827,841–851,889–904`，参数只引用 MODEL。环境限 [BUILD §1](../docs/BUILD.md#1-支持环境) ENV-A，CPU 选择见 D0018，范围边界只见 MODEL §4。
- 知识分工：ROUTER 唯一契约正文，MODEL 模型依据与验证边界，BUILD 操作入口，D0018 选择理由；本项只留过程及证据，不重写历史 D 或既有 oracle。

### 最终证据

- ENV-A；最终工件 `17036aad01434142fc9ddcf9c38a2827495ae2fb`，VERSION 0.1.20。`git clone --no-local --single-branch --branch p0023-router-oracle` 至 `/tmp/p0023-clean-ba5xnmu5/repo`，起止 dirty=false；命令均从 `/tmp` 执行。
- 配置 `cpu-debug`/`cpu-release` 并构建 `df_quant_exec`；`ctest --test-dir <checkout>/build/<preset> -R '^core\.router(\.expert)?$' --verbose --output-on-failure --no-tests=error -j 2` 各 2/2，内部路由 24/24、接口 5/5，无 skip、构建告警或非预期报告。绝对路径单跑路由及指定双执行器的接口命令通过；ASan 仅配置枚举，新增路由注册为零，未跑旧验收。
- 独立 Fraction/格点、Taylor 指数与对数级数检查全部 trace；九种错误路由模型以编号或阶段差异检出。首次乘法改写探针覆盖普通/trace × 文本/图像；容量匹配的未用坏 bias 先于投影溢出拒绝。16 种控制模式 × 2 种 sticky 状态覆盖成功、参数及运行中拒绝；Decimal 完整 context/flags 不变，认证预算耗尽与期望独立性负控通过。
- 合成接口执行文本/图像各选中专家的 24 个生产 ordinary/trace 请求，核对实际编号、权重及专家阶段；错矩阵/错权重显式模型的最终 BF16 不同。缺失、不可执行或假执行器失败，不宣称生产变体门禁。
- 干净检出严格检查及 `--audit --range d48d67c..HEAD --strict` 均 E=0 W=0 I=0，主树 diff 与修改页链接/锚点通过；数值、验证、文档及最后修复独立复核无阻塞。完整命令、日志与 summary 在 `scratch/p0023-router-oracle/closure/`；本轮收尾继承证据，不重跑旧 oracle、全域 parity 或真实权重验收。

### 评审处置

- 2026-10-15：用户确认范围并授权起草，登记 draft。
- 2026-10-15：用户批准含 D0018 的开工方案、必要归档、增量验证与分支提交；D0018 accepted，本项 active。
- 2026-10-15：独立审阅加固预检、快照与环境；低精度故障注入导致病态大整数，终止该轮并改为预算耗尽，最终干净复现通过，oracle 数值未变。
- 2026-10-15：用户人工审议改动通过，授权按 D0011 精简、结项、提交与合入。

### 提交

- `540a6c6`：接受 D0018，登记 active 并同步 STATUS。
- `b52078a`：参照、独立验证、专家衔接与知识归档；携带 `Oracle: tests/core/test_router.py`。
- `17036aa`：修复认证失败负控，oracle 数值未变；携带同一 `Oracle:`。
- `21bad3b`：回填最终证据并等待人工审议；本次压缩前全文指针。
- 本次结项提交记录人工验收并同步 STATUS；合并及 patch 递增由 `Merge P0023` 提交记录。

## 收尾

- 验收或停止依据：八项验收成立；用户于 2026-10-15 审议通过并授权依 D0011 精简、结项及合入。技术证据见上，不以合并代替验收，覆盖边界以 MODEL §4 为准。
- 知识处置：updated；[ROUTER](../docs/ROUTER.md) 归档契约，MODEL §3.5/§4 归档模型依据与验证边界，BUILD §7 归档复现入口，D0018 保留理由；用户已审议知识处置。
- 剩余义务：无本项阻塞或移交义务，followups 为空；生产 router、完整 MoE 等后续工作另立 P，不重开本项或既有专家工作，P0005 阻塞不解除。
