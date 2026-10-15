---
id: P0024
title: 建立单 token 主干 router 的生产 CPU scalar 与专家接口闭环
status: active
created: 2026-10-15
updated: 2026-10-15
knowledge: pending
docs: [docs/ROUTER.md, docs/MODEL.md, docs/BUILD.md]
bench: []
decisions: [D0011, D0018, D0019]
followups: []
---

## 意图

接续 [P0023](p0023-router-oracle.md)，将已验收的主干路由参照落为生产 C++ scalar，使 token 激活经生产 router 生成的专家编号与 FP32 权重能直接进入 [P0022](p0022-expert-scalar.md) 生产专家。成功边界为生产路由及所选专家调用，不扩为完整 MoE。既有数值契约见 [ROUTER](../docs/ROUTER.md)，模型依据与验证边界见 [MODEL §3.5/§4](../docs/MODEL.md#35-路由srccore)。

## 验收

- [x] **开工决策**：[D0019](decisions/d0019-router-scalar.md) 经用户接受，本项 active；非线性算法及资源上限、认证失败语义、结果与工作区所有权、容量/别名/预检、失败原子性、环境保证和构建接线明确，不改变 D0018 的数值阶段与确定性选择。
- [x] **生产入口**：公共头与实现归 `include/df/core/`、`src/core/`，普通 C++ 消费者只依赖公共头与生产 target 即可从 BF16 激活、Gate 矩阵及 FP32 bias 得到配对编号与 FP32 权重；文本/图像 bias 及回退符合 ROUTER。固定主干算法参数，保留合成 N/K；普通与私有观测入口共用计算核心，不依赖 Python、测试执行器或 Torch/TileLang 运行。
- [x] **逐阶段 parity**：生产普通/观测结果直接对照 `ref/router.py::route/trace_route`，核对投影至最终权重的全部约定阶段及顺序；生产后续阶段消费生产前阶段。覆盖抵消与归约次序、次正规与溢出拒绝、softplus 阈值/下溢和 sqrt 双舍入、top-k 截止/平局/零符号、两种 bias、防零及极小权重；定向边界复用 P0023 已验收的独立期望，不以宿主 libm、Torch 或生产副本作真值。
- [x] **存储与失败**：验证计数/字节溢出、形状、容量、别名、非有限输入及未选用 bias 的完整预检；依开工决策区分参数拒绝、运行中非有限与认证失败，核对输出原子性、scratch 可变范围、只读输入和写范围外字节不变。合法对象/span、公共消费者与适用 sanitizer 提供存储证据，认证预算耗尽不得伪装为成功结果。
- [x] **错误检出与环境**：针对归约/舍入、非线性阶段、bias 选择、平局、未加 bias 分数取权及归一化的临时生产变体，以完成调用后的编号或阶段数值 mismatch 检出，保留同例正确基线；异常、拒绝与协议故障不计数值证据。普通、观测、成功及失败路径覆盖 BUILD 的 MXCSR 模式和初始状态，满足开工决策的环境保证；不放宽既有生产 quant 契约。
- [x] **生产专家衔接**：合成不同 FP4 三矩阵，实际消费生产 router 返回的编号和权重，调用既有专家普通/观测入口；对照 P0023 参照组合，错误编号映射或权重接线有完成调用后的数值反例。不新增专家调度或输出聚合。
- [ ] **增量复现**：新增无权重测试、公共消费者及受实际改动影响的回归，在 ENV-A 的 Debug/Release/适用 ASan 通过；已提交干净检出、源码目录外复现，无 skip、构建告警或非预期报告。继承既有组件证据，不重跑旧全量测试、SiLU 全域证明或真实权重验收；实现提交携带适用的 `Oracle:`，严格检查、范围审计和修改页引用检查通过，不设性能门槛。
- [ ] **知识与人工收尾**：按 D0011 将生产契约归 ROUTER、操作入口归 BUILD、覆盖边界归 MODEL §4，新增 D 保留实现理由；用户审议实现、证据及知识处置通过后，授权精简、结项与合入，同步 STATUS。

## 记录

### 范围与依据

- 前置为 P0023 的路由参照与 P0022 的生产完整专家；官方来源为只读镜像 `model.py:792–827,841–851,889–904`，模型参数只引用 MODEL，环境限 [BUILD §1](../docs/BUILD.md#1-支持环境) ENV-A。
- 生产 router 归 core；复用既有整数 FP32 原语所需的最小内部边界调整随本项处理，不将 router 放入 quant，不展开通用数学库或全仓重构。FP32 输入 softplus 的算法、依赖与资源/失败取舍见 D0019，不因实现不便改写 oracle。
- 范围外：完整 MoE、多专家聚合、共享 FP8 专家、真实 Gate 读取工具、真实模型激活/质量、batch/DSpark 配置、attention/mHC/Engram、SIMD/CUDA、pack/生产加载器、缓存与 serve/deploy；不承接 P0005、CI 或 bench 工作流，不宣称官方 GPU 等价或性能提升。
- 知识分工沿用 D0011；生产取舍归 D0019，契约归 ROUTER，操作归 BUILD，覆盖边界归 MODEL §4，本项只留过程与最终证据。用户已审议改动并授权提交合入；先补齐已提交干净检出与范围审计，再结项，不以未提交快照代替。

### 最终证据

- ENV-A；分支 `p0024-router-scalar`，基于 `1f07d1e` 的未提交工作树，VERSION 未改。配置并构建 `cpu-debug`、`cpu-release` 的 `df_core_exec/df_router_consumer/df_quant_exec`，`cpu-asan-core` 只构建前两者；最终构建无告警。
- `ctest --preset <preset> -R '^core\.(router\.(consumer|parity|contract|harness|combination)|compile_flags)$' --output-on-failure --no-tests=error -j 2`：Debug/Release 各 6/6，core ASan/UBSan 5/5，无 skip 或非预期报告。含 38 个普通/观测定向请求、除法/sqrt 边界、softplus 精度容量与区间、八模式 MXCSR × 两种状态位初值、参数/运行中拒绝、trace 原子性及合法尾部共享；ASan 故意读毒负控按预期非零并报告。
- 生产路由编号/权重驱动专家普通/观测 24 个请求，逐阶段对照既有上一级；错编号/权重完成后数值不同。`router_sensitivity.py` 八个临时生产变体全部以普通及观测的阶段/编号差异检出，异常和协议失败不计证据；同例基线、恢复和工作树 hash 保护通过。
- `/tmp` 下绝对路径 Release parity 4/4；受影响的 `bb.compile_flags` 在 Debug/Release 各 1/1。原 `cpu-asan` 仅配置枚举，仍 quant-only，未执行旧验收；既有 expert consumer 本地未构建，不据枚举宣称可运行。既有 oracle、SiLU 全域和真实 checkpoint 均未重跑。
- 两路审阅发现并修复私有预算预检、trace 独立观测、共享尾部与 byte 探针证据缺口，定点复核无阻塞；最终三配置及八变体在修正后通过。日志在 `scratch/p0024-router-scalar/validation/`；已提交干净检出与提交范围审计尚待获准提交，不以工作树结果替代。

### 评审处置

- 2026-10-15：用户同意单 token 主干 router 生产 scalar 的范围并授权按流程起草，登记 draft。
- 2026-10-15：用户批准含 D0019 的开工方案，授权实现、精简知识归档与增量验证；登记 active，不提交 Git，最终人工验收另行进行。
- 2026-10-15：实现及增量验证完成，两路独立审阅与定点复核无阻塞。
- 2026-10-15：用户审查同意改动，授权按 D0011 精简、提交及合入；补齐已提交干净检出与范围审计后据证据记录结项。

### 提交

尚未提交。

## 收尾

- 验收或停止依据：实现与工作树增量证据到位；已提交干净检出、提交范围审计及最终人工审议待完成，本项保持 active。
- 知识处置：pending；ROUTER §6、MODEL §4、BUILD 与 D0019 已补齐归档，知识充分性待人工审议，不据内容写入自行结项。
- 剩余义务：待结项审议；当前未登记移交义务，followups 为空。
