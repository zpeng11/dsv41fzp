---
id: P0024
title: 建立单 token 主干 router 的生产 CPU scalar 与专家接口闭环
status: completed
created: 2026-10-15
updated: 2026-10-15
knowledge: updated
docs: [docs/ROUTER.md, docs/MODEL.md, docs/BUILD.md]
bench: []
decisions: [D0011, D0018, D0019]
followups: []
---

## 意图

接续 [P0023](p0023-router-oracle.md)，将主干路由参照落为生产 C++ scalar，并将实际编号与 FP32 权重交给 [P0022](p0022-expert-scalar.md) 生产专家；不扩完整 MoE。契约归 [ROUTER](../docs/ROUTER.md)，模型依据及验证边界归 [MODEL §3.5/§4](../docs/MODEL.md#35-路由srccore)。

## 验收

- [x] **开工决策**：[D0019](decisions/d0019-router-scalar.md) 经用户接受，生产接口、算法、认证资源及失败/环境保证明确，继承 D0018。
- [x] **生产入口**：core 公共入口按 ROUTER §6 返回配对编号/权重，普通与私有观测共用核心，不依赖 Python 或测试执行器运行。
- [x] **逐阶段 parity**：普通/观测直接对照 `ref/router.py`，另核对既有独立边界期望；覆盖投影、非线性、选择与权重阶段，不以最终偶合代替。
- [x] **存储与失败**：元数据及内容预检、容量/别名、只读输入与写范围、参数原子性、运行中拒绝及认证耗尽均有证据，包含私有 trace 不变性及合法尾部共享。
- [x] **错误检出与环境**：八种临时生产变体以完成调用后的阶段/编号差异检出，普通/观测均有正确基线；成功及失败覆盖 BUILD 的 MXCSR 控制模式与状态初值。
- [x] **生产专家衔接**：实际编号/权重进入既有专家普通/观测，逐阶段核对上一级，错矩阵/权重有完成调用后的数值反例，不新增调度或聚合。
- [x] **增量复现**：ENV-A 已提交干净检出、源码目录外的三配置新增测试通过，无 skip、构建告警或非预期报告；严格检查、提交范围审计及修改页引用通过，继承旧组件证据，不设性能门槛。
- [x] **知识与人工收尾**：用户审查同意改动及知识处置，授权依 D0011 精简、提交、结项与合入，同步 STATUS。

## 记录

### 范围与依据

- 前置为 P0023/P0022；官方来源 `model.py:792–827,841–851,889–904`，模型参数只引用 MODEL，环境限 [BUILD §1](../docs/BUILD.md#1-支持环境) ENV-A，生产取舍见 D0019。
- 知识分工：ROUTER §6 唯一生产契约，BUILD 操作入口，MODEL §4 覆盖边界，D0019 选择理由。本项不承接 P0005、CI 或 bench 工作流；具体覆盖及范围外项只见 MODEL §4。

### 最终证据

- 最终实现 `45bfaa3d3b425ad42a2ae30a12ce54c94e5aa12c`，VERSION 0.1.21。`git clone --no-local --single-branch --branch p0024-router-scalar` 至 `/tmp/p0024-clean-gk8bh6mj/repo`，起止 dirty=false；全部配置、构建与测试命令从 `/tmp` 执行。
- 构建 Debug/Release 的 `df_core_exec/df_router_consumer/df_quant_exec`，core ASan/UBSan 只构建前两者。`ctest --test-dir <checkout>/build/<preset> -R '^core\.(router\.(consumer|parity|contract|harness|combination)|compile_flags)$' --output-on-failure --no-tests=error -j 2` 分别 6/6、6/6、5/5，无 skip、构建告警或非预期报告。认证容量、除法/sqrt、38 个普通/观测请求、存储/拒绝及八模式 MXCSR × 两种状态初值通过，ASan 故意读毒负控有效。
- 生产路由驱动专家普通/观测 24 个请求，阶段对照既有上一级；错编号/权重完成后数值不同。`router_sensitivity.py` 八变体全部由普通/观测数值差异检出；异常及协议失败不计证据，同例基线、恢复和工作树 hash 保护通过，继承本实现提交前的同源码证据。
- 干净检出 `check_repo.py --audit --range 1f07d1e..HEAD --strict` 为 E=0 W=0 I=0；修改页链接/锚点及 diff 检查通过。受影响编译选项定向检查通过，原 quant-only ASan 仅枚举，未执行旧 oracle、全量 parity、SiLU 全域证明或真实 checkpoint 验收。
- 数学、存储/测试及知识分工独立审阅的必要修正已完成，定点复核无阻塞。最终日志及身份在 `scratch/p0024-router-scalar/validation/`；压缩前完整记录见 `45bfaa3`。

### 评审处置

- 2026-10-15：用户确认范围、批准含 D0019 的开工方案，授权实现、必要归档与增量验证。
- 2026-10-15：用户审查同意改动，授权按 D0011 减负提交合入；已提交干净复现及范围审计通过，据该授权记录结项。

### 提交

- `45bfaa3`：生产路由、独立验证与单一来源归档；携带 `Oracle: tests/core/test_router_parity.py`。
- 本次结项提交精简记录并同步 STATUS；版本递增由 `Merge P0024` 合并提交记录。

## 收尾

- 验收或停止依据：八项验收成立；用户已审查同意并授权提交合入，已提交干净检出补证通过，不以合并代替验收。覆盖边界以 MODEL §4 为准。
- 知识处置：updated；ROUTER §6、MODEL §4、BUILD 与 D0019 分别归档契约、覆盖边界、操作及理由，用户已审议；不重复保留规范正文。
- 剩余义务：无本项阻塞或移交义务，followups 为空。完整 MoE 等后续另立 P，不重开本项，P0005 阻塞不解除。
