---
id: P0025
title: 建立单 token 主干 routed 分支的 CPU 参照与多专家聚合验证
status: completed
created: 2026-10-16
updated: 2026-10-16
knowledge: updated
docs: [docs/MOE.md, docs/MODEL.md, docs/BUILD.md, docs/ROUTER.md, docs/EXPERT.md]
bench: []
decisions: [D0011, D0018, D0019, D0020]
followups: []
---

## 意图

接续 [P0023](p0023-router-oracle.md)、[P0024](p0024-router-scalar.md) 与 [P0020](p0020-expert-oracle.md)/[P0022](p0022-expert-scalar.md)，把已验收的路由和完整 FP4 专家参照组合为单 token 主干 routed 分支，补齐多专家聚合的独立数值验证，为后续生产聚合建立上一级。交付保留共享支路接入所需精度的 routed 部分和，不以此宣称完整 MoE 已完成。

## 验收

- [x] **决策审议**：[D0020](decisions/d0020-routed-branch-reference.md) 经用户随候选实现审议接受，既有路由与专家契约不变。
- [x] **routed 分支参照**：`ref/moe.py` 普通与 trace 共用核心，按 [MOE §1～§4](../docs/MOE.md) 完成真实路由、所选专家调用及聚合，不以预制输出替代组合入口。
- [x] **独立聚合期望**：独立精确算术/格点判定逐步部分和，不调用被测聚合或其舍入 helper；上游期望复用既有独立参照。
- [x] **组合与边界**：合成不同三矩阵核对实际编号、权重、专家阶段和累加轨迹；顺序、抵消、舍入、零、次正规及非有限边界有定向证据。
- [x] **校验与环境**：完整预检、快照、输入不变、上游失败传播及约定环境保证通过，普通/trace 不交付部分结果。
- [x] **错误检出**：六类错误链均有同例正确基线和完成调用后的数值差异，异常或参数拒绝不计数值证据。
- [x] **增量复现**：ENV-A 已提交干净检出，源码目录外新增 Debug/Release CTest 通过，无 skip、告警或非预期报告；严格范围审计与引用检查通过，沿用旧组件证据，不设性能门槛。
- [x] **知识与人工收尾**：用户审议实现、证据及知识处置通过；按 D0011 归档并精简记录，干净复现补证成立后结项，同步 STATUS。

## 记录

### 范围与依据

- 前置、模型依据与覆盖边界分别见 [ROUTER](../docs/ROUTER.md)、[EXPERT](../docs/EXPERT.md)、[MODEL §4](../docs/MODEL.md#4-精确性链条)；官方来源 `model.py:841–851,889–904`，环境限 [BUILD §1](../docs/BUILD.md#1-支持环境) ENV-A。
- MOE 唯一契约正文，D0020 留选择理由，MODEL 留覆盖边界，BUILD 留操作入口。本项不承接生产聚合、完整 MoE、P0005、CI 或 bench 工作流；旧 oracle 与冻结工件不变。

### 最终证据

- 最终实现 `46c47ce37221e71d6148ac70d1d49aded4240e48`，VERSION 0.1.22。`git clone --no-local --single-branch --branch p0025-routed-branch-oracle` 至 `/tmp/p0025-clean-ev5djr2m/repo`，起止 dirty=false；配置、CTest 命令均从 `/tmp` 执行，不构建 C++。
- `ctest --test-dir <checkout>/build/<preset> -R '^core\.moe$' --verbose --output-on-failure --no-tests=error`：Debug/Release 各 1/1，内部各 17/17，无 skip、告警或非预期报告。两种 sanitizer scope 只配置枚举，新增注册为零；未执行旧组件测试、真实权重验收或性能测试。
- 独立 Fraction/格点核对逐步加法与真实文本/图像组合；六类错误链完成调用后检出。顺序/提前 BF16 另用注入输出的聚合反例，不冒充真实权重证据。16 环境模式 × 2 sticky 初值覆盖真实普通/观测、注入次正规普通/观测、参数拒绝及运行失败，共 192 次，控制位、已有 sticky 与完整恢复通过。
- 干净检出 `check_repo.py --audit --range d032e37..HEAD --strict` 为 E=0 W=0 I=0；源码指纹与审议版本一致，diff 与修改页引用通过。实现/契约及独立验证分域审阅、修正后的定点复核无阻塞。命令、日志及身份在 `scratch/p0025-routed-branch-oracle/closure/`；压缩前记录见 `46c47ce`。

### 评审处置

- 2026-10-16：用户确认范围并授权起草；随后批准设计，授权依 D0011 实现及必要归档，登记 active。
- 2026-10-16：分域审阅后修正覆盖矩阵歧义、docs 状态重复及验证盲点，最终复核无阻塞。
- 2026-10-16：用户人工审议通过候选实现及知识处置，要求继续推进；D0020 accepted，已提交干净复现及范围审计补证通过，据此记录结项。

### 提交

- `46c47ce`：参照、独立/组合验证、接受 D0020 及单一来源归档；携带 `Oracle: tests/core/test_moe.py`。
- 本次结项提交精简最终证据并同步 STATUS；版本递增由 `Merge P0025` 合并提交记录。

## 收尾

- 验收或停止依据：八项验收成立；用户已人工审议通过，已提交干净检出增量复现与严格范围审计补证通过，不以合并代替验收。覆盖边界以 MODEL §4 为准。
- 知识处置：updated；MOE、MODEL §4、BUILD §7 与 D0020 分别归档契约、覆盖边界、操作及理由，用户已审议，不重复保留规范正文。
- 剩余义务：无本项阻塞或移交义务，followups 为空。生产聚合与共享专家等后续另立 P，不重开本项，P0005 阻塞不解除。
