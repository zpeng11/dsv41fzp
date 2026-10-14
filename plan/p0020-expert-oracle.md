---
id: P0020
title: 建立单 token FP4 专家的 CPU 数值参照与三矩阵组合验证
status: completed
created: 2026-10-15
updated: 2026-10-15
knowledge: updated
docs: [docs/EXPERT.md, docs/MODEL.md, docs/BUILD.md, docs/COMPUTE.md]
bench: []
decisions: [D0015]
followups: []
---

## 意图

接续已结项 [P0019](p0019-expert-linear-scalar.md)，以既有量化及点积参照补齐非线性与三矩阵慢路径，为单 token FP4 专家生产实现建立独立验证的上一级。契约归 [EXPERT](../docs/EXPERT.md)，边界只见 [MODEL §4](../docs/MODEL.md#4-精确性链条)。

## 验收

- [x] **数值方案审议**：[D0015](decisions/d0015-expert-oracle.md) 已接受，SiLU、阶段与环境边界明确。
- [x] **单专家参照**：`ref/expert.py` 满足 EXPERT §1～§3，复用既有量化与点积；普通与 trace 共用核心，不改既有参照。
- [x] **非线性独立验证**：依 EXPERT §4 以独立区间及格点期望验证钳位、SiLU、逐次乘法、BF16 与特殊值，不共用被测 exp/舍入核心。
- [x] **三矩阵逐阶段组合**：单组、多组、非方阵的全部阶段对照独立期望链；覆盖参数拒绝、输入不变及中间失败边界。
- [x] **错误检出**：遗漏/错误钳位、交换分支、提前/绕过 BF16、遗漏/错放权重的显式错误链均有数值反例及正确基线，不以异常替代检出。
- [x] **增量复现**：依 BUILD §7 完成本项提交后干净检出、目录外命令及 Debug/Release 新 CTest 入口，无 skip，严格检查与提交范围审计通过；不重跑 P0019 验收。
- [x] **知识与收尾**：依 D0011 归档契约和路由，完成用户已授权的提交后证据及结项记录，同步 STATUS；合并为后续机械步骤，不代替验收。

## 记录

### 范围与依据

- P0019 结项证据按用户确认直接继承。来源为官方 `model.py:830–851`、`kernel.py:478–484,550–557`；取舍见 D0015，数值不重复归档。
- 知识分工：EXPERT 唯一契约正文，MODEL 唯一覆盖边界，BUILD 操作入口，COMPUTE 只补组合路由；不扩生产实现或真实矩阵验收范围。

### 最终证据

- ENV-A；提交 `2b8d366`，VERSION 0.1.17，dirty=false。`git clone --no-local --single-branch --branch p0020-expert-oracle` 至 `/tmp/p0020-clean-wj1ctiwl/repo`，从 `/tmp` 以脚本绝对路径执行 `python3 -B <checkout>/tests/quant/test_expert.py`：17/17、无 skip。
- 同一干净检出配置 `cpu-debug`/`cpu-release` 后，按 BUILD §7 的 `-R '^quant\.expert$' --no-tests=error` 各通过 1/1 新入口，内部各 17 项；未构建 C++ target、未跑旧套件或 P0019 复验。运行后检出仍干净，日志在 `scratch/p0020-expert-oracle/closure/`。
- 独立期望检查非线性、单/多组及非方阵全部阶段、钳位恢复 Inf、真正中间非有限拒绝和最终 Inf/NaN。显式错误链反例不是源码变异门禁。四种舍入 × 四种 FTZ/DAZ × 两种初始状态及接受/拒绝路径验证 EXPERT 环境边界；不宣称完整浮点环境无副作用。
- 干净检出 `check_repo.py --strict`、主工作树 `--audit --range main..HEAD --strict` 均 E=0 W=0 I=0；范围 diff 检查及修改页链接/锚点通过。新增 ref 文件之外未改既有数值实现、生产代码、冻结工件或只读镜像；实现提交不改 VERSION。

### 评审处置

- 2026-10-15：用户同意范围并授权起草推进；确认 P0019 已结项，无需复验。
- 2026-10-15：Decimal 内部新增 inexact 导致原完整环境不变探针失败，明确改用 D0015 的 sticky 边界；修复线性 Inf 在钳位前过早拒绝，增加可恢复与真实失败用例。
- 2026-10-15：用户审议代码通过，授权按 D0011 精简归档并继续提交、合并；D0015 转 accepted，完成剩余机械证据后结项。

### 提交

- `2b8d366`：参照、独立验证、CTest 及 D0011 单一来源归档；携带 `Oracle: tests/quant/test_expert.py`。
- 结项提交同步本项与 STATUS；合并及 patch 递增由后续 `Merge P0020` 提交记录。

## 收尾

- 验收或停止依据：七项验收成立；用户于 2026-10-15 审议通过并授权提交合并，已提交工件 `2b8d366` 的干净检出增量复现及范围审计通过。覆盖边界以 MODEL §4 为准，未将合成 CPU 证据扩作 GPU、真实模型质量或生产专家验收。
- 知识处置：updated；[EXPERT](../docs/EXPERT.md) 归档契约，MODEL/BUILD/COMPUTE 仅保留各自路由，D0015 保留选择理由。
- 剩余义务：无本项阻塞或移交义务；后续同契约生产专家需另立 P，不重开本项或 P0019。
