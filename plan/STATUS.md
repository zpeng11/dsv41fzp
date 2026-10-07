# STATUS — 状态页

状态信息的唯一归宿（状态不进 docs，README 只做路由）。本页属生命周期簿记，更新免用户侧二次确认。

## 当前阶段

骨架期，向密集开发期过渡。目录结构与治理基线已落：引擎版本方案、plan 编号、git 规范（`D0001`），机械执法器入宪（`D0002`）。机械执法器 `tools/check_repo.py` v1 随 `P0001` 交付；P0001 经用户明确确认，结项为 completed / knowledge: covered，补齐 D0002 引用与收尾记录，与 STATUS 同提交同步；P0003 已修复合法决策替代误报并补充回归测试，全部 57 项规则测试通过，严格历史审计 E=0 W=0。用户已确认修复完成并授权提交合入与推送；修复提交 `85f7faa` 同步 P0001 与 STATUS 结项，P0003 以 --no-ff 合入，merge 本体递增 VERSION 至 0.1.3 并同步结项为 completed / knowledge: covered。工作项结项与知识处置解耦已按用户审阅确认入库：`D0003` 宪法、模板与条目字段迁移直接提交 main（`390c404`）；`P0002` 校验器与测试实现（`c6bf346`）以 --no-ff 合入，merge 本体递增 VERSION 至 0.1.2，并同步结项为 completed / knowledge: covered。49 项规则测试通过，合入前严格历史审计及实现分支范围检查均为 E=0 W=0；临时仓库验证修宪、实现合入与结项流程通过。CI 接线待「构建」工作流定案。引擎实现（`src/`）、oracle（`ref/`）、构建、CI 均未动工；宪法「构建」「bench」两处待填。

## 文档进度

- **已落盘**：`AGENTS.md`（宪法）；`docs/MODEL.md`（模型事实单一引用源）；`docs/HARDWARE.md`（目标硬件、支持矩阵与资质门槛）；`plan/` 条目与决策模板、`D0001`、`D0002`（机械执法六项决议）
- **治理修订**：`D0003`（结项与知识处置契约）；P 模板采用 completed / knowledge / followups / 收尾结构，已有 P0001 补齐字段。
- **已结项工作项**：`P0002`（校验器配套实现），completed / knowledge: covered；结项依据与规则映射见条目。
- **已结项工作项**：`P0001`（v1 校验器），completed / knowledge: covered，结项与修复同提交记录。
- **已结项工作项**：`P0003`（决策替代误报修复），completed / knowledge: covered；修复依据与回归测试见条目。
