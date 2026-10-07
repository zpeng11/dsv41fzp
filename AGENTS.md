# AGENTS.md

dsv41fzp 是 DeepSeek-V4.1-Flash 的单机柜推理引擎：单一模型特化，只把这一个模型在一种硬件形态（双路 CPU + 2/4 卡 NVLink GPU 的单机柜）上跑好。模型事实见 [docs/MODEL.md](docs/MODEL.md)，目标硬件见 [docs/HARDWARE.md](docs/HARDWARE.md)。

**本文件是执法文件**：只收规则与路由。知识在 `docs/`（单一信息源，本文件不复述），状态在 `plan/`（本文件不记载当前进度）。一条内容是否属于这里，看它六个月后是否仍然成立：规则留下，知识与状态移走。文中 `待填` 为骨架占位，由维护者替换；替换前对应条目不生效。

## 仓库地图

| 位置 | 角色 |
| --- | --- |
| `docs/` | 事实的唯一归宿；每个数字带来源标记（[卡][cfg][代码]…）与测量环境 |
| `plan/` | 意图与决策记录；条目完成后结论沉淀进 `docs/`，随即归档 |
| `ref/` | Python FP32 oracle，验证链之根，先于引擎存在 |
| `include/df/` | 公共头，镜像 `src/` 模块 |
| `src/` | 引擎，按域分模块（划分见 README；`src/quant` 是全仓库量化格式的唯一真源） |
| `tests/` | 按域分树：quant / core / cuda / cpu / attn / platform；`tests/data/` 金标准数据字节冻结 |
| `bench/` | `results/` 日期-主题实验记录（入库）；`prompts/` 冻结提示词 |
| `tools/` | pack 转换、热度画像、硬件资质、质量门禁、运维 MCP |
| `data/` | 专家热度画像、放置计划、路由 trace |
| `third_party/reference/` | 官方仓库非权重镜像，只读，只做仲裁（见其 PROVENANCE.md） |
| `deploy/` | 机柜部署器；启动前执行硬件资质预检 |
| `serve/` | OpenAI/Anthropic 兼容 API + monitor（测试与代码同目录） |

## 规则（项目宪法）
1. **每个工件可追溯**——commit、测试、bench 记录引用 plan 编号（编号格式 `待填`）。
2. **agent 是一等公民**——无 GPU 可跑的测试、面向 agent 的操作手册、MCP 化运维。
3. **维护简洁comment**——代码comment一般情况仅允许覆盖函数/类等代码块作为代码块的整体解释， 不允许将细节编码改动体现在comment中。
4. **文档防膨胀**—— `docs/` 和 `plan/` 目录严格由用户管控，仅由用户确认下允许进行写入改动。如果需要临时文档归纳请使用系统的临时目录。

补充约束：
- 每个论断都有测量，文档里不写没有硬件环境与方法说明的数字，模型数字一律引用 `docs/MODEL.md`。
- 每段快代码都有已验证的慢参照，kernel 的 parity 只对照验证链上的上一级，绝不对照自身副本。
- 优化路径必须带运行时开关，开关两侧都要有测量数字（进 bench 记录与 docs）。

## 每次改动伴随什么

| 你改了 | 必须同时 |
| --- | --- |
| kernel（`src/kernels` `src/attn` `src/aq` `src/spec`） | 更新对应 parity 测试；只对照验证链上一级 |
| 量化编解码 | 定义只落在 `src/quant`，别处引用不复制；`tests/data/` 金标准向量同步更新 |
| 性能 / 质量论断 | 同一提交含 `bench/results/<日期-主题>/`（README + JSON + 配置 + 日志），挂引擎版本；文档数字回链该目录 |
| docs 中的事实 | 与用户确认来源标记与测量环境同步 |
| 完成一项 plan 条目 | 与用户确认后结论沉淀进 docs，条目标记完成或归档 |

## 工作流

- **git**：`待填`——分支模型与提交规范（git 初始化后补）。
- **构建**：`待填`。
- **无 GPU 可跑的测试**：`待填`——首批测试落地后必须在此列出命令，这是宪法第 4 条的落点。
- **bench**：`待填`。
- **资质预检**：`tools/qualify_hw.py`（`待填`：命令；门槛对应 HARDWARE.md，不过即拒启）。
- **MCP 运维**：`待填`。
- **引擎版本方案**：`待填`——所有性能数字挂版本的前提，须在第一行性能数字落盘前定。

## 禁区

- `third_party/reference/` 只读：只做仲裁，不参与构建。
- `bench/prompts/` 与 `tests/data/` 字节冻结（`.gitattributes` 已含规则）：改动即视为新工件，须重算 manifest 并说明。
- `ref/` 不因引擎实现不便而修改；确认 oracle 自身有误时，单独提交并写明依据。
- `docs/` 不写无测量环境的数字；状态不进 docs（进 plan）。
- 生成物不进 git（`.gitignore` 已列；bench 入库内容经人审）。
