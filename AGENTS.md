# AGENTS.md

dsv41fzp 是 DeepSeek-V4.1-Flash 的单机柜推理引擎：单一模型特化，只把这一个模型在一种硬件形态（双路 CPU + 2/4 卡 NVLink GPU 的单机柜）上跑好。模型事实见 [docs/MODEL.md](docs/MODEL.md)，满足资质的目标硬件见 [docs/HARDWARE.md](docs/HARDWARE.md)。

**本文件是执法文件**：只收规则与路由。知识在 `docs/`（单一信息源，本文件不复述），状态在 `plan/`（本文件不记载当前进度）。文中 `待填` 为骨架占位，由维护者替换；替换前对应条目不生效。

## 仓库地图

| 位置 | 角色 |
| --- | --- |
| `docs/` | 事实的唯一归宿；每个数字带来源标记（[卡][cfg][代码]…）与测量环境 |
| `plan/` | 意图、决策记录与状态页（`STATUS.md`）；条目完成后结论沉淀进 `docs/` |
| `ref/` | Python FP32 oracle，验证链之根，先于引擎存在 |
| `include/df/` | 公共头，镜像 `src/` 模块 |
| `src/` | 引擎，按域分模块（划分见 README；`src/quant` 是全仓库量化格式的唯一真源） |
| `tests/` | 按域分树：quant / core / cuda / cpu / attn / platform；`tests/data/` 金标准数据字节冻结 |
| `bench/` | `results/` 日期-主题实验记录（入库）；`prompts/` 冻结提示词 |
| `tools/` | pack 转换、热度画像、硬件资质、质量门禁、长文探针、运维 MCP |
| `data/` | 专家热度画像、放置计划、路由 trace |
| `third_party/reference/` | 官方仓库非权重镜像，只读，只做仲裁（见其 PROVENANCE.md） |
| `deploy/` | 机柜部署器；启动前执行硬件资质预检 |
| `serve/` | OpenAI/Anthropic 兼容 API + monitor（测试与代码同目录） |
| `scratch/` | Agent转用的临时归纳与跨会话草稿；git-ignore 不进库，按需创建和清理 |

## 规则（项目宪法）
1. **每个工件可追溯**——commit、测试、bench 记录引用 plan 编号（`P<NNNN>` 工作项 / `D<NNNN>` 决策，扁平递增、四位零填充、永不复用；生命周期与模板见工作流「plan 条目」）。
2. **维护简洁comment**——代码comment一般情况仅允许覆盖函数/类等代码块作为代码块的整体解释， 不允许将细节编码改动体现在comment中。
3. **文档防膨胀**——`docs/` 与 `plan/` 的**内容性写入**（新建或修改正文、结论、决策）须用户确认，对话中的明确指示即为确认
4. **生命周期簿记**——免二次确认：`plan/`的按模板新建条目 stub、状态翻转与互链回填、`plan/STATUS.md` 的状态更新。临时归纳与跨会话草稿进 `scratch/`。
5. **宪法自缚**——本文件的修订须用户确认；agent 不得改写自身受约束的执法依据。


## 工作流

- **plan 条目**：工作项 `plan/p<NNNN>-<slug>.md`，决策 `plan/decisions/d<NNNN>-<slug>.md`，模板在对应目录。编号扁平递增、四位零填充、永不复用，ID 进文件名。P 状态机 `draft → active → sedimented | abandoned`（sedimented = 结论已沉淀进 docs 的终态）；D 状态机 `proposed → accepted | rejected | superseded`。归档 = 原地改 front matter 状态，文件永不移动删除——commit 里的编号引用永远可解析。决策被取代时新建 D 文件，两者 `replaces` / `superseded-by` 互链，不改写旧文。状态翻转与 `plan/STATUS.md` 同步在同一提交内完成，禁止事后补记。
- **git**：trunk-based。`main` 始终可构建，未完成路径靠运行时开关默认关（见补充约束）。实现类改动走短命分支 `p<NNNN>-<slug>`，`--no-ff` 合入，合并提交 subject 以 `Merge P<NNNN>` 开头。提交 subject 格式 `<P|D 编号> <area>: <论断句>`，area 与 `src/` 域名或仓库级域名（docs / plan / bench / ci / tools / serve / deploy）对齐。义务表落 trailer：性能论断必带 `Bench: bench/results/<日期-主题>`，kernel 与量化变更必带 `Oracle: <parity 测试路径>`。`meta:` 前缀仅限不改行为与论断的仓库维护提交（构建配置、格式化、ignore 等）；docs/ 与 plan/ 的改动经用户确认后可直接提交 main。
- **构建**：`待填`。
- **bench**：`待填`。
- **引擎版本**：单一真源为根目录 `VERSION` 文件（`0.<minor>.<patch>`，`0.x` 期无兼容承诺；仅由合并流程递增，不手工改）。初始 0.1.0；每次 `--no-ff` 合入后 patch+1，minor 递增绑定 plan 条目声明的里程碑，major 固定 0 直到 1.0 决策。CMake 读入注入编译定义，Python 工具与 CI 直接读文件；构建时工作树脏则版本串加 `+dirty` 后缀；引擎启动日志打印版本，使 bench 落盘日志自描述。量化/pack 格式版本独立为 `src/quant` 常量，不随引擎版本走。bench manifest 必含 `version`（须与 VERSION 一致）、`git_sha`、`dirty`。文档数字引用语法固定：`引擎 <x.y.z>; bench/results/<日期-主题>`。tag 仅打在有 bench 记录背书的里程碑（`v<x.y.z>`），不逐提交打。
- **机械执法**：工作流与禁区的可机械化子集由 `tools/check_repo.py` 校验（`python3 tools/check_repo.py [--range A..B] [--audit] [--strict]`，提交格式与 area、编号引用可解析、trailer 义务与真伪、VERSION 纪律、禁区触碰、plan 条目一致性；规则映射与豁免见 `plan/p0001-check-repo.md`），CI 与每次合入前必跑；首个 D 条目入库 commit 之前的提交豁免。
## 禁区

- `third_party/reference/` 只读：只做仲裁，不参与构建。
- `bench/prompts/` 与 `tests/data/` 字节冻结（`.gitattributes` 已含规则）：改动即视为新工件，须重算 manifest 并说明。
- `ref/` 不因引擎实现不便而修改；确认 oracle 自身有误时，单独提交并写明依据。
- `docs/` 不写无测量环境的数字；状态不进 docs（进 `plan/STATUS.md`）。
- 生成物不进 git（`.gitignore` 已列；bench 入库内容经人审）。

## 补充约束：
- 每个论断都有测量，文档里不写没有硬件环境与方法说明的数字，模型数字一律引用 `docs/MODEL.md`。
- 每段快代码都有已验证的慢参照，kernel 的 parity 只对照验证链上的上一级，绝不对照自身副本。
- 优化路径必须带运行时开关，开关两侧都要有测量数字（进 bench 记录与 docs）。