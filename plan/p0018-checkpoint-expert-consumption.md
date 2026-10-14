---
id: P0018
title: 建立真实 checkpoint 路由专家权重的有界读取与 CPU 数值消费闭环
status: completed
created: 2026-10-14
updated: 2026-10-14
knowledge: updated
docs: [docs/MODEL.md, docs/BUILD.md, docs/ACTQUANT.md, docs/COMPUTE.md]
bench: []
decisions: [D0011, D0012, D0013]
followups: []
---

## 意图

接续 [P0016](p0016-checkpoint-inventory.md) 的 checkpoint 清点与 [P0017](p0017-act-quant-scalar.md) 的生产组合测试，补上真实权重载荷进入计算链的证据。成功的样子：指定路由专家的权重行及对应 scale 可被有界、可追溯地读取，在完整 K 维上由既有 Python 参照与生产 C++ scalar 分别消费，结果逐位一致；读取正确性另由独立合成期望验证，不以两端共用读取结果形成循环论证。

## 验收

- [x] **开工方案**：细化按需读取、执行器连接与测试入口，经用户审议后转 active；沿用既有数值契约，不把本项扩成生产矩阵 API 或新计算路线。
- [x] **有界读取**：基于 P0016 的结构校验与独立预期，核对所选张量的名称、存储 dtype、shape、权重与 scale 配对及行范围；按需读取载荷，完整保留所选行的 K 维。非法选择、越界、截断与短读明确失败，不加载或扫描整套权重载荷。
- [x] **独立读取验收**：在临时目录构造手工可核对的小型 safetensors，验证数据区基址、非零张量偏移、首/中/末行偏移、打包字节顺序与 scale 行配对。相邻行及分组使用可区分数据；期望不由被测读取 helper 生成，测试不写入冻结的 `tests/data/`。
- [x] **真实数值闭环**：按下述固定范围分别验证 w1/w3/w2；确定性有限 BF16 输入覆盖非零、正负混合及跨组幅值变化，不只依靠退化零向量。Python 量化与点积参照的组合直接作为生产组合上一级，FP32/BF16 两类输出逐位比较；生产量化结果缓冲直接供生产点积，不经 Python 重量化或数值替换。失败可定位至输入用例、张量、输出行及量化或点积阶段。
- [x] **失败传播**：合成反例覆盖错误的张量配对、越界、短读与非法编码；读取错误、执行器或协议错误、计算拒绝、结果不符均使总命令非零退出，不将库拒绝、工具异常或缺权重记为 parity 通过。对读取错位、scale 错配与结果篡改设置定向检出验证，并保留同用例无故障对照。
- [x] **复现与回归**：合成测试进入 CPU Debug/Release 常规测试，不依赖本机权重或网络；真实 checkpoint 验证用显式命令执行，不以缺路径 skip。证据记录 ENV-A、快照 revision、代码 SHA/版本/dirty、输入构造、张量与所选行、实际读取区间及载荷指纹、命令和比较结果；在已提交干净检出、自源码目录外复现。若调整 C++ 测试连接，另跑现有 sanitizer 范围；相关回归无 skip、编译告警及非预期报告，仓库严格检查与提交范围审计通过。不设置性能门槛。
- [x] **知识与人工收尾**：操作入口与读取约束归 BUILD，验证链新增边界归 MODEL；仅在新实测推翻既有事实时修订相应事实，既有数值契约以引用为准。用户评审实现、证据与知识处置后结项并同步 STATUS。

## 记录

### 范围与依据

- 前置：P0016 提供结构校验、独立张量预期及已清点快照；P0017 与 [P0013](p0013-expert-grouped-dot-scalar.md) 提供生产组合组件。存储布局见 [MODEL §2.1](../docs/MODEL.md#21-权重分量格式表)，数值契约见 [ACTQUANT](../docs/ACTQUANT.md) 与 [COMPUTE](../docs/COMPUTE.md)，既有验证边界见 [MODEL §4](../docs/MODEL.md#4-精确性链条)。环境与本机快照路径引用 [BUILD](../docs/BUILD.md)，不在本项复制。
- 固定切片：沿用 MODEL 的 [ckpt] 快照，选择 `layers.0.ffn.experts.0` 的 w1/w3/w2 权重及 scale；分别取每个矩阵的首行、中间行（`N // 2`）与末行，K 维不裁剪。矩阵逻辑尺寸由配置/官方构造规则独立确定并核对 checkpoint，不把被检头部自身当作尺寸期望；三个矩阵独立计算，w2 使用独立构造的输入，不串接 MLP。小型合成矩阵可检查全部输出行。
- 实现边界：拟采用标准库 Python 按需读取与验证入口，复用 `tools/ckpt_inventory.py` 的结构能力、`ref/` 数值参照和现有 C++ 测试连接。格式解释仍走既有量化真源，不另写一份解码/点积；必要的执行器调整仅服务本项测试，不建立公共矩阵接口或生产序列化格式。
- 范围外：通用 C++ artifact 读取器、pack、完整专家 MLP、路由、真实模型激活采集、W4A16/K-E1 或整数改写/K-E2、SIMD/CUDA、缓存放置和性能优化。真实权重加人工输入只验证既有 CPU 契约的数值消费，不验收官方 GPU 等价、模型质量或端到端一致。
- 知识去向：按 [D0011](decisions/d0011-doc-single-source.md) 在既有 BUILD/MODEL 中归档必要增量，不另建重复契约页；取舍延续 [D0012](decisions/d0012-checkpoint-inventory.md) 的按需工具路线与 [D0013](decisions/d0013-act-quant-scalar.md) 的生产组合边界。运行生成物留 `build/` 或 `scratch/`，P 只记录最终证据摘要与复现指针。

### 最终证据

- 环境：[BUILD §1](../docs/BUILD.md#1-支持环境) ENV-A；2026-10-14 验收实现提交 `f8b4fe3da4caa1eefce18c32b4fa1838821a8d98`、VERSION 0.1.15、dirty=false。全新 detached 检出为 `/tmp/p0018-final-exu5iqym/checkout`，测试和真实快照命令从其源码目录外 `/tmp/p0018-final-exu5iqym` 运行，执行前后工作树均干净。
- 真实快照：revision `2cba9e42aa026125f3ed06c6d98c1db82f7ca027`；按 BUILD §7 的 `python3 <checkout>/tools/ckpt_expert.py <snapshot> --exe <checkout>/build/<preset>/tests/quant/df_quant_exec --json <logs>/real-<preset>.json` 分别运行全新构建的 Debug/Release 执行器，两次均通过。w1/w3/w2 共 9 行、每行 2 个输入模式、36 个 FP32/BF16 输出；每次激活比较 315,168、点积阶段比较 159,954，总比较 475,158。所选输出均有限且逐位相同，Debug/Release 输入、载荷区间与结果一致；每次仅消费 19,992 字节权重/scale 载荷，另以独立 seek/read 复核报告全部 18 段区间的 SHA256，核对执行器指纹及源码身份。
- 合成测试 `tests/tools/test_ckpt_expert.py`：22 项在 Debug/Release 均通过、无 skip；包含手算 13.5 锚点及合法溢出 NaN 分类。错行/错 scale 注入不改变两端 parity 通过，但被独立字节期望检出；两类输出篡改均产生数值 mismatch，均有无故障对照。
- 干净检出回归：在检出根执行 `cmake --preset <preset>`、`cmake --build --preset <preset>`，在源码目录外执行 `ctest --test-dir <checkout>/build/<preset> --output-on-failure --no-tests=error`；cpu-debug 27/27、cpu-release 27/27 通过，无编译告警或 skip。没有修改生产 C++、执行器或 sanitizer 范围，本轮不重跑 cpu-asan。
- 治理：干净检出执行 `check_repo.py --repo <checkout> --range 412d38f..HEAD --audit --strict`，E=0 W=0 I=0；`git diff --check 412d38f..HEAD` 通过。范围核对仅涉及本项工具、合成测试、CTest 注册及 BUILD/MODEL/P/STATUS；`ref/`、`src/`、`include/`、只读镜像、冻结数据与 VERSION 未改。最终日志、真实报告及独立复核摘要位于 `scratch/p0018-final-exu5iqym/`（上文 `<logs>`），为临时生成物；原工作树阶段记录留在 `scratch/p0018-acceptance/`，不作为最终工件身份依据。

### 评审处置

- 2026-10-14：用户认可真实权重最小消费闭环的范围建议并授权起草 P0018，登记 draft。
- 2026-10-14：用户审议通过本项设计，授权按流程转 active 并实施；建立 `p0018-checkpoint-expert-consumption` 分支，采用 Python 按需读取与既有测试执行器组合入口，合成测试接入 CPU Debug/Release，结项保留人工验收。
- 2026-10-14：用户审议同意实现、测试与知识处置改动并授权继续推进；提交实现，补齐已提交干净检出复现与范围审计后按流程收尾。
- 2026-10-14：用户再次明确已审议通过 P0018 的代码实现与进度，并要求按流程验收收尾；据此完成已提交工件的干净检出复现与范围审计，条件全部通过后登记 completed，并在同一收尾提交同步 STATUS，不另设重复人工审批。

### 提交

- `f8b4fe3`：真实专家权重有界读取、独立合成验收、CPU 数值消费闭环及知识归档；为本项干净检出验收的实现工件。
- 本次收尾提交仅回填最终证据、人工授权与知识处置，同步本项及 STATUS；不改已验收实现。

## 收尾

- 验收或停止依据：用户审议通过并授权按流程收尾；`f8b4fe3` 的干净检出 Debug/Release 常规回归、独立读取验收、真实切片数值验证与严格范围审计均通过，全部约定验收成立。
- 知识处置：updated；操作入口、读取约束与证据字段归 [BUILD §7](../docs/BUILD.md#7-测试清单)，集成验证边界归 [MODEL §4](../docs/MODEL.md#4-精确性链条)，既有数值契约以 [ACTQUANT](../docs/ACTQUANT.md) 与 [COMPUTE](../docs/COMPUTE.md) 为准。知识处置已随改动审议通过，本次未产生需另归档的契约或 checkpoint 事实。
- 剩余义务：无本项遗留义务，followups 为空；不接收 P0016/P0017 的遗留义务。生产 artifact、完整专家 MLP、GPU parity、模型质量与性能均属既定范围外，仓库 CI 与 P0005 硬件阻塞不据此解除。
