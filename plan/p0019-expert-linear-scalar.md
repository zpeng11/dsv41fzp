---
id: P0019
title: 建立单 token FP4 专家线性层的生产 CPU scalar 与完整输出验证
status: completed
created: 2026-10-14
updated: 2026-10-14
knowledge: updated
docs: [docs/ACTQUANT.md, docs/COMPUTE.md, docs/BUILD.md, docs/MODEL.md]
bench: []
decisions: [D0011, D0013, D0014]
followups: []
---

## 意图

接续 [P0018](p0018-checkpoint-expert-consumption.md) 的真实权重选定行消费，将 [P0017](p0017-act-quant-scalar.md) 的激活量化与 [P0013](p0013-expert-grouped-dot-scalar.md) 的点积提升为可独立调用的完整线性算子。一次生产量化供全部输出行消费，以既有 Python 参照组合和真实 checkpoint 验证，不改变已有数值契约。

## 验收

- [x] **开工前置决策**：生产接口、模块、工作缓冲、别名及矩阵级失败语义由 [D0014](decisions/d0014-expert-linear-scalar.md) 明确，经用户审议接受后转 active；保留 D0013 的历史结论。
- [x] **生产完整线性入口**：满足 [COMPUTE §7](../docs/COMPUTE.md#7-单-token-专家线性层)，普通 C++ 消费者只依赖公共头与库即可调用；两种维度方向与输出类型均覆盖。生产量化结果直接供全部行点积，不另写数值核心。
- [x] **全向量 parity**：直接对照 `ref/act_quant.py` 与 `ref/compute.py` 组合，按既有位比较/NaN 分类规则检查全部输出及量化工作缓冲。覆盖非方阵、多组、行间不同 scale、首末行、BF16 舍入、正负混合、零符号和跨组幅值；期望不用生产 helper 生成，不由 Python 替换生产中间结果。
- [x] **矩阵级失败与存储契约**：按 COMPUTE §7 验证维度、溢出、容量、别名及内容拒绝，尤其末行非法 scale 的全输出和工作缓冲原子性；定向验证校验次序、输入及写范围外字节不变和边界访问，不以单点积保证代替矩阵证据。测试构造须满足 C++ 对象和 span 前提。
- [x] **独立布局与故障检出**：手算合成矩阵独立核对权重行步长、scale 配对和 nibble 次序，读取字节与计算分开验收。错行、scale 错配、nibble 反序、遗漏末输出及结果篡改须检出，且有同用例无故障对照；读取、执行器、协议或库拒绝均使验证失败，不计为 parity 通过或数值敏感性证据。
- [x] **真实完整矩阵闭环**：下述固定 w1 的全部 N 行、完整 K 和两类输出均通过生产入口对照参照组合；配置及官方构造规则提供独立尺寸期望。显式离线命令缺权重或执行器失败、不 skip；按 BUILD §7 记录环境、快照、源码/执行器身份、输入与载荷指纹、完整读取区间、命令和比较结果，错误定位输入与输出行。
- [x] **复现与回归**：合成测试不依赖权重或网络；Debug/Release 全套及适用 sanitizer 回归、MXCSR 和新增接口边界测试通过，无 skip、编译告警或非预期报告。已提交干净检出、自源码目录外复现；量化提交携带有效 `Oracle:`，严格检查及提交范围审计通过。不设置性能门槛。
- [x] **知识与人工收尾**：按 D0011 将矩阵契约归 COMPUTE、操作与测试范围归 BUILD、验证链增量归 MODEL；ACTQUANT 数值规则只引用。用户评审实现、证据及知识处置，全部条件成立后结项并同步 STATUS。

## 记录

### 范围与依据

- 前置与环境：P0018 的结构校验和有界读取，P0017/P0013 的生产数值组件；[BUILD §1](../docs/BUILD.md#1-支持环境) ENV-A，不依赖 P0005 目标机。
- 契约与边界：[COMPUTE §7](../docs/COMPUTE.md#7-单-token-专家线性层)、[ACTQUANT](../docs/ACTQUANT.md)、[MODEL §4](../docs/MODEL.md#4-精确性链条)；取舍见 D0014。完整专家 MLP、生产加载/pack、快路径与硬件工作另行立项。
- 固定真实验收：MODEL 的 [ckpt] 快照，`layers.0.ffn.experts.0.w1` 全 N×K，确定性有限 BF16 人工输入 `mixed`/`group_ramp`。w2 方向由合成矩阵覆盖，真实 w3/w2 不作为最小门槛；不扫描整套载荷，不串接专家。
- 知识去向：COMPUTE/BUILD/MODEL 为可维护正文，D0014 保留设计时点理由；依 [D0011](decisions/d0011-doc-single-source.md) 以引用代替重复。运行生成物留 `build/` 或 `scratch/`。

### 最终证据

- ENV-A；验收工件 `9a034fa95b24431139bd4d54968368c5bff4ded5`，VERSION 0.1.16，dirty=false。全新 detached 检出 `/tmp/p0019-final-8yu5zoky/checkout`，构建后从其源码目录外 `/tmp/p0019-final-8yu5zoky` 运行测试及真实验证，执行前后工作树均干净。日志与报告在 `scratch/p0019-final-8yu5zoky/`，替代工作树阶段证据。
- 在检出根运行 `cmake --preset <preset>`、`cmake --build --preset <preset>`，在源码目录外运行 `ctest --test-dir <checkout>/build/<preset> --output-on-failure --no-tests=error`：Debug 32/32、Release 32/32、ASan 20/20 通过，无告警、skip 或非预期报告；sanitizer 环境按 BUILD。线性 parity 64 例、contract 385 例各覆盖 16 种 MXCSR 状态；工具合成测试 15/15。
- 按 BUILD §7 以绝对路径分别运行 Debug/Release 真实验证：revision `2cba9e42aa026125f3ed06c6d98c1db82f7ca027`，完整 2304×5120，各 9216 项输出全部有限且逐位一致；输入、输出、载荷区间及指纹相同。独立 seek/read 复核 4608 段区间与矩阵指纹，共 6,266,880 字节，并核对执行器 SHA256、源码 SHA/版本/dirty。
- 在源码目录外运行 `<checkout>/tests/quant/{linear_sensitivity,act_quant_sensitivity}.py`：线性五种变体在两类输出上 10/10 数值检出，既有激活量化 10/10 通过；进程/协议故障不算数值证据，源码哈希未变。
- 干净检出执行 `check_repo.py --repo <checkout> --range 6f90171..HEAD --audit --strict`，E=0 W=0 I=0；`git diff --check 6f90171..HEAD` 及文档链接/锚点通过。实现范围内未改受保护目录、AGENTS 或 VERSION；该版本仅由随后合并流程递增。

### 评审处置

- 2026-10-14：用户确认范围并授权起草，登记 draft。
- 2026-10-14：用户通过 D0014 开工方案，授权实现及 COMPUTE/BUILD/MODEL 必要增补，转 active。
- 2026-10-14：修复执行器共享尾部类型化读取、BF16 NaN 高位漏检；FP32 input/out 不合法的完整类型化重叠改报协议错误，BF16 用同一真实数组子范围验收。静态复核通过后重跑验证，生产契约不变。
- 2026-10-14：用户审议代码通过，授权依 D0011 精简文档并继续提交、验收和合并；据此完成剩余机械证据后按流程结项。

### 提交

- `9a034fa`：完整生产线性入口、合成与真实矩阵验证、D0014 和 D0011 文档减负；携带 `Oracle: tests/quant/test_linear_parity.py`。
- 本次收尾提交：回填已提交工件的干净检出证据、知识处置，完成结项并同步 STATUS。

## 收尾

- 验收或停止依据：用户审议代码通过并授权依 D0011 继续提交合并；`9a034fa` 的干净检出全回归、真实完整矩阵验证、敏感性与范围审计全部通过，约定验收成立。
- 知识处置：updated；接口与失败契约归 [COMPUTE §7](../docs/COMPUTE.md#7-单-token-专家线性层)，操作及测试入口归 [BUILD §7](../docs/BUILD.md#7-测试清单)，验证链边界归 [MODEL §4](../docs/MODEL.md#4-精确性链条)，理由留 D0014。按用户本次授权及 D0011 删除重复契约与过程，已接受 D 原文保留。
- 剩余义务：无本项遗留义务，followups 为空；不接收前项遗留义务，范围外工作仍按 MODEL §4 另行立项，P0005 硬件阻塞不解除。
