---
id: P0017
title: 建立单行 BF16 动态激活量化的生产 scalar 与组合 parity
status: completed
created: 2026-10-13
updated: 2026-10-14
knowledge: updated
docs: [docs/ACTQUANT.md, docs/BUILD.md, docs/MODEL.md]
bench: []
decisions: [D0007, D0009, D0010, D0013]
followups: []
---

## 意图

接续 [P0014](p0014-act-quant-oracle.md) 的已验收参照，交付同契约生产 C++ scalar，并与 [P0013](p0013-expert-grouped-dot-scalar.md) 在测试中组合：生产量化输出由生产分组点积直接消费，全链结果与已验证的 Python 参照组合一致。数值及生产契约只引用 [ACTQUANT](../docs/ACTQUANT.md)，验证链边界只引用 [MODEL §4](../docs/MODEL.md#4-精确性链条)。

## 验收

- [x] **开工前置决策**：按 [D0010](decisions/d0010-act-quant-reference.md) 后果，另立生产 C++ 决策并经用户审议接受后转 active。
- [x] **生产入口与失败契约**：满足 ACTQUANT §7；逐类验证非法输入、容量、重叠与 trace 拒绝。用精确分配、偏移共享及 ASan 加毒验证校验次序和失败原子性，协议错误不计为库拒绝；成功时核对输入及允许写入范围外字节不变。
- [x] **同契约整数数值路径**：满足 ACTQUANT §3、§7；复用既有整数基础，普通入口与私有观测共享生产核心。
- [x] **逐阶段 parity 与覆盖**：输出及全部阶段位直接对照 `ref/act_quant.py`。覆盖有限 BF16 amax 穷举、全部有限元素的不同量级锚点、非有限拒绝、ACTQUANT §3 边界及多组、跨 128 元素布局；沿用 P0014 用例依据，随机仅补充，不用生产 helper 生成期望，不以 checkpoint 实测 scale 范围限制输入。
- [x] **数值环境无关**：按 [BUILD §7](../docs/BUILD.md#7-测试清单) 八模式 MXCSR，在状态位清空与预置下验证普通、trace、接受及拒绝路径；结果一致、完整 MXCSR 不变、无陷入。
- [x] **测试敏感性与失败传播**：临时生产变体覆盖指数下取整/就近取整、四组共享 scale、FP8 截断/中点远离零、负零规范化、删除下限、BF16 次正规冲零、商次正规冲零及商中点远离零，全部由指定用例的数值或阶段位比较检出；异常、崩溃或 sanitizer 报告不算数值敏感性。另验证执行器、协议、trace 形状和存储检查故障使驱动失败，同筛选无故障须成功，未知筛选失败，无 skip。
- [x] **生产组合闭环**：量化结果缓冲直接供生产点积，两种输出对照 Python 参照组合；向量含非退化数值、零符号及不同 scale。组合范围与接口遵循 ACTQUANT §7。
- [x] **构建、治理与复现**：三种 CPU preset 全套回归通过，无 skip、编译告警及非预期 sanitizer 报告；源码目录外及已提交干净检出复现。量化提交携带有效 `Oracle:`，`--strict` 与提交范围审计 E=0 W=0 I=0。不设置性能门槛。
- [x] **知识与人工收尾**：增补 ACTQUANT 生产契约、BUILD 操作入口与 MODEL 上一级路由，遵守 [D0011](decisions/d0011-doc-single-source.md) 单一来源；用户评审实现、证据与知识处置后结项并同步 STATUS。

## 记录

### 范围与依据

- P0014 是量化上一级，P0013 是组合消费者；复用 P0010/P0011 整数基础。官方来源与阶段见 ACTQUANT §1、§3，生产契约见 §7，选择理由见 [D0013](decisions/d0013-act-quant-scalar.md)。
- 实现归 `df_quant`，组合仅在测试中；不建设 `src/aq` 调度框架，不修改既有 oracle、只读镜像与冻结数据。后续主线优先真实权重消费或具体模型子模块，另行立项。

### 最终证据

- 环境：[BUILD §1](../docs/BUILD.md#1-支持环境) ENV-A；验收工件 `6ad7276867a27456d22ca15911644b12df445a6f`，VERSION 0.1.14。`git worktree add --detach /tmp/p0017-clean-ul8556yv/checkout 6ad7276` 建立独立已提交检出，验证前后工作树 clean；Debug/Release 版本探针均报告该 SHA、dirty=0、display=0.1.14。
- 2026-10-14 从源码目录外执行 `cmake --preset <preset> -S <检出目录>`、`cmake --build <检出目录>/build/<preset>`、`ctest --test-dir <检出目录>/build/<preset> --output-on-failure --no-tests=error`：cpu-debug 26/26、cpu-release 26/26、cpu-asan 16/16 通过。sanitizer 环境变量按 BUILD §3 设置，无编译告警、测试 skip 或非预期报告；harness 的故意非法读取按预期失败。此已提交干净检出结果取代工作树阶段构建证据。
- 定稿 parity 452、contract 2,602（含 trace 数据别名/尾部共享 330）、combination 20，均经 16 种 MXCSR 状态；amax 穷举 32,640 个幅值，两套元素构造各覆盖 65,280 个有限 BF16 编码。非有限保留 256 码四位置拒绝，另补普通/trace 组内中间位置 60 例。
- 新增描述对象独立/输出别名及共享间隙各 2 例；harness 的 103 个成功用例在输入改写故障下均被检出，共享间隙改写也被检出。故障传播均有同筛选无故障对照及原因核对，未知筛选失败；临时移除描述快照、改写输入或共享间隙均使对应回归失败。
- 干净检出执行 `PYTHONDONTWRITEBYTECODE=1 python3 <检出目录>/tests/quant/act_quant_sensitivity.py`：10/10 数值变体检出，按验收条目顺序 mismatch 为 112、60、258、756、252、46、142、4、62、22，errors=0/skips=0；恢复与 pristine 重测通过，受保护工件哈希不变。
- `python3 tools/check_repo.py --range c4aa755..6ad7276 --strict` 与 `--audit --strict` 均 E=0 W=0 I=0；`git diff --check` 通过，内部 Markdown 链接/锚点 broken=0。已提交工件日志 `scratch/p0017-acceptance-rdfowm9g/`，定向注错材料 `scratch/p0017-review-fixes/`，均为临时工件。

### 评审处置

- 2026-10-13：用户同意范围并授权起草 P0017，登记 draft。
- 2026-10-14：用户接受含 D0013 的开工方案，授权转 active、实现及 ACTQUANT/BUILD/MODEL 必要增补。
- 2026-10-14：早期复核补齐 trace 数据缓冲区别名测试、修正尾部共享期望；共享存储改用 C++20 原始分配建立 implicit-lifetime 数组生命周期。
- 2026-10-14：用户授权处理本轮评审问题并减负可修改文档；trace 入口复制描述对象，contract 合并筛选并加无故障对照，成功路径增加输入与写范围外字节检查，补组内非有限用例。
- 2026-10-14：P 的契约、环境和验证边界改为章节引用，敏感性门禁留本项及 P0014，STATUS 仅留进度与阻塞。D0013 按 D0011 保留已接受原文，ACTQUANT 仍是生产契约的可维护正文。
- 2026-10-14：用户同意实现、证据与知识处置验收并授权按流程推进；已提交工件的干净检出与范围审计通过，本次结项并同步 STATUS。

### 提交

- `6ad7276`：生产实现、回归与敏感性工具、D0013 和文档减负；携带 `Oracle: tests/quant/test_act_quant_parity.py`。
- 本提交：记录干净检出与人工验收，完成结项并同步 STATUS。

## 收尾

- 验收或停止依据：九项验收成立；已提交工件的三 preset、敏感性、干净检出及范围审计证据见记录，2026-10-14 用户评审通过。验证边界以 MODEL §4 为准。
- 知识处置：updated；生产契约归 [ACTQUANT §7](../docs/ACTQUANT.md#7-生产-c-scalarsrcquant)，命令与环境归 [BUILD](../docs/BUILD.md)，验证链归 [MODEL §4](../docs/MODEL.md#4-精确性链条)，理由留 D0013；用户已认可知识处置，可修改正文按 D0011 收敛，已接受 D 原文保留。
- 剩余义务：无本项遗留义务，无移交项；范围外能力按 MODEL §4 另行立项。
