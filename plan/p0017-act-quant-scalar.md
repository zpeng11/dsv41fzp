---
id: P0017
title: 建立单行 BF16 动态激活量化的生产 scalar 与组合 parity
status: active
created: 2026-10-13
updated: 2026-10-14
knowledge: pending
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
- [ ] **构建、治理与复现**：三种 CPU preset 全套回归通过，无 skip、编译告警及非预期 sanitizer 报告；源码目录外及已提交干净检出复现。量化提交携带有效 `Oracle:`，`--strict` 与提交范围审计 E=0 W=0 I=0。不设置性能门槛。
- [ ] **知识与人工收尾**：增补 ACTQUANT 生产契约、BUILD 操作入口与 MODEL 上一级路由，遵守 [D0011](decisions/d0011-doc-single-source.md) 单一来源；用户评审实现、证据与知识处置后结项并同步 STATUS。

## 记录

### 范围与依据

- P0014 是量化上一级，P0013 是组合消费者；复用 P0010/P0011 整数基础。官方来源与阶段见 ACTQUANT §1、§3，生产契约见 §7，选择理由见 [D0013](decisions/d0013-act-quant-scalar.md)。
- 实现归 `df_quant`，组合仅在测试中；不建设 `src/aq` 调度框架，不修改既有 oracle、只读镜像与冻结数据。后续主线优先真实权重消费或具体模型子模块，另行立项。

### 最终证据

- 环境：[BUILD §1](../docs/BUILD.md#1-支持环境) ENV-A；分支 `p0017-act-quant-scalar`，基于 `c4aa7558df31d7a6c480db14c1030b9026e0949e`，VERSION 0.1.14，dirty=true。
- 2026-10-14 修复后各执行 `cmake --preset <preset> --fresh`、`cmake --build --preset <preset> --clean-first`、`ctest --preset <preset> --output-on-failure`：cpu-debug 26/26、cpu-release 26/26、cpu-asan 16/16 通过，无编译告警、测试 skip 或非预期 sanitizer 报告；harness 的故意非法读取按预期失败。
- 源码目录外全新构建：`cmake -S /root/dsv41fzp -B /tmp/p0017-review-external.iaMVPe -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON`，随后 `cmake --build <目录>`、`ctest --test-dir <目录> --output-on-failure --no-tests=error`，26/26 通过。此为未提交工作树的独立构建，不代替已提交干净检出。
- 定稿 parity 452、contract 2,602（含 trace 数据别名/尾部共享 330）、combination 20，均经 16 种 MXCSR 状态；amax 穷举 32,640 个幅值，两套元素构造各覆盖 65,280 个有限 BF16 编码。非有限保留 256 码四位置拒绝，另补普通/trace 组内中间位置 60 例。
- 新增描述对象独立/输出别名及共享间隙各 2 例；harness 的 103 个成功用例在输入改写故障下均被检出，共享间隙改写也被检出。故障传播均有同筛选无故障对照及原因核对，未知筛选失败；临时移除描述快照、改写输入或共享间隙均使对应回归失败。
- `PYTHONDONTWRITEBYTECODE=1 python3 tests/quant/act_quant_sensitivity.py`：10/10 数值变体检出，按验收条目顺序 mismatch 为 112、60、258、756、252、46、142、4、62、22，errors=0/skips=0；恢复与 pristine 重测通过，受保护工件哈希不变。
- `python3 tools/check_repo.py --strict` 为 E=0 W=0 I=0；`git diff --check` 通过；一次性 Markdown 链接/锚点检查 broken=0。复验日志 `scratch/p0017-review-verification/`，定向注错材料 `scratch/p0017-review-fixes/`，均为临时工件。
- 提交范围审计与已提交干净检出待本轮工件提交后执行。

### 评审处置

- 2026-10-13：用户同意范围并授权起草 P0017，登记 draft。
- 2026-10-14：用户接受含 D0013 的开工方案，授权转 active、实现及 ACTQUANT/BUILD/MODEL 必要增补。
- 2026-10-14：早期复核补齐 trace 数据缓冲区别名测试、修正尾部共享期望；共享存储改用 C++20 原始分配建立 implicit-lifetime 数组生命周期。
- 2026-10-14：用户授权处理本轮评审问题并减负可修改文档；trace 入口复制描述对象，contract 合并筛选并加无故障对照，成功路径增加输入与写范围外字节检查，补组内非有限用例。
- 2026-10-14：P 的契约、环境和验证边界改为章节引用，敏感性门禁留本项及 P0014，STATUS 仅留进度与阻塞。D0013 按 D0011 保留已接受原文，ACTQUANT 仍是生产契约的可维护正文。
- 2026-10-14：用户同意验收并授权按流程推进；本轮提交工件、完成干净检出与范围审计后结项并合入 main。

### 提交

尚未提交。

## 收尾

- 验收或停止依据：active；修复后技术复验通过，提交范围审计、已提交干净检出及最终人工评审尚未完成。
- 知识处置：pending；长期契约、命令及验证链分别归 ACTQUANT、BUILD、MODEL，选择理由归 D0013，验收证据留本项；充分性待人工评审。
- 剩余义务：上述验收与知识处置仍由本项承担，暂无移交项。
