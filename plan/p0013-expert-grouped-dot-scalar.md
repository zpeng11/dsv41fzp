---
id: P0013
title: 建立专家分组点积的生产 scalar 基线
status: completed
created: 2026-10-12
updated: 2026-10-13
knowledge: updated
docs: [docs/COMPUTE.md, docs/BUILD.md, docs/MODEL.md]
bench: []
decisions: [D0009]
followups: []
---

## 意图

接续 [P0012](p0012-expert-grouped-dot-oracle.md)，按 [COMPUTE](../docs/COMPUTE.md) 交付单输出专家分组点积的生产 C++ scalar 入口，对照 `ref/compute.py` 逐阶段验证同一生产执行路径。成功标准：FP32/BF16 结果的有限值（含带符号零）与 Inf 按位一致、NaN 按分类一致，且与调用线程浮点环境无关。取舍见 [D0009](decisions/d0009-compute-scalar-integer-path.md)，覆盖边界见 [MODEL §4](../docs/MODEL.md#4-精确性链条)。

## 验收

- [x] **生产入口与失败契约**：`include/df/quant/compute.h`，失败类按 D0009；每类拒绝按两种输出验证输出不变；短缓冲区在 `cpu-asan` 下验证校验次序；执行器协议错误不计为库拒绝。
- [x] **整数构造数值路径**：逐条实现 COMPUTE §3，全程定宽整数、不读写浮点环境；乘 scale 前分流非有限值；舍入不依赖输入性质。
- [x] **逐阶段计算 parity**：经私有观测对照 `trace_expert_grouped_dot` 的全部阶段；4,064 码点对、scale 编码边界、多块、128 边界、次序敏感、舍入与特殊值；Inf 保留、块贡献 −0、NaN 传播定向证明。
- [x] **定宽整数加法**：同一生产 helper 对照 `ref/compute.py::_add_fp32`，覆盖大指数差、中点、sticky、近相消、进位、次正规、零与特殊值；`cpu-asan` 无 UBSan 报告。
- [x] **数值环境无关**：公开 parity 与加法专项在八模式 MXCSR 下一致、不陷入、MXCSR 不变。
- [x] **测试敏感性与失败传播**：十类注入（scale 每四块共享、提前 BF16、跨块倒序、−0 规范化、非有限误送乘法、丢失 sticky、nibble 颠倒、平衡树归约、scale 交换、scale 折叠）各由指定定向用例数值失败检出；harness 覆盖执行器故障与 trace 形状。
- [x] **构建、治理与复现**：三个 preset 通过，compute 进入 `cpu-asan`；`Oracle:` 与审计通过；源码目录外与干净检出复现。
- [x] **知识与人工收尾**：COMPUTE §6、BUILD §7、MODEL §4 增补；用户评审后结项。

## 记录

### 范围与依据

- 前置 P0012 completed；复用 P0010/P0011 的 codec、有限乘法、BF16 舍入、执行器协议与 MXCSR 矩阵。
- 范围外：矩阵/批量接口、K-E1/K-E2、SIMD/CUDA、性能论断、有损编码与 scale 生成、端到端验证。

### 验收证据

- 环境：ENV-A；干净克隆 `82b89b7` 自源码目录外三个 preset 结果相同，克隆保持干净。
- 套件：`quant.compute.parity` 309 个请求；`quant.compute.contract` 104 个请求（86 拒绝，含 22 加毒元数据与 24 偏移重叠；18 成功，含 16 恰好相邻与 2 个 cap=1），`ref/` 同类输入抛 `ValueError`；`quant.compute.add` 9 组定向加 20,000 对随机；`quant.compute.harness` 7 项。parity/contract/add 在八模式 MXCSR 下以状态位清空与置满各跑一遍。
- 回归：`cpu-debug`/`cpu-release` 各 20/20、`cpu-asan` 12/12，零告警；`--strict` 与 `--range main..HEAD` E=0 W=0 I=0。
- 敏感性检出用例：四块共享 → `layout.group128_sensitive.fp32`；提前 BF16 → `rounding.bf16_tie_up.fp32`；跨块倒序 → `order.cross_block_sensitive.fp32`；−0 规范化 → 仅 `rounding.underflow_negative_zero.trace` 与一个随机 trace；非有限误送、scale 交换、折叠 → `order.scale_order.fp32`；丢失 sticky → `add.sticky`；nibble 颠倒 → `layout.nibble_order.trace`；平衡树 → `order.tree_sensitive.fp32`。

### 评审处置

- 2026-10-12：用户选定整数构造扩展到加法，否决硬件 FP32 加法加 MXCSR 纪律；接受 D0009，转 active。
- 实现发现：加法初稿 39 位对齐使 sticky 永不影响结果、丢失 sticky 注入不可检出；改为 3 位对齐加 jamming 再 RNE，论证在 helper 函数级注释。
- 评审以变异体证明四处测试盲区（BF16 输出越界不可见、内容先于元数据读取、只拒绝同起点重叠、清除状态位）；`82b89b7` 改为精确长度分配、ASan 加毒、偏移重叠与 `--status=set`，四个变异体均被检出。生产代码无误。
- 2026-10-13：用户人工评审通过，同意结项并合入。

### 提交

- `a3bd6c5` D0009，`078ba1c` 实现，`82b89b7` 测试盲区修复，`564fd2d` 知识归档，`796bc05` 结项；合并提交 `7eef3f2`，VERSION 0.1.12。
- 压缩前全文见提交 `b60526d`（依 [D0011](decisions/d0011-doc-single-source.md) 压缩）。

## 收尾

- 验收或停止依据：八项验收全部成立。技术证据见上方验收证据（提交 `078ba1c`、`82b89b7`，干净克隆复现于 `82b89b7`）；知识归档于 `564fd2d`；2026-10-13 用户人工评审通过。未验证 SIMD/CUDA、K-E1/K-E2、官方 GPU、端到端一致及非 GCC/非 Linux x86_64 环境。
- 知识处置：updated；经用户确认，生产入口的表示、失败、数值与环境、验证边界归档 COMPUTE §6，§5 改为快路径只对照 §6；测试入口与 `cpu-asan` 范围归档 BUILD §7；MODEL §4 链路图更新。选择理由留 D0009，过程与证据留本项。
- 剩余义务：无。同契约快路径、K-E1/K-E2 与官方 GPU 验证尚未立项，不属于本项义务。
