---
id: P0012
title: 建立官方专家 FP8×FP4 分组点积与累加后乘 scale 的计算 oracle
status: completed
created: 2026-10-12
updated: 2026-10-13
knowledge: updated
docs: [docs/MODEL.md, docs/QUANT.md, docs/COMPUTE.md, docs/BUILD.md]
bench: []
decisions: [D0008]
followups: []
---

## 意图

接续 P0006、P0010 与 P0011，建立官方 FP8×FP4 阶段约束下的单输出 CPU 规范参照，覆盖块内点积、给定 scale 应用与跨块累加；交付按 [COMPUTE](../docs/COMPUTE.md) 独立验证的 Python 实现，供后续同契约 C++ scalar 对照。覆盖边界见 [MODEL §4](../docs/MODEL.md#4-精确性链条)。

## 验收

- [x] **开工前置决策**：D0008 经用户接受，明确归属、验证链位置、归约、scale 次序、累加顺序、舍入、融合、特殊值、输出边界与环境，区分源码事实、项目语义与待 GPU 仲裁项；MODEL §4 与 K-Q2 路由同步修订。
- [x] **一个计算路径、一个输出**：输入、分组与失败按 COMPUTE §2；固定每 32 元素一组，不支持 128 元素组，不扩展为矩阵算子。
- [x] **计算与舍入独立验证**：期望由独立精确算术逐步舍入构造；块内、scale 后、跨块累加与最终输出分别按位检查，不用 allclose。
- [x] **边界与布局**：相消、顺序敏感、RNE 中点、进位、次正规、上下溢、零符号、极端 scale；相邻不同 scale 与跨 128 边界错组用例；全部数据码点与 scale 编码边界，随机仅补充。
- [x] **测试敏感性与失败传播**：scale 错组、提前 BF16、跨块累加顺序三类注入被定向用例识别；错误或异常非零退出，无 skip。
- [x] **CPU 独立复现**：单条标准库命令，源码目录外与干净检出复现；接入 CPU CTest，不扩大 `cpu-asan`。
- [x] **知识与人工收尾**：契约归档 COMPUTE，MODEL 路由、BUILD 测试入口；用户评审后结项。

## 记录

### 范围与依据

- 依据 [D0008](decisions/d0008-expert-grouped-dot-reference.md)；官方阶段与待仲裁项见 [COMPUTE §1](../docs/COMPUTE.md#1-来源与参照边界)。
- 新增加法在计算模块内以整数有效数与显式 RNE 构造，不改写格式 oracle，不建通用软件浮点 API。
- 范围外：K-E1/K-E2、生产 C++、GEMM/GEMV 与专家 MLP、SIMD/CUDA、有损编码与 scale 生成、性能论断、端到端验证。

### 最终证据

- 开工前标准库探针（ENV-A）：全部 4,064 个有限 E4M3FN×E2M1 乘积均精确，元素乘法舍入不可作公开输入边界；构造块内顺序、scale 次序、跨块顺序、融合、提前 BF16 与负零阶段六类敏感向量，作为测试定向用例来源。
- 交付：`ref/compute.py` 普通入口与不可变阶段观测共享执行路径；`tests/quant/test_compute.py` 30 项，无 skip。
- 敏感性：scale 每四块共享、提前 BF16、跨块倒序在各自 16 项定向套件中分别造成 5/9/9 项数值失败，异常与 skip 为零；计算异常与结果篡改使普通及 marker 入口退出 1。
- 干净检出复现见收尾；CTest Debug/Release 各 16/16、`cpu-asan` 8/8。

### 评审处置

- 2026-10-12：用户确认固定 32 元素分组、排除 128 元素组；澄清本项不直接验收 K-E1/K-E2、不改 W4A16 规划；接受 D0008 并发布 COMPUTE。
- 环境审查发现 `struct.unpack('<f')` 承载的次正规值在 DAZ 下被冲刷；计算模块改用精确 binary64 构造与整数移位 RNE，不改契约与格式 oracle。四种舍入×四种 DAZ/FTZ 组合下负次正规链结果一致。
- 2026-10-12：用户评审通过，授权归档 BUILD 并推进结项。

### 提交

- `932ada0` draft，`a9aacab` D0008 与契约，`2a6b0c7` 实现，`2ed3e19` 结项；合并提交 `35c925f`，VERSION 0.1.11。
- 压缩前全文见提交 `b60526d`（依 [D0011](decisions/d0011-doc-single-source.md) 压缩）。

## 收尾

- 验收或停止依据：全部验收成立，2026-10-12 用户评审通过并授权按流程归档推进。实现提交 `2a6b0c76537a2d85546cb9541bf369ed83105f33` 的干净检出在源码目录外运行完整标准库验收 30/30 通过，从空 build 配置构建三个 CPU preset，Debug/Release 各 16/16、cpu-asan 8/8 通过；仓库全量及提交范围 strict 检查均 E=0 W=0 I=0，复现前后 `git status --porcelain` 均为空。方法及环境沿用上文，原始日志为 `scratch/p0012/clean-reproduction.log`，结果摘要为 `scratch/p0012/clean-reproduction.json`。据证据记录 completed，不以合并代替验收。
- 知识处置：updated；数值契约、来源与边界见 [COMPUTE](../docs/COMPUTE.md)，理由见 [D0008](decisions/d0008-expert-grouped-dot-reference.md)，参照路由见 [MODEL §4](../docs/MODEL.md#4-精确性链条) 与 [QUANT](../docs/QUANT.md)；实际入口、复现方法及测试边界已归档 [BUILD §7](../docs/BUILD.md#7-测试清单)，过程与验收证据保留本项，不重复契约正文。
- 剩余义务：本项无未完成交付或转交义务，followups=[]。生产 C++、K-E1 同路径参照与跨路径质量、K-E2 整数改写、SIMD/CUDA、官方 GPU parity 及端到端一致均为已明确排除的未来范围，另项决定，不据本项声称已验收。
