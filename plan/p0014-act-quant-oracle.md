---
id: P0014
title: 建立官方动态激活量化（BF16→E4M3FN+UE8M0，每 32 元素）的 CPU 规范参照
status: completed
created: 2026-10-13
updated: 2026-10-13
knowledge: updated
docs: [docs/ACTQUANT.md, docs/MODEL.md, docs/QUANT.md, docs/COMPUTE.md, docs/BUILD.md]
bench: []
decisions: [D0010]
followups: []
---

## 意图

接续 [P0012](p0012-expert-grouped-dot-oracle.md) 与 [P0013](p0013-expert-grouped-dot-scalar.md)，为官方 `act_quant(x, 32, "ue8m0", float8_e8m0fnu)` 建立单行 CPU 规范参照，输出可直接作为 COMPUTE 的 `a_codes`/`scale_a`。契约见 [ACTQUANT](../docs/ACTQUANT.md)，取舍见 [D0010](decisions/d0010-act-quant-reference.md)，覆盖边界见 [MODEL §4](../docs/MODEL.md#4-精确性链条)。

## 验收

- [x] **开工前置决策**：用户审议通过 D0010 与本项范围后转 active。
- [x] **一个入口、一行输入**：`ref/act_quant.py` 按 ACTQUANT §2 的输入域、输出与失败契约；不改写既有 `ref/` 模块。
- [x] **量化与舍入独立验证**：期望不调用被测 helper；scale 指数对全部有限 BF16 amax 与精确 ceil 逐一比较并证明 clamp 不可达；全部有限码点在边界指数下量化，非有限全部拒绝；每组与每元素阶段位逐一检查，不以最终编码替代。
- [x] **边界与布局**：E4M3FN 中点及两侧、次正规边界、448 上界、零符号、`448×2^k` 指数、低于下限与全零组、低于下限的非零组、BF16 次正规输入、商的 binary32 次正规与下溢中点、多组与跨 128 边界错组用例；随机仅补充。
- [x] **与计算参照组合**：输出喂给 `ref/compute.py::expert_grouped_dot` 与独立期望一致；只作测试，不新增组合入口。
- [x] **测试敏感性与失败传播**：指数 floor/round、四组共享 scale、FP8 截断或中点远离零、负零规范化、去掉下限、BF16 次正规冲零、商次正规冲零、商中点远离零，各由指定用例数值或阶段位失败检出；仅异常或进程故障不作证据；错误或异常非零退出，无 skip。
- [x] **CPU 独立复现**：单条标准库命令，接入 Debug/Release CTest，不扩大 `cpu-asan`；源码目录外与干净检出复现；审计 E=0 W=0 I=0。
- [x] **知识与人工收尾**：契约归档 ACTQUANT，QUANT/COMPUTE/MODEL 路由，BUILD 测试入口；用户评审后结项。

## 记录

### 范围与依据

- 官方来源：`kernel.py:22–38,41–124`，`model.py:27–30,181–198,841–851`。
- BF16 解码只复用 `bf16_bits << 16` 位映射，在新模块内以整数位域构造，不调用值级 `decode_bf16`/`fp32_from_bits`。
- 范围外：inplace 与 KV 量化、`fp4_act_quant`、128 元素组与非幂 scale、生产 C++、SIMD/CUDA、矩阵接口、K-E1/K-E2、端到端验证。

### 最终证据

- 开工前探针（ENV-A）：`fl32(1/448)` = `0x3B124925`；全部正有限 BF16 amax 下源码阶段指数与精确 ceil 无一不同，范围 −22～120；下限 `1e-4` 的具体浮点表示不改变指数。DAZ 下既有 `decode_bf16` 把 `0x0001/0x8001` 冲为零，故新模块不复用值级解码。
- 数值：34 项必选测试，无 skip。32,640 个非负有限 BF16 amax 逐组比较全部阶段；65,280 个有限码在自身量级组与最大 amax 锚点组中量化；256 个非有限码在 32 个位置被两个入口拒绝；六个向量接入计算参照（含一个有意全零编码向量）。
- 敏感性门禁：数值与阶段比较抛 `NumericMismatch`，校验顺序哨兵抛 `ValidationOrderMismatch`，marker 分别记为 `numeric-fail`、`validation-fail`、`error`；指定检测用例必须为数值类别，普通 `AssertionError`、其他异常、混合异常或 skip 一律拒绝。十个数值变体均由指定用例检出、恢复后通过；删除下限由全组 `2^-15` 检出（全组 `2^-14` 不能区分）。另以“逐组校验并计算”变异把坏内容放在后续组，证明完整校验先于任何组计算。精确倒数替换不列为必选变体（数值事实见 ACTQUANT §3）。
- 环境：八模式 MXCSR 下普通与 trace 一致，接受与拒绝路径 MXCSR 不变；源码 AST 检查整数数值与依赖边界。
- 干净检出：`git clone --no-local` 于 `d522379`，源码目录外 `test_act_quant` 34/34、`test_formats` 19/19、`test_compute` 30/30；`cpu-debug`/`cpu-release` 零告警、各 21/21；构建后克隆仍干净。`--strict` 与 `--range main..HEAD --strict` E=0 W=0 I=0。

### 评审处置

- 2026-10-13：复评补齐契约缺口：商为一次 binary32 RNE、渐进下溢并保号；纠正“低于下限的非零组被清零”的错误表述。用户审议通过 D0010，转 active，授权实现。
- 测试评审发现第一组坏内容无法检出“完整校验先于计算”违约，新增后续组变异；加固 AST 增量除法/幂通道与组合向量非退化断言。
- 用户评审授权修复：敏感性门禁改为上述分类；非一维 `memoryview` 在长度与内容校验前抛 `TypeError`；精确倒数替换的无差异范围限定为指数、scale 与最终编码。
- 2026-10-13：用户人工评审通过全部改动并授权提交结项。

### 提交

- `aaaffa1` D0010，`a63eafd` 实现与 CTest，`80216f8` 契约归档与路由，`d522379` 证据与 STATUS，`bf3fa16` 结项；合并提交 `8d25cf8`，VERSION 0.1.13。
- 压缩前全文见提交 `b60526d`（依 [D0011](decisions/d0011-doc-single-source.md) 压缩）。

## 收尾

- 验收或停止依据：七项验收全部成立。实现与测试见提交 `a63eafd`、知识归档见 `80216f8`、干净克隆复现于 `d522379`；2026-10-13 用户人工评审通过。未验证 SIMD/CUDA、K-X1 kernel、K-E1/K-E2、官方 GPU、端到端一致及非 glibc/非 Linux x86_64 环境。
- 知识处置：updated；契约、来源、阶段观测、失败、环境与验证边界归档 [ACTQUANT](../docs/ACTQUANT.md)，路由与测试入口见 QUANT/COMPUTE/MODEL/BUILD；选择理由留 D0010，过程与验收证据留本项。
- 剩余义务：无。同契约生产 C++ scalar、SIMD/CUDA、官方 GPU 仲裁、K-E1/K-E2 与端到端验证不属于本项，待立项时另建 P 并引用本项。
