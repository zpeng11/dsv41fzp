---
id: P0027
title: 建立 FP8×FP8 块缩放单输出点积 CPU 参照
status: completed
created: 2026-10-16
updated: 2026-10-16
knowledge: updated
docs: [docs/MODEL.md, docs/QUANT.md, docs/ACTQUANT.md, docs/COMPUTE.md, docs/BUILD.md]
bench: []
decisions: [D0011, D0021]
followups: []
---

## 意图

接续 [P0026](p0026-routed-branch-scalar.md) 后向共享专家推进，先补齐其缺失的 FP8×FP8 块缩放计算参照。交付单输出 Python 普通与阶段观测入口及独立验证，供后续同契约生产 scalar 对照；复用既有格式与激活量化，不以逐元素反量化或 FP8×FP4 点积替代新计算契约。

## 验收

- [x] **开工前置决策**：[D0021](decisions/d0021-fp8-grouped-dot-reference.md) 经用户审议接受；[COMPUTE §8](../docs/COMPUTE.md#8-fp8-块缩放单输出点积) 明确数值、输出与环境边界，区分官方阶段、项目选择及设备待仲裁项，不继承 FP8×FP4 验收。
- [x] **单输出参照**：E4M3FN 激活与权重、给定 E8M0 scale，固定模型使用的 32 元素 K 分组；普通与 trace 共用核心，覆盖块内点积、两次 scale 应用、跨块累加及 FP32/BF16 输出，不扩展为矩阵 API。
- [x] **独立逐阶段期望**：独立精确算术与格点判定核对各阶段及正式输出，不调用被测计算或舍入 helper 生成期望；有限值、零与 Inf 按位比较，NaN 按分类比较，不用 allclose。
- [x] **边界与最小组合**：合法码点、scale 边界、相消、顺序敏感、RNE、次正规、上下溢及零符号；测试侧单行提取验证二维 scale、输出行跨块及尾块，直接消费既有 act_quant 实际输出。
- [x] **校验与环境**：类型、位宽、长度、分组、NaN 编码拒绝，输入不变，失败不返回部分结果；普通与 trace 在 ENV-A 定向验证完整环境保证及独立阶段期望，不以最终输出相同替代阶段检查。
- [x] **错误检出**：错误权重解码、二维 scale 错映射、移动/合并 scale、错误归约顺序、提前 BF16 均有同例独立正确基线及完成调用后的数值差异；异常或参数拒绝不算数值检出。
- [x] **增量复现**：标准库新增套件已在已提交干净检出、源码目录外通过，接入 Debug/Release CTest；缺依赖、错误、异常或 skip 非零退出。沿用旧组件验收证据，不重跑其全量套件，不扩大 sanitizer scope；严格范围审计及引用检查通过。
- [x] **知识与人工收尾**：用户审议实现、证据及知识处置通过；干净复现补证成立，按 D0011 单一来源归档并同步 STATUS，不以提交或合并替代验收。

## 记录

### 范围与依据

- 前置为 P0006/P0012/P0014；共享专家与权重布局只见 [MODEL §2.1](../docs/MODEL.md#21-权重分量格式表)，routed 停点见 [MOE §2](../docs/MOE.md#2-聚合与精度落点)，环境限 [BUILD §1](../docs/BUILD.md#1-支持环境) ENV-A。
- 官方来源、数值正文唯一归 COMPUTE §8，选择理由留 D0021，覆盖矩阵归 MODEL §4，操作入口归 BUILD §7。二维 scale 选行及量化组合仅在测试侧，不改旧 oracle 或冻结工件，不建立通用软件浮点层；本项不承接 P0005、CI 或 bench 工作流。

### 最终证据

- 实现 `de6e97e6af5fed925ecd4c622e155c09f868af36`，VERSION 0.1.24。`git clone --no-local --single-branch --branch p0027-fp8-grouped-dot-oracle` 至 `/tmp/p0027-clean-w2mg4nma/repo`，起止 dirty=false，参照/测试/CTest 接线指纹与用户审议版一致；全部命令从 `/tmp` 执行。
- `PYTHONDONTWRITEBYTECODE=1 python3 <checkout>/tests/quant/test_fp8_compute.py` 为 24/24。外置空 build 配置 Debug/Release 后，`ctest --test-dir <build> -R '^quant\.fp8_compute$' --verbose --output-on-failure --no-tests=error` 各 1/1，内部各 24/24，无 skip；无需 C++ build，未重跑旧组件全量套件。两种 sanitizer scope 只核对注册，均未新增本套件。
- 逐阶段覆盖全部 64,516 对合法 FP8 码点、舍入及特殊值边界；二维选行覆盖 N=35、K=96 的行 0/31/32/34，并直接消费既有量化输出。八模式 MXCSR × 两种状态位初值 × 普通/trace × FP32/BF16 的五组成功向量及拒绝共 384 次调用，完整环境不变；恢复后核对独立阶段与正式输出，四份快照增缩均计算前拒绝，runner 故障/skip/缺依赖负控通过。
- 七个临时源码变体加测试侧二维错映射均完成四类公共调用。FP32 正确→变体：权重当 FP4 解码 `41000000→00000000`；scale 移入乘积 `00000000→7fc00000`（合法 NaN 结果）；交换/合并 scale 各 `7f800000→40000000`；块内倒序 `00000000→3a800000`；跨块倒序 `00000000→3f800000`；提前 BF16 `3b800000→00000000`；二维错映射 `44400000→43400000`。同例独立正确基线及完整阶段证据随干净检出日志保留，不冒充生产矩阵验证。
- 干净检出 `check_repo.py --repo <checkout> --audit --range 92408b2..HEAD --strict` 为 E=0 W=0 I=0，链接与锚点 broken=0，差异检查通过。最终命令、日志及身份归 `scratch/p0027-fp8-grouped-dot-oracle/closure/`；本次压缩前全文见 `de6e97e`，不重复保留候选树证据。

### 评审处置

- 2026-10-16：用户先批准 P，再审议接受 D0021/COMPUTE §8 设计并授权实现。
- 2026-10-16：核心环境状态位污染与快照长度缺口、测试环境正式结果/变体语义/二维基线/runner 负控缺口均修正，独立复核无阻塞；旧 oracle 不变。
- 2026-10-16：用户人工审议通过改动及知识处置，授权按 D0011 提交合入；已提交干净复现和严格范围审计补证通过，据此记录结项。

### 提交

- `de6e97e`：参照实现、独立验收及单一来源归档，携带 `Oracle: tests/quant/test_fp8_compute.py`。
- 本次结项提交精简最终证据并同步 STATUS；版本递增由 `Merge P0027` 合并提交记录。

## 收尾

- 验收或停止依据：八项验收成立，用户人工审议通过；已提交干净检出新增套件、Debug/Release 及严格范围审计补证通过，不以合并替代验收。覆盖边界以 MODEL §4 为准。
- 知识处置：updated；COMPUTE §8、MODEL §4 与 BUILD §7 分别归档数值契约、覆盖边界及操作入口，选择理由留 D0021；用户已审议，按 D0011 不重复规范正文。
- 剩余义务：无本项阻塞或移交义务，followups 为空。生产 FP8 scalar、共享专家与完整 MoE 等方向另立 P，范围以 MODEL §4 为准；P0005 阻塞不解除。
