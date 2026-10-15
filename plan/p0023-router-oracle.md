---
id: P0023
title: 建立单 token MoE 路由的 CPU 数值参照与独立验证
status: active
created: 2026-10-15
updated: 2026-10-15
knowledge: updated
docs: [docs/ROUTER.md, docs/MODEL.md, docs/BUILD.md]
bench: []
decisions: [D0011, D0015, D0018]
followups: []
---

## 意图

接续已结项 [P0022](p0022-expert-scalar.md)，补齐完整 FP4 专家入口上游的专家选择与权重生成。以单 token 激活和模型 gate 参数为输入，建立覆盖 gate 投影、打分、选择及归一化的 CPU 慢参照，为后续生产 router 提供独立验证的上一级。

成功边界为从单 token 激活得到有独立数值依据的专家编号与 FP32 路由权重，不止于给定分数后的 top-k；不实现完整 MoE。模型参数引用 [MODEL §1](../docs/MODEL.md#1-概览) 与 [§3.5](../docs/MODEL.md#35-路由srccore)，不另建通用路由配置体系。

## 验收

- [x] **开工前置决策**：[D0018](decisions/d0018-router-reference.md) 经用户审议接受，本项转 active；明确 gate 累加与舍入、softplus/sqrt 阶段、top-k 平局和返回顺序、权重归约、失败及环境边界，不套用专家点积或推定官方 GPU 规则。
- [x] **完整 Gate 参照**：`ref/router.py` 按 [ROUTER §1～§4](../docs/ROUTER.md#1-接口与输入) 完成单 token Gate；普通与 trace 共用核心、输入不变，输出编号和 FP32 权重原始编码。
- [x] **逐阶段独立验证**：以独立精确算术或认证区间建立投影、非线性、选择及权重的期望链，不调用被测数值 helper，不直接用 Torch/libm 输出充当真值。对全部返回编号、权重和约定 trace 阶段逐项比较；期望的后续阶段消费期望的前阶段结果，不以最终编号或权重偶合代替阶段验证。
- [x] **边界与失败**：覆盖投影抵消及累加次序敏感输入、非线性极端值与舍入边界、top-k 截止处近邻和平局、bias 改变选择但不参与权重、文本/图像 bias 分支、归一化防零与下溢。覆盖形状、长度、编码及内容拒绝，按新 D 检查输入不变、失败传播及普通/trace/拒绝路径的环境保证；不以随机样本替代定向边界。
- [x] **错误检出**：偏置进入权重、归一化与选择次序错误、权重取错专家、遗漏或错误缩放，以及违背已定投影/舍入/top-k 规则的定向错误链，均由指定用例的编号或阶段数值 mismatch 检出，并有同用例正确基线；异常、协议故障或参数拒绝不充当数值检出。
- [x] **专家接口衔接**：少量无权重文件的合成用例将生成的专家编号与 FP32 权重接入既有 `ref/expert.py` 和 P0022 生产专家执行器，核对所选矩阵、权重对应关系及专家结果。复用既有专家契约和验证入口，不修改其数值语义，不新增多专家调度或聚合入口，不据此认定完整 MoE 已验收。
- [x] **增量复现**：新增验证限 BUILD 的 ENV-A，使用 Python 标准库、无需 GPU、网络或 checkpoint，提供源码目录外可执行命令并接入 Debug/Release CTest。已提交干净检出复现本项及受实际改动影响的回归，无 skip 或非预期报告；严格检查、提交范围审计与修改页引用检查通过。继承 P0022 及既有组件证据，不无差别重跑旧验收，不设置性能门槛。
- [ ] **知识与人工收尾**：按 D0011 将路由 CPU 契约归专页、选择理由留新 D、验证链边界归 MODEL §4、环境及复现入口归 BUILD。用户审议实现、证据及知识处置后方可结项，同步本项与 STATUS；不以公式复刻、测试通过或代码合入单独代替验收。

## 记录

### 范围与依据

- 前置：P0022 的完整生产专家及 [P0020](p0020-expert-oracle.md) 的专家参照。二者消费外部权重，不负责生成权重或选择专家，边界见 [MODEL §4](../docs/MODEL.md#4-精确性链条)。既有参照不因新实现方便而改写。
- 官方来源：只读镜像 `third_party/reference/inference/model.py:792–827` 的 Gate；`841–851` 的专家内权重位置；`889–904` 的 MoE 连接关系。按主干 MoE 配置验证，文本/图像 bias 分支仅表示 token 属性，不引入视觉计算。DSpark 特殊配置与 softmax/sigmoid 等其他打分模式不在本项验收范围。
- CPU 数值取舍见 D0018，不绑定宿主 Torch/libm 或替官方 GPU 确定归约与平局行为。[D0015](decisions/d0015-expert-oracle.md) 仅作建立独立非线性参照的先例，不直接决定本项 softplus/sqrt 契约。
- 范围外：生产 C++ router、真实 checkpoint Gate 闭环、共享 FP8 专家、多专家聚合/分布式归约、完整 MoE、batch、调度、加载器/pack/缓存、SIMD/CUDA/融合、性能、官方 GPU parity、模型质量、attention/KV/mHC/Engram、CI/bench 工作流及 P0005 硬件资质。
- 实施工件为 `ref/router.py` 与 core 域测试；API 与数值边界归 [ROUTER](../docs/ROUTER.md)，选择理由留 D0018，MODEL 只补模型依据与覆盖边界，BUILD 只补操作入口，本项只留意图、过程和证据。

### 最终证据

- ENV-A；最终工件 `17036aad01434142fc9ddcf9c38a2827495ae2fb`，VERSION 0.1.20。`git clone --no-local --single-branch --branch p0023-router-oracle` 干净检出至 `/tmp/p0023-clean-ba5xnmu5/repo`，起止 dirty=false；以下命令均从 `/tmp` 执行。
- 对该检出配置 `cpu-debug`/`cpu-release` 并构建 `df_quant_exec`；`ctest --test-dir <checkout>/build/<preset> -R '^core\.router(\.expert)?$' --verbose --output-on-failure --no-tests=error -j 2` 各 2/2。内部路由 24/24、接口 5/5，无 skip、构建告警或非预期报告；绝对路径单跑路由及分别指定双配置执行器的接口命令也通过。ASan 仅 configure 和枚举，确认新增路由注册为零，未运行旧 ASan 验收。
- 独立 Fraction/格点、Taylor 指数与对数级数期望检查全部 trace 阶段；覆盖投影抵消、逐乘/逐加溢出和次正规、非线性边界、双舍入、top-k 截止/平局、两种 bias、EPS/极小权重。九种显式错误路由模型以编号或阶段 mismatch 检出，不宣称生产变体门禁。
- 首次乘法时注入输入改写，覆盖普通/trace × 文本/图像四组合；容量匹配的未用 bias 非有限优先于合法投影溢出拒绝。环境覆盖 16 种控制模式 × 2 种初始 sticky 状态，普通/完整 trace、参数拒绝与两种运行中溢出均保持约定；Decimal context 含完整 flags 前后一致。认证预算不足与异常传播、期望链不调用被测数值入口均有负控。
- 接口以合成不同三矩阵验证文本/图像各选中专家的实际编号和权重消费，共 24 个生产专家 ordinary/trace 请求；专家阶段对照既有独立期望与参照。错矩阵/错权重显式模型完成后阶段和最终 BF16 不同；执行器缺失、不可执行或假协议均失败，未建立聚合入口。
- 干净检出严格检查、`--audit --range d48d67c..HEAD --strict` 均 E=0 W=0 I=0；主树 diff 与修改页 114 项本地链接/锚点检查通过。数值、独立验证及文档三路审阅与最后负控修复复核无剩余阻塞。命令、构建/测试日志、summary 在 `scratch/p0023-router-oracle/closure/`，临时复现脚本在其上级目录。
- 只复现 P0023 新增范围；继承 P0022 和既有组件，不重跑旧 oracle、全域 parity 或真实权重验收。未改既有数值实现、执行器、只读镜像、冻结数据、AGENTS 或 VERSION。覆盖边界仍以 MODEL §4 为准，等待用户人工审议。

### 评审处置

- 2026-10-15：用户同意单 token 路由 CPU oracle、独立验证及少量专家接口衔接的边界，授权起草 P0023，登记 draft。
- 2026-10-15：用户批准含 D0018 的开工方案，授权短命分支实现、必要知识归档、增量验证与本地提交；D0018 accepted，本项 active，最终结项与合入保留人工审议。
- 2026-10-15：独立审阅加固首次乘法前校验与快照、完整 Decimal flags 及参数/运行中拒绝环境覆盖；新增低精度认证负控造成病态大整数，终止该轮并改为认证预算耗尽，最终干净检出通过，未改变 oracle 数值。

### 提交

- `540a6c6`：接受 D0018，登记 P0023 active 并同步 STATUS。
- `b52078a`：路由参照、独立验证、专家接口与 ROUTER/MODEL/BUILD 归档；携带 `Oracle: tests/core/test_router.py`。
- `17036aa`：认证失败负控改为预算耗尽，避免低精度测试病态资源消耗；oracle 数值未变，携带同一 `Oracle:`。
- 本次证据提交回填最终增量复现并同步 STATUS，保持 active 等待人工验收。

## 收尾

- 验收或停止依据：技术交付与增量证据已到位，见最终证据；最后“知识与人工收尾”尚待用户审议，状态保持 active，不以测试通过代替结项。
- 知识处置：updated；[ROUTER](../docs/ROUTER.md) 为契约正文，MODEL §3.5/§4 为模型依据与验证边界，BUILD §7 为复现入口，D0018 保留理由；充分性待用户评审。
- 剩余义务：本项仍须人工验收及后续结项/合入授权，当前无移交项，followups 为空。生产 router 与完整 MoE 另立 P，不重开已结项专家工作。
