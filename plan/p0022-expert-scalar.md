---
id: P0022
title: 建立单 token FP4 完整专家的生产 CPU scalar 与真实三矩阵闭环
status: active
created: 2026-10-15
updated: 2026-10-15
knowledge: updated
docs: [docs/EXPERT.md, docs/BUILD.md, docs/MODEL.md]
bench: []
decisions: [D0011, D0014, D0015, D0016, D0017]
followups: []
---

## 意图

接续已结项 [P0021](p0021-swiglu-bridge-scalar.md)，将已有激活量化、专家点积与 SwiGLU 生产组件组合为可独立调用的完整专家入口。调用方给定一个 token、一个已选 FP4 专家的三张矩阵和可选外部路由权重，生产 C++ 完成专家计算，逐阶段符合 [P0020](p0020-expert-oracle.md) 的 CPU oracle，并以一个真实专家的完整权重验证布局与数值闭环。

成功边界为单 token、单个已选专家，不扩为完整 MoE；数值契约引用 [EXPERT §1～§3](../docs/EXPERT.md#1-接口与输入)，不重新定义量化、点积或非线性语义。

## 验收

- [x] **开工前置决策**：[D0017](decisions/d0017-expert-scalar.md) 经用户审议接受，明确完整 API、独立阶段工作缓冲、共享量化、别名与校验次序；参数拒绝全存储不变，运行中 hidden 非有限拒绝允许 scratch/trace 有效前缀改写而完整 out 不变。本项据此转 active。
- [x] **完整生产入口**：普通 C++ 消费者仅依赖公共头与 `df_quant` 即可调用。按 EXPERT §2 共享一次输入量化供 W1/W3 消费，直接调用已验收 SwiGLU，再量化 hidden 并执行 W2；公共结果为 BF16。无权重与显式权重保持既有区别，不生成或归一化权重；调用方持有工作存储，不动态分配，不复制数值核心。
- [x] **逐阶段组合 parity**：完整入口直接对照 `ref/expert.py::expert`/`trace_expert`，核对两次量化编码与 scale、W1/W3 BF16、bridge 各阶段及 W2 全部输出，沿用既有位比较与 NaN 分类规则。覆盖单组、多组、非方阵、不同矩阵与行 scale、首末输出及两种权重入口；生产后续阶段必须消费生产前一阶段结果。普通与必要的内部观测共用执行核心，不能仅凭最终 BF16 相同认定组合正确。
- [x] **专家级存储与失败**：依新 D 验证三矩阵形状、计数溢出、长度、容量、别名及内容错误，包括 W2 末行非法 scale；全部外部元数据先于内容读取，预检拒绝先于写入。另覆盖线性 Inf 被钳位恢复、hidden 非有限在第二次量化边界拒绝及最终 W2 非有限合法输出。检查正式输出、各工作缓冲、只读输入与范围外字节的约定状态；sanitizer 和边界探针遵守合法 C++ 对象/span 前提。
- [x] **组合错误检出与环境**：分支接反、权重行或 scale 步长错误、遗漏第二次量化、路由权重错放、提前或绕过 BF16 落点等定向生产变体均由数值或阶段 mismatch 检出，并有同用例正确基线；异常、协议故障及存储拒绝不充当数值检出。按 BUILD 的 MXCSR 模式与初始状态覆盖普通、内部观测、接受及拒绝路径，结果环境独立且完整 MXCSR 不变。
- [x] **真实三矩阵闭环**：固定下述专家的完整 W1/W3/W2，以确定性有限人工 BF16 输入通过本项生产入口执行，全部阶段与 P0020 oracle 对照。复用既有结构校验与有界读取，独立核对尺寸、行布局与实际载荷；离线报告可追溯环境、checkpoint 快照、源码及执行器身份、输入/载荷指纹、读取区间、命令与比较结果。缺权重、读取或执行器失败均不得 skip 或计为通过，不引入生产加载器。
- [x] **增量复现与构建**：新增合成验证不依赖权重或网络，接入既有 Debug/Release 及适用 ASan 体系并覆盖普通消费者链接；已提交干净检出、自源码目录外复现新增验收与受实际改动影响的既有回归，无 skip、编译告警或非预期报告。既有已验收组件证据直接继承，不另立重复的 SiLU 全域证明或无差别重跑旧验收。量化提交携带有效 `Oracle:`，严格检查、范围审计及修改页引用检查通过；不设置性能门槛。
- [ ] **知识与人工收尾**：按 D0011 将完整生产契约归 EXPERT、操作与复现入口归 BUILD、验证链增量归 MODEL §4，方案理由留新 D。用户审议实现、证据和知识处置后方可结项，同步本项与 STATUS；不以组件已验收或代码合入代替本项验收。

## 记录

### 范围与依据

- 前置：[P0017](p0017-act-quant-scalar.md) 的生产激活量化、[P0019](p0019-expert-linear-scalar.md) 的单矩阵层、P0020 完整 oracle 与 P0021 bridge；共享量化、窄接口与复用取舍见 D0017。
- 环境限 [BUILD §1](../docs/BUILD.md#1-支持环境) ENV-A。完整 API、独立工作缓冲与分层拒绝语义由已接受 D0017 决定，已有 D0014/D0016 保留历史结论。
- 固定真实验收：MODEL 的 [ckpt] 快照中 `layers.0.ffn.experts.0` 的完整 `w1/w3/w2`，不裁剪 N/K、不扫描其他专家；输入沿用现有确定性人工输入方法。离线命令与无权重合成测试分离，不进入默认 CTest；只证明真实权重布局与 CPU 数值链，不证明真实模型激活、质量或 GPU 等价。
- 范围外：router/top-k/权重生成、多专家聚合、共享 FP8 专家、batch、prefill/decode 调度、生产 checkpoint loader/pack/缓存、SIMD/CUDA/融合、性能测量、attention/KV/mHC/Engram、CI/bench 工作流及 P0005 硬件资格。不新增完整专家 FP32 输出或通用 tensor/计算图/内存管理框架。
- 知识去向：EXPERT 为完整生产契约唯一正文，BUILD 为环境与命令入口，MODEL §4 为覆盖边界；D0017 只留选择与理由，本项只留验收证据与归档指针。

### 最终证据

- ENV-A；最终实现 `dde35d99017aa64b9b560514cae26c5a05e053f2`，VERSION 0.1.19。生产/执行器/工具工件 `b0073f9b2861331558218e150005d651cb15e4f0` 经 `git clone --no-local --single-branch --branch p0022-expert-scalar` 检出至 `/tmp/p0022-clean-EuevPg/repo`，dirty=false；后继提交仅增加敏感性脚本的仓库导入路径，其余文件逐项 diff 确认未变，沿用下述证据，不重复重型 oracle。
- 在该检出构建三个 CPU preset；从源码目录外以 `ctest --test-dir <checkout>/build/<preset> -R '^(quant\.expert\.(parity|contract|harness|consumer)|tools\.ckpt_(expert_mlp|expert|linear)|quant\.(codec|dequant|compute|act_quant|linear|swiglu)\.(contract|harness)|bb\.compile_flags|quant\.compile_flags)$' --output-on-failure --no-tests=error -j 4` 完成 Debug 20/20、Release 20/20、ASan 17/17。无编译告警、skip 或非预期报告，sanitizer 环境按 BUILD §3；未重跑旧 oracle、旧 parity 全域或旧真实权重验收。
- 新专家 parity 16 例、contract 44 例各覆盖 16 种 MXCSR 状态，另含生产共用计数/字节区间探针、消费者、四项 harness；新工具合成测试 15/15。runtime 固定 fixture 核验实际 hidden 全部为 BF16 NaN、阶段前缀已写、第二量化缓冲未写及完整 out 不变，并有提前拒绝负控；这不是独立调用计数探针。
- 同一干净工件用 BUILD §7 的完整专家命令重复传入 Debug/Release 两个 `--exe`，固定快照 `2cba9e42aa026125f3ed06c6d98c1db82f7ca027`，四个输入/权重组合只生成一次各自 oracle trace。完整 w1/w3 为 2304×5120、w2 为 5120×2304（尺寸来源 MODEL §2），共 407168 项比较通过：ordinary 157504、trace 249664，有限/零/Inf 逐位、NaN 按类。独立读取 safetensors 头与 seek/read 复核 19456 段区间、18800640 字节及三矩阵载荷指纹；两个执行器的阶段结果与自身 SHA256 复核一致。
- 导入修复提交另经干净 clone 至 `/tmp/p0022-fixed-Z8JXBR/repo`，从 `/tmp` 运行脚本绝对路径：六个生产组合变体 6/6 均完成调用后以阶段数值 mismatch 检出，每项同用例恢复基线、最终 pristine 与受保护源码哈希通过；崩溃/协议故障不计数值证据。该最终检出的 `check_repo.py --repo <checkout> --range 9300229..HEAD --audit --strict` E=0 W=0 I=0，范围 diff 与修改页链接/锚点通过，两个检出运行后仍干净。
- 日志、完整命令、执行器指纹、真实报告及独立读取复核留 `scratch/p0022-final/`（临时工件，不入库）。未触碰 oracle、只读镜像、冻结数据、AGENTS 或 VERSION；不作性能、官方 GPU、真实模型激活或质量论断。

### 评审处置

- 2026-10-15：用户同意单 token、单个已选 FP4 专家生产 scalar、逐阶段 parity 与固定真实专家三矩阵闭环的范围，授权起草 P0022，登记 draft。
- 2026-10-15：用户通过草案与含 D0017 的开工方案，授权实现、必要文档增补、增量验证和短命分支本地提交；D0017 转 accepted，本项转 active，最终结项保留人工审议。
- 2026-10-15：独立评审指出短容量掩盖别名、工具 adapter 隐含重算 oracle、超宽 NaN、读取区间自证、混型对象构造、共享间隙与运行中提前拒绝假阳性；修复为容量充分的合法构造、正式预计算接口与调用负控、位宽前置、独立行区间、共用字节探针、有效写并集及实际 hidden 阶段诊断，限定复核无剩余阻塞。

### 提交

- `7b5823a`：接受 D0017、登记 P0022 active 并同步 STATUS。
- `b0073f9`：完整生产专家、公共消费者、组合验收、真实三矩阵工具及 EXPERT/BUILD/MODEL 归档；携带 `Oracle: tests/quant/test_expert_parity.py`。
- `dde35d9`：修复敏感性脚本从源码目录外运行的仓库导入路径；生产源码、执行器、parity 与真实工具未变，携带同一 `Oracle:`。

## 收尾

- 验收或停止依据：实现与机械证据已到位，见最终证据；本项保持 active，等待用户审议实现、证据及知识处置后决定结项，不以独立静态评审替代人工验收。
- 知识处置：updated；[EXPERT §6](../docs/EXPERT.md#6-生产完整专家) 归档完整生产契约，BUILD §7 归档操作入口，MODEL §4 归档验证链边界，D0017 保留选择理由；充分性待用户评审。
- 剩余义务：最终人工验收及结项处置待办；无已识别实现阻塞或已登记移交义务，followups 为空。
