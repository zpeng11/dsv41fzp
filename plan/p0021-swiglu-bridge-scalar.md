---
id: P0021
title: 建立单 token SwiGLU 非线性桥的生产 CPU scalar 与逐阶段 parity
status: active
created: 2026-10-15
updated: 2026-10-15
knowledge: pending
docs: [docs/EXPERT.md, docs/BUILD.md, docs/MODEL.md]
bench: []
decisions: [D0015, D0016]
followups: []
---

## 意图

接续已结项 [P0020](p0020-expert-oracle.md)，为 [EXPERT §2](../docs/EXPERT.md#2-运算与舍入) 已独立验证的单 token SwiGLU 桥建立同契约生产 C++ scalar：消费等长 BF16 gate/up，执行非对称钳位、正确舍入 SiLU、逐次 FP32 RNE 乘法、可选 route weight 与 BF16 落点，为后续完整 FP4 专家生产实现提供已验收的非线性上一级。

本项只补非线性桥，不把已有专家线性 primitive 误作 W1/W3 双支路或完整专家实现。成功以生产入口及其逐阶段结果直接对照 `ref/expert.py::swiglu`/`trace_swiglu` 成立为准，不以最终 BF16 偶合替代中间数值验证。

## 验收

- [x] **开工前置决策**：[D0016](decisions/d0016-swiglu-scalar.md) 已接受构建期全域查表、纯整数运行时、双重载接口、存储/失败语义及完整 MXCSR 不变承诺；不因实现方便静默改成 exp/add/div 分步 FP32 舍入。
- [x] **生产桥入口**：普通 C++ 消费者只依赖公共头与 `df_quant` 即可调用满足 EXPERT §1～§3 的桥接 scalar；等长 gate/up、可选有限 FP32 route weight、空输入策略、输出容量和允许写入前缀均服从已接受 D。实现不扩成通用 SiLU/libm API，不生成或归一化路由权重。
- [x] **逐阶段数值 parity**：钳位后 gate/up、SiLU、`silu*up`、可选权重乘积及 BF16 输出直接对照 `trace_swiglu`，每次乘法保持独立 FP32 RNE。覆盖非对称钳位边界、正负零、次正规、溢出、NaN/Inf 传播、`None` 完全省略乘法及提前 BF16、交换分支、重排乘法会改变结果的反例。
- [x] **SiLU 正确舍入验证**：穷举全部 BF16 gate 编码经精确拓宽与钳位后的 SiLU 映射，直接对照 P0020 已验收 oracle；另以不复用生产核心的独立边界/格点期望验证舍入中点附近、饱和端、渐进下溢和特殊值。异常、超时、拒绝或 skip 不计为数值一致。
- [x] **存储与失败契约**：逐类验证长度、容量、route weight、别名和溢出拒绝；元数据检查先于内容读取，全部拒绝先于写入。使用精确分配、偏移共享与 sanitizer 验证输入不变、失败时输出完全不变、成功时仅约定前缀可写及范围外字节不变；测试构造满足 C++ 对象与 span 前提。
- [x] **环境与确定性**：依已接受生产 D，在 BUILD 的八模式 MXCSR、状态位清空与预置下覆盖普通、内部观测、接受及拒绝路径；结果不随调用方舍入/FTZ/DAZ 改变且不陷入，控制位、既有 sticky 位及可能新增状态严格按声明核对。不得由 Python Decimal 的环境边界推定生产实现行为。
- [x] **错误检出与执行器边界**：定向生产变体至少覆盖遗漏/错误钳位、错误 SiLU 舍入、交换 gate/up、提前 BF16、融合或遗漏乘法、route weight 错放及特殊值错误处理，均由指定数值或阶段比较检出且有无故障基线；崩溃、协议错误、存储拒绝或 sanitizer 报告不算数值敏感性证据。
- [x] **构建、复现与回归**：接入既有 CPU Debug/Release/ASan 体系及普通消费者链接测试，全套适用回归通过，无 skip、编译告警或非预期报告；已提交干净检出、自源码目录外复现，量化提交携带有效 `Oracle:`，严格检查与范围审计通过。不设置性能门槛，不作快路径论断。
- [ ] **知识与人工收尾**：按 [D0011](decisions/d0011-doc-single-source.md) 将生产桥契约归 EXPERT、环境与测试入口归 BUILD、验证链增量归 MODEL §4，生产方案理由留新 D；用户评审实现、证据及知识处置后方可结项并同步 STATUS。

## 记录

### 范围与依据

- 直接上一级为 [P0020](p0020-expert-oracle.md) 的 `swiglu`/`trace_swiglu` 与 [D0015](decisions/d0015-expert-oracle.md) 接受的 CPU 数值语义；P0019 仅提供可分别消费 W1/W3/W2 的线性 primitive，不包含非线性或专家串接。
- 本项实现归 `df_quant`，验证环境限 [BUILD §1](../docs/BUILD.md#1-支持环境) ENV-A，不依赖 P0005 目标机、权重、网络或 CUDA。生产算法与接口取舍见已接受的 [D0016](decisions/d0016-swiglu-scalar.md)。
- 范围外：W1/W3 双投影、共享一次输入量化、第二次激活量化、W2、完整专家失败原子性、router top-k/权重生成、多专家聚合、共享专家、batch、生产加载/pack、真实 checkpoint、SIMD/CUDA、融合与性能测量。完整单专家生产实现须以后续 P 直接消费本项入口，不在本项顺带实现。
- 知识去向：EXPERT 为生产桥可维护契约，BUILD 为命令与环境入口，MODEL §4 为验证链覆盖边界；新 D 只保留生产算法与接口取舍，P0021 只保留验收证据。

### 最终证据

- ENV-A；验收工件 `c5634d100b934ce9056d8300a4b7f9c5e0c5b470`，VERSION 0.1.18，dirty=false。`git clone --no-local --single-branch --branch p0021-swiglu-bridge-scalar` 至 `/tmp/p0021-clean2-20rjuh0f/repo`，配置并构建 cpu-debug/cpu-release/cpu-asan 后从源码目录外 `ctest --no-tests=error --output-on-failure`：Debug 37/37、Release 37/37、ASan 24/24，无编译告警、skip 或非预期报告；sanitizer 环境按 BUILD §3。
- 同一检出从源码目录外运行 `tests/quant/swiglu_sensitivity.py`：9/9 生产变体以完成调用的数值/阶段 mismatch 检出，pristine 复测与受保护文件哈希不变；协议故障均不计数值证据。三 preset 生成的 `silu_table.h` 逐字节一致（SHA-256 `9489af97…b48b3`）。
- 覆盖：默认 MXCSR 下穷举全部 65,536 个 BF16 gate 编码的普通+trace 逐阶段 parity；定向与契约集合（60 例，含全部零偏移写前缀别名拒绝、相邻/只读/尾部接受、空输入、非法权重、trace 形状及 64 位边界计数溢出探针）在 16 种 MXCSR 状态下通过；harness 含协议负例、快照/间隙/加毒及驱动退出门禁。
- 干净检出 `check_repo.py --range 8f0e240..HEAD --audit --strict` E=0 W=0 I=0；`git diff --check` 通过，改动 Markdown 路径/锚点可解析；未触碰 `ref/`、只读镜像、冻结数据、AGENTS 与 VERSION；运行后检出仍干净。日志与审计在 `scratch/p0021-final/`，为临时工件。

### 评审处置

- 2026-10-15：用户确认生产侧缺口应先拆为 SwiGLU 非线性桥，再做完整专家，并授权按流程起草 P0021；登记 draft，未授权实现或生产方案。
- 2026-10-15：用户审阅草案通过，并批准含 D0016 的开工方案、必要 docs 更新、实现、验证与提交；D0016 转 accepted，本项转 active。
- 2026-10-15：独立评审发现混型 span 构造、协议负例假阳性、全域故障参数遗漏及拒绝输入快照缺口；修复为共用字节校验、精确进程退出负控、参数传递与全存储快照，并补齐混型探针元数据前置保护。静态复核确认全部闭环，可提交；不代替用户验收。
- 2026-10-15：回填时发现字节计数溢出分支无法经合法 span 构造触达；提取生产共用标量计数检查并以 64 位边界探针验收后补齐该项。

### 提交

- `ed20ca2`：接受 D0016、登记 P0021 active 并同步 STATUS。
- `b08ca28`：生产 SwiGLU 双重载、构建期 SiLU 表、执行器/consumer/parity/敏感性测试与 EXPERT/BUILD/MODEL 归档；携带 `Oracle: tests/quant/test_swiglu_parity.py`。
- `c5634d1`：共用标量计数检查与边界溢出探针，补齐失败契约验收；携带同一 `Oracle:`。

## 收尾

- 验收或停止依据：active；除知识与人工收尾外七项验收已成立，最终工件的干净检出复现与范围审计通过（见最终证据）。待用户审议实现、证据及知识处置后结项并同步 STATUS；不以内部评审与机械检查代替人工验收。
- 知识处置：pending；生产契约已写入 [EXPERT §5](../docs/EXPERT.md#5-生产-swiglu-bridge)，BUILD/MODEL 路由及 D0016 已到位，处置结论待用户审议。
- 剩余义务：本项完成后，完整单 token FP4 专家生产实现仍须另立后续 P，组合 W1/W3、非线性桥、第二次量化与 W2，并独立处理三矩阵统一预校验、workspace 和失败原子性。
