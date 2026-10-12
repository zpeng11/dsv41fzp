---
id: P0012
title: 建立官方专家 FP8×FP4 分组点积与累加后乘 scale 的计算 oracle
status: completed
created: 2026-10-12
updated: 2026-10-12
knowledge: updated
docs: [docs/MODEL.md, docs/QUANT.md, docs/COMPUTE.md, docs/BUILD.md]
bench: []
decisions: [D0008]
followups: []
---

## 意图

接续 [P0006](p0006-ref-format-oracle.md)、[P0010](p0010-quant-scalar-baseline.md) 与 [P0011](p0011-quant-dequant-scalar.md)，建立官方 FP8×FP4 阶段约束下的单输出 CPU 规范参照，覆盖块内点积、给定 scale 应用与跨块累加。

交付按 [COMPUTE](../docs/COMPUTE.md) 契约独立验证的 Python 实现，供后续同契约 C++ scalar 对照。不验收 K-E1/K-E2 或官方 GPU，不改变 W4A16 默认规划；各路径参照与跨路径质量边界见 [MODEL §4](../docs/MODEL.md#4-精确性链条)。

## 验收

- [x] **开工前置决策**：在实现前形成并经用户确认接受计算参照决策，明确参照归属、验证链位置、块内乘法与归约规则、乘 scale 顺序、跨块累加顺序、各步精度与舍入、FMA/融合语义、次正规及特殊值处理、FP32/BF16 输出边界与运行环境要求。区分官方源码可确认的顺序、本项目选定的 CPU 参照语义，以及仍需 GPU 仲裁的归约/cast/FTZ 细节；关键语义未明确时不得进入实现。同步定向修订 MODEL §4 与 K-Q2 的参照路由，区分官方 FP8×FP4、K-E1 W4A16 与 K-E2 整数改写各自所需参照；不将本项挂为逐元素 dequant 的已验证后继算法或 K-E1/K-E2 的直接 parity 依据，不以本项为由改变默认生产路径。
- [x] **一个计算路径、一个输出**：提供 Python 慢参照，以已编码 E4M3FN 激活、沿 K 打包的 E2M1 权重及给定 E8M0/UE8M0 激活与权重 scale 为输入，计算单个输出。激活与权重均固定沿 K 每 32 元素一组；第 j 个 32 元素块分别使用激活 `scale_a[j]` 与权重 `scale_b[j]`。K 须整除 32，激活长度为 K、packed 权重字节数为 K/2，两份 scale 长度各为 K/32；覆盖单块和多个相邻块。分组是已确认的模型路径边界，不留给前置决策重新选择，不支持 128 元素激活组或其他可配置组大小，不另立格式配置中心。明确非法输入和空输入契约，不隐式广播、补齐或生成 scale，不扩展为矩阵算子。
- [x] **计算与舍入独立验证**：期望值由独立精确算术和按已接受契约逐步舍入构造，不调用被测点积、累加或舍入辅助函数生成期望值。块内结果、应用 scale 后的块结果、跨块累加与最终输出分别检查；有限结果按声明的 FP32/BF16 位模式比较，零符号、Inf 与 NaN 按已接受契约检查，不以统一 allclose 掩盖顺序或舍入错误。
- [x] **边界与布局**：覆盖正负值、相消、累加顺序敏感、RNE 中点及两侧、指数进位、正常/次正规边界、下溢、上溢、零符号、极端 scale；多块用相邻不同的激活/权重 scale 验证两类每 32 元素分组索引，并以跨越 128 元素边界的定向用例识别错误地每四块共享激活 scale。覆盖全部数据码点和 scale 编码边界，不要求穷举所有点积向量；随机用例仅作固定种子的补充。合法输入造成的中间特殊值不得被默默忽略或钳制。
- [x] **测试敏感性与失败传播**：在独立临时副本分别注入 scale 错组、提前 BF16 舍入、错误跨块累加顺序，定向用例均识别对应错误；恢复后测试通过。测试入口的错误结果或异常必须导致非零退出，必选用例无 skip，不把舍入顺序恰好无差异的用例当作敏感性证据。
- [x] **CPU 独立复现**：一条明确的 Python 命令运行全部必选验收，仅依赖标准库，不初始化 Torch/TileLang/CUDA，不依赖目标机、真实权重或官方 GPU dump。测试引用 P0012，记录解释器、CPU/平台、方法、命令和结果；从源码目录外及干净检出复现，既有格式/反量化与仓库检查回归通过。新测试接入既有 CPU CTest 入口，不因本项扩大 quant 专用的 cpu-asan 范围。
- [x] **知识与人工收尾**：经用户确认后归档计算契约、来源、复现方法和验证边界，MODEL 提供参照链路由、BUILD 更新测试入口，必要知识缺失阻塞结项。用户评审后才结项；明确本项不证明生产 C++、K-E1 W4A16 的实现 parity 或相对官方路径的质量偏差、K-E2 整数改写、SIMD/CUDA、官方 GPU parity 或端到端一致。不得将 MODEL §4 既有的激活侧偏差表述当作已取得本项未提供的质量实测证据。

## 记录

### 范围与依据

- 前置能力：P0006 的格式 oracle、P0010 的生产 codec 与 P0011 的给定 scale 反量化均已结项。本项新增计算参照，不修改既有 oracle 以迁就后续引擎实现。
- 官方源码阶段、分组与未决 GPU 细节见 [COMPUTE §1](../docs/COMPUTE.md#1-来源与参照边界)；计算选择依据为 [D0008](decisions/d0008-expert-grouped-dot-reference.md)，验证链及 K-Q2 路由见 [MODEL §4](../docs/MODEL.md#4-精确性链条)。
- 范围外：K-E1 W4A16 计算参照与质量偏差验收、sm80 默认数值策略调整、K-E3 与 K-E1 端到端一致锁、128 元素激活组及其他可配置组大小、生产 C++ 计算实现、完整 GEMM/GEMV 与专家 MLP、整数点积改写与 scale 折叠优化、VNNI/SIMD/CUDA、性能论断与优化开关、有损编码与 amax/scale 生成、动态激活/KV 量化、真实权重与 pack 容器、调度/缓存系统、完整 FP32 模型 oracle 及端到端验证。

### 执行与知识处置

- 2026-10-12：用户确认实现方案并授权按流程开工；P0012 翻转 active，同步 STATUS，沿用短命分支 `p0012-expert-grouped-dot-oracle`。按单块、跨块与阶段观测、独立测试、敏感性及复现的顺序实现，不扩项；不将开工授权当作人工结项验收。
- 可复用既有格式解码、FP32 乘法和 BF16 转换；新增加法在计算模块内以整数有效数、指数对齐与显式 RNE 构造结果，不改写格式 oracle，不建设通用软件浮点 API 或第二个格式配置中心。
- 分阶段检查可达边界：输入正负零、scale 后负零、相消后正零与负 FP32 次正规值转 BF16 负零分别验证。全部码点覆盖包含合法计算和 NaN 拒绝；不可达的元素乘法舍入用例不充作公开路径证据。定向用例先于固定种子随机补充，覆盖块内归约、scale 次序/错组、跨块重排、融合及提前 BF16；错组边界与失败传播按上述验收执行。
- CTest 注册放在 quant 专用 scope 返回之后，不扩大 cpu-asan。COMPUTE 已归档契约，D0008 保留理由，本项保留过程与证据；实际入口及本轮验证已到位，BUILD 定向归档经用户评审确认后落实；提交后的干净检出复现成立，知识处置为 updated；未建后续项前不填写虚构 followups。
- 2026-10-12：用户确认上一轮范围判断，并要求新建 P、清理旧 agent 工作残留与整理 STATUS，先不 commit。本次仅登记 draft 和同步状态入口；未建立或接受新 D，未授权计算 oracle 实现，未开始编码或运行本项数值验收。
- 2026-10-12：用户确认激活和权重均固定沿 K 每 32 元素一组，K 整除 32，两份 scale 各为 K/32；128 元素激活组及其他可配置组大小明确排除。补充跨越 128 元素边界的错组敏感用例要求。此为范围澄清，不接受尚未定案的数值语义决策，不翻转 active，仍不 commit。
- 2026-10-12：用户确认澄清官方 FP8×FP4 oracle 与 K-E1 W4A16 的参照边界：本项只供同数值契约的后续实现对照，不直接验收 K-E1 或 K-E2，不修改 W4A16 默认路径规划，不承接跨路径质量偏差与双 arch 一致锁。K-E1 所需同路径参照另项决定；本次仍仅更新 draft，不建立或接受新 D，不实现、不 commit。
- 2026-10-12：用户确认当前 P0012 覆盖范围并授权按流程提交计划与 STATUS；本次仅提交已确认的 draft，不翻转 active，不建立或接受新 D，不授权实现，不合并或递增 VERSION。
- 2026-10-12：用户明确接受开工前 review 的计算参照决策和验证链路由，授权按治理落实对应改进。登记 D0008 accepted、发布 COMPUTE 契约并定向修订 MODEL §4/K-Q2、QUANT 边界链接及 STATUS；第一项前置验收成立，其余验收条件不变。本次不启动编码、不翻转 active、不提交或合并、不递增 VERSION，也不将该确认当作人工结项验收。
- 2026-10-12：用户确认精简后的 D0008 关联改动并授权提交。仅提交决策、契约与路由文档；数值语义、验收条件和评审证据不变，不授权实现、合并或结项，P0012 保持 draft、knowledge=pending，VERSION 不变。

### 开工前评审证据

- 环境：Python 3.12.8（conda-forge，GCC 13.3.0）、Linux 5.15.0-72-generic x86_64 / glibc 2.43、Intel Xeon Platinum 8350C。方法：`PYTHONDONTWRITEBYTECODE=1 python3 -` 临时标准库探针，以 `Fraction` 独立位域解码、精确运算与 `tests/quant/test_formats.py` 的相邻格点 binary32 RNE 方法构造结果，不调用被测点积；无 GPU、权重或在线来源。以下仅为数值设计证据，不替代后续 P0012 独立测试与敏感性验收。
- 全部有限 E4M3FN×E2M1 编码组合共 4,064 对，FP32 不精确乘积为 0；不能将元素乘法不精确舍入当作公开输入可达边界。块内顺序累加仍有舍入差异。
- 块内向量 `7×(448×6) + 2^-10 − 7×(448×6)`，按该顺序补零至 32 项：逐次 FP32 累加为 `0x00000000`，精确求和后一次舍入为 `0x3A800000`。合法码点为 E4M3FN `0x7E/0xFE/0x01`、E2M1 `0x7/0x1`，两份 scale 均为 `0x7F`。
- 块 dot=2，scale_a=`0xFE`、scale_b=`0x00`：先乘激活 scale 上溢为 `0x7F800000`，再乘权重 scale 仍为 Inf；折叠 scale 或交换次序得 `0x40000000`。
- 三块贡献 `[2^24, 1, -2^24]`：按 K 顺序累加为 `0x00000000`，先相消为 `0x3F800000`。由块 dot `[1,1,-1]`、激活 scale `[0x97,0x7F,0x97]` 与权重 scale 全 `0x7F` 构造。
- 两块 dot 均为 `2^-10`，激活 scale 全 `0x00`、权重 scale `[0x73,0x72]`：分别乘法及加法 RNE 得 `0x00000001`，第二块最后一次乘法与累加融合得 `0x00000002`。
- 两块贡献 `[1+2^-8,-1]`：仅最终转 BF16 得 `0x3B80`，每块提前转 BF16 得 `0x0000`。两份 scale 全 `0x7F`；第一块由 `1×1 + 2^-9×2` 构造。
- 负零阶段边界：dot=`-2^-10`、scale_a=`0x00`、scale_b=`0x72` 的块贡献为 FP32 `0x80000000`，从正零累加后为正零；scale_b 改为 `0x73` 得最终 FP32 `0x80000001`，再转 BF16 得 `0x8000`。须分阶段检查，不以最终输出代替中间负零验证。
- 评审基线命令：`PYTHONDONTWRITEBYTECODE=1 python3 tests/quant/test_formats.py`（19 项通过）、既有 cpu-debug 构建目录 `ctest --test-dir build/cpu-debug --output-on-failure -R '^quant\\.(codec|dequant)\\.'`（7 项通过）、同目录 `-R '^bb\\.compile_flags$'`（通过）、`PYTHONDONTWRITEBYTECODE=1 python3 tests/tools/test_check_repo.py`（71 项通过）、`PYTHONDONTWRITEBYTECODE=1 python3 tools/check_repo.py --strict`（E=0 W=0 I=0）。本轮评审未重建 C++，未进行干净检出复现，未运行尚不存在的计算 oracle 验收。

### 实现与本轮验证证据

- 2026-10-12：新增 `ref/compute.py` 的普通入口与不可变阶段观测，共享同一执行路径；新增 `tests/quant/test_compute.py`，期望由独立 Fraction 位域及相邻格点 RNE 构造，另以整数 RNE 交叉复核。输入校验、逐阶段有限/零/Inf 按位与 NaN 分类、FP32/BF16 模式独立性已验证；既有 `ref/` 文件、官方镜像及冻结数据均未修改。
- 环境：Python 3.12.8（conda-forge，GCC 13.3.0），Linux 5.15.0-72-generic x86_64 / glibc 2.43，KVM 报告 Intel Xeon Platinum 8350C；GCC 15.2.0、CMake/CTest 4.4.4、Ninja 1.13.2。方法：标准库独立精确算术、逐步舍入、定向向量及固定种子补充，无 Torch/TileLang/CUDA、GPU、真实权重、dump 或在线来源。
- `PYTHONDONTWRITEBYTECODE=1 python3 tests/quant/test_compute.py`：30 项通过，无 skip。覆盖 4,064 个有限 E4M3FN×E2M1 码点对、NaN 拒绝、双方 scale 编码边界、双方不同 scale 的相邻块与 K=160 错组边界、归约及 scale 次序、独立乘加舍入、仅最终 BF16、正常/次正规边界进位、下溢/上溢、零符号与 Inf/NaN 传播；随机仅作补充，不宣称穷举所有点积向量。
- 敏感性入口包含原版副本先过、三类分别变异及原版复测：scale 每四块共享、提前 BF16、跨块倒序在各自 16 项定向套件中分别造成 5/9/9 项数值失败，指定检测用例均失败，异常与 skip 均为零；真实计算异常和结果篡改分别造成 error/failure，普通及 marker 入口均退出 1。诊断结果同时核对计数、用例标识和摘要，不以导入失败或任意非零退出代替数值敏感性。源码字节前后不变。
- 环境审查发现 `struct.unpack('<f')` 的次正规承载值在 DAZ 下被冲刷；只在新增计算模块内改用精确 binary64 构造及给定 E8M0 的整数移位 RNE，不改变契约、不修改既有格式 oracle。临时子进程探针 `scratch/p0012/check_environment.py` 在四种舍入方向×四种 DAZ/FTZ 组合下，负次正规链均得到 `after_a=0x80001000`、`block/fp32=0x80000001`、`bf16=0x8000`，调用前后 MXCSR 控制位不变且 finally 恢复；不承诺异常状态标志不变，不将本机平台探针加入跨平台必选入口。
- 三个 CPU preset configure/build 成功；`ctest --preset cpu-debug`、`cpu-release` 各 16/16 通过，`cpu-asan` 8/8 通过且注册范围未扩大。最终补强的 `quant.compute` 在 Debug/Release 分别复测通过；既有格式/反量化 19 项、仓库治理测试 71 项通过，`python3 tools/check_repo.py --strict` 为 E=0 W=0 I=0，`git diff --check` 通过。
- 从 `/tmp` 用测试脚本绝对路径运行全部 30 项通过。隔离源码快照从空 build 重建三个 preset 并通过全部 CTest，最终补强版 compute 再通过；该快照叠加了未提交改动，**不是干净 Git 检出**。本轮未授权提交/合并，干净检出子项保留未验收，VERSION 未改。
- 原始本轮日志和复现脚本在 git-ignore 的 `scratch/p0012/`，BUILD 内容修订仍为待确认草稿。以上为实现验证证据，不替代人工结项评审，不表示生产 C++、K-E1/K-E2、官方 GPU、跨路径质量或端到端一致已验收。

### 归档与结项复现

- 2026-10-12：用户评审通过，授权按流程归档和推进。按已评审草稿定向更新 BUILD 的计算测试入口及复现边界；先提交实现与同步簿记，再从该提交的干净 Git 检出完成全部复现，证据成立后记录结项并按 no-ff 合入 main。此次授权不扩展计算范围，不取得新的性能、设备 parity 或跨路径质量结论。

## 收尾

- 验收或停止依据：全部验收成立，2026-10-12 用户评审通过并授权按流程归档推进。实现提交 `2a6b0c76537a2d85546cb9541bf369ed83105f33` 的干净检出在源码目录外运行完整标准库验收 30/30 通过，从空 build 配置构建三个 CPU preset，Debug/Release 各 16/16、cpu-asan 8/8 通过；仓库全量及提交范围 strict 检查均 E=0 W=0 I=0，复现前后 `git status --porcelain` 均为空。方法及环境沿用上文，原始日志为 `scratch/p0012/clean-reproduction.log`，结果摘要为 `scratch/p0012/clean-reproduction.json`。据证据记录 completed，不以合并代替验收。
- 知识处置：updated；数值契约、来源与边界见 [COMPUTE](../docs/COMPUTE.md)，理由见 [D0008](decisions/d0008-expert-grouped-dot-reference.md)，参照路由见 [MODEL §4](../docs/MODEL.md#4-精确性链条) 与 [QUANT](../docs/QUANT.md)；实际入口、复现方法及测试边界已归档 [BUILD §7](../docs/BUILD.md#7-测试清单)，过程与验收证据保留本项，不重复契约正文。
- 剩余义务：本项无未完成交付或转交义务，followups=[]。生产 C++、K-E1 同路径参照与跨路径质量、K-E2 整数改写、SIMD/CUDA、官方 GPU parity 及端到端一致均为已明确排除的未来范围，另项决定，不据本项声称已验收。
