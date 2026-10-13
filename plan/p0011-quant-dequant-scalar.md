---
id: P0011
title: 建立给定 scale 六类反量化的生产 scalar 基线
status: completed
created: 2026-10-11
updated: 2026-10-13
knowledge: updated
docs: [docs/MODEL.md, docs/QUANT.md, docs/BUILD.md]
bench: []
decisions: [D0001, D0005, D0006, D0007]
followups: []
---

## 意图

在 [P0010](p0010-quant-scalar-baseline.md) 之上交付 `ref/dequant.py` 对应的六类给定 scale 反量化生产入口，逐位对照 `ref/`。成功标准是独立 C++ 消费者在声明的环境与输入契约下获得与 `ref/` 一致的 FP32 与 BF16 位模式。覆盖边界见 [MODEL §4](../docs/MODEL.md#4-精确性链条)。

## 验收

- [x] **六类入口**：专家、普通 FP8 权重、Engram、SWA、主 KV、索引 K，各有 FP32 与 BF16 位模式输出；布局与 scale shape 遵守 QUANT，不广播、不转置、不推断。
- [x] **输入与失败契约**：按 D0007 拒绝 rank/shape/长度/scale shape/容量/组不整除/NaN/非正或非有限 scale/换算溢出/重叠；元数据先于读、内容先于写，失败输出不变，允许零元素；短缓冲区在 `cpu-asan` 下验证，坏值置末元素。
- [x] **数值环境无关**：整数位运算构造；默认及向上、向下、向零、FTZ、DAZ、解除溢出掩码下逐位一致、不陷入、MXCSR 不变；`-448 × 2^127` 向上舍入仍为 `-inf`，次正规乘积与 E8M0 `0x00` 不被冲刷。
- [x] **乘 scale parity**：全部有限数据码与合法 scale 码组合、极端 scale、符号零、下溢、上溢，FP32/BF16 逐位；NaN 比分类。
- [x] **布局边界**：多行、多组、批维、相邻不同 scale、非对称、双尾块、组边界、零元素，错轴 scale 不能通过。
- [x] **可复现、越界检查与测试敏感性**：三个 preset、源码目录外与干净检出；逐行 scale 索引、二维尾块索引、批维步长、BF16 舍入四类注入被识别。
- [x] **知识、治理与人工收尾**：QUANT §7 与 BUILD 增补，`Oracle:` 指向 C++ parity；审计通过，用户评审后结项。

## 记录

### 范围与依据

- 前置 P0010 completed、[D0007](decisions/d0007-quant-integer-dequant.md) accepted；沿用 P0010 模块、协议与 `cpu-asan`。契约见 [QUANT §7](../docs/QUANT.md#7-给定-scale-的生产-scalar-反量化)。
- 范围外：有损编码、scale 生成、动态量化、点积、GEMM/GEMV、真实权重、CUDA/SIMD、性能论断。

### 验收证据

- 环境：ENV-A（KVM）；最终提交态 `a211439`（实现 `bdbda5d`），工作仓库与全新克隆无补丁且测试后干净。
- `cpu-debug`/`cpu-release` 各 15/15，`cpu-asan` 8/8，零告警、无 sanitizer 报告；`/tmp` 下 15/15。
- parity：64 个批量请求；E2M1×E8M0 4,080 对、E4M3FN×E8M0 64,770 对、E2M1×正 E4M3FN 2,016 对；批维 rank 1/3/4，普通 FP8 权重 35×67 双尾块。
- contract：324 个请求覆盖全部拒绝类与零元素，短缓冲区 ASan 无越界读，失败时输出不变。
- 私有乘法：48 个定向对加 seed 21 的 20,000 个随机对对照 `ref/formats.py::multiply_fp32`；`0x3F900001 × 0x3FE38E37` → `0x40000000`，`0x7F100001 × 0x3FE38E37` 进位得 Inf。
- 环境：公开 parity 与私有乘法在八模式 MXCSR 下一致，含拒绝路径。
- 敏感性：四类注入使 `quant.dequant.parity` 失败（CTest 8）；私有中点不取偶、漏进位、冲刷次正规三类使 multiply 失败；恢复后通过。
- 耗时（Debug，测试连接，52 请求时测量）：期望生成约 2.10s，每环境执行约 0.41s。
- 审计：非空 `main..HEAD` E=0 W=0 I=0。

### 评审处置

- 2026-10-11：用户指出 glibc `fegetround()` 读 x87 控制字、解除溢出掩码会触发 SIGFPE，环境检查改以 MXCSR 为准并加上溢敏感用例。
- 2026-10-11：探针确认合法输入的乘积要么精确要么上溢；D0007 取代 D0006，改整数乘法并删去调用时 MXCSR 检查。
- 实现审阅修复：向零 MXCSR 常量、私有探针坏请求分类、harness 递归、溢出用例 scale shape、协议空分量/重复字段；补舍入进位锚点。
- 2026-10-12：用户收束边界：codec 专用 helper 留匿名命名空间，共享函数走 `detail::`；`-fno-exceptions` 只按 quant 库与执行器 target 检查。nm 与合成编译命令证实。
- 2026-10-12：`.zcode/plans/` 精确忽略（`f612026`）；用户确认验收并授权合入。

### 提交

- `bdbda5d` 实现，`a211439` 知识归档，`f612026` ignore；合并提交 VERSION 0.1.10。
- 压缩前全文见提交 `b60526d`（依 [D0011](decisions/d0011-doc-single-source.md) 压缩）。

## 收尾

- 验收或停止依据：2026-10-12 用户确认验收代码与文档并授权推进合入；提交态干净复现、非空范围审计与逐条证据见上方。本合并提交同步 completed、STATUS 与合并流程的 VERSION patch+1，版本为 0.1.10。不证明 GEMM、累加后乘 scale、点积 scale 折叠、提前 BF16 舍入、scale 生成、动态量化、SIMD/CUDA、真实权重或端到端模型一致。
- 知识处置：updated；入口、存储/逻辑 shape、失败、环境无关性与验证边界归档到 [QUANT §7](../docs/QUANT.md#7-给定-scale-的生产-scalar-反量化)，新增测试与执行器/编译要求归档到 [BUILD §7](../docs/BUILD.md#7-测试清单)；选择理由保留在 [D0007](decisions/d0007-quant-integer-dequant.md)，过程证据保留在本项。
- 剩余义务：无；followups 为空。范围外的计算类 scalar 参照与验证链完整修订仍按 STATUS 的既有路由在首个计算类 kernel 工作项开工前决定，不以本项验收推定完成。
