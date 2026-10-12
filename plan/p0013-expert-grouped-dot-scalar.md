---
id: P0013
title: 建立专家分组点积的生产 scalar 基线
status: active
created: 2026-10-12
updated: 2026-10-12
knowledge: pending
docs: []
bench: []
decisions: [D0009]
followups: []
---

## 意图

接续 [P0012](p0012-expert-grouped-dot-oracle.md)，按 [COMPUTE](../docs/COMPUTE.md) 契约交付单输出专家分组点积的生产 C++ scalar 入口，并对照 `ref/compute.py` 逐阶段验证同一生产执行路径。成功标准：独立 C++ 消费者调用生产库，得到与 oracle 一致的 FP32 与 BF16 结果，有限值（含带符号零）及 Inf 按位一致，NaN 按分类一致；结果与调用线程的浮点环境无关。

本项是该契约 kernel 的上一级参照（[MODEL §4](../docs/MODEL.md#4-精确性链条)），数值路径、接口与观测方式的取舍见 [D0009](decisions/d0009-compute-scalar-integer-path.md)。不验收 SIMD/CUDA、K-E1/K-E2、官方 GPU 或端到端一致，不改变 W4A16 decode 默认规划。

## 验收

- [x] **生产入口与失败契约**：`src/quant` 提供单输出入口，公共头 `include/df/quant/compute.h`；输入长度与分组按 COMPUTE §2，表示、输出重载与失败类按 D0009。K 为零或不整除 32、长度不匹配、输出容量不足、NaN 激活或 scale 编码、四份输入与实际输出前缀重叠，均抛 `std::invalid_argument`；每类按两种输出分别验证失败时整个输出缓冲区不变。坏内容置于输入末端，元数据错误配合短缓冲区，在 `cpu-asan` 下验证元数据校验先于内容读取、内容校验先于输出写入。Python 侧的类型、宽度与输出字符串拒绝仍由 oracle 验证；超宽 token 与未知输出模式由执行器按协议错误拒绝，不计为库拒绝。解码复用 P0010 codec，不扩展为矩阵算子。
- [x] **整数构造数值路径**：逐条实现 COMPUTE §3，全部阶段按 D0009 以定宽整数构造，不执行浮点运算，不读写浮点环境。乘 scale 前先分流非有限操作数；合法计算产生的 Inf/NaN 照常传播，不是调用失败。舍入保持通用，不以"E4M3FN×E2M1 有限乘积恒精确"之类的输入性质作为正确性前提；不新增公共通用 IEEE 接口。
- [x] **逐阶段计算 parity**：经 D0009 的私有观测连接对照 `ref/compute.py::trace_expert_grouped_dot`，比较每个元素乘积、每次块内累加、每块 `dot_bits` / `after_a_bits` / `block_bits`、每次 `acc_bits` 与最终 FP32/BF16；两种公开入口与对应观测结果一致。覆盖：全部有限 E4M3FN×E2M1 码点对（4,064 对）、双方 scale 编码边界、相邻不同 scale 多块、跨 128 元素边界的错组敏感布局、归约与 scale 次序敏感向量、RNE 中点及两侧、指数进位、正常/次正规边界、下溢到零、上溢到 Inf、零符号与 Inf/NaN 传播。定向证明：第二次 scale 乘法保留第一次产生的 Inf；块贡献 −0 在中间阶段检出；异号 Inf 产生的 NaN 在后续累加中传播。输入 NaN 只作库拒绝用例；随机用例仅作固定种子补充；不以最终输出、codec 或逐元素 dequant parity 代替。
- [x] **定宽整数加法**：按 D0009 的表示实现，不允许移位越界、带符号整数溢出或丢弃影响舍入的低位。同一生产 helper 经私有连接对照 `ref/compute.py::_add_fp32`，覆盖大指数差、RNE 中点及两侧、sticky 敏感舍入、近相消、舍入与指数进位、正常/次正规边界、符号零与 Inf/NaN；公开 parity 另含 `0x7F000000 + 0x00000001`（`2^127 + 2^-149`）。全部专项在 `cpu-asan` 下无 UBSan 报告。
- [x] **数值环境无关**：公开 parity 与加法专项在执行器既有的八种 MXCSR 设置下各运行一遍，结果与 `ref/` 一致，进程不陷入，每次调用前后（含拒绝路径）完整 MXCSR 不变。敏感用例含负次正规链、上溢带符号 Inf 与非默认舍入方向。
- [x] **测试敏感性与失败传播**：在 scratch 独立副本中分别注入以下十类错误，每类由指定的定向用例产生数值比较失败：
  - scale 每四块共享；
  - 提前 BF16；
  - 跨块倒序；
  - 块贡献 −0 规范化：最终输出相同时由中间阶段检出；
  - 非有限中间值误送有限乘法；
  - 加法丢失 sticky：须命中指定舍入用例；
  - nibble 高低位颠倒：高低 nibble 不同且对应激活不对称，在乘积阶段检出；
  - 相邻配对平衡树归约：指定向量的最终结果须与顺序累加不同；
  - 两份 scale 交换；
  - 两份 scale 折叠。

  scale 交换与折叠的锚点为 dot=2、scale_a=`0xFE`、scale_b=`0x00`：规定次序得 Inf，交换或折叠得 2。恢复后全部测试通过，交付源字节不变。harness 沿用 P0010/P0011 协议，覆盖崩溃、非零退出、乱码、少响应与坏请求；trace 响应须检查阶段数量与位宽。错误结果或异常必须导致非零退出，但不以任意非零退出代替数值敏感性。
- [x] **构建、治理与复现**：三个 CPU preset 构建并通过既有与新增测试；新增测试接入 `tests/quant` 的 quant CTest 序列，compute 进入 `cpu-asan` 所构建的 quant 生产库，不新增 preset。数值提交带 `Oracle:`；`python3 tools/check_repo.py --strict` 与提交范围审计 E=0 W=0 I=0；从源码目录外及干净检出复现，既有 oracle 测试回归通过。
- [ ] **知识与人工收尾**：经用户确认后，生产入口与失败契约、特殊值分流、验证边界与环境要求最小增补到 COMPUTE（与 Python 拒绝类分开），测试入口增补到 BUILD，必要时更新 MODEL §4 与 STATUS。用户评审后才结项。

## 记录

### 范围与依据

- 前置：P0012 completed；P0010/P0011 的生产 codec、有限乘法、BF16 舍入、跨语言 parity 协议与 MXCSR 环境矩阵可复用。逐元素反量化的有限输入前提不能外推到计算中间值，Python 拒绝类也不能直接转为 C++ 失败类。
- 范围外：矩阵算子与多输出/批量接口、K-E1 同路径参照与跨路径质量偏差、K-E2 整数改写与 scale 折叠、SIMD/CUDA、性能论断与优化开关（运行时开关义务在优化路径）、有损编码与 scale 生成、动态量化、真实权重与 pack 容器、完整 FP32 模型 oracle 与端到端验证。

### 完成声明边界

- 验证链新增"同契约 C++ scalar"一级；快路径以本项为上一级，不直接对照 Python oracle。
- 不新增性能论断或优化开关。

### 执行与知识处置

- 前置：D0009 accepted 后才进入 active。
- 分支 `p0013-expert-grouped-dot-scalar`；提交用 `P0013 quant:` 并带 `Oracle:`。建议顺序：接口与特殊值分流、定宽加法及其专项，再做单块逐阶段 parity、多块与跨块、失败契约、环境矩阵与敏感性注入。
- 知识预计为 updated（COMPUTE 生产契约、BUILD 测试入口）；必要知识缺失阻塞结项。

### 设计记录

- 2026-10-12：P0012 结项后确定下一项为生产 C++ 计算基线；用户拍板数值路径沿用整数构造并扩展到加法，否决硬件 FP32 加法加 MXCSR 纪律。
- 2026-10-12：用户确认两轮设计评审补强：同路径逐阶段 parity、特殊值分流、三层失败类分离、定宽加法专项、八种环境矩阵、四类新增敏感性注入、输出 span 及无元数据溢出拒绝类的理由。
- 2026-10-12：用户授权按第三轮评审压缩本项：与 COMPUTE §2/§3 重复的契约正文改为引用，数值路径、接口失败类与观测方式的选择理由移入 D0009（proposed）。验收条件不变。仍为 draft。

### 开工与实现记录

- 2026-10-12：用户确认 D0009 与当前范围并授权推进；D0009 accepted，本项转 active 并同步 STATUS，在 `p0013-expert-grouped-dot-scalar` 实现。新增 CTest 命名为 `quant.compute.{parity,contract,add,harness}`，与 Python oracle 自身的 `quant.compute` 在 BUILD 中区分。
- 2026-10-12：实现提交 `078ba1c`。公共入口 `expert_grouped_dot` 的 FP32/BF16 重载与私有 `detail::trace_expert_grouped_dot` 共用同一计算核心（trace 指针为空即公开路径）；`binary_to_fp32` 提到 `detail` 供乘法与加法共用。加法初稿的对齐位为 39，实现中发现这样宽的对齐使 sticky 永不影响结果，丢失 sticky 的注入无法检出，也不符合 D0009 的 guard/round/sticky 设计；改为 3 位对齐加就近舍入到奇数。指数差不超过 3 时精确（覆盖全部近相消）；更大时较大操作数为正规数，和超过 2^25，舍入至少丢弃两位，保证与精确和的 RNE 一致。论证写在 helper 的函数级注释中。
- 执行器协议：payload 按 `;` 分段且各操作要求固定段数；计算请求四段，`overlap=a|w|sa|sb` 令指定输入与输出共用一块分配；trace 的 cap 须恰为 2K+4K/32+1；`add_fp32` 接受任意位模式对。codec/dequant/multiply 收到 `overlap=` 或段数不符即判协议错误。

### 验收证据

环境：Intel Xeon Platinum 8350C，Linux 5.15.0-72-generic x86_64，GCC 15.2.0，CMake 4.4.4，Python 3.12.8。

- 套件：`quant.compute.parity` 309 个请求（两份 4,064 对穷举向量各含 trace/fp32/bf16、49 组 scale 编码边界、多块与 128 边界布局、nibble 次序、树形/跨块/scale 次序敏感、21 类舍入与特殊值定向、24 个固定种子随机向量）；`quant.compute.contract` 42 个拒绝及 cap=1 成功用例，另验 `ref/` 对同类输入抛 `ValueError`；`quant.compute.add` 9 组定向加 20,000 对随机；`quant.compute.harness` 6 项。parity、contract、add 均在八种 MXCSR 下运行，并在请求前后核对控制字。
- 回归：`cpu-debug` 20/20、`cpu-release` 20/20、`cpu-asan` 12/12，构建零告警，无 sanitizer 报告；`check_repo.py --strict` 与 `--range main..HEAD` 均为 E=0 W=0 I=0。从源码目录外用 `ctest --test-dir` 对全新克隆（`078ba1c`）运行三个 preset，结果相同，克隆工作树保持干净。
- 敏感性：在 scratch 独立副本中逐项注入十类错误，每类都使指定用例的数值比较失败：scale 每四块共享 → `layout.group128_sensitive.fp32`；提前 BF16 → `rounding.bf16_tie_up.fp32`；跨块倒序 → `order.cross_block_sensitive.fp32`；块贡献 −0 规范化 → 只有 `rounding.underflow_negative_zero.trace` 与另一随机 trace 失败，最终输出不变；非有限值误送有限乘法、scale 交换、scale 折叠 → `order.scale_order.fp32`；丢失 sticky → `add.sticky`（另有 `add.tie_neighbors` 与 `add.random`）；nibble 颠倒 → `layout.nibble_order.trace`；相邻配对平衡树 → `order.tree_sensitive.fp32`。注入前后的干净对照均通过，交付源逐字节未变。临时脚本与原始输出不入库。
- 未验证：SIMD/CUDA、K-E1/K-E2、官方 GPU、端到端一致、非 GCC 或非 Linux x86_64 环境。

## 收尾

- 验收或停止依据：尚未启动，未实现生产计算入口，未运行 parity。
- 知识处置：pending；既有契约见 docs/COMPUTE.md，未来生产入口契约与测试入口归 COMPUTE/BUILD。
- 剩余义务：无转交义务；技术验收已有证据，待用户确认 COMPUTE/BUILD 增补并评审后结项。
