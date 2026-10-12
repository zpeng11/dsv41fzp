# COMPUTE.md — 专家分组点积 CPU 参照契约

本页定义官方 FP8×FP4 路径阶段约束下的单输出 CPU 规范，不宣称官方 GPU 逐位等价。选择理由见 [D0008](../plan/decisions/d0008-expert-grouped-dot-reference.md)，实施与验收见 [P0012](../plan/p0012-expert-grouped-dot-oracle.md)，验证链见 [MODEL §4](MODEL.md#4-精确性链条)。模型事实只引用 MODEL，格式与既有舍入只引用 [QUANT](QUANT.md)。

## 1. 来源与参照边界

**来源标记**：**[代码]** 官方只读镜像，来源见 [PROVENANCE](../third_party/reference/PROVENANCE.md)；**[格式]** QUANT 的格式与舍入定义；**[D]** D0008 接受的项目计算语义。

- `model.py:27–30,186–194` 明确 UE8M0 scale 及激活、权重沿 K 每 32 元素分组，覆盖 kernel 的激活组默认值。[代码] 模型用途与格式见 [MODEL §2](MODEL.md#2-结构与格式)。
- `kernel.py:478–507,525–557` 给出 FP4→FP32→FP8、块内 GEMM、`(block_dot * scale_a) * scale_b`、跨块累加及最终输出转换的阶段；累加为 FP32，默认输出为 BF16。E2M1 有限值可由 E4M3FN 精确表示，CPU 直接复用解码值。[代码][格式]
- 源码不确定 `T.gemm` 内部归约、设备有效累加精度、编译后融合/重排、cast 舍入及 FTZ；以下顺序 FP32 RNE、无融合和渐进下溢是项目选择。[D]

以下数字是规格与算法约束，不是性能测量；CPU 评审探针的环境、方法和结果见 P0012。

## 2. 输入与失败契约

输入为四份一维原始编码序列：E4M3FN 激活 `a_codes`、沿 K 打包的 E2M1 权重 `w_packed`、E8M0/UE8M0 激活 scale `scale_a` 与权重 scale `scale_b`，仅计算一个输出。[D]

- `K = len(a_codes)`，要求 `K > 0` 且整除模型固定组大小 32；权重字节数为 `K/2`，两份 scale 长度各为 `K/32`。第 j 个块分别使用 `scale_a[j]` 与 `scale_b[j]`。[代码][D] 不支持 128 元素激活组或可配置组大小。
- 每个 packed 字节的低 nibble 对应先出现的 K 元素，后接高 nibble；长度严格匹配，不广播、不补齐、不推断或生成 scale，不接受矩阵、stride 或隐式 flatten。[格式][D]
- 原始编码只收位宽内非负 Python 整数，拒绝 bool、浮点或负的 int8 数值。NaN 激活编码 `0x7F/0xFF` 与 NaN scale 编码 `0xFF` 拒绝；E8M0 有限编码均为严格正值，不能编码零或 Inf。全部 E2M1 nibble 含正负零均合法。[格式][D]
- 输出模式只接受 `"fp32"` 与 `"bf16"`，默认 `"fp32"`。前者返回精确承载 binary32 结果的 Python float，后者返回 BF16 原始整数位模式；模式不改变块内和跨块计算。[D]
- 空输入、长度或分组错误、超宽/负编码、NaN 输入与非法输出模式抛 `ValueError`；非整数编码抛 `TypeError`。元数据先校验，随后全部输入内容校验，最后计算；不会返回坏输入的部分结果。[D] 本入口校验不宣称官方所有入口也如此拒绝。

## 3. 逐步计算与舍入

令 `R32` 表示一次 binary32 RNE（中点取偶）；各步结果立即落到 FP32，不能跨步骤保留额外精度。每个块处理 32 项，块及项均按 K 索引递增。[D]

```text
acc = +0
for j in ascending block order:
    dot = +0
    for r in 0..31:
        p = R32(decode_e4m3fn(a_codes[32*j+r]) * decode_e2m1(weight_code[32*j+r]))
        dot = R32(dot + p)
    after_a = R32(dot * decode_e8m0(scale_a[j]))
    block = R32(after_a * decode_e8m0(scale_b[j]))
    acc = R32(acc + block)
fp32_result = acc
bf16_result = RNE_BF16(fp32_result)
```

禁止树形归约、精确求和后只舍入一次、跨块重排、折叠或交换两份 scale、先逐元素 dequant 后点积、乘加融合以及提前 BF16 舍入。BF16 仅在全部跨块累加完成后进行一次转换，不是 Python binary64 直接转 BF16。[D][格式]

合法输入的中间下溢按渐进下溢处理，上溢产生带符号 Inf；不钳制，不将非有限块贡献当作零或跳过。NaN 传播，异号 Inf 相加产生 NaN，同号 Inf 相加保留 Inf，Inf 与有限数相加保留 Inf。[D] 两份 scale 有限且严格为正，中间 Inf 乘下一份 scale 仍为原符号 Inf。

有限非零乘法的零符号遵守操作数符号异或，下溢到零保留乘积符号；RNE 加法的精确异号相消为正零，同号零相加保留符号、异号零相加为正零。NaN 不承诺 payload、quiet/signaling 状态或符号；有限值（含零）及 Inf 按位检查，NaN 按分类检查。最终 BF16 转换复用 QUANT 的 RNE 与特殊值规则。[D][格式]

## 4. 实现与验证约束

数值语义不依赖宿主浮点加乘、环境舍入或 FTZ，不读取或修改调用方浮点环境，也不改写既有格式 oracle。执行仅依赖 Python 标准库，不初始化官方 GPU 栈或依赖真实权重、设备 dump、在线来源；平台支持只能按实测证据声明。[D]

分阶段观测记录同一执行路径的块内结果、两次 scale 乘法、每次跨块累加和最终输出位模式，不另写测试专用点积。独立期望由位域定义、精确算术与每步独立舍入构造，不调用被测计算或舍入辅助。[D]

## 5. 验证链路由

独立验收后的 Python 参照可供同契约生产 C++ scalar 对照；scalar 验收后，快路径才可对照它。该计算分支不是逐元素 dequant 的已验证后继。[D]

本契约不直接验收 K-E1 W4A16、K-E2 整数改写、官方 GPU 或端到端一致，不改变 W4A16 默认规划。各路径需要同路径参照；相对官方路径的质量偏差还须从共同量化前输入覆盖动态量化链另验。契约接受不代表实现完成。[D]
