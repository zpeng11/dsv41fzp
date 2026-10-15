# MODEL.md — DeepSeek-V4.1-Flash 模型事实

**本页是模型事实的单一引用源。** 其他文档、代码注释、AGENTS.md 引用模型数字时一律指向本页，不得另立数字。

**来源标记**：
- **[卡]** HF 模型卡 · **[cfg]** 官方 `config.json` · **[代码]** 官方 `inference/` 参考实现（repo commit `93cb8a5`，本地镜像 `third_party/reference/`，见其 PROVENANCE.md）· **[报告]** 官方技术报告 arXiv:2609.19969 · **[ckpt]** 官方 checkpoint 实测：HF 快照 `2cba9e42`（48 shard），经 `tools/ckpt_inventory.py` 解析全部 shard 头部并扫描全部张量的原始编码（方法与命令见 [BUILD §7](BUILD.md#7-测试清单)，证据见 [P0016](../plan/p0016-checkpoint-inventory.md)）· **[算]** 推导算术（附式可复核）· **[待]** 待确认（附途径）
- 权威序：代码 > 报告 > 模型卡（三者冲突时以代码为准并在此页标注）；checkpoint 的存储 dtype、shape 与字节数以 [ckpt] 为准

---

## 1. 概览

| 项 | 值 | 来源 |
| --- | --- | --- |
| 架构 | `DeepseekV41ForCausalLM`；**CED = Causal Encoder-Decoder**：40 层 = 20 层因果编码器（0–19）+ 20 层解码器（20–39），解码器全局 KV 由编码器末层隐状态投影产生（详见 §2.8） | [卡][报告][代码] |
| 参数 | backbone 552B + **Engram 条件记忆 196B = 总 748B**；激活 prefill 8B / decode 16B（构成见 §2.8） | [卡][报告] |
| 层数 / hidden | 40 / 5120；词汇表 129280 | [cfg] |
| 注意力 | 64 头 × head_dim 512（qk_rope 64）；KV 为 MQA 式共享 latent；q_lora 1280 / o_lora 1024 / **o_groups 8（块对角分组，见 K-A4）** | [cfg][代码] |
| 上下文 | 训练 64K，YaRN factor 16 外推至 1,048,576（theta 10000）；4K→1M decode FLOPs 仅增约 1/4 | [卡][cfg][报告] |
| MoE | 每层 1 共享专家 + 384 路由专家（激活 6）；moe_intermediate 2304；专家张量共 47,232 个（backbone 46,080 + DSpark 1,152） | [cfg][代码][ckpt] |
| 残差流 | **mHC 超连接：4 份并行残差拷贝（hc_mult 4）+ Sinkhorn 平衡的 4×4 混合矩阵（hc_sinkhorn_iters 20）——推理期每 token 每子层都执行，非训练专用**（详见 §2.8） | [代码][报告] |
| 投机解码 | DSpark：3 个 nextn 层（40–42），各带 128 专家 MoE（激活 3）、SWA 128；Markov 头 rank 256；block size 5；noise token 128799 = `<｜System｜>` 特殊 token（详见 §2.7） | [cfg][代码] |
| 多模态 | 32 层 ViT（hidden 1024，patch 14，≤1024 image tokens），权重 BF16；视觉塔与 aligner 独占 shard 1，图像分隔嵌入 `image_start/end/newline` 在 shard 2；kernel 见 K-V1 | [cfg][代码][ckpt] |

## 2. 结构与格式

### 2.1 权重分量格式表

格式列是 checkpoint 的存储 dtype；运行时 dtype 不同者在格式列注明。

| 分量 | 格式 | 规模 | 来源 |
| --- | --- | --- | --- |
| 路由专家 w1/w2/w3（backbone + DSpark 层） | **E2M1 以 int8 打包**（`[out, in/2]`，沿 K 每字节 2 值）+ **UE8M0 scale `[out, in/32]`**（即 MXFP4 权重结构）；运行时视为 `float4_e2m1fn_x2` | 557,171,343,360 个 FP4 值 → **296.00 GB**（打包权重 278.59 + scale 17.41） | [ckpt]；[代码] convert.py:174–181、config `expert_dtype: fp4` |
| **Engram 两张表**（层 1、14） | **FP8 E4M3 `[rows, 256]` + E8M0 scale `[rows, 8]`**；行数 384,006,168 / 384,016,682 | **202.76 GB，占 checkpoint 39.7%**；各与本层 `engram.wkv`、`q/k_weight` 同在 shard 47/48，两 shard 不含其他张量 | [ckpt]；[cfg]；[代码] model.py:307 |
| 注意力 wq_a/wq_b/wkv/wo_a/wo_b、共享专家、indexer.wq_b、engram.wkv、mtp.main_proj | FP8 E4M3 + UE8M0，块 32×32；`wo_a` 由转换器反量化为 BF16 后运行 | 7.41 GB | [ckpt]；[代码] convert.py:157–173 |
| indexer.wk/weights_proj/k_norm、compressor wkv/wgate/norm、engram q/k_weight、embed/head、markov embed/head、confidence proj、视觉塔、aligner、各 RMSNorm | BF16；compressor wkv(r>1)/wgate、head、markov head、confidence proj 与视觉塔 RMSNorm 运行时构造为 FP32，加载时拓宽 | 3.95 GB | [ckpt]；[代码] model.py:444–448,1005–1006,1092；vision.py:28 |
| hc_\*、attn_sink、gate.bias/bias_vl | FP32 | 42,307,282 元素（0.17 GB） | [ckpt] |
| 运行时激活 | FP8 E4M3 dynamic + UE8M0 | — | [cfg] |

**全量编码扫描** [ckpt]：E4M3 与 E8M0 中没有 NaN 编码，BF16 与 FP32 中没有 NaN/Inf。E8M0 scale 编码范围 109–128（2^−18～2^1），其中路由专家 scale 为 115–128。路由专家的 E2M1 码点 8（负零）共 32,765,814,399 个，占全部 FP4 值的 5.88%，与码点 0（32,766,317,383 个）数量相当，因此 [QUANT §2](QUANT.md#2-位模式与零符号) 所述 `ref/` 与官方 FP4_TABLE 的零符号差异在真实权重中普遍出现。DSpark 层的 `gate.bias`/`bias_vl` 全为 +0。

### 2.2 体积核算与内存预算

checkpoint 总量 **510.29 GB / 48 shard** [卡][ckpt]；分解：专家 FP4 296.00 + engram 表 202.76 + 其余 FP8 7.41 + BF16 3.95 + FP32 0.17 = 510.29 GB，与 index 的 `total_size`（510,286,023,000 字节）逐字节相等 [ckpt]

**[算] 分层预算的硬事实**：专家 296.0GB + engram 202.8GB = **498.8GB，500GB RAM 上仅余约 1.2GB，装不下系统与其余常驻项** →
- 非专家热路径（FP8+BF16 ≈11.4GB）全部常驻 VRAM，无压力；
- **NVMe 兜底是必需**，或 engram 行级缓存（每 token 仅 24 行×2 层 ≈ 12.7KB 随机读，行访问呈 Zipf 分布，热点行缓存可行性是 bench 议题）；
- VRAM 预算（160/192/320/384GB）对 296GB 专家热集分层不变。

### 2.3 数值格式速查（`src/quant/` 规格）

位模式、特殊值、FP32 运算与 BF16 舍入契约见 [QUANT.md](QUANT.md)；格式 oracle 不替代生产 `src/quant` 或 GPU/kernel parity 验收。

| 格式 | 关键性质 |
| --- | --- |
| E2M1 | 幅值 {0,.5,1,1.5,2,3,4,6}；**×2 → 精确整数 {0,1,2,3,4,6,8,12}** → int8 数据通路位精确映射（IMMA/dp4a/VNNI）的根本依据 [算]；积 ≤12×127，32 项累加 ≤48,768，int32 无溢出 [算] |
| UE8M0 | 8 位纯指数；折出点积为精确指数运算 [算] |
| E4M3 | →BF16 精确；sm120 原生 TC，sm80 需反量化或 int8 格映射 |
| KV 双格式 | **主 KV：E2M1 + E4M3 scale/16ch（RoPE 之后量化）**；**索引 K：E2M1 + UE8M0 scale/32ch（MXFP4）**；SWA KV：FP8 E4M3 + UE8M0/32 [代码] |
| 参考实现注记 | 官方专家路径把 FP4 经 FP32 转为 FP8 参与 GEMM，再对累加值应用 scale；KV 的 inplace 量化回写输入 dtype，不可总括为所有 FP4 路径先反量化到 BF16。索引 QK 与专家 GEMM 的整数格改写须另行验证运算顺序、舍入与 parity，格式解码精确不单独证明计算等价 | [代码] kernel.py:167–172、538–553；详见 [QUANT §5](QUANT.md#5-验证链边界) |

### 2.4 KV cache（890 B/token 精确分解，已验证）

全局持久 KV 仅由 4 个 kv_source 层产生 [代码]：编码器 3 层（2/8/14，ratio 2，每 2 token 压成 1 latent）+ 解码器 1 层（20，ratio 1）= **每 token 2.5 个 latent**。

| 项 | 每 latent 字节 | 说明 |
| --- | --- | --- |
| 主 KV 数据 | 512 × 0.5 = 256 B | E2M1 |
| 主 KV scale | 512/16 = 32 × 1 = 32 B | E4M3 |
| 索引 K 数据 | 128 × 0.5 = 64 B | E2M1 |
| 索引 K scale | 128/32 = 4 × 1 = 4 B | UE8M0 |
| **小计** | **356 B** | ×2.5 latent/token = **890 B/token 整** [算，与官方数字精确吻合] |

- 1M 上下文全局 KV ≈ 0.93 GB [算]；890B 为渐近值（ratio-2 尾部不完整组存 fp32 状态）[代码]
- **SWA KV 不持久**：40 层各一个 128 槽 FP8 环形缓冲；生产中 SWA 重放缓存驻留主机 DRAM（~10%，分钟级 TTL），全局 KV 持久 ≥72h [报告]
- **compress_ratios 精确值（43 项 = 40 backbone + 3 DSpark，假设已证实）**：`[0,0] + [2]×18 + [1]×20 + [0,0,0]`；层 0-1 纯 SWA，2-19 每 2→1，20-39 每 1→1，40-42（DSpark）纯 SWA [代码]
- Compressor：r>1 用可学习 softmax 门池化（fp32 计算），r=1 退化普通投影；压缩 latent 的 RoPE 用独立 theta=160000；decode 不完整组缓存在 kv_state/score_state [代码]
- 注意力输出做**逆 RoPE**（缓存保持已旋转共享形态）；可学习 attn_sink [代码]

### 2.5 CSA2 = Compressed Sparse Attention 2

每层静态指派三模式之一 [报告][代码]：

| 模式 | 层 | 行为 |
| --- | --- | --- |
| **Full** | [2,8,14,20]（= kv_source ∩ index_source） | 自跑 Compressor 产 latent + 从 latent 投影索引 K + 跑 indexer 选 top-512 |
| **Reindex** | [24,28,32,36] | 复用共享 main KV 与索引 K，仅算自己的索引 Q 重打分，**只在候选池内检索** |
| **Reuse** | 其余 compress_ratio>0 层 | 直接复用最新 top-512 索引 |

- 层分组：编码器 2-19 为 3 组×6 层（组首 Full + 5 Reuse）；解码器 20-39 为 5 组×4 层（首组 1 Full + 3 Reuse，后 4 组 1 Reindex + 3 Reuse）[报告]
- **层级稀疏索引器**：层 20（candidate_source）块级候选 = top-2048 块 × 块 8 = **16,384 候选位置**（含 query 的不完整块强制 pin）；Reindex 层仅在池内检索 → **索引成本与上下文长度无关** [报告][代码]
- Indexer 打分：fp4 化 Q 与共享索引 K einsum → `relu` → 乘 `weights_proj(x)`（32 头加权求和）；压缩组仅在 query 越过其最后 token 后可见；topk 后按位置重排 [代码]
- 主注意力：拼接 `[SWA 128; 全局 top-512]` 一次核调用，**每 query 最多 640 个 KV**；在线 softmax + 无效槽置零 + attn_sink [代码]

### 2.6 Engram 条件记忆（层 1、14；K-G1 规格）

两个模块各置于层 1/14 的块 attention 之前；表格式与规模见 §2.1，内存预算与预取议题见 §2.2。规格来源 `inference/engram.py` + `model.py` [代码]，每 token 流程：

**① 压缩词表映射**（129,280 → 99,092 类）：token 逐个 decode 后归一化，同归一化结果的 token 折叠为同类（`" The"`/`"the"`/`"THE"` 同哈希）。归一化链：NFKC → NFD → StripAccents → Lowercase → 空白折叠（`[ \t\r\n]+` → 单空格）→ 孤空格经私用区哨兵 `\ue000` 保护 → Strip → 还原；含 U+FFFD 的部分字节 token 以原始 token 串为键；归一化后为空回退原文。按首次出现顺序编号。
**不变量 [代码]**：所有哈希乘子由压缩词表大小推导（int64 防溢出界同样依赖它），不一致会**静默重排整张表**——官方在初始化 assert 锁定；本引擎的 `token_map` / `multipliers` / `primes` 三表必须在 pack 时预生成并与之逐位一致。

**② 乘子**：每（层 × 回看阶）一个；RNG 种子 `10007 × layer_id`（各层哈希不同）；在 `[0, int64max/V/2)` 均匀采样后 ×2+1（恒奇，`token × multiplier` 不溢出 int64）。

**③ 素数桶**：每（阶 × 头）一个独立素数，从 15,999,999（`engram_vocab_size−1`）向上按序抽取、跨层全局不重复 → 桶区间互斥。每层 3 阶 × 8 头 = 24 段；两表行数 384,006,168 / 384,016,682 恰为各自 24 个素数之和（差 10,514 = 第二层续抽所致）[算，交叉验证 ✓]。

**④ 哈希构造**：压缩 id 序列上，每位置回看 0..3 共 4 个 token；越界或遇 **DEAD（−1，图像 span 掩码）** 即阻断（阻断累积——n-gram 不跨图像 span 与序列起点）；阻断位填 pad 的压缩 id。`rolling = XOR(token × multiplier)`，第 i 步后的 rolling 即 (i+1)-gram 哈希 → `mod` 该（阶, 头）素数 → `+` 段偏移 = 表行号。输出 `[B, L, 2 层, 24 列]`。

**⑤ 状态**：压缩 id 缓存 `[B, max_seq_len]` int64 跨 prefill/decode 维护——**会话缓存设计的一个状态项**。

**⑥ 消费**（model.py）：24 行 × 256 维 FP8 反量化（每行 264B）→ `wkv` 投影 key(4×5120) + value(5120) → 门 `sigmoid(copysign(sqrt(clamp(|归一化点积|,1e-6)), dot))` → `h + gate·value` 广播到 4 份残差。

### 2.7 DSpark 投机解码

- 3 个 DSpark 块（层 40-42），纯 SWA 128 滑窗；各自 128 专家 MoE top-3 [代码]
- mtp.0 带 `main_proj`（拼接主干 37/38/39 层注意力输入、对 hc 4 份取均值）[代码]
- **起草**：输入 `[真实token, noise×4]`（noise=128799=`<｜System｜>`，不携带信息）；5 个草稿位一次并行前向，注意 `[128 环形窗 + 块内 5]`，**块内无因果掩码**；forward_head 逐位加 Markov bigram 偏置后采样，产出 1+5=6 个 id [代码]
- **验证**：confidence head（fp32）预测逐位置条件接受概率 → 前缀存活率 → 调度器结合引擎吞吐曲线动态选验证长度 [报告]；**验证/调度循环不在仓库**（见 §2.9）
- 训练时机：预训练后单独阶段训练，backbone 冻结（与 V3 MTP 不同）[报告]

### 2.8 CED 与 mHC（残差流）

**CED** [报告式(1)+代码]：
- 解码器（21-39）全局 KV 不由自身隐状态产生：层 20 的 Compressor 从编码器末层隐状态产 latent，经 `shared_attn.compress_kv` 供全部解码层复用（代码实现为共享一份；报告通式为逐层 W^KV_l 投影，以代码为准）
- **prefill：多数 token 只跑前 20 层（≈8B 激活），后 20 层全局 KV 由投影获得；仅 prompt 末 128 token 过解码器层（Decoder SWA Bounded Replay）→ prefill 计算近似减半**（面向 input-heavy agent 负载）[报告]
- decode：每 token 走全部 40 层。激活核算 [算]：每层 ≈376.6M（注意力 126.6M + MoE 247.7M + gate/hc 2.5M）×40 ≈15.1B + embed/head 1.32B ≈ **16B** ✓
- Engram 196B 与 DSpark 层不计入激活参数（稀疏查表 / 仅投机时运行）[代码]

**mHC（mega hyper-connections）** [代码+报告]：
- 残差流为 4 份并行拷贝（`h.unsqueeze(2).repeat(1,1,4,1)`）
- 每子层由 `hc_mixes` 一次投影导出 pre（sigmoid+eps）/ post（2·sigmoid）/ comb（4×4）系数；**comb 经 Sinkhorn 20 次行列归一化为双随机矩阵——kernel.py 中实现在推理路径上，每 token 每子层执行**；系数被下一子层消费（Single-Pass）；生产融合为 Mega-mHC 单核

### 2.9 参考实现与生产引擎的差异（本项目的实现风险项）

官方 `inference/` 自述"可读参考实现，非生产引擎"。**以下特性只有报告描述、没有参考代码**，是本引擎需要自行设计并最先 bench 验证的项：

1. **编码器-only prefill**（参考实现对 prefill 也跑满 40 层）
2. **SWA Bounded Replay**（编码器侧前缀重放 + 解码器侧末 128 token 重放；含主机 DRAM 池与 TTL 语义）
3. **DSpark 置信度调度验证循环**（仓库只有 draft 前向 + confidence 打分）

## 3. 本项目重点 kernel 清单

编号 K-\<域>\<序>；parity 参照必须来自 §4 链条的上一级。**[R]** = 无参考代码，规格来自报告（对应 §2.9）。

### 3.1 格式编解码（`src/quant/`）

| 编号 | 内容 | parity 参照 |
| --- | --- | --- |
| K-Q1 | E2M1 ⇄ int8 整数格（LUT、nibble 打包/展开，沿 K 每字节 2 值） | `ref/`（与官方 convert.py 逐位一致，E2M1 码点 8 除外：本仓解码为负零，官方 FP4_TABLE 为正零，见 [QUANT §2](QUANT.md#2-位模式与零符号)） |
| K-Q2 | UE8M0 / E8M0 指数应用与折叠 | 逐元素 scale 应用对照 [QUANT](QUANT.md) 的格式/反量化参照；点积 scale 折叠须按计算契约另验（见 §4） |
| K-Q3 | E4M3 ⇄ BF16；KV 双 scale 变体（主 KV E4M3/16、索引 K UE8M0/32、SWA FP8/32） | 格式转换与逐元素反量化对照 `ref/formats.py`、`ref/dequant.py`（[QUANT](QUANT.md)） |

### 3.2 专家路径（`src/kernels/`）

| 编号 | 内容 | 备注 |
| --- | --- | --- |
| K-E1 | sm80 W4A16：E2M1 飞行反量化→BF16 分组 GEMV/GEMM（6 专家 ×3 矩阵 ×5120×2304） | decode 默认路径 |
| K-E2 | sm80 W4A8-int8：整数格 IMMA m16n8k16.s8 + 每 32K UE8M0 epilogue | prefill/批量；quality_gate 门禁，默认关 |
| K-E3 | sm120 原生块缩放 FP4 MMA | 与 K-E1 端到端贪心一致锁 |
| K-E4/K-E5 | sm120 原生 FP8 / sm80 FP8→BF16 反量化 GEMM（注意力、共享专家、engram.wkv、main_proj） | |
| K-E6 | cpu AVX-512 VNNI `vpdpwssd` 整数格（NUMA 亲和） | RAM 层专家 |
| K-E7 | 共享专家 + DSpark 128 专家小 MoE（K-E1/E2/E3 缩幅复用；共享专家为 FP8 权重） | |

### 3.3 注意力（`src/attn/`）

| 编号 | 内容 | 备注 |
| --- | --- | --- |
| K-A1 | Indexer：fp4 Q × 共享索引 K（可用 K-E2 同款格映射）→ relu → 32 头加权 → top-512；Full/Reindex/Reuse 三模式；候选池 16,384 限制 | 规格全确定 |
| K-A2 | 主稀疏注意力：`[SWA 128; 全局 512]` 拼接、每 query ≤640 KV、在线 softmax、attn_sink、无效槽、逆 RoPE、**内联 KV 双格式反量化** | |
| K-A3 | Compressor：softmax 门池化（fp32）、theta 160000 RoPE、decode 部分组状态 | |
| K-A4 | 投影组：wq_a/wq_b/wkv；**wo_a 块对角分组 einsum（8 组 ×4096→1024）+ wo_b 跨组混合**——需分组 GEMM，勿按普通 Linear 处理 | 代码注释明示 fp8 分组 GEMM 可省半内存 |
| K-A5 | SWA 128 槽环形缓冲（40 层）+ FP8 量化（RoPE 后）；**[R] Bounded Replay（编码器/解码器两侧）** | |

### 3.4 mHC 残差（`src/core/`，新增域）

| 编号 | 内容 |
| --- | --- |
| K-H1 | mHC 混合：hc_mixes 投影 → pre/post 门 + 4×4 Sinkhorn(20 迭代) 双随机化，每 token 每子层；生产形态 Mega-mHC 融合单核。与所有子层耦合 |

### 3.5 路由（`src/core/`）

| 编号 | 内容 |
| --- | --- |
| K-R1 | **精确算式**：gate 全 fp32 → linear → sqrt(softplus) → `(scores+bias).topk(6)`（**bias 只选不缩放**；图像 span 用 bias_vl）→ 权重取未加 bias 的分 → 被选 6 个内归一化（防零 1e-20）→ ×1.5；专家内 swiglu_limit=10 钳位（up ±10 / gate ≤10，为低精度激活保范围）。sqrt(softplus) 与 topk+gather 的融合实现是 kernel 点 |

### 3.6 Engram 检索（`src/attn/` 子模块）

| 编号 | 内容 |
| --- | --- |
| K-G1 | Engram 检索：完整编码与哈希规格见 **§2.6**（压缩词表 → 24 素数桶 XOR 滚动哈希 → 行号）。kernel 侧 = token_map 查表 + 滚动哈希 + 24 路行取数（FP8 行 264B，反量化）+ `wkv` 投影 key(4×5120)+value(5120) + 门控 → 4 份残差；含跨 prefill/decode 的压缩 id 缓存状态（§2.6 ⑤）；生产经 RDMA 预取自主机内存（第 1 模块预取与第 1 块计算重叠）→ 本项目即 RAM/NVMe 预取设计 |

### 3.7 CED prefill（`src/prefill/`）

| 编号 | 内容 |
| --- | --- |
| K-C1 | **[R]** 编码器-only prefill：prompt 主体仅 20 层 + H_20 投影全局 KV + 末 128 token 过解码器层；与 chunked prefill、CPU-GPU 重叠合流 |

### 3.8 激活量化 / 数据搬运 / 投机（`src/aq/`、`src/platform/`、`src/spec/`）

| 编号 | 内容 |
| --- | --- |
| K-X1 | 动态激活量化 E4M3+UE8M0（sm120 原生路径）；Ampere 默认 A16；int8 变体同 K-E2 门禁 |
| K-M1 | GPU P2P 专家拉取（NVLink 或 PCIe，拓扑与资质见 [HARDWARE.md](HARDWARE.md)）、pinned staging、NVMe direct I/O（**含 engram 行预取通道**）、NUMA 本地放置、逐层 checksum |
| K-D1 | DSpark 起草前向（块内无因果掩码的 640-KV 注意力复用 K-A2 缩幅）+ **[R]** 置信度调度验证循环 |

### 3.9 视觉（`src/vision/`）

| 编号 | 内容 |
| --- | --- |
| K-V1 | 视觉塔 32 层 ViT（BF16，hidden 1024，patch 14，downsample 3）+ aligner + 图像预处理（min_pixels 295936，≤1024 image tokens）；GEMM 复用 K-E5（BF16 常规路径），无量化特殊性；图像 span 触发路由 `bias_vl`（K-R1）。规格对象：`third_party/reference/inference/{vision.py,image_processor.py}` |

## 4. 精确性链条

各箭头表示参照与验收依赖，不表示源码推导即取得逐位等价或端到端证据；实现与验收状态以 `plan/STATUS.md` 和对应 P 条目为准。

```
官方参考实现 `third_party/reference/inference/`（布局与运算阶段来源）
  ├─ 格式与逐元素反量化：规格定义 + ref/formats.py、ref/dequant.py
  │    └─ src/quant/ C++ scalar codec 与给定 scale dequant
  │         └─ 同契约格式/逐元素 dequant kernel parity
  └─ 计算：官方源码阶段约束 + 独立精确算术 + 明确的 CPU 数值契约
       ├─ BF16 动态激活量化 CPU 规范：ACTQUANT → Python 量化参照
       │    └─ src/quant/ C++ scalar 激活量化 → 同契约激活量化 kernel parity
       ├─ 官方 FP8×FP4 阶段约束下的 CPU 规范：COMPUTE → Python 计算参照
       │    └─ src/quant/ C++ scalar 专家分组点积 → 同契约计算 kernel parity
       ├─ K-E1 W4A16：BF16 激活、权重反量化落点及累加规则一致的计算参照
       └─ K-E2 W4A8-int8：整数改写计算参照与相对官方路径的独立验证
```

格式与逐元素反量化契约见 [QUANT](QUANT.md)；BF16 动态激活量化 CPU 规范见 [ACTQUANT](ACTQUANT.md)，选择理由见 [D0010](../plan/decisions/d0010-act-quant-reference.md)，其输出可直接供给计算参照。官方 FP8×FP4 阶段约束下的 CPU 规范见 [COMPUTE](COMPUTE.md)，选择理由见 [D0008](../plan/decisions/d0008-expert-grouped-dot-reference.md)。每段快代码只对照上一级已验证的同契约 scalar；格式正确不证明归约、移动 scale、提前 BF16 或整数改写等价。官方 GPU 的归约、融合及 cast/FTZ 须设备仲裁，不能由 CPU 契约推定。

**验证链覆盖矩阵**（契约与上一级路由的唯一边界来源；各契约页只链接本表，验收状态仍以对应 P 条目为准）：

| 入口 | 契约 | 是谁的上一级 | 不能据此验收 |
| --- | --- | --- | --- |
| `ref/formats.py` | [QUANT §2～§3](QUANT.md#2-位模式与零符号) | 生产 codec | 有损编码、scale 生成 |
| `ref/dequant.py` | [QUANT §4](QUANT.md#4-反量化-api-与布局) | 生产给定 scale 反量化 | GEMM/GEMV、累加后乘 scale、移动或折叠 scale、提前 BF16、整数点积、动态量化 |
| `df_quant` codec | [QUANT §6](QUANT.md#6-生产-c-codecsrcquant) | 同契约格式 kernel；覆盖 K-Q1 与 K-Q3 的 E4M3FN/BF16 解码和 FP32→BF16 | 乘 scale（由下一行验收）、BF16→E4M3 编码、点积 scale 折叠 |
| `df_quant` 反量化 | [QUANT §7](QUANT.md#7-给定-scale-的生产-scalar-反量化) | 逐元素 dequant 类 kernel | 同 `ref/dequant.py` 行；不笼统表示 K-Q1～K-Q3 全部完成 |
| `ref/compute.py` | [COMPUTE §1～§4](COMPUTE.md#1-来源与参照边界) | 生产专家分组点积 | 逐元素 dequant 的后继、K-E1 W4A16、K-E2 整数改写、scale 生成 |
| `df_quant` 专家分组点积 | [COMPUTE §6](COMPUTE.md#6-生产-c-scalarsrcquant) | 同契约计算快路径 | 同 `ref/compute.py` 行 |
| `ref/act_quant.py` | [ACTQUANT](ACTQUANT.md) | 同契约生产激活量化 scalar | K-X1 kernel（须经生产 scalar）、inplace/KV 量化、`fp4_act_quant`、权重量化、K-E1/K-E2、完整专家 MLP |
| `df_quant` 单行激活量化 | [ACTQUANT §7](ACTQUANT.md#7-生产-c-scalarsrcquant) | 同契约激活量化快路径 | 上一行除 K-X1 kernel 外的边界同样适用；单行入口不单独验收完整线性层 |
| `ref/expert.py` | [EXPERT](EXPERT.md)；复用激活量化、逐行点积并独立验证非线性 | 同契约生产 SwiGLU bridge 与单 token FP4 专家生产实现 | batch、路由选择/多专家聚合、共享专家、生产加载器、真实完整三矩阵与真实模型激活、模型质量；合成逐阶段验证不替代这些证据 |
| `df_quant` SwiGLU bridge | [EXPERT §5](EXPERT.md#5-生产-swiglu-bridge)；上一级为 `ref/expert.py::swiglu` 及独立区间边界期望 | 同契约完整专家生产实现的非线性桥 | W1/W3 双投影、共享输入量化、第二次激活量化、W2、完整专家统一失败原子性、路由权重生成、多专家/共享专家、SIMD/CUDA 与融合 |
| `df_quant` 专家线性层 | [COMPUTE §7](COMPUTE.md#7-单-token-专家线性层)；上一级为 Python 激活量化与逐行点积参照组合 | 同契约单 token 完整线性快路径 | batch、非线性与专家串接、路由和共享专家、K-E1/K-E2；不能由组件已验收替代矩阵级验收 |
| `df_quant` 完整 FP4 专家 | [EXPERT §6](EXPERT.md#6-生产完整专家)；上一级为 `ref/expert.py::expert/trace_expert`，证据归 [P0022](../plan/p0022-expert-scalar.md) | 同契约单 token 单专家快路径；组合共享输入量化、W1/W3、bridge、第二次量化及 W2 | router/权重生成、多专家聚合、共享 FP8 专家、batch、生产加载器、真实模型激活和质量；参数与运行中拒绝的保证不同，不承诺 scratch 全事务回滚 |
| `tools/ckpt_inventory.py` | 本页 §2.1～§2.2 的 [ckpt] 事实；运行方法见 [BUILD §7](BUILD.md#7-测试清单) | 未来 `src/artifact` 读取器的张量清单 | 码点统计以外的数值语义、pack 格式与转换、读取性能、其他快照 |
| `tools/ckpt_expert.py` | §2.1 路由专家行布局与既有 ACTQUANT/COMPUTE 契约；读取和运行约束见 [BUILD §7](BUILD.md#7-测试清单) | 真实权重选定行的集成验证入口，不新增数值 oracle；读取由独立合成字节验收，计算对照既有 Python 参照组合 | 未选权重载荷、真实模型激活、完整线性层/专家 MLP、生产 artifact 读取器；人工输入的局部 parity 不证明模型质量 |
| `tools/ckpt_linear.py` | §2.1 路由专家布局与 [COMPUTE §7](COMPUTE.md#7-单-token-专家线性层)；读取和运行约束见 [BUILD §7](BUILD.md#7-测试清单) | 固定专家完整 w1 的集成验证入口，不新增数值 oracle；独立验收读取字节，全输出对照 Python 参照组合 | 未选矩阵、真实模型激活、完整专家 MLP、生产 artifact 读取器；人工输入 parity 不证明模型质量 |
| `tools/ckpt_expert_mlp.py` | §2.1 路由专家布局与 [EXPERT §6](EXPERT.md#6-生产完整专家)；读取和运行约束见 [BUILD §7](BUILD.md#7-测试清单) | 固定专家完整 w1/w3/w2 的集成验证入口；读取由独立合成字节核对，生产完整调用逐阶段对照既有专家 oracle | 其他专家载荷、真实模型激活、MoE 聚合、生产 artifact 读取器和模型质量；不新增数值 oracle |

全部条目均不覆盖 SIMD/CUDA、官方 GPU 逐位等价、checkpoint 转换、K-E1 质量偏差与端到端一致；验证环境限 [BUILD §1](BUILD.md#1-支持环境) 的 ENV-A。

W4A16 decode 默认规划不变。实现 parity 与相对官方 FP8 dynamic 路径的质量偏差分别验收；后者须从共同量化前激活覆盖动态量化链并记录环境，不能以路径差异本身代替质量测量。双 arch 顶层锁仍按冻结 prompt 集独立验收 sm80/sm120 端到端贪心逐 token 一致，不由局部 oracle 保证。

§2.9 三项 **[R]** 特性无参考实现，其验证只能靠质量门禁三件套（困惑度 / needle / 贪心 parity）+ 官方 API 输出对照。Engram 三表（token_map/multipliers/primes）与官方逐位一致是 §2.6 不变量的硬要求。


## 5. 采样建议（官方）

temperature 1.0，top_p 0.95 或 1.0，max_tokens ≥ 256K。[卡] chat_template.jinja（15KB）在 repo 根目录 [代码]。

## 6. 官方推理栈（已收编：`third_party/reference/`）

repo 自带最小推理栈 `inference/`（torch≥2.10、tilelang==0.1.8；无 HF transformers 建模文件）；其中 `engram.py` 为 Engram 编码/哈希官方实现（§2.6）。`encoding/` 为**聊天编码**官方实现（chat 模板、DSML 工具调用、思维链与 reasoning effort、图像占位；含 5 组输入/期望输出测试）。来源、commit 与维护规则见 `third_party/reference/PROVENANCE.md`（只读镜像，分析结论写入 docs/ 不写回）。

