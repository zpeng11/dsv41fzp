# MODEL.md — DeepSeek-V4.1-Flash 模型事实

**本页是模型事实的单一引用源。** 其他文档、代码注释、AGENTS.md 引用模型数字时一律指向本页，不得另立数字。

**来源标记**：
- **[卡]** HF 模型卡 · **[cfg]** 官方 `config.json` · **[代码]** 官方 `inference/` 参考实现（repo commit `93cb8a5`，本地镜像 `third_party/reference/`，见其 PROVENANCE.md）· **[报告]** 官方技术报告 arXiv:2609.19969 · **[算]** 推导算术（附式可复核）· **[待]** 待确认（附途径）
- 权威序：代码 > 报告 > 模型卡（三者冲突时以代码为准并在此页标注）

---

## 1. 概览

| 项 | 值 | 来源 |
| --- | --- | --- |
| 架构 | `DeepseekV41ForCausalLM`；**CED = Causal Encoder-Decoder**：40 层 = 20 层因果编码器（0–19）+ 20 层解码器（20–39），解码器全局 KV 由编码器末层隐状态投影产生（详见 §2.8） | [卡][报告][代码] |
| 参数 | backbone 552B + **Engram 条件记忆 196B = 总 748B**；激活 prefill 8B / decode 16B（构成见 §2.8） | [卡][报告] |
| 层数 / hidden | 40 / 5120；词汇表 129280 | [cfg] |
| 注意力 | 64 头 × head_dim 512（qk_rope 64）；KV 为 MQA 式共享 latent；q_lora 1280 / o_lora 1024 / **o_groups 8（块对角分组，见 K-A4）** | [cfg][代码] |
| 上下文 | 训练 64K，YaRN factor 16 外推至 1,048,576（theta 10000）；4K→1M decode FLOPs 仅增约 1/4 | [卡][cfg][报告] |
| MoE | 每层 1 共享专家 + 384 路由专家（激活 6）；moe_intermediate 2304；专家张量共 47,232 个（backbone 46,080 + DSpark 1,152） | [cfg][代码] |
| 残差流 | **mHC 超连接：4 份并行残差拷贝（hc_mult 4）+ Sinkhorn 平衡的 4×4 混合矩阵（hc_sinkhorn_iters 20）——推理期每 token 每子层都执行，非训练专用**（详见 §2.8） | [代码][报告] |
| 投机解码 | DSpark：3 个 nextn 层（40–42），各带 128 专家 MoE（激活 3）、SWA 128；Markov 头 rank 256；block size 5；noise token 128799 = `<｜System｜>` 特殊 token（详见 §2.7） | [cfg][代码] |
| 多模态 | 32 层 ViT（hidden 1024，patch 14，≤1024 image tokens），权重 BF16 在 shard 1；kernel 见 K-V1** | [cfg][代码] |

## 2. 结构与格式

### 2.1 权重分量格式表

| 分量 | 格式 | 规模 | 来源 |
| --- | --- | --- | --- |
| 路由专家 w1/w2/w3（backbone + DSpark 层） | **E2M1 以 int8 打包**（`[out, in/2]`，沿 K 每字节 2 值）+ **UE8M0 scale `[out, in/32]`**（即 MXFP4 权重结构） | ≈555.5B 元素 → **≈302 GB** | [代码] convert.py:174、config `expert_dtype: fp4` |
| **Engram 两张表**（层 1、14） | **FP8 E4M3 `[rows, 256]` + E8M0 scale `[rows, 8]`**；行数 384,006,168 / 384,016,682 | **≈202.8 GB，占 checkpoint 39.7%**（各独占 shard 47/48） | [代码] model.py:307 |
| 注意力 wq_a/wq_b/wkv/wo_a/wo_b、共享专家、indexer.wq_b、engram.wkv、mtp.main_proj | FP8 E4M3 + UE8M0，块 32×32 | ≈7.4 GB | [代码] |
| indexer.wk、weights_proj、compressor.wkv(r=1)、embed/head、markov embed/head、视觉塔、aligner | BF16 | ≈4 GB | [代码] |
| compressor wkv(r>1)/wgate、hc_\*、attn_sink、gate.bias/bias_vl、engram q/k_weight | FP32（confidence proj 在 ckpt 为 BF16） | ≈42M 元素 | [代码] |
| 运行时激活 | FP8 E4M3 dynamic + UE8M0 | — | [cfg] |

### 2.2 体积核算与内存预算

checkpoint 总量 **510.29 GB / 48 shard** [卡]；分解：专家 FP4 302 + engram 表 203 + 其余 FP8 7.4 + BF16 ≈4 GB ✓（字节核算残差 0.08% [代码]）

**[算] 分层预算的硬事实**：专家 302GB + engram 203GB = **505GB，单独这两项已超过 500GB RAM** →
- 非专家热路径（FP8+BF16 ≈11GB）全部常驻 VRAM，无压力；
- **NVMe 兜底是必需**，或 engram 行级缓存（每 token 仅 24 行×2 层 ≈ 12.7KB 随机读，行访问呈 Zipf 分布，热点行缓存可行性是 bench 议题）；
- VRAM 预算（160/192/320/384GB）对 302GB 专家热集分层不变。

### 2.3 数值格式速查（`src/quant/` 规格）

| 格式 | 关键性质 |
| --- | --- |
| E2M1 | 幅值 {0,.5,1,1.5,2,3,4,6}；**×2 → 精确整数 {0,1,2,3,4,6,8,12}** → int8 数据通路位精确映射（IMMA/dp4a/VNNI）的根本依据 [算]；积 ≤12×127，32 项累加 ≤48,768，int32 无溢出 [算] |
| UE8M0 | 8 位纯指数；折出点积为精确指数运算 [算] |
| E4M3 | →BF16 精确；sm120 原生 TC，sm80 需反量化或 int8 格映射 |
| KV 双格式 | **主 KV：E2M1 + E4M3 scale/16ch（RoPE 之后量化）**；**索引 K：E2M1 + UE8M0 scale/32ch（MXFP4）**；SWA KV：FP8 E4M3 + UE8M0/32 [代码] |
| 参考实现注记 | 官方参考实现中 FP4 反量化回 BF16 参与计算（"FP4 只省存储"）；本引擎对索引 QK 与专家 GEMM 做格映射加速是对官方行为的**精确等价改写**（parity 锁定） | [代码] |

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
| K-Q1 | E2M1 ⇄ int8 整数格（LUT、nibble 打包/展开，沿 K 每字节 2 值） | `ref/`（与官方 convert.py 逐位一致） |
| K-Q2 | UE8M0 / E8M0 指数应用与折叠 | 同上 |
| K-Q3 | E4M3 ⇄ BF16；KV 双 scale 变体（主 KV E4M3/16、索引 K UE8M0/32、SWA FP8/32） | 同上 |

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

```
官方参考实现 `third_party/reference/inference/`（torch+tilelang，贪心逐 token 一致自证）
  └─ ref/  Python FP32 oracle（精确 MX 反量化，语义镜像 model.py）
       └─ src/quant/  C++ scalar 编解码（与 ref/ 及官方 convert.py 逐位一致）
            └─ 各 kernel parity（只对照 quant 已验证 scalar 路径）
                 └─ 双 arch 顶层锁：sm80 与 sm120 端到端贪心逐 token 一致（冻结 prompt 集）
```

权重侧零容差（仅累加顺序差异）；激活侧（A16 vs 官方 FP8 dynamic）为已量化记录的偏差项。§2.9 三项 **[R]** 特性无参考实现，其验证只能靠质量门禁三件套（困惑度 / needle / 贪心 parity）+ 官方 API 输出对照。Engram 三表（token_map/multipliers/primes）与官方逐位一致是 §2.6 不变量的硬要求。


## 5. 采样建议（官方）

temperature 1.0，top_p 0.95 或 1.0，max_tokens ≥ 256K。[卡] chat_template.jinja（15KB）在 repo 根目录 [代码]。

## 6. 官方推理栈（已收编：`third_party/reference/`）

repo 自带最小推理栈 `inference/`（torch≥2.10、tilelang==0.1.8；无 HF transformers 建模文件）；其中 `engram.py` 为 Engram 编码/哈希官方实现（§2.6）。`encoding/` 为**聊天编码**官方实现（chat 模板、DSML 工具调用、思维链与 reasoning effort、图像占位；含 5 组输入/期望输出测试）。来源、commit 与维护规则见 `third_party/reference/PROVENANCE.md`（只读镜像，分析结论写入 docs/ 不写回）。

