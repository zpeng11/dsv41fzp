# dsv41fzp

DeepSeek-V4.1-Flash 单机柜推理引擎。单一模型特化：不追求通用架构，只把这一个模型在一种硬件形态上跑好。

- **目标模型**：[deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)（552B 总参数；MXFP4 专家权重 + FP8 其余权重，checkpoint 约 510 GB）
- **目标硬件**：入门级单机柜 —— 双路 8/12 通道 DDR4/5 服务器 CPU（>500 GB RAM、AVX-512/VNNI）+ 2/4 张 NVLink 互联的 A100/A800/RTX PRO 6000（单卡 <100 GB VRAM）

## 状态

**骨架阶段。** 目录结构已建立；`docs/` 与 `plan/` 下的文档**尚未落盘**——它们是影响后续走向的重要文档，正在逐条核实确认内容后填入。在本 README 引用的文档存在之前，以下事实页即为唯一信息源。

## 目录

```
plan/           分阶段计划与决策记录（待确认后落盘）
docs/           事实文档：模型、包格式、硬件、原理、基准（待确认后落盘）
ref/            Python FP32 oracle，先于引擎存在
include/df/     公共头，镜像 src 模块
src/
  platform/     NUMA、大页/pinned、direct I/O、NVLink 拓扑、CPU 特性分派
  quant/        E2M1 / UE8M0 / E4M3 编解码——全仓库唯一的格式真源
  artifact/     safetensors 读取、pack 转换（重排器）、模型结构代码生成
  kernels/
    cuda/sm80/  Ampere：W4A16 飞行反量化 + W4A8-int8 IMMA 格映射
    cuda/sm120/ Blackwell：原生块缩放 FP4/FP8 tensor core
    cpu/        AVX-512 VNNI lattice 路径
  core/         session / graph / router
    expert_cache/   三级缓存：VRAM 热集 / RAM 全量 / NVMe 兜底
    placement/      放置计划加载（由 data/ 的热度画像离线生成）
    remote_experts/ NVLink P2P 跨卡专家服务
  attn/         CSA2 稀疏注意力 + SWA replay + FP4 KV 编解码
  prefill/      chunked prefill、CPU-GPU 重叠
  spec/         DSpark 投机解码（外部 drafter + 置信度调度验证）
  aq/           激活量化（A16 直通 / A8-int8，带质量门禁）
  vision/       视觉塔（ViT + aligner + 图像预处理，BF16 路径）
  program/      可执行入口
serve/          OpenAI/Anthropic 兼容 API + monitor（测试与代码同目录）
tests/          按域分树：quant / core / cuda / cpu / attn / platform + 金标准数据
bench/
  results/      日期-主题实验记录（README + JSON + 配置 + 日志）
  prompts/      冻结提示词（字节哈希，禁止行尾转换）
tools/          pack 转换、热度画像、硬件资质、质量门禁、长文探针、运维 MCP
data/           数据资产：专家热度画像、放置计划、路由 trace
third_party/    vendored 依赖；reference/ = 官方仓库非权重镜像
                （inference/ 参考实现、encoding/、evaluation/、config、
                tokenizer、技术报告；来源见其 PROVENANCE.md，只读）
deploy/         机柜部署器 + 部署前硬件资质检查
.github/        自托管 CI（跑在真实机柜上）
```

## 纪律（项目宪法，先于一切文档）

宪法正文位于 [AGENTS.md](AGENTS.md)——agent 每次会话必读的执法文件：规则在那里，知识在 `docs/`，状态在 `plan/`。

## 文档状态

- **已落盘**：`AGENTS.md` — 开发纪律与执法文件（宪法、伴随义务、工作流、禁区、演化规则；部分条目待填）；`docs/MODEL.md` — 模型事实单一引用源（官方权重格式 + 重点 kernel 清单 K-Q/E/A/R/G/X/M/D/V 编号体系）；`docs/HARDWARE.md` — 目标硬件、支持矩阵与部署资质门槛（待后续测试）
- **待落盘**（逐条确认后填入）：`plan/PLAN.md`、`docs/PACK_FORMAT.md`、`docs/DETAILS.md`、`docs/HOW_IT_WORKS.md`、`docs/AI_SETUP.md`、`docs/MCP_SERVER.md`、`LICENSE`
