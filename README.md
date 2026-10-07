# dsv41fzp

DeepSeek-V4.1-Flash 单机柜推理引擎：单一模型特化，不追求通用架构，只把这一个模型在一种硬件形态（双路 CPU + 2/4 卡 NVLink GPU 的单机柜）上跑好。

**路由**：模型事实 → [docs/MODEL.md](docs/MODEL.md)；目标硬件 → [docs/HARDWARE.md](docs/HARDWARE.md)；宪法与执法 → [AGENTS.md](AGENTS.md)（agent 每次会话必读：规则在宪法，知识在 `docs/`，状态在 `plan/`）；当前状态与文档进度 → [plan/STATUS.md](plan/STATUS.md)。

## 目录

```
plan/           分阶段计划、决策记录与状态页（STATUS.md；P/D 编号）
docs/           事实文档：模型、包格式、硬件、原理、基准
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
.github/        自托管 CI
scratch/        临时归纳与跨会话草稿（git-ignore，不进库，按需创建）
```
