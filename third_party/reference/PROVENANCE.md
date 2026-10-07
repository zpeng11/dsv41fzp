# third_party/reference — DeepSeek-V4.1-Flash 官方仓库镜像（非权重部分）

| 项 | 值 |
| --- | --- |
| 上游 | `deepseek-ai/DeepSeek-V4.1-Flash`（HF: huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash · ModelScope 同名镜像） |
| commit | `93cb8a5`（"Adding chat template (#68)"） |
| 取回日期 | 2026-10-06 |
| 取回方式 | `git clone --depth 1 --filter=blob:none`（ModelScope）+ sparse checkout；**权重从未下载**（仓库内 safetensors 仅为 135 字节 LFS 指针桩） |
| 本地许可 | 随仓库 LICENSE（见本目录 LICENSE）；技术报告 arXiv:2609.19969 |

## 内容

| 路径 | 说明 |
| --- | --- |
| `inference/` | 官方可读参考实现（torch≥2.10、tilelang==0.1.8）：model.py / kernel.py / convert.py / generate.py / engram.py（**Engram 压缩词表与哈希官方实现，MODEL.md §2.6**）/ vision.py / image_processor.py。**本项目 §4 精确性链条的 oracle 顶端**。注意其自述"reference implementation, not a production engine"——encoder-only prefill / Bounded Replay / DSpark 验证循环均未实现（MODEL.md §2.9） |
| `encoding/` | **聊天编码**官方实现（encoding.py：chat 模板、DSML 工具调用、思维链与 reasoning effort、图像占位；+ 5 组输入/期望输出测试）——serve 域规格对象（serve 文档规划中，不入 MODEL.md）。注：Engram 哈希**不在**此目录，在 `inference/engram.py` |
| `evaluation/` | dsh-minimal.patch（用途待分析，bench 域） |
| `config.json` / `configuration.json` | MODEL.md **[cfg]** 标记的权威来源 |
| `model.safetensors.index.json` | 96,085 张量 → shard 映射（pack 工具的输入之一） |
| `chat_template.jinja` / `tokenizer.json` / `tokenizer_config.json` | serve/ 层规格对象（serve 文档规划中） |
| `report/` | 技术报告 PDF + 提取文本（**[报告]** 标记来源） |
| `README.md` | 上游模型卡（**[卡]** 标记来源） |

## 维护规则

- **只读**：本目录是外部镜像，不做任何修改；分析结论写入 `docs/`，不写回这里。
- 更新上游时：新增 commit 按 整体替换并在本表记录新 commit，`docs/` 中引用的 file:line 需复核。
