---
id: P0016
title: 清点本机官方 checkpoint，机械核对并更正 MODEL §2 的格式与体积事实
status: active
created: 2026-10-13
updated: 2026-10-13
knowledge: pending
docs: [docs/MODEL.md]
bench: []
decisions: [D0012]
followups: []
---

## 意图

让 [MODEL §2](../docs/MODEL.md#2-结构与格式) 的格式与体积事实由真实 checkpoint 机械核对，而不只来自读代码，并更正已发现的专家字节数错误；同时建立仓库第一个读取真实权重的入口，作为后续 pack 转换、热度画像与模型级 oracle 的起点和慢参照。成功的样子：全部张量与独立预期逐个一致，或差异有原因并已更正进 MODEL；全量扫描给出全称统计；MODEL §2.1～§2.2 中可由 checkpoint 判定的数字都带 [ckpt] 标记。取舍见 [D0012](decisions/d0012-checkpoint-inventory.md)。

## 验收

- [x] **开工前置决策**：用户审议通过 D0012 与本项范围后转 active。
- [ ] **结构完整性**：按 D0012 校验 48 个 shard 头部与 index；每类结构错误由合成反例触发并使工具非零退出。
- [ ] **独立预期比对**：预期只由 `config.json` 与官方源码规则（注明行号）生成，不读被检 checkpoint；真实 checkpoint 的名称、存储 dtype、shape 与计数全部一致，或每处差异记录原因并归入 MODEL 更正；合成反例证明多余、缺失、dtype 不符与 shape 不符各能被报告。
- [ ] **全量扫描**：全部张量的原始码直方图，码点分类经 `ref/formats`；报告 NaN 与非有限计数、E8M0 编码范围、E2M1 各码点频次（含码点 8）；扫描在合成数据上与逐元素朴素计数一致。
- [ ] **MODEL 更正**：§2.1～§2.2 中可由 checkpoint 判定的数字全部核对；专家元素数与字节数、专家与 Engram 合计、核算残差按实测更正；dtype 列区分存储与运行时；新增 [ckpt] 标记并写明快照 revision 与方法；新旧值对照留本项证据。
- [ ] **复现与治理**：合成测试接入 `cpu-debug`/`cpu-release` CTest，不进 `cpu-asan`；真实 checkpoint 的运行命令写入 BUILD §7；`--strict` 与 `--range` 审计 E=0 W=0 I=0；干净检出复现合成测试。
- [ ] **知识与人工收尾**：MODEL 与 BUILD 归档；用户评审后结项。

## 记录

### 范围与依据

- 起因：2026-10-13 会话只读头部探查。快照 `2cba9e42`，48 shard，510,286,023,000 字节，96,085 个张量，114 种名称模式；名称已是官方推理代码的模块名，无需 HF 名称映射。
- 待本项正式核对的已知差异：
  - 路由专家 FP4 + scale 为 295.997 GB，MODEL 写 ≈302 GB；元素数 47,232 × 5120 × 2304 ≈ 557.2B，MODEL 写 ≈555.5B。
  - 专家与 Engram 合计 498.76 GB，MODEL §2.2 写 505 GB 且称"单独已超过 500GB RAM"；§2.2 分解相加为 516.4 GB，与总量不符，"残差 0.08%"不成立。结论"500 GB 内存的机器需要 NVMe 兜底"仍成立，但算式须更正。
  - dtype 列：compressor `wkv`(r>1)/`wgate` 在 checkpoint 中为 BF16，MODEL 列为 FP32，即运行时构造的 dtype（`model.py:446,448`）；`engram.q_weight`/`k_weight` 在 checkpoint 中为 BF16，运行时同为 BF16（`model.py:347–348` 按 `generate.py:118` 的默认 dtype 构造），MODEL 列为 FP32；`head` 在 MODEL 列为 BF16，与 checkpoint 一致，但运行时构造为 FP32（`model.py:1006`）。
- 开工前探针见 D0012 决策驱动（PFS 单流读与标准库直方图速度）。
- safetensors 格式规范 pin 于 `huggingface/safetensors@e246a256`（2026-09-22）：README 的 Format 一节，以及 `safetensors/src/tensor.rs:10,390–425,540,610–647,794–875`（头长上限、元数据读取、按偏移排序、偏移与长度校验、dtype 位宽）。
- 前置：[P0006](p0006-ref-format-oracle.md) 的 `ref/formats`（码点分类）。
- 范围外：pack 格式与转换、C++ 读取器、放置与热度画像、性能论断、超出码点统计的数值语义验证、修改 `third_party/reference/`（含 PROVENANCE）。
- 尚未产生的归档目标：BUILD §7 的运行入口与测试条目；MODEL §2 的 [ckpt] 更正。

### 最终证据

尚未开工。

### 评审处置

- 2026-10-13：用户同意以 checkpoint 清点为下一项方向，并把专家字节数更正纳入范围；本项以 draft、D0012 以 proposed 登记。
- 2026-10-13：用户审议通过 D0012 与本项范围并授权推进；engram `q_weight`/`k_weight` 的运行时 dtype 按用户意见以 checkpoint 为准。D0012 accepted，本项 active。

### 提交

尚无。

## 收尾

- 验收或停止依据：尚未启动。
- 知识处置：pending；预期归档 MODEL §2（更正与 [ckpt] 来源）与 BUILD §7（运行与测试入口）。
- 剩余义务：待开工后确定。
