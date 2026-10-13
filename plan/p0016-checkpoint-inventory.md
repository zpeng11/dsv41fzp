---
id: P0016
title: 清点本机官方 checkpoint，机械核对并更正 MODEL §2 的格式与体积事实
status: active
created: 2026-10-13
updated: 2026-10-13
knowledge: pending
docs: [docs/MODEL.md, docs/BUILD.md]
bench: []
decisions: [D0012]
followups: []
---

## 意图

让 [MODEL §2](../docs/MODEL.md#2-结构与格式) 的格式与体积事实由真实 checkpoint 机械核对，而不只来自读代码，并更正已发现的专家字节数错误；同时建立仓库第一个读取真实权重的入口，作为后续 pack 转换、热度画像与模型级 oracle 的起点和慢参照。成功的样子：全部张量与独立预期逐个一致，或差异有原因并已更正进 MODEL；全量扫描给出全称统计；MODEL §2.1～§2.2 中可由 checkpoint 判定的数字都带 [ckpt] 标记。取舍见 [D0012](decisions/d0012-checkpoint-inventory.md)。

## 验收

- [x] **开工前置决策**：用户审议通过 D0012 与本项范围后转 active。
- [x] **结构完整性**：按 D0012 校验 48 个 shard 头部与 index；每类结构错误由合成反例触发并使工具非零退出。
- [x] **独立预期比对**：预期只由 `config.json` 与官方源码规则（注明行号）生成，不读被检 checkpoint；真实 checkpoint 的名称、存储 dtype、shape 与计数全部一致，或每处差异记录原因并归入 MODEL 更正；合成反例证明多余、缺失、dtype 不符与 shape 不符各能被报告。
- [x] **全量扫描**：全部张量的原始码直方图，码点分类经 `ref/formats`；报告 NaN 与非有限计数、E8M0 编码范围、E2M1 各码点频次（含码点 8）；扫描在合成数据上与逐元素朴素计数一致。
- [x] **MODEL 更正**：§2.1～§2.2 中可由 checkpoint 判定的数字全部核对；专家元素数与字节数、专家与 Engram 合计、核算残差按实测更正；dtype 列区分存储与运行时；新增 [ckpt] 标记并写明快照 revision 与方法；新旧值对照留本项证据。
- [x] **复现与治理**：合成测试接入 `cpu-debug`/`cpu-release` CTest，不进 `cpu-asan`；真实 checkpoint 的运行命令写入 BUILD §7；`--strict` 与 `--range` 审计 E=0 W=0 I=0；干净检出复现合成测试。
- [ ] **知识与人工收尾**：MODEL 与 BUILD 归档；用户评审后结项。

## 记录

### 范围与依据

- 起因：2026-10-13 会话只读头部探查发现 MODEL §2 专家体积与 dtype 列有误；开工前探针见 D0012 决策驱动。
- safetensors 格式规范 pin 于 `huggingface/safetensors@e246a256`（2026-09-22）：README 的 Format 一节，以及 `safetensors/src/tensor.rs:10,390–425,540,610–647,794–875`。
- 存储与运行时 dtype：默认相同；源码另有说明的按源码（`convert.py:157–181`，`model.py:444–448,1005,1092`）；其余 FP32 参数源码未定存储，接受 FP32 或 BF16（`generate.py:133` 加载时转换）并报告实际选择。engram `q/k_weight` 按用户意见以 checkpoint 为准，与 `generate.py:118` 默认 dtype 一致。
- 前置：[P0006](p0006-ref-format-oracle.md) 的 `ref/formats`。范围外：pack 格式与转换、C++ 读取器、放置与热度画像、性能论断、码点统计以外的数值语义、修改 `third_party/reference/`。

### 最终证据

- 真实 checkpoint（ENV-A，快照 `2cba9e42`，命令见 BUILD §7）：头部核对 PASS，48 shard、96,085 个张量与独立预期逐个一致（缺失、多余、dtype、shape 不符均为 0），载荷 510,286,023,000 字节等于 index `total_size`，两份配置与镜像逐字节相同。全量扫描 `--jobs 96` PASS，用时约 8 分钟，计数见 MODEL §2.1。
- 新旧值对照：

| 项 | MODEL 旧值 | 实测 |
| --- | --- | --- |
| 路由专家 | ≈555.5B 元素，≈302 GB | 557,171,343,360 个 FP4 值，296.00 GB |
| 专家 + Engram | 505 GB，"已超过 500GB" | 498.76 GB |
| §2.2 分解 | 302+203+7.4+4，"残差 0.08%" | 296.00+202.76+7.41+3.95+0.17 = 510.29 GB |
| dtype 列 | compressor wkv(r>1)/wgate、engram q/k 为 FP32 | 均存 BF16；compressor 运行时 FP32，engram q/k 运行时 BF16 |
| shard 归属 | Engram 表"各独占 47/48"；视觉"在 shard 1" | 47/48 各含一层完整 Engram 模块；视觉塔与 aligner 独占 shard 1，分隔嵌入在 shard 2 |

- 合成测试 `tests/tools/test_ckpt_inventory.py`：32 项无 skip。临时副本注错 6 处（漏数高半字节、丢分块尾部、去掉按偏移排序、dtype 只认首选存储、正负零对调、漏掉 `bias_vl`），各由对应用例检出。
- 回归：`cpu-debug`/`cpu-release` 各 22/22、零告警；`--strict` 与 `--range main..HEAD --strict` E=0 W=0 I=0；`scratch/check_links.py` broken=0。干净克隆 `ae7dd7d` 自源码目录外合成测试 32/32、真实头部核对 PASS，克隆保持干净。

### 评审处置

- 2026-10-13：用户同意以 checkpoint 清点为下一项方向，并把专家字节数更正纳入范围；本项以 draft、D0012 以 proposed 登记。
- 2026-10-13：用户审议通过 D0012 与本项范围并授权推进；engram `q_weight`/`k_weight` 的运行时 dtype 按用户意见以 checkpoint 为准。D0012 accepted，本项 active。

### 提交

- `eaeb45b` D0012 接受与登记，`71c6f92` 清点工具与测试，`ae7dd7d` MODEL/BUILD 归档。

## 收尾

- 验收或停止依据：尚未结项；六项技术验收已成立，待用户评审。
- 知识处置：pending；已归档 MODEL §2（[ckpt] 更正与扫描结论）、MODEL §4 覆盖矩阵与 BUILD §7（运行与测试入口），结项时改为 updated。
- 剩余义务：待结项时确定。
