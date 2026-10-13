---
id: D0012
title: checkpoint 清点以标准库 Python 工具实现并作为未来 src/artifact 读取器的慢参照，预期表由 config 与官方源码独立生成，MODEL 以 [ckpt] 标记实测事实
status: accepted
created: 2026-10-13
updated: 2026-10-13
replaces: []
superseded-by: []
---

## 背景与问题

本机有官方 checkpoint（HF 快照 `2cba9e42`，48 shard）。2026-10-13 会话中只读头部的探查显示，[MODEL §2.1～§2.2](../../docs/MODEL.md#2-结构与格式) 的路由专家体积、专家与 Engram 合计及核算残差与真实文件不符，且 §2.1 的 dtype 列混用了运行时 dtype 与 checkpoint 存储 dtype（差异清单见 [P0016](../p0016-checkpoint-inventory.md)）。MODEL §2 的数字至今只来自读代码与算术，从未对照真实文件做机械检查。

[P0016](../p0016-checkpoint-inventory.md) 要建立第一个读取真实权重的入口，须决定：工具的位置与语言、预期从哪里来、全量数据扫描的方式、测试与执行入口，以及实测事实在 MODEL 中的来源标记。

## 决策驱动

- 技术约束：safetensors 头部是 JSON；C++ 侧目前没有 JSON 依赖，引入须另立依赖决策。量化载荷约 506 GB。开工前探针（ENV-A，快照 `2cba9e42`，shard 10，单线程）：PFS 单流读 1.70 GB/s；标准库 `collections.Counter` 字节直方图 26 MB/s。
- 项目约束：预期不得由被检 checkpoint 生成，否则比对是循环论证；`src/quant` 是格式唯一真源，工具不另立格式常量；docs 不写无测量环境的数字；生成物不进 git；`third_party/reference/`（含 PROVENANCE）只读；测试不 skip。
- 宪法补充约束：每段快代码须有已验证的慢参照。未来 pack 转换所需的 C++ 读取器需要一个上一级。

## 备选方案

### 工具位置与语言

- 标准库 Python 工具（暂名 `tools/ckpt_inventory.py`），完成头部解析、预期比对与全量扫描，并作为未来 `src/artifact` C++ 读取器的慢参照：采纳，推荐。头部解析有标准库 `json`；按张量分片的多进程扫描在本机受 I/O 限制，可在分钟级完成。
- 现在就建 C++ `src/artifact`：须先定 JSON 依赖并补 parity 测试。读取器的真实接口需求（pack 转换、mmap、direct I/O）尚未出现：不采纳，留待 pack 转换立项。
- Python 加 numpy：扫描更快，但 numpy 不在 ENV-A，须另立依赖决策；本项规模下标准库并行已够用：不采纳。

### 预期表来源

- 工具内按 `config.json` 字段与官方 `model.py`/`convert.py` 的模块构造规则（注明行号）生成名称、存储 dtype、shape 与计数，与 checkpoint 逐张量比对，多余、缺失与不符逐项报告：采纳。
- 从 checkpoint 头部生成并冻结为期望：循环论证，否决。
- 只核对汇总（总字节、张量数）：发现不了 dtype 列这类错误：不采纳。

### 结构完整性

- 按 safetensors 格式规范校验（规范来源与版本在开工时 pin 并归档）：8 字节小端头长及上界、头部为合法 JSON 且无重复键、`data_offsets` 连续无洞无重叠且恰好覆盖数据区、dtype 字节数乘 shape 等于区间长度；`model.safetensors.index.json` 的 `weight_map` 与各 shard 头部一一对应，`metadata.total_size` 等于载荷总和：采纳。

### 全量数据扫描

- 对全部张量做原始码直方图，按张量分片并行；码点分类（NaN、零、非有限、E8M0 指数）调用 `ref/formats` 的解码一次性生成分类表，工具内不写格式常量。输出 NaN 与非有限计数（对照 QUANT/COMPUTE 入口拒绝 NaN 的契约在真实数据上是否触发）、E8M0 编码范围（整数格路径溢出分析的输入）、E2M1 各码点频次（含码点 8，即 `ref/` 与官方 LUT 零符号差异的实际影响面）：采纳。
- C++ 扫描器：更快，但须慢参照与 parity，超出本项：不采纳。
- 只扫抽样：给不出"NaN 计数为 0"这类全称结论：不采纳。

### 测试与执行入口

- 合成的小型 safetensors 在临时目录现场生成，不写 `tests/data/`；覆盖每类结构错误、预期不符报告与扫描计数，注册 Debug/Release CTest。真实 checkpoint 运行依赖本机路径与 510 GB 数据，不进 CTest，以显式命令执行，证据记入 P：采纳。
- CTest 读取真实 checkpoint，路径缺失时 skip：违背无 skip 惯例：否决。

### 知识去向与来源标记

- MODEL 新增来源标记 **[ckpt]**（checkpoint 头部或载荷实测，注明快照 revision 与方法）；§2.1/§2.2 按实测更正，§2.1 dtype 列改为存储 dtype，运行时不同者注明；扫描得到的全称结论写入 MODEL §2；工具用法与测试入口写入 BUILD §7；完整清单输出是生成物，不入库：采纳。
- 新建 `docs/CHECKPOINT.md`：内容不足一页，且模型事实的单一来源是 MODEL：不采纳；pack 契约出现时再定。
- 快照 revision 写入 PROVENANCE：该文件位于只读镜像内，否决。

## 结论

2026-10-13 经用户审议通过，接受本决策及 P0016 当前范围；本决策转为 accepted，P0016 同步转为 active。结论：

1. 以标准库 Python 工具完成 checkpoint 结构校验、独立预期比对与全量原始码扫描；它是未来 C++ 读取器的慢参照，不承担 pack 转换。
2. 预期表只由 `config.json` 与官方源码规则生成；码点分类经 `ref/formats`，工具不另立格式常量。
3. 合成数据测试进 CTest，真实 checkpoint 以显式命令执行；MODEL 引入 [ckpt] 标记，更正与全称结论归 MODEL §2，运行入口归 BUILD §7。

## 后果

- P0016 以本决策为开工前置，本决策被接受前 P0016 不进入 active。
- 未来 C++ `src/artifact` 读取器以本工具的清单输出为 parity 上一级；pack 格式、转换与放置另立决策。
- MODEL 的来源标记表增加一项；既有 [代码][算] 数字被 [ckpt] 推翻时以实测为准，并在 P0016 证据中保留旧值对照。
