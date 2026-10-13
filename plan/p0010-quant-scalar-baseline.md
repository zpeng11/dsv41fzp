---
id: P0010
title: 建立低精度格式 codec 的生产 scalar 基线与跨语言 parity 连接
status: completed
created: 2026-10-11
updated: 2026-10-13
knowledge: updated
docs: [docs/MODEL.md, docs/QUANT.md, docs/BUILD.md]
bench: []
decisions: [D0001, D0005, D0006, D0007]
followups: []
---

## 意图

在 P0006 格式 oracle 与 P0009 构建入口之上，建立可链接的生产 `src/quant` codec 与 C++ 对照 `ref/` 的跨语言 parity 连接。成功标准是独立 C++ 消费者调用生产库，在声明的环境与输入契约下逐位对上 `ref/`。给定 scale 反量化由 [P0011](p0011-quant-dequant-scalar.md) 完成。

## 验收

- [x] **生产模块与链接边界**：`src/quant` 编译库、公共头在 `include/df/quant/`，只依赖标准库；消费者链接库，不含生产 `.cpp` 或算法副本。
- [x] **格式覆盖**：E2M1、E4M3FN、E8M0、BF16 解码，FP32→BF16 RNE，E2M1 打包/展开与整数格双向，遵守 QUANT。
- [x] **接口与失败契约**：按 D0007 的视图、shape、输出缓冲区与两类异常拒绝全部无效输入；元数据先于读数据、内容先于写输出，失败输出不变，允许零元素；元数据错误配短缓冲区在 `cpu-asan` 下验证，内容错误置末元素；列出静态类型排除项，执行器不先窄化。
- [x] **数值环境无关**：整数位运算构造；默认、向上舍入、FTZ、DAZ 四种 MXCSR 下逐位对上 `ref/`。
- [x] **码点穷举 parity**：E2M1 16、E4M3FN/E8M0 各 256、BF16 65,536、packed byte 256、整数格全部格点及代表性非格点；NaN 比分类，零检符号。
- [x] **BF16 舍入边界**：全部有限相邻格点中点及两侧、进位、次正规边界、上溢、下溢、零、Inf、NaN。
- [x] **格式契约版本**：`src/quant` 常量独立于 VERSION，初值与递增条件归档 QUANT，测试断言一致。
- [x] **跨语言连接与失败传播**：执行器十六进制批量协议，`$<TARGET_FILE:...>` 定位；执行器故障、协议损坏或比较失败使 CTest 非零，不 skip；记录一次 parity 耗时。
- [x] **编译与越界检查**：严格选项生效且不污染无关 target；新增 `cpu-asan`，注入越界读被识别。
- [x] **可复现与测试敏感性**：三个 preset、源码目录外与干净检出复现；nibble 次序、BF16 中点、负零、E8M0 次正规四类注入被识别。
- [x] **知识、治理与人工收尾**：QUANT §6 与 BUILD 增补，`Oracle:` 指向 C++ parity；审计通过，用户评审后结项。

## 记录

### 范围与依据

- 依据 [D0007](decisions/d0007-quant-integer-dequant.md)（取代 D0006）；契约见 [QUANT §6](../docs/QUANT.md#6-生产-c-codecsrcquant)、[BUILD](../docs/BUILD.md)。K-Q 覆盖与不覆盖见 [MODEL §4](../docs/MODEL.md#4-精确性链条)。
- 范围外：给定 scale 反量化（P0011）、有损编码与 scale 生成、点积、pack、CUDA/SIMD、性能论断。

### 验收证据

- 环境：ENV-A；实现提交 `9201b8a`，工作仓库与全新克隆结果相同。
- `cpu-debug`/`cpu-release` 各 11/11，`cpu-asan` 4/4，零告警；`/tmp` 下 `cpu-release` 11/11。
- parity：上列码点穷举、多行打包、整数格 15 格点、FP32→BF16 中点及两侧 195,840 个、18 个特殊值、20,000 个固定种子随机值；输出缓冲区多 3 元素确认只写前缀。四种 MXCSR 由执行器回报并核对。
- 失败契约：9 个入口逐类拒绝（含 `(2^63, 2^63, 0)` 按零元素接受）；短缓冲区下 ASan 无报告；末元素坏值时输出不变且 `ref/` 同样拒绝；格式契约版本为 1。
- 耗时（Debug，测试连接，非性能论断）：262,144 元素单请求期望生成 0.27s、执行比较 0.22s。
- 故障传播由永久 `quant.codec.harness` 覆盖；四类数值注入分别定位到 `pack.all_bytes[1]`、`fp32_to_bf16.rounding[1]`、`e2m1.all[8]`、`e8m0.all[0]`，越界读注入报 heap-buffer-overflow。
- 审计：`--audit --range main..HEAD --strict` E=0 W=0 I=0。

### 评审处置

- 2026-10-11：评审指出 FTZ 风险、越界证据缺失、参照说法过宽、缺 D、条目过大；拆出 P0011，起草 D0006，新增 `cpu-asan`。
- 2026-10-11：D0007 取代 D0006（反量化整数化、取消调用时 MXCSR 检查、异常减为两类）；授权开工。
- 2026-10-11：MODEL 码点 8 例外与 kernel parity 表述改为单独 docs 修订（`68b1861`）。
- 开发中发现：UBSan 报执行器零字节输出向 `memcmp` 传空指针，已在 `ci` 提交修复，生产库无报告。
- 用户评审发现：驱动用 `int(v, 16)` 接受带符号/前缀 token，`-0000001` 被当作 NaN 掩盖协议损坏；`01c18c9` 改为定宽小写十六进制完整匹配并加回归。
- 2026-10-11：用户评审通过并授权合入。

### 提交

- `3433675`、`283849d`、`e75f5e7`、`9201b8a`、`01c18c9` 实现与测试，`528cfc8` cpu-asan，`260379b` 知识归档；合并提交 VERSION 0.1.9。
- 压缩前全文见提交 `b60526d`（依 [D0011](decisions/d0011-doc-single-source.md) 压缩）。

## 收尾

- 验收或停止依据：2026-10-11 用户评审通过并授权合入；验收逐条见上方验收证据。`cpu-debug`、`cpu-release` 各 11/11、`cpu-asan` 4/4 通过，构建零告警，源码目录外与全新克隆复现相同；codec 在四种 MXCSR 下逐位对上 `ref/`，每类拒绝、失败时输出不变与零元素张量均有测试，四类数值错误与越界读的临时注入均被识别，执行器故障由 `quant.codec.harness` 永久覆盖。评审发现的协议解析缺陷已在 `01c18c9` 修复并加回归用例。本合并提交同步结项与 STATUS，VERSION 为 0.1.9。不证明给定 scale 反量化（P0011）、BF16→E4M3 编码、点积 scale 折叠、SIMD/CUDA 路径或非 GCC/非 Linux x86_64 环境。
- 知识处置：updated；C++ 接口、失败契约、静态类型排除、环境无关性、格式契约版本与验证边界归档到 [docs/QUANT.md §6](../docs/QUANT.md#6-生产-c-codecsrcquant)，`cpu-asan`、`df_quant`/`df_sanitize` target、`DF_TEST_SCOPE` 与测试清单归档到 [docs/BUILD.md](../docs/BUILD.md)；选择理由在 [D0007](decisions/d0007-quant-integer-dequant.md)，过程证据保留在本项。
- 剩余义务：无；followups 为空。P0011 是开工前的范围拆分，沿用本项的模块、执行器协议与 `cpu-asan`，不是本项移交的义务。
