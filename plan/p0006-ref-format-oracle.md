---
id: P0006
title: 建立模型低精度格式的 CPU 编解码与反量化 oracle
status: completed
created: 2026-10-09
updated: 2026-10-10
knowledge: updated
docs: [docs/MODEL.md, docs/QUANT.md]
bench: []
decisions: []
followups: []
---

## 意图

先建立可独立验收、无需目标 GPU 或模型权重的数值参照，为后续量化格式与计算路径提供验证根。本项只交付 `ref/` 的第一块格式 oracle，不等同于完整 FP32 模型 oracle；不实现引擎或宣称与官方 GPU 执行逐位一致。

## 验收

- [x] **覆盖格式与输出边界**：提供 E2M1 原始 nibble、E4M3FN 原始 byte、UE8M0/E8M0 原始 byte 到 FP32 的解码；明确具体格式变体，不以笼统的“FP8”代替。提供 BF16 位模式到 FP32，以及按有依据的舍入规则生成 BF16 位模式的转换。精确 FP32 解码、FP32 乘 scale、落到 BF16 的舍入分别验收，不抹掉官方路径中的 BF16 落点。
- [x] **码点穷举与独立依据**：覆盖 E2M1 全部 16 个码点、E4M3FN 与 E8M0 各 256 个字节编码，以及 BF16 全部 65,536 个位模式的解码。正常值、次正规值、零及特殊编码按各格式实际语义判定；预期值来自可追溯的格式定义、独立位域/精确算术或官方常量，不调用被测函数生成，也不只复制被测 LUT。官方代码未充分定义的行为须补足权威依据，依据或关键语义未明确时不得验收。
- [x] **E2M1 字节布局**：穷举 256 个 packed byte，验证低半字节在前及 `byte → nibble → byte` 逐位可逆；多行、不对称码点用例验证沿最后一维配对、不跨行打包。非法码点、奇数逻辑长度与不匹配的 shape 明确报错，不隐式截断或填充。不把 `byte → 数值 → byte` 当成全部编码可逆的验收条件。
- [x] **E2M1 精确整数格**：全部码点的整数格结果与 FP32 解码值乘 2 精确相等，不使用近似误差。零码点与整数零之间的信息损失及零符号策略明确；若提供整数格反向接口，仅接受精确格点、拒绝非格点，并明确零的规范编码，不要求从整数格恢复全部原始位信息。
- [x] **给定 scale 的反量化与布局**：覆盖专家 E2M1 + E8M0 沿 K 每 32 元素分组、普通 FP8 权重 E4M3 + E8M0 的二维 32×32 块、Engram/SWA 的 E4M3 + E8M0 沿最后一维每 32 元素分组、主 KV 的 E2M1 + E4M3 每 16 元素分组、索引 K 的 E2M1 + E8M0 每 32 元素分组。每类以多行、多组或多块、相邻不同 scale、行列不对称及组边界用例检查 scale 索引，避免错轴仍通过。二维块路径覆盖非整块尾部及向上取整的 scale shape；逐行分组路径要求最后一维整除组大小，错误 shape 明确拒绝。此处只覆盖已编码数据与给定 scale，不代表已实现 KV 动态量化、RoPE 或 cache 管理。
- [x] **精度、特殊值与异常契约**：编码、打包字节及整数格精确比较；有限 FP32 解码与按 FP32 运算规则得到的反量化结果精确比较；BF16 输出比较位模式，编码方向覆盖舍入中点及其两侧、指数进位、正常/次正规边界、上溢/下溢和特殊输入。NaN 比较分类，除非有独立依据不承诺 payload 一致；零符号按声明策略检查。底层 codec 完整识别编码，模型反量化入口拒绝 NaN 和非正 scale；有效输入的 FP32 乘积上溢/下溢按明确浮点规则处理，不静默钳制。不以统一 `allclose` 掩盖格式错误，也不承诺任意 scale 乘积或 BF16 落点都能精确表示。
- [x] **CPU 独立执行与证据**：`ref/` 代码与 `tests/quant/` 测试引用 P0006；一条明确的 CPU 命令可运行全部必选验收，不依赖 CUDA、TileLang 初始化、模型权重、目标机或官方 GPU dump，不导入官方 GPU kernel 生成预期结果。必选穷举和边界测试不得因某 dtype 的 CPU 支持缺失而跳过后宣称通过；随机测试仅作补充且可复现。记录解释器、依赖版本、CPU/平台、方法、命令及结果；CPU 库对照只有在确认所用操作受支持后才作为补充证据。
- [x] **知识与验证边界处置**：格式依据、特殊值策略、FP32 运算及 BF16 舍入契约归档到 `docs/`，测试命令与环境、验收证据指针回填本项；不把长期规格重复留在计划正文。若落盘金标准字节到 `tests/data/`，按冻结区契约记录来源、生成方式并重算 manifest。结项时明确本项不证明动态量化、GPU cast、kernel parity 或端到端模型一致；完成 knowledge 处置并据实际情况回填真实接收项，不以本项完成替代 `src/quant` 或 K-Q1～K-Q3 的引擎实现验收。

## 记录

- 2026-10-09：用户确认讨论中的覆盖范围与验收设计，授权新增本计划并刷新 STATUS；明确先不开始实现。本项保持 draft，未创建数值代码、测试或金标准数据，未运行格式验收。
- 范围内交付为 `ref/` 的格式慢参照与 `tests/quant/` 的独立测试；不增加生产格式配置中心，`src/quant` 作为全仓量化格式唯一真源的约定不变。本项首建 oracle，不因引擎实现便利而调整参照。
- 范围外：任意 FP32 到 E2M1/E4M3 的有损量化、由 amax 生成 scale、动态激活/KV 量化及融合反量化、点积 scale 折叠与 GEMM/GEMV、C++ scalar/CUDA/SIMD 实现和性能优化、checkpoint 转换与真实权重加载、完整模型与端到端验证。格式解码不等于 scale 生成，反量化不等于计算路径验收。
- 既有依据：[MODEL §2.1～§2.3](../docs/MODEL.md#2-结构与格式)、[K-Q1～K-Q3](../docs/MODEL.md#31-格式编解码srcquant)、[精确性链条](../docs/MODEL.md#4-精确性链条)；官方只读仲裁点为 `third_party/reference/inference/convert.py` 的 FP4_TABLE 与 nibble 展开、`model.py` 的权重/Engram/KV 格式用途，以及 `kernel.py` 的数值运算与 cast 位置。官方镜像来源见 [PROVENANCE](../third_party/reference/PROVENANCE.md)；不把 torch dtype 名称存在视为对应 CPU 操作可用的证据。
- 知识拟归档到 `docs/QUANT.md`，由 `docs/MODEL.md` 路由；该文档尚未创建，仅在此记录归档目标，不在 front matter 填入悬空路径。本次不修改 docs，未来内容性写入仍按宪法取得用户确认；验收前须完成必要知识归档。
- 本项不依赖 P0005 的目标机验证，也不依赖尚未定案的 C++ 构建或性能 bench 工作流；未来引擎、硬件与性能验收仍须满足各自前置条件。本项不新增性能论断。
- 2026-10-09：用户确定本项 `ref/` 相关提交沿用 `quant` area，不修宪；检查器对新建 `ref/` 文件的误报由 P0007 处理，提交 oracle 代码前须待其合入。同一分支内修改本分支新建的 `ref/` 文件仍会提醒，合入前整理提交，使每个新文件只由一个提交引入。
- 2026-10-10：用户授权按机械路径开工并同步清理 STATUS，本项由 draft 转为 active；P0007 已合入，下一步在 `p0006-ref-format-oracle` 分支实现 `ref/` 格式 oracle 与 `tests/quant/` 独立测试。本次仅更新生命周期与编码入口，不改验收条件，不新增数值代码、测试或金标准数据，不宣称格式验收通过。
- 2026-10-10：用户授权开始编码，网络访问使用其提供的代理；本分支已实现 `ref/formats.py`、`ref/dequant.py` 与 `tests/quant/test_formats.py`，不新增生产 `src/quant` 或冻结区数据。环境为 Python 3.12.8（conda-forge，GCC 13.3.0）、Linux 5.15.0-72-generic x86_64 / glibc 2.43、Intel Xeon Platinum 8350C，依赖仅标准库；命令 `python3 tests/quant/test_formats.py` 的 19 项测试通过，包含必选码点穷举、BF16 舍入边界及反量化布局/异常用例，无跳过。独立期望值、方法与官方常量对照见该测试；另在临时副本注入零符号、格式 bias、E8M0 最小编码、nibble 次序、BF16 舍入、逐行与二维块 scale 索引共 7 类错误，均被对应测试识别。数值日志与反例日志临存 `scratch/p0006/cpu-acceptance.log`、`scratch/p0006/negative-controls.log`；来源清单及必要知识草稿临存 `scratch/p0006/sources.md`、`scratch/p0006/QUANT-proposed.md`，不替代 docs 归档。治理回归 `python3 tests/tools/test_check_repo.py` 的 71 项测试通过，当前分支 `--range main..HEAD --audit --strict` 为 E=0 W=0 I=0；临时克隆中以 `P0006 quant:` 和 `Oracle: tests/quant/test_formats.py` 演练新增实现提交，提交级审计亦为 E=0 W=0 I=0，工作仓库未提交实现。尚未写入 docs、完成人工评审或结项，验收清单不提前勾选。
- 2026-10-10：用户明确评审通过并要求继续流程；已核对实现文件与评审时的 SHA256 一致，确认归档必要知识至 docs/QUANT.md，并由 docs/MODEL.md 路由及澄清官方专家与 KV 路径的舍入落点。验收清单据用户评审与 CPU 证据勾选；本实现提交保持 active，合入时同步 completed 与 STATUS，不以本项验收替代生产量化或计算路径验收。

- 2026-10-10：实现提交 `ea4960b`；合入前重跑数值与治理测试均通过，`python3 tools/check_repo.py --range main..HEAD --audit --strict` 为 E=0 W=0 I=0，`git diff --check` 通过。本次 `--no-ff` 合入提交同步状态 completed、清理 STATUS 并递增 VERSION 至 0.1.7。

## 收尾

- 验收或停止依据：2026-10-10 用户评审通过并授权继续流程；格式 oracle、独立测试及必要知识归档已到位。命令 `python3 tests/quant/test_formats.py` 的 19 项测试全部通过，无跳过；`python3 tests/tools/test_check_repo.py` 的 71 项治理测试与合入前分支审计通过。实现提交 `ea4960b`，本合并提交同步结项与 STATUS，VERSION 为 0.1.7；不证明动态量化、GPU cast/FTZ、kernel parity 或端到端一致。
- 知识处置：updated；位模式、特殊值、FP32 运算、BF16 舍入、反量化入口与验证边界归档到 [docs/QUANT.md](../docs/QUANT.md)，[docs/MODEL.md](../docs/MODEL.md#23-数值格式速查srcquant-规格) 提供路由并澄清官方运算落点。scratch 原始来源快照与日志仅为临时证据，长期契约不重复保留在计划正文。
- 剩余义务：无；范围内交付、评审、必要知识归档与合入结项均完成，无需移交，followups 为空。生产 src/quant、K-Q1～K-Q3 与后续动态量化、kernel 及端到端验证属于本项明确排除的未来工作，不宣称已实现或验收。
