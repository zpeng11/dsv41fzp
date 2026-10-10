---
id: P0009
title: 建立 CPU-only C++ 构建与测试基线
status: active
created: 2026-10-10
updated: 2026-10-10
knowledge: pending
docs: []
bench: []
decisions: [D0001, D0005]
followups: []
---

## 意图

在实现生产 `src/quant` 前，建立可独立验收的第一套 C++ 构建与测试入口，明确工具链、target 边界、数值编译前提和版本元数据刷新规则。成功标准是无需 CUDA 或模型权重即可从干净检出完成配置、构建和测试，且后续 quant 能直接接入同一入口；不是交付推理引擎、量化实现或完整基础设施。

本项承接 P0006 之后的工程前置缺口，不接收 P0006 的未完成义务。CMake 读取 VERSION、构建时 dirty 后缀、生成物不入库及官方镜像不参与构建沿用既有规则；下述新增选择是待评审的设计提案，不以建立 draft 视为新决策已接受。先评审构建契约，再启动实现；本次仅落盘设计。

## 验收

- [ ] **CPU-only、离线与独立构建**：干净检出后按归档命令完成配置、构建和 CTest；不寻找 CUDA toolkit、不初始化 GPU、不要求 Torch/TileLang、模型权重或目标机。普通 configure/build/test 不联网、不隐式下载工具或依赖，已有工具缺失时明确失败并指出前置条件。源码树不产生构建文件，生成物留在忽略的 build 目录，官方镜像不作为编译或链接依赖。
- [ ] **可判定的首个支持环境**：记录实际 CMake/CTest、生成器、C++ 编译器、Python、CPU/平台、编译选项与命令；实际使用 C++20 编译并链接 smoke 库与消费者。首批支持声明只覆盖本项实测配置，不从单一工具链通过推出其他编译器、旧版依赖、其他操作系统或未来 NVCC 支持；依赖版本下界的声明须有对应证据。
- [ ] **target 边界真实可用**：测试目录中的极小 C++ 库及测试消费者构成跨 translation unit 编译、链接验证，消费者链接该库而非包含其源文件。include、语言要求、编译/链接选项与依赖均按 target 管理；通过编译命令及对照 target 验证数值选项不污染无关 target。项目受控的通用 target 默认无 fast-math、全局 native ISA 或无依据的 exceptions/RTTI 禁用；不生成占位业务模块。
- [ ] **数值编译前提可检查**：数值基线 target 在 Debug 和 Release 下显式禁用 fast-math 与浮点收缩；用运行时输入的 smoke 检查 binary32 前提、RNE 环境、正负零与渐进下溢，不以常量折叠结果冒充运行证据。临时反例能识别舍入环境或 FTZ/DAZ 条件不符合要求；异常环境明确失败，不静默接受。用户传入的矛盾选项不得无声破坏严格 target，处理策略和生效选项须可审计。此项只验证构建前提，不替代 quant 的格式/乘法/舍入 parity，也不新增软件浮点实现。
- [ ] **统一测试与失败传播**：CTest 必选注册 C++ smoke、现有 `tests/quant/test_formats.py` 和 `tests/tools/test_check_repo.py`；任一失败均使统一入口非零退出。Python 解释器路径显式传入，不猜测 build 路径；从源码目录外运行 CTest 仍可定位源码。缺少必选执行器/依赖不能 skip 后报通过，临时失败注入须验证失败能传到 CTest。
- [ ] **开发配置**：CPU Debug 与 Release 配置有明确入口，均运行全部必选测试。
- [ ] **版本信息构建时刷新**：每次 build 读取 VERSION、HEAD SHA 与 dirty 状态，产物可读取三者；显示版本只在 dirty 时加 `+dirty`，不把后缀写回 VERSION。状态未变时不触发重编。在工作仓库中验证：干净时 configure 并 build 无后缀；修改已跟踪文件后只 build、不重跑 configure，读到 `+dirty`；撤销修改再 build，后缀消失。
- [ ] **知识、治理与人工收尾**：构建契约、命令、支持环境、版本刷新策略与边界归档到 docs；长期选择的理由形成用户确认的 D 条目，必要知识不得仅留在本计划。现有 Python 验收、仓库检查与合入前提交范围审计通过，证据回填本项；用户评审通过后才结项，同步 STATUS 与真实剩余义务。不得把本项完成写成 quant、CUDA 或部署资质已验收。

## 记录

### 拟议构建契约

以下构建契约的长期选择已由 D0005 接受；评审如改变验收范围，须先确认并更新本项，不在实现过程中静默扩大支持矩阵。

- **平台与工具链**：首个验收环境为 Linux x86_64、GCC；以本机 GCC 15.2.0 与 Python 3.12.8 为首轮候选，不以它们推导未来最低 GCC/Python 版本。宿主 C++ 标准拟定 C++20，关闭编译器专有语言扩展；这不批准未来 CUDA translation unit 的语言标准。CMake 3.24 为拟议配置版本门槛，CTest 与 CMake 同发行，Ninja 为首轮生成器；实测版本另行记录，未经验证不声称门槛版本已获支持。CMake、CTest 与 Ninja 装入运行测试的同一 conda 环境（经 conda 或 pip，同一工具不混用两种来源），安装命令与实际版本归档到 BUILD；构建本身不自动安装，本项不做容器体系。
- **target 组织**：根 CMake 入口只建立必要构建规则和测试接线，源文件显式列举。可复用的语言/告警设置、严格数值设置、版本元数据分别组织，按需附到 target，不通过全局 CXX flags、include_directories 或全仓 ISA 设置传递。语言需求仅在公共头确有要求时传播给消费者；数值选项不作为无关消费者的隐式接口。严格数值选项在首轮 GCC 下拟用 `-fno-fast-math`、`-ffp-contract=off`，实际生效与运行环境通过验收证明，而非仅检查配置文本。
- **构建配置**：拟提供 `cpu-debug`、`cpu-release` 两个 preset，二进制目录位于 `build/<preset>`。没有 CUDA 目标时不启用 CUDA 语言，不为未来后端制造假 target 或开关。
- **测试连接**：使用 CTest 和 Python 标准库，不新增第三方 C++ 测试框架或 pybind11。smoke 库与消费者仅放在 `tests/cpu/build_baseline/`，用于证明编译、链接与版本元数据；不放入 `src/quant`、公共业务头或虚假的引擎入口。Python 调用 C++ 的连接方式随首个 quant parity 数据协议一并设计，本项不预设。
- **版本元数据**：版本来源遵守 D0001 和 AGENTS.md。每次 build 由始终执行的自定义 target 查询 Git 状态，内容变化才改写生成头，生成头只被单个版本 TU 包含；生成元数据不入源码树。dirty 拟按 Git 已跟踪改动（含暂存）与非忽略未跟踪文件判定，忽略的 build/scratch 不算 dirty。未在可识别 Git 检出中构建时明确拒绝，不伪造 clean 或 SHA。测试进程打印版本信息，未来引擎启动日志的消费责任不由本项冒充完成。
- **范围内接线**：现有 Python oracle 和治理回归进入 CTest，不修改 `ref/`、官方镜像或现有测试的数值语义。工作树检查与提交范围审计仍按既有流程独立执行。拟议根目录命令为 `cmake --preset cpu-debug`、`cmake --build --preset cpu-debug`、`ctest --preset cpu-debug --output-on-failure`；其他配置对应替换 preset。这些命令是交付目标，当前尚不能运行。
- **提交追溯**：构建与测试基线实现拟沿用现有仓库级 `ci` area，subject 为 `P0009 ci: <论断句>`；计划和知识提交分别使用 plan/docs area，不新增 build/cpp/test area，不为本项修改执法规则。这里的 ci 表示构建与测试集成，不表示已接通远端 CI。实现仍走 `p0009-cpu-cpp-build-baseline` 短命分支与既有合入流程，不以 `meta:` 代替带 P 编号的行为及验收论断。

### 范围与执行顺序

- 范围内：最小 CMake/CTest 与 preset、测试侧 C++ 编译链接工件、现有 Python 测试接线、严格数值前提的正反例、版本构建时刷新和必要知识归档。构建本身不产生模型格式版本，也不手工递增引擎 VERSION。
- 范围外：生产 quant/C++ 业务 API、通用 Tensor/graph、CUDA 版本矩阵及 sm80/sm120 策略、SIMD 分派或 native 优化、真实权重和 pack、依赖管理框架、安装导出/稳定 ABI、部署容器、全套 CI/bench/发布体系、性能基线和 P0005 目标机认证。以下在评审中移出本项：AddressSanitizer/UndefinedBehaviorSanitizer 检查配置、Python 调用 C++ 的执行器探针、版本信息的临时 Git 仓库状态矩阵（HEAD/VERSION 变化、暂存、未跟踪与忽略文件等组合）。未验证的扩展留待相应后续 P，不在本项提前占号。
- 拟议顺序：长期选择已由 D0005 接受；先完成 CPU 构建及 smoke；接入统一测试与版本刷新；运行各配置及临时正反例；最后归档知识、人工评审和结项。
- 后续 quant P 使用本项构建入口实现 scalar，并另行对照 `ref/` 验收；不因构建测试通过宣称 K-Q1～K-Q3 完成。P0005 仍需 CUDA 探针、目标机与实验前置，本项 CPU-only 交付不解除这些阻塞。必要构建知识缺失阻塞本项结项；明确范围外的未来工作不自动视为移交义务。
- 知识拟归档到 `docs/BUILD.md`，只收当前真实构建契约、操作和支持证据，不重复量化规格或保存状态；长期选择的理由留 D，过程与验收指针留本项。该文件尚未创建，不在 front matter 填悬空路径。未来 docs/决策内容写入按规则确认；本项不修改 AGENTS.md，填充其构建路由另须用户明确授权。

### 设计记录

- 2026-10-10：用户同意将 CPU-only C++ 构建与测试基线作为 P0009，并要求写设计。新增本 draft 与 STATUS 路由，未安装工具、实现构建、创建已接受决策或运行 C++ 验收；不提交或改变 VERSION。
- 2026-10-10：只读环境探测：Linux 5.15.0-72-generic x86_64 / glibc 2.43；GCC `g++ (Ubuntu 15.2.0-16ubuntu1) 15.2.0`；Python 3.12.8（conda-forge，GCC 13.3.0）。PATH 中未找到 CMake、CTest、Ninja 或 Clang；这是开工环境缺口，不是已通过的工具链支持证据。普通构建不自动补装，编码前显式准备并记录实际工具版本。
- 2026-10-10：用户评审确认裁剪范围：移出 sanitizer 配置、Python 调用 C++ 探针与版本增量刷新矩阵，版本信息收窄为 configure 注入并验证干净/dirty 两态。工具提供方式（系统包或 Python 环境）未定，留待 D 条目讨论。
- 2026-10-10：起草 [D0005](decisions/d0005-cpu-build-toolchain.md)（proposed），含本机 GCC 15.2.0 FMA 收缩探测；工具来源与 dirty 判定时机待用户选定，接受前不开始实现。
- 2026-10-10：用户选定工具由 conda/pip 管理、dirty 在构建时判定；版本信息验收改为构建时刷新，仍只验证干净与 dirty 两态，状态组合矩阵保持移出。
- 2026-10-10：用户接受 D0005 并授权推进；P0009 翻转为 active，实现走 `p0009-cpu-cpp-build-baseline` 分支。

## 收尾

- 验收或停止依据：仅设计，尚未启动实现；验收清单未勾选，无 C++ 构建或数值前提通过的论断。
- 知识处置：pending；必要知识拟归档到 docs/BUILD.md，长期选择理由待评审形成 D 条目，完成后回填实际归档引用。
- 剩余义务：本项交付尚未执行；followups 为空，不用尚未建立的后续 quant P 代替本项验收或知识归档。
