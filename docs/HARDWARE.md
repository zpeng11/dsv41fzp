# HARDWARE.md — 目标硬件、支持矩阵与部署资质

本页定义目标硬件形态、型号支持范围与**部署资质门槛**（P0004 / D0004）。`tools/qualify_hw.py` 是硬件预检的约定入口，`deploy/` 启动前必须执行；资格不成立或证据不完整时拒启。实现与实机验证进度见 [STATUS](../plan/STATUS.md)，不以本页的型号列表代替整机认证。

**来源标记**：**[厂]** 厂商公开规格（资料索引见 §7）· **[算]** 规格推导（附式）· **[门槛]** 项目资格要求，不是已测结果 · **[待测]** 必须在目标机实测。厂商数字仅适用于所列型号及版本，不代表引擎性能；项目实测须记录硬件、方法与软件环境。

---

## 1. 目标形态与非目标

**单机柜**：1 台**双路**服务器（2×CPU，8/12 内存通道/路，即 16/24 通道合计 [算：每路通道×2]，RAM + NVMe）+ **2/4 张同型号 GPU [门槛]**。内存按 NUMA 分两个域管理。

**GPU 互联**：NVLink **非必选**。允许 §3 白名单中的 NVLink 配置或 RTX PRO 6000 Blackwell **PCIe 5.0 P2P** 配置；全部 GPU 必须通过 §5 的逐对双向 P2P 资格检查。PCIe 代际、卡名或 NVLink 标识本身均不是充分条件。

**非目标**：多机分布式、云弹性调度、Windows、消费卡、异型号混插，以及不能提供合格 GPU P2P 的配置。不因缺少 NVLink 单独排除 PCIe 配置，也不自动扩大到所有 PCIe GPU。

## 2. CPU 与内存平台

下表容量与指令集是项目门槛；带宽是理论推导或待测验收线，非目标机性能结果。本次互联修订不放宽 CPU、内存或 NVMe 门槛。

| 项 | 门槛 | 说明 |
| --- | --- | --- |
| 内存通道 | **≥8/路 ×2 路（≥16 通道合计）且插满 [门槛]** | 尽可能使用满 CPU 带宽 |
| 内存代际 | **DDR4/DDR5 [门槛]** | 8 通道/路、双路、每通道 8B：DDR4-3200 为 409.6 GB/s，DDR5-4800 为 614.4 GB/s [算：传输率×8B×8×2]；不代表可持续带宽 |
| 持续带宽 | **每域**流测试 ≥ 该域理论 80% [门槛/待测] | STREAM 类测试；CPU 计算绑定 NUMA 域与内存，记录 CPU、DIMM 配置、频率、线程亲和与方法 |
| 容量 | 总量 ≥500GB；**每域可用 ≥240GB [门槛]** | 可用量按系统保留与 NUMA 分布核验 |
| 指令集 | AVX-512（F/BW/DQ/VL）**+ AVX512_VNNI（`vpdpwssd`）[门槛]** | 整数格路径依赖；无 VNNI 直接拒启 |
| 参考平台 | Intel Xeon SPR / GNR 双路，或 AMD EPYC 9004/9005 等 | 具体 SKU 仍须核验通道、指令集及实测门槛，不凭系列名称放行 |

## 3. GPU 支持矩阵

下表是**设计支持范围**，不是实机通过清单。每台机器仍须满足 CPU、内存、链路、P2P 与软件环境要求。显存带宽与 GPU 间带宽是不同指标，不能互相替代。

| 卡（同型号组卡） | 架构 | VRAM | 显存带宽 | GPU 间路径与资格 | 计算路径 |
| --- | --- | --- | --- | --- | --- |
| A100 80GB **SXM** | sm_80 [厂 S5] | 80GB HBM2e [厂 S6] | 2039 GB/s [厂 S6] | 保留 NVLink 平台；标称 600 GB/s [厂 S6]，按实际底板拓扑验收，不假定任意 SXM 配置自带 NVSwitch | K-E1 / K-E2 |
| A800 80GB **SXM** | Ampere；具体模组资格核验 | 80GB（限定型号） | 按具体模组厂商规格核对，不沿用未核实的带宽数字 | 保留既有 NVLink 配置范围，NVSwitch 与可达性按实际整机核验 | 同 sm80 路径，须核验设备能力 |
| RTX PRO 6000 Blackwell **Server Edition** | sm_120 [厂 S5] | 96GB GDDR7 ECC [厂 S1] | 1597 GB/s [厂 S1] | **无 NVLink [厂 S3]**；PCIe 5.0 x16 [厂 S3] GPU P2P，须通过 §5 | K-E3 / K-E4 |
| RTX PRO 6000 Blackwell **Workstation Edition** | sm_120 [厂 S5] | 96GB GDDR7 ECC [厂 S2] | 1792 GB/s [厂 S2] | 按 PCIe 5.0 x16 [厂 S2] GPU P2P 纳入，不依赖 NVLink；须满足服务器安装、供电、散热与 §5 | K-E3 / K-E4 |
| RTX PRO 6000 Blackwell **Max-Q Workstation Edition** | sm_120 [厂 S5] | 96GB GDDR7 [厂 S4] | 最高 1792 GB/s [厂 S4] | **无 NVLink [厂 S4]**；PCIe 5.0 x16 [厂 S4] GPU P2P，须满足服务器安装、供电、散热与 §5 | K-E3 / K-E4 |

- **型号边界**：PRO 6000 指上述 Blackwell 版本，不泛指旧代 RTX 6000。各版分别资格验证，不用一版实测替另一版背书。A100/A800 PCIe、H100/B100 等及消费卡仍不在白名单；这是项目范围限制，不宣称这些卡因 PCIe 形态而必然没有 NVLink 或 P2P。
- **配置边界**：同一台机器采用同型号、同版本的整卡 GPU；不支持 sm80/sm120 或不同 PRO 版本混插。MIG 实例组合不在本契约范围，不能用整卡检查替代 MIG 资格。
- **放置拓扑**：PRO 6000 不按“NVLink 桥接对”分簇。platform、placement 与 K-M1 必须使用实际 PCIe/NVLink 拓扑、NUMA 归属与实测有向传输矩阵；不能假定 PCIe 等价于 NVLink，也不能假定跨 CPU 插槽的路径可用。
- **能力边界**：CUDA P2P 能力依设备、驱动与拓扑而定 [厂 S7]。产品页中的 PCIe 规格、GPUDirect RDMA 或营销算力不是 GPU→GPU P2P 可用性证明。所需原子操作等额外能力应由实际搬运协议单独查询与验证。

## 4. NVMe 兜底层

| 项 | 门槛 |
| --- | --- |
| 容量 | ≥150GB（专家尾部 + engram 冷行），建议 ≥1TB [门槛] |
| 顺序读 | ≥5 GB/s [门槛/待测] |
| 4K 随机读（QD1） | 延迟 ≤120 μs [门槛/待测，首台定标] |

实测记录 NVMe 型号、固件、文件系统、direct I/O 设置、块大小、队列深度与方法；不以盘的标称参数代替结果。

## 5. GPU P2P 与系统资格检查

### 5.1 环境与拓扑

- 记录 GPU 精确型号/版本、PCI BDF、UUID、主板/服务器、CPU、NUMA 归属、PCIe switch/root complex 路径、BIOS、Linux 内核、驱动与 CUDA 版本。
- PCIe PRO 6000 配置须核验 GPU 链路支持并在负载下实际协商到 **PCIe 5.0 x16 [门槛]**；最大链路能力不代替实际状态。采集 `nvidia-smi topo -m`、`lspci` 与逐卡链路信息。同一 switch/root complex 是拓扑线索，不是 P2P 合格证明；跨 root/跨 CPU 路径不得凭插槽位置放行。
- 记录 IOMMU 与 ACS 状态。CUDA 对 Linux 裸金属 PCIe P2P 的要求是禁用 IOMMU，以避免静默显存损坏 [厂 S7]；不得仅凭 `iommu=pt` 字样声称满足该要求。ACS 可能将 PCIe peer 流量重定向至 CPU root complex [厂 S7/S8]，须检查实际桥路径，不将其误述为必然拖慢 NVLink。
- 不自动更改 BIOS、内核参数、IOMMU 或 ACS；这些改动涉及隔离与安全，须由部署者在确认环境与风险后操作。虚拟化直通的 IOMMU/VFIO 要求不同于裸金属 [厂 S7]，必须单独验证，不直接套用裸金属配置。
- GPU→pinned 池 DMA 必须可达；绑定本地 NUMA 并记录跨域损耗 [待测]。`RLIMIT_MEMLOCK` ≥ pinned 池大小 [门槛]；THP 建议 `madvise`。

### 5.2 逐对正确性与性能证据

- 对**全部不同 GPU 的有向卡对**查询 `cudaDeviceCanAccessPeer`，并在正确设备上下文调用 `cudaDeviceEnablePeerAccess` 验证启用；双向结果分别检查 [厂 S7]。任一方向不可达则本配置不合格，不通过 host staging 伪装成 GPU P2P。
- 通过真实远端 GPU 内存读写、peer copy、同步与结果校验验证数据正确性；覆盖全部卡对及多卡并发，记录失败与超时。CUDA `simpleP2P` 可作基础正确性探针 [厂 S8]，单次两卡示例不能覆盖全部部署资格。
- `nvidia-smi topo -p2p p` / `-p2p n` 仅作诊断线索；“OK”或能力 API 返回真不能代替数据正确性测试 [厂 S8]。PCIe 配置必须证明选用直接 P2P 路径，不能把通信库的 CPU/RAM 回退结果计为通过。
- 用 `nvbandwidth` 或可复现的 CUDA 探针记录每方向带宽、延迟、单向/双向与多卡并发竞争 [厂 S8/待测]；记录传输大小、预热、重复次数、同步方式、功耗/时钟及拓扑。性能资格阈值须由首台实验与工作负载定标，不预设 PCIe 等价 NVLink，也不将显存带宽或理论链路速度写成实测值。
- 原始证据进入 `bench/results/<日期-主题>`，manifest 含 `version`、`git_sha`、`dirty` 并附环境与方法。**正确性失败、P2P 不可用、缺少资格证据均拒启**；未经定标的性能配置不得宣称完成部署认证。

## 6. OS 与驱动栈

Linux 裸金属服务器，或经过独立验证、NUMA 映射正确且具备 GPU 直通条件的虚拟化服务器。供电、散热与插槽承载能力须匹配 GPU 的具体版本。驱动/CUDA 组合按具体机型资格报告固定；不臆定一个覆盖所有 PRO 6000 版本与服务器拓扑的最低版本。目标机环境与认证进度见 STATUS 及接收工作项。

## 7. 资料索引与研究边界

P0004 于 2026-10-08 核对以下公开资料，方法为产品规格与 CUDA/NCCL 契约核查，非本项目硬件测量。NVIDIA 产品资料与 Lenovo OEM 规格分开标注；论坛/社区报告不作型号保证。没有从“产品页未提 NVLink”单独推导无 NVLink，也不以 GPUDirect RDMA 文档替代 GPU 间 P2P 契约。

- **S1 [厂/NVIDIA]**：[RTX PRO 6000 Blackwell Server Edition 产品页](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/)：Server 的显存与显存带宽。
- **S2 [厂/NVIDIA]**：[Workstation Edition 产品页](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/)及[官方 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/data-center/rtx-pro-6000-blackwell-workstation-edition/workstation-blackwell-rtx-pro-6000-workstation-edition-nvidia-us-3519208-web.pdf)：Workstation 的显存、显存带宽与 PCIe 接口；不作为 P2P 保证。
- **S3 [厂/Lenovo OEM]**：[ThinkSystem Server Edition 产品指南 LP2263](https://lenovopress.lenovo.com/lp2263)：规格表明确 `NVLink support: No`、`Host Interface: PCIe 5.0 x16`。
- **S4 [厂/Lenovo OEM]**：[ThinkSystem Max-Q Workstation Edition 产品指南 LP2364](https://lenovopress.lenovo.com/lp2364)：规格表明确 `NVLink support: No`、PCIe 接口、显存与显存带宽。
- **S5 [厂/NVIDIA]**：[CUDA GPU Compute Capability](https://developer.nvidia.com/cuda-gpus)：A100 为 8.0，PRO 6000 Blackwell 三版为 12.0。
- **S6 [厂/NVIDIA]**：[A100 产品页](https://www.nvidia.com/en-us/data-center/a100/)：A100 80GB SXM 规格；互联数字是产品规格，不是目标整机测试。
- **S7 [厂/NVIDIA]**：[CUDA Programming Guide — Multi-GPU Systems](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html)，§3.4.2.2 Peer-to-Peer Memory Access 与 §3.4.2.5 Host IOMMU Hardware, PCI Access Control Services, and VMs：拓扑/设备相关的能力查询、peer access 启用及裸金属/虚拟化区别。
- **S8 [厂/NVIDIA]**：[NCCL GPU Troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html)：GPU-to-GPU 诊断、`simpleP2P`、`nvbandwidth`、IOMMU 与 ACS。

A800 SXM 保留既有型号范围，但本次未重新取得完整模组规格证据，因此不延续其未经本次核对的互联/显存带宽数字；部署前仍须取得对应模组厂商资料并通过上述整机资格检查。
