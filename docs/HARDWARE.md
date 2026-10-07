# HARDWARE.md — 目标硬件、支持矩阵与部署资质

本页定义目标硬件形态、支持矩阵与**部署资质门槛**（将由 `tools/qualify_hw.py` [待实现]实现，`deploy/` 启动前执行，不过即拒启）。

**来源标记**：**[厂]** 厂商公开规格 · **[算]** 由公开规格推导（附式）· **[待测]** 以实测为准 ·

---

## 1. 目标形态与非目标

**单机柜**：1 台**双路**服务器（2×CPU，**8/12 通道/路、16/32 通道合计**，RAM + NVMe）+ 2/4 张 NVLink 互联 GPU。内存按 NUMA 分两个域管理。

**非目标**：多机分布式、云弹性调度、Windows、无 NVLink 的 PCIe 卡。

## 2. CPU 与内存平台

| 项 | 门槛 | 说明 |
| --- | --- | --- |
| 内存通道 | **≥8/路 ×2 路（≥16 通道合计）且插满** | 尽可能使用满cpu带宽 |
| 内存代际 | **DDR4/DDR5** | 8×DDR4-3200双域聚合409.6 GB/s，8×DDR5-4800双域聚合614.4 GB/s  |
| 持续带宽 | **每域**流测试 ≥ 该域理论 80%（STREAM 类）[待测] | CPU 计算绑定 NUMA域与内存 |
| 容量 | 总量 ≥500GB（如 16×32GB=512GB） | **每域可用 ≥240GB** |
| 指令集 | AVX-512（F/BW/DQ/VL）**+ AVX512_VNNI（`vpdpwssd`）** | 整数格路径依赖；无 VNNI 直接拒启 |
| 参考平台 | Intel Xeon SPR / GNR 双路（8 通道 DDR5/路）[厂] | 或AMD EPYC 9004/9005 等 12 通道/路[厂] |


## 3. GPU 支持矩阵

| 卡 | 架构 | VRAM | 显存带宽 | 卡间互联 | 计算路径 |
| --- | --- | --- | --- | --- | --- |
| A100 80GB **SXM** | sm_80 | 80GB HBM2e | ≈2.0 TB/s [厂] | NVLink **600 GB/s**，NVSwitch **全互联** [厂] | K-E1 / K-E2（INT8 TC 624 TOPS、FP16 312 TFLOPS dense [厂]） |
| A800 80GB **SXM** | sm_80 | 80GB HBM2e | ≈2.0 TB/s [厂] | NVLink **400 GB/s**，NVSwitch 全互联 [厂] | 同 A100 |
| RTX PRO 6000 Blackwell **Server** | sm_120 | 96GB GDDR7 ECC | ≈1.6 TB/s（512-bit；Workstation 版 1792 GB/s）[厂] | **2-way NVLink 桥**，标称 ~400 GB/s 量级 [厂/待测] | 原生块缩放 FP4/FP8 TC；厂商标称 ~4 PFLOPS 级为稀疏口径，dense 折半 [待测] |

- **排除**：A100/A800 PCIe 版（无 NVLink）；H100/B100 等（超范围）；消费卡。
- **拓扑注记 [厂→规格]**：A100/A800 SXM 经 NVSwitch 全互联（4 卡任意点对点）；**PRO6000 的 NVLink 是桥接对——4 卡仅两两成对、无全互联**，K-M1（remote_experts）的放置拓扑必须按"对"分簇设计。
- **混插**：sm80 与 sm120 混插不支持

## 4. NVMe 兜底层

| 项 | 门槛 |
| --- | --- |
| 容量 | ≥150GB（专家尾部 + engram 冷行），建议 ≥1TB   |
| 顺序读 | ≥5 GB/s [待测] |
| 4K 随机读（QD1） | 延迟 ≤120 μs [待测，首台定标] |

## 5. PCIe / 系统设置

- GPU→pinned 池 DMA 可达；跨 NUMA 访问损耗实测记录[待测]。
- IOMMU/ACS 对 P2P 的影响必须记录（ACS 强制经根复合体时会显著拖慢 NVLink P2P 路径）[待测]。
- `RLIMIT_MEMLOCK` ≥ pinned 池大小；THP 建议 `madvise` 。

## 6. OS 与驱动栈

BareMetal 服务器，或有正确 NUMA 映射的虚拟化服务器，Linux（发行版/内核版本 [待测]）；NVIDIA 驱动与 CUDA toolkit 在 `deploy/` 定版后写入本节。
