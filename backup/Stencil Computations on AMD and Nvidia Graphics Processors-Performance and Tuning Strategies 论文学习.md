> 在 AMD/Nvidia 显卡上的 Stencil 评测和优化策略

摘要：

In this study, we evaluate the performance and energy efficiency of stencil computations on modern datacenter graphics processors, and propose a tuning strategy for fusing cache‐heavy stencil kernels.

> 提出了一种融合缓存密集型的 Stencil Kernel，从一维到三维

介绍：

模板计算从邻域中以固定模式来更新信息的算法，举例 中值滤波(取窗口内所有像素值的中位数来替代当前像素值，相对于平均滤波保留边缘能力强)，另外还可以通过卷积等Stencil 操作来实现 Conv/边缘检测等算法。

- 元胞自动机：

通过 Moore 邻域、Von Neumann 邻域等方式来决定影响规则，进而形成演化规则。
可以简单理解为离散的 Stencil 计算：通过状态跳转（查表、位运算）等方式来实现

- 基于有限差分的直接数值模拟：

简单理解为连续的 Stencil 计算，把 PDE 离散化为差分格式(从定义角度出发，逼近导数)，

[Scalable communication for high-order stencil computations using CUDA-aware MPI](https://www.themoonlight.io/extension/dist/web/pdf.html?origin=https%3A%2F%2Farxiv.org%2Fpdf%2F2406.08923&file=blob%3Ahttps%3A%2F%2Fwww.themoonlight.io%2Fa3877b5d-df17-4970-8e9c-626af5813d03&paperId=8a823000-7090-4748-a2e3-d8d3aee2d27f&isPaperOwner=true#5)

[Magnetohydrodynamics with GAMER](https://www.themoonlight.io/extension/dist/web/pdf.html?origin=https%3A%2F%2Farxiv.org%2Fpdf%2F2406.08923&file=blob%3Ahttps%3A%2F%2Fwww.themoonlight.io%2Fa3877b5d-df17-4970-8e9c-626af5813d03&paperId=8a823000-7090-4748-a2e3-d8d3aee2d27f&isPaperOwner=true#6)

<img width="525" height="308" alt="Image" src="https://github.com/user-attachments/assets/c241228e-ac88-4afa-8508-8dc83d645fa2" />

以一维 Stencil 计算为例，通过将处理边界后的输入经过模板函数的处理得到输出。

> 并介绍了一些在本文使用的术语。

Nvidia 和 AMD 的 GPU 制造上的差异(缓存配置)：


在Nvidia Volta及更新的GPU上，L1的一部分可以分配为共享内存(On Chip)，用于在CU上运行的一组线程之间的协作。在AMD CDNA 2 GPU上，共享内存分配在单独的内存单元上，称为本地数据共享（LDS），它位于CU之外。内存区域的容量也不同。例如，MI250X的共享内存容量大约比A100小2.5倍，但其每个CU的计算FP64性能大约高2.4倍。因此，程序必须以更低共享内存容量实现更高的运算强度，才能在MI250X上达到机器平衡。

<img width="948" height="289" alt="Image" src="https://github.com/user-attachments/assets/8a5913e4-a67e-488d-8b99-c3a6a45cdeb6" />

---

结论：

模板计算本质上受内存限制且对缓存敏感。每个网格点仅参与少量邻居更新，导致算术强度非常低。与GEMM不同，模板内核无法通过大量计算摊销全局内存延迟，因此严重依赖缓存或显式管理的共享/LDS内存进行数据重用。

本文旨在提出一种新的模板离散化或数值方案。相反，它解决了一个关键的性能工程问题：为Nvidia GPU开发的模板调整策略在现代AMD架构上是否仍然有效。通过对缓存和内存密集型模板内核进行广泛的跨平台基准测试，作者们证明了模板优化高度依赖于架构。他们进一步确定了关于缓存使用、显式共享/LDS内存管理、内核融合以及能效权衡的特定平台调整策略。

优化维度 | NVIDIA GPU（A100 / H100 / V100） | AMD GPU（MI100 / MI250X / CDNA2）
-- | -- | --
架构核心特征 | SM 内部 L1 与 Shared 复用同一片 on-chip SRAM | LDS 是独立存储单元，与 CU 分离
硬件缓存可靠性 | ✅ L1/L2 非常可靠，stencil 可高度依赖 | ⚠️ L1/L2 对 stencil 的稳定性不如 NVIDIA
共享内存（shared/LDS）角色 | “可选加速器”：不用也能跑得不错 | “必需组件”：不用 LDS 性能大幅下降
典型最优策略核心 | ✅ Cache-first → 再考虑 shared | ✅ LDS-first → cache 作为补充
Kernel Fusion（多时间步融合） | ⭐ 有收益，但不是决定性 | ⭐⭐⭐⭐⭐ 几乎是刚需
时间步复用（temporal reuse） | 可部分由 L1/L2 自动完成 | 必须显式放入 LDS 并跨 time-step 复用
空间复用（spatial reuse） | 由 L1 自动提供一部分 | 必须通过 LDS tile 布局显式控制
Block / Workgroup 尺寸敏感性 | ✅ 相对不敏感（容错性强） | ❗ 高度敏感（与 wavefront、LDS 冲突强相关）
Warp/Wavefront 模型影响 | 32 线程 Warp，block 设计自由度高 | 64 线程 Wavefront，block 设计受限更强
Occupancy 容忍度 | ✅ 即使 occupancy 稍低也能隐藏 latency | ❗ 过低 occupancy 会直接拖垮性能
Register 压力容忍度 | ✅ 容忍度较高 | ❗ register 多 → occupancy 迅速下降
典型性能上限瓶颈 | ✅ 多数 stencil 最终是 HBM-bandwidth-bound | ❗ 多数 stencil 是 LDS-bound / reuse-bound
共享内存冲突影响 | 相对温和 | ❗ bank 冲突 + LDS 端口压力非常致命
Naive stencil kernel（无 LDS / 无 fusion） | ✅ 还能有“中高等性能” | ❌ 性能通常非常差
调优主要目标 | 把 stencil 推到 HBM 屋顶线（roofline top） | 把 stencil 从 LDS 限制中“解放出来”
对 tile 尺寸的要求 | 中等即可（如 16×16，32×8） | 必须精细匹配 LDS（如 8×8×Z，小而密）
对 kernel 结构的要求 | 单 kernel + 硬件 cache 已可接受 | 必须重构 kernel 结构（fusion / split）
代码“可迁移性” | ✅ CUDA stencil kernel 迁移性很好 | ❌ CUDA 思路直接搬到 HIP 上风险极大
能效最优点（Perf/Watt） | 多数与“性能最优点”接近 | ❗ 多数 早于性能最优点出现
是否适合“通用 stencil 模板” | ✅ 适合 | ❌ 不适合，必须架构定制
工程调优难度 | ⭐⭐（相对友好） | ⭐⭐⭐⭐⭐（明显更硬核）

NVIDIA 的 L1 与共享内存位于同一片片上 SRAM，L1 对 stencil 的局部性非常友好且行为稳定，因此在很多情况下可以仅依靠硬件缓存而不显式使用 shared memory。而在 AMD GPU 上，L1 cache 的行为对 stencil 的空间与时间复用并不稳定，无法保证数据持续驻留在片上缓存中。尽管 AMD 的 LDS 与 NVIDIA 的 shared 一样是低延迟片上存储，但它提供的是“可控、确定性的复用”。由于 stencil 的算术强度极低，无法通过计算隐藏 HBM 延迟，因此在 AMD 上必须依赖显式 LDS 管理来获得可预测的高性能。

---

扩展AMD 的 GPU 架构 GCN -> RDNA/CDNA

对比维度 | GCN（Graphics Core Next） | RDNA（Radeon DNA） | CDNA（Compute DNA）
-- | -- | -- | --
首次发布时间 | 2012 | 2019 | 2020
设计初衷 | 通用：图形 + 计算二合一 | 纯游戏 / 图形 | 纯 HPC / AI 计算
产品线 | RX 480、Vega 64、MI50、MI100 | RX 5700、RX 6800、RX 7900 | MI100、MI250X、MI300
是否支持图形渲染 | ✅ 支持 | ✅ 极强 | ❌ 不支持
是否支持光线追踪 | ❌ | ✅ | ❌
是否面向超算 | ⚠️ 勉强支持 | ❌ 不支持 | ✅ 核心目标
是否面向大模型 / AI | ❌ | ❌ | ✅ 核心目标
目标精度类型 | FP32 为主 | FP32 / FP16 为主 | ✅ FP64 / FP32 / BF16
FP64 与 FP32 比例 | 1/2 ～ 1/4 | ❌ 1/16 ～ 1/32 | ✅ 1/2 或更高
HBM 显存支持 | ⚠️ 少数型号支持 | ❌ 全部不支持 | ✅ 全系列标配
片上共享存储（LDS） | ✅ 有 | ✅ 有 | ✅ 极其核心
LDS 是否为关键性能资源 | ⚠️ 中等 | ❌ 次要 | ✅ 绝对核心
L1 Cache 行为特点 | 偏旧式 cache | 面向图形流 | ❌ 对 stencil 不稳定
LDS 与 L1 是否复用物理 SRAM | ❌ 否 | ❌ 否 | ❌ 否
LDS 是否可编程控制 | ✅ 是 | ✅ 是 | ✅ 是
L1 Cache 是否可编程控制 | ❌ 否 | ❌ 否 | ❌ 否
执行单元名称 | CU（Compute Unit） | WGP（Work Group Processor） | CU（Compute Unit）
最小执行粒度 | Wavefront = 64 | 可 32 / 64 | Wavefront = 64
面向的线程模型 | OpenCL / HIP | DX12 / Vulkan | HIP / ROCm / MPI
是否支持 Tensor / Matrix Core | ❌ | ❌ | ✅
是否适合 Stencil / PDE | ⚠️ 可用但不高效 | ❌ 不适合 | ✅ 最佳架构
Stencil 的主要瓶颈 | HBM + cache | 完全不适配 | ✅ LDS 复用与冲突
Stencil 是否必须显式用 LDS | ⚠️ 可选 | ❌ 不重要 | ✅ 必须
Kernel Fusion 在 stencil 中的重要性 | ⭐⭐ | ❌ 无意义 | ⭐⭐⭐⭐⭐ 刚需
是否适合 GEMM / BLAS | ⚠️ 一般 | ❌ | ✅ 极强
是否适合 CFD / 气候模式 | ⚠️ 勉强 | ❌ | ✅ 最佳
是否适合大模型训练 | ❌ | ❌ | ✅ MI250X / MI300
典型能效目标 | 中等 | 面向功耗墙下的帧率 | ✅ Exascale 性能/瓦
是否适合 CUDA 风格直接迁移 | ⚠️ 勉强 | ❌ 完全不适合 | ✅ HIP 可迁移但需重调
是否适合 Triton / CUTLASS 风格 | ❌ | ❌ | ⚠️ 需专门适配
AMD 当前的战略地位 | ❌ 已淘汰 | ✅ 游戏主力 | ✅ 超算与 AI 战略核心








