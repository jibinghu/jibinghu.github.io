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











