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

