ConvStencil: Transform Stencil Computation to Matrix Multiplication on Tensor Cores

链接：https://dl.acm.org/doi/pdf/10.1145/3627535.3638476
引用：Yuetao Chen, Kun Li, Yuhao Wang, Donglin Bai, Lei Wang, Lingxiao Ma, Liang Yuan, Yunquan Zhang, Ting Cao, and Mao Yang. 2024. ConvStencil: Transform Stencil Computation to Matrix Multiplication on Tensor Cores. In Proceedings of the 29th ACM SIGPLAN Annual Symposium on Principles and Practice of Parallel Programming (PPoPP '24). Association for Computing Machinery, New York, NY, USA, 333–347. https://doi.org/10.1145/3627535.3638476

---
![image](https://github.com/user-attachments/assets/8c65ef9d-829a-4a31-b993-cdefbe66482f)
- A00 80GB PCIe:
	- 使用标准的 PCIe（Peripheral Component Interconnect Express）接口，方便与标准服务器进行兼容。
	- 这种形式的 GPU 可以插入任何支持 PCIe 插槽的主板，不需要特殊的硬件支持。
- A100 80GB SXM:
	- 使用 NVIDIA 的 SXM（Scalable Matrix Extension）接口，通常需要与专用的 NVIDIA HGX 服务器平台或具有 SXM 支持的系统一起使用。
	- SXM 设计允许更高的带宽和更紧密的集成，但对硬件有更高的要求。
---

Stencil 本身就是 Memory-bound 的算法；

im2col 方法的缺点：

将输入通道转换为矩阵形式后，相邻两行之间至少有[(kernel_size - 1) / kernel_size]重复元素，