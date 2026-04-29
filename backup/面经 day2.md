## 一、  GPU架构，包括硬件架构和软件编程架构

#### 1. 硬件架构：

1. 核心执行单元是 SM/CU：

名称 | NVIDIA | AMD
-- | -- | --
计算簇 | SM (Streaming Multiprocessor) | CU (Compute Unit)
最小调度单元 | Warp = 32 threads | Wavefront = 64 threads
向量执行 | 32 FP ALU / tensor core | 64 ALU / MFMA

SM/CU 内部包含：
  标量/向量 ALU（FP32/FP16/BF16/INT8）
  Tensor Core / Matrix Core (NVIDIA)
  MFMA Matrix Core (AMD)
  寄存器文件（Register File）
  共享内存 / LDS（SRAM）
  L1 Cache（可与共享内存融合）
  
2. 层次化存储结构：

层级 | 类型 | 延迟 | 带宽 | 大小
-- | -- | -- | -- | --
寄存器 | SRAM | 几个 cycle | 极高 | 每线程几十～上百
Shared Memory / LDS | SRAM | 10~20 cycles | 很高 | 48–164 KB/SM
L1 Cache | SRAM | ~20 cycles | 高 | 128 KB ~ 256 KB
L2 Cache | SRAM | 数百 cycles | 中 | 数 MB
HBM/DRAM | DRAM | 几百 cycles | 高但慢于SRAM两个数量级 | GB级

GPU 计算性能非常强，但 DRAM 访问极慢，因此一切优化都围绕减少 HBM 访问展开。

3. 线程模型

抽象层 | NVIDIA硬件 | 意义
-- | -- | --
Thread | 线程 | 最小执行主体
Warp | 32 threads | 同步执行，SIMD
Block (CTA) | 多个 warp | 可共享 shared memory
Grid | 多个 block | 分派到多个 SM

> AMD 类似，只是最小 SIMD 是 wavefront = 64。

4. . 指令调度与并发

核心思想：GPU 要保持 SM 处于 “满载（high occupancy）” 状态。

如果 kernel：
  - 寄存器使用太多 → warp 数减少 → occupancy 降低
  - 访存不 coalesce → L1/L2 miss → 访存 stall → SM idle

5. Tensor Core / MFMA（矩阵计算）

- Tensor Core（NVIDIA）
- MFMA（AMD）

都是专门的矩阵-FMA 并行阵列，用于：
  - GEMM
  - Attention
  - Convolution

#### 2. 软件编程架构

软件编程架构分为：
  - 底层：驱动 + Runtime + JIT
  - 编程语言：CUDA/HIP、Triton、cuTile
  - 高层库：cuBLAS/CUTLASS/cuDNN 等
  
1. 驱动层：

NVIDIA：
  - NVIDIA Driver (libcuda.so)
  - CUDA Runtime (libcudart.so)
  
AMD：
  - ROCm driver
  	- HIP runtime
  
作用：
  创建 GPU 上下文
  分配显存
  启动 kernel
  管理 stream & events
  设备同步

cudaMalloc, cudaMemcpy, hipMemcpy 都属于这一层。

2. 编程层：CUDA/HIP

明确暴漏 Grid/Block/Thread
共享内存用 __shared__ 管理，区分 host/device/global 程序，asm 嵌入，内置函数等。

3. Triton

Triton（软件层 DSL）
Triton，它把 GPU 编程抽象成：
  program = tile/block
  内部没有 warp 的概念，由编译器负责
  显式创建向量维度 (tl.arange)
  自动做 coalescing / vectorization / pipelining
  Triton 的抽象让注意力类 kernel 简化到几十行即可获得接近 CUTLASS 水平性能。

4. 高层库

CUTLASS：
它是 CUDA C++ 的 GEMM 内核生成器。

特征：
可控 tile 尺寸、warp shape、thread shape
Tensor Core 配置
epilogue（softmax/activation/bias）可融合
性能接近 cuBLAS

cuBLAS：
cuBLAS 是黑盒高性能库
CUTLASS 是你自己构建内核的框架，可以扩展

cuTile：
cuTile / TileLang 更抽象，让你只描述算法分块

cuDNN：
深度学习算子库

5. Runtime 调度（Streams / Graphs）

GPU kernel 的调度包括：
  - Stream：GPU 上的工作队列
  - Event：同步 primitive
  - CUDA Graph：用 DAG 表达 kernel + memcpy，减少 kernel launch overhead
  
#### 3. 硬件架构/软件架构对应：

软件抽象 | 硬件实体
-- | --
threadIdx.x | ALU lane
warp | SIMD 执行单元
block (CTA) | SM（调度单位）
shared memory | SM 内部 SRAM
register | RF（寄存器文件）
global memory | HBM
Tensor Core API | GPU 的矩阵加速单元


