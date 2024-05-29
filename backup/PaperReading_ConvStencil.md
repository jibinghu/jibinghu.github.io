##### *PAPER READING*

**@address: https://dl.acm.org/doi/10.1145/3627535.3638476**
**@github: https://github.com/microsoft/ConvStencil**

### ConvStencil: Transform Stencil Computation to Matrix Multiplication on Tensor Cores

##### 关键词：
    模版计算，卷积，张量核，矩阵乘

##### 摘要：

文章提出了ConvStencil，通过有效地将stencil模版计算转化为在张量核Tensor Core上的矩阵计算来实现。
    文章提出三种技术来实现ConvStencil：
* 使用sencil2row方式实现内存高效布局转换
* 基于Dual Tessel-lation and kernel fusion的计算密度自适应算法
* 使用可扩展表和脏位实现性能提升冲突的消除

##### 引言：

> 引言提供了将张量核心单元（TCU）集成到现代处理器中以增强矩阵乘法性能的背景信息。它指出了TCU在模板计算中未得到充分利用的问题，并提出了ConvStencil作为解决方案。主要贡献包括性能模型、内存高效布局转换、计算适配和冲突消除技术。

Tensor Core已经被完成的工作证实可以用作一些简单的reduction和scan原语操作，目前的工作，比如TCStencil关注Tensor Core是否可以应用于更复杂的类似stenci的操作。但目前，一方面TCStencil仅支持精度FP16且只适用于对称矩阵乘操作，而大多数模版计算需要FP64精度下的非对称矩阵乘；另一方面，TCStencil会遇到全局内存的非合并内存访问以及共享内存的bank conflict，从而限制Tensor Core的充分利用。

通过im2row(col)方式使得在张量核上将卷积操作转化为矩阵乘操作，所以本文的关键在于在Tensor Core和Stencil计算之间通过im2row方式进行连接和转化。但是
1. im2row将卷积操作转化为矩阵乘，然而，这种转换会导致矩阵-向量乘法，因为在每次迭代中，模板核和通道的数量都是1，这可能导致显著的内存膨胀和低张量核心（Tensor Core）利用率。
2. 另外，对于高精度的FP64位数据，张量核只适合小型的非对称矩阵计算。
3. 此外，算法的实现和设计可能会遇到影响性能的冲突，如warp分歧和bank冲突，导致性能大幅下降。

> im2row:在卷积运算中，我们需要对图像的每一个局部区域（称为感受野）进行卷积操作，将其与卷积核进行元素级乘法和求和。传统的卷积操作是通过滑动窗口在输入图像上移动卷积核来实现的。然而，这种操作在计算上不够高效。im2row 操作通过将卷积操作转化为矩阵乘法，从而利用高度优化的矩阵乘法实现进行加速。

**im2row 的具体步骤：**

假设我们有一个大小为  $H \times W$ 的输入图像和一个大小为 $K \times K$ 的卷积核。`im2row` 操作的步骤如下：
* 提取感受野：对于输入图像中的每一个 $K \times K$ 的感受野，将其展开为一个行向量。
* 构建矩阵：将所有的行向量堆叠成一个矩阵，每一行对应一个感受野。
* 矩阵乘法：将这个矩阵与展开后的卷积核进行矩阵乘法，得到卷积操作的结果。

文章中提出的Convstencil方式解决了上述的三种问题；
* 对于Layout Transformation， 引入了stencil2row方式减少了内存空间占用；
* 在计算适配中，提出双镶嵌（Dual Tessellation）方法，通过矩阵镶嵌来提高张量核心（Tensor Core）的利用率。同时利用kernel fusion减少矩阵稀疏度提高核心利用率。
* 在减少冲突中，提出查找表来减少大量的地址计算以及额外的消耗；使用脏位空间来填充脏位来避免条件分支。

相对于同样使用Tensor Core的TCStencil，全局内存非合并访存以及共享内存bank冲突都有大幅度减少。




EXPAND：

1. Stencil 计算
Stencil 计算是科学和工程计算中常见的计算模式。它通过迭代更新网格上的每个点，依赖于其邻域点的值。常见应用包括：流体动力学：计算流体流动。地球建模：地震波传播模拟。天气模拟：气象预报。

Stencil 计算及其原理：
> Stencil计算是一种在科学和工程领域中广泛应用的数值计算技术，主要用于求解偏微分方程（PDEs）、模拟物理现象（如流体动力学、地球建模和气象预报）等。Stencil计算的基本原理是在多维空间网格上迭代更新每个点的值，该值是该点及其邻近点在上一个时间步的加权和。具体来说，Stencil计算包含一个预定义的模式，该模式规定如何使用网格上一个点及其周围邻近点的值来计算该点在下一个时间步的值。

* 基本组成部分
  * 空间网格: 一个d维网格，用来表示计算区域，每个网格点都保存一个数值。
  * Stencil核: 一个定义了每个网格点如何通过自身及其邻近点的值来更新的权重模式。常见的Stencil核形状包括星形和盒形。
  * 时间步: Stencil计算通过时间步迭代更新网格点的值，每个时间步表示一个独立的计算过程。

* Stencil计算原理
  * 初始化: 在第一个时间步开始时，根据初始条件设置网格中每个点的初始值。
  * 迭代更新: 对于每个时间步，根据Stencil核的定义计算网格中每个点的新值。新的值是该点及其邻近点在上一个时间步的加权和。
  * 边界条件: 处理网格边界上的点，边界条件决定了如何计算这些点的值（例如，固定值或周期性条件）。

* Stencil 计算公式：
对于一个二维网格上的Stencil计算，假设时间步为 $t$，空间坐标为 $(i, j)$，则Stencil计算的一般公式为：

  $$u_{i,j}^{(t+1)} = \sum_{k,l} w_{k,l} \cdot u_{i+k, j+l}^{(t)}$$

    其中， $u_{i,j}^{(t+1)}$ 是在时间步 $t+1$ 时网格点 $(i, j)$ 的值， $w_{k,l}$ 是Stencil核的权重， $u_{i+k, j+l}^{(t)}$ 是时间步 $t$时相应邻近点的值。

* 优化技术
Stencil计算的性能优化是一个重要的研究领域，常见的优化技术包括：
  * 内存布局优化: 通过调整数据存储方式减少内存访问冲突，提高内存带宽利用率。
  * 计算密度优化: 增加计算过程中有意义的计算操作比例，减少无效计算和数据传输。
  * 并行计算技术: 使用多线程、GPU加速、向量化等技术提高计算效率。
  * 重用数据: 通过数据重用减少内存访问次数，例如缓存优化和数据局部性优化。
  * 时间步融合: 将多个时间步的计算合并到一次计算中，减少内存传输和同步开销。



* 延伸与应用
Stencil计算可以应用于多种科学和工程领域，以下是一些典型的应用和延伸：
  * 流体动力学模拟: 用于模拟流体流动和行为，例如风洞实验和海洋流动模拟。
  * 地球物理建模: 用于模拟地震波传播、地下水流动和地质结构分析。
  * 气象预报: 用于天气预报模型，模拟大气现象和气候变化。
  * 图像处理: 用于图像滤波、边缘检测和图像去噪。
  * 并行计算优化: Stencil计算具有高度的数据并行性，可以在高性能计算（HPC）平台上进行优化，例如使用GPU或多核处理器来加速计算。

1. Tensor Core
Tensor Core 是 NVIDIA GPU 的一个硬件单元，专门用于加速矩阵乘法。它在深度学习训练和推理中发挥重要作用，能够高效地执行混合精度运算（FP16 和 FP32）。

1. 矩阵乘法 (Matrix Multiplication, MM)
矩阵乘法是线性代数中的基本操作，将两个矩阵相乘以生成第三个矩阵。Tensor Core 特别优化了这一操作，可以显著加速计算。

1. 性能模型 (Performance Model)
性能模型是一种理论工具，用于预测和分析算法在特定硬件上的性能。论文中，性能模型用于指导 Tensor Core 上的算法设计和优化。

1. 内存高效的布局转换（stencil2row 方法）
内存高效的布局转换是指将数据重新排列成一种便于计算和存储的格式。stencil2row 方法通过减少不必要的数据重复和占用，显著降低了内存使用。

传统方法（im2row）：通常将输入数据展开为行，这会导致大量的冗余数据。
stencil2row 方法：通过更紧凑的方式重新排列数据，减少内存占用 70% 到 96.4%。

6. 双重镶嵌（Dual Tessellation）
双重镶嵌是一种将计算任务划分为更小块的技术，使其更适合 Tensor Core 的计算模式。通过分块处理，可以最大化 Tensor Core 的利用率。

7. 内核融合（Kernel Fusion）
内核融合是将多个计算内核合并为一个内核，以减少内存访问和数据传输。通过内核融合，可以提高计算密度和效率，减少稀疏性问题。

8. 冲突消除
冲突消除是指通过优化数据访问模式，减少内存访问冲突和银行冲突。

查找表：预计算并存储指针偏移，减少运行时的计算。
脏位填充（Dirty Bits Padding）：通过在数据中添加填充位，避免条件分支和冲突，提高数据访问效率。

**实验与评估**
论文通过在 AMD EPYC 7V13 处理器和 NVIDIA A100 Tensor Core GPU 上进行实验，验证了 ConvStencil 系统的性能提升。使用了多种 Stencil 核（如 Heat-1D、Box-2D9P、Heat-3D 等）进行基准测试。

**结果与结论**
性能提升：相比于其他优化框架，ConvStencil 展示了显著的加速效果。
内存访问优化：显著减少了非合并全局内存访问和银行冲突。
通用性：适用于多种 Stencil 核，具有良好的通用性和扩展性。

**总结**
ConvStencil 提供了一种创新的方法，通过将 Stencil 计算转换为矩阵乘法来充分利用 Tensor Core 的计算能力。通过内存优化、计算适配和冲突消除，ConvStencil 显著提升了 Stencil 计算的性能，展示了在高性能计算领域的广泛应用前景。


#### 和分解(sum factorization)
    化矩阵向量乘为矩阵矩阵乘，

谱元，谱变换，傅立叶谱方法，高阶有限元，

```cpp
#include <iostream>
#include <cuda_runtime.h>
#include <cusparse_v2.h>

// 定义矩阵和向量大小
const int N = 100;  // 网格大小
const int NNZ = 300;  // 非零元素数量

// 检查CUDA错误的宏
#define CHECK_CUDA(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(err); \
    } \
} while (0)

void initialize_matrices(double* h_D, double* h_G, double* h_u) {
    // 初始化矩阵和向量（用户自定义）
    // 这里只是简单示例，用户可以根据需要填充实际值
    for (int i = 0; i < N * N; ++i) {
        h_D[i] = static_cast<double>(rand()) / RAND_MAX;
        h_G[i] = static_cast<double>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < N; ++i) {
        h_u[i] = static_cast<double>(rand()) / RAND_MAX;
    }
}

void sum_factorization(double* h_D, double* h_G, double* h_u, double* h_result) {
    double *d_D, *d_G, *d_u, *d_intermediate, *d_result;
    size_t size_DG = N * N * sizeof(double);
    size_t size_u = N * sizeof(double);

    // 分配设备内存
    CHECK_CUDA(cudaMalloc((void**)&d_D, size_DG));
    CHECK_CUDA(cudaMalloc((void**)&d_G, size_DG));
    CHECK_CUDA(cudaMalloc((void**)&d_u, size_u));
    CHECK_CUDA(cudaMalloc((void**)&d_intermediate, size_u));
    CHECK_CUDA(cudaMalloc((void**)&d_result, size_u));

    // 将数据从主机内存传输到设备内存
    CHECK_CUDA(cudaMemcpy(d_D, h_D, size_DG, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_G, h_G, size_DG, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_u, h_u, size_u, cudaMemcpyHostToDevice));

    // 创建cuSPARSE句柄
    cusparseHandle_t handle;
    cusparseCreate(&handle);

    // 计算D * u
    const double alpha = 1.0;
    const double beta = 0.0;
    cusparseDnMatDescr_t matD, matG;
    cusparseDnVecDescr_t vecU, vecIntermediate, vecResult;

    // 创建cuSPARSE描述符
    cusparseCreateDnMat(&matD, N, N, N, d_D, CUDA_R_64F, CUSPARSE_ORDER_ROW);
    cusparseCreateDnMat(&matG, N, N, N, d_G, CUDA_R_64F, CUSPARSE_ORDER_ROW);
    cusparseCreateDnVec(&vecU, N, d_u, CUDA_R_64F);
    cusparseCreateDnVec(&vecIntermediate, N, d_intermediate, CUDA_R_64F);
    cusparseCreateDnVec(&vecResult, N, d_result, CUDA_R_64F);

    // 进行矩阵向量乘法D * u
    cusparseDnMatVec(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matD, vecU, &beta, vecIntermediate, CUDA_R_64F);

    // 进行矩阵向量乘法G * (D * u)
    cusparseDnMatVec(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matG, vecIntermediate, &beta, vecResult, CUDA_R_64F);

    // 进行矩阵向量乘法D^T * (G * (D * u))
    cusparseDnMatVec(handle, CUSPARSE_OPERATION_TRANSPOSE, &alpha, matD, vecResult, &beta, vecIntermediate, CUDA_R_64F);

    // 将结果从设备内存传输到主机内存
    CHECK_CUDA(cudaMemcpy(h_result, d_intermediate, size_u, cudaMemcpyDeviceToHost));

    // 释放设备内存
    CHECK_CUDA(cudaFree(d_D));
    CHECK_CUDA(cudaFree(d_G));
    CHECK_CUDA(cudaFree(d_u));
    CHECK_CUDA(cudaFree(d_intermediate));
    CHECK_CUDA(cudaFree(d_result));

    // 销毁cuSPARSE描述符和句柄
    cusparseDestroyDnMat(matD);
    cusparseDestroyDnMat(matG);
    cusparseDestroyDnVec(vecU);
    cusparseDestroyDnVec(vecIntermediate);
    cusparseDestroyDnVec(vecResult);
    cusparseDestroy(handle);
}

int main() {
    double *h_D = new double[N * N];
    double *h_G = new double[N * N];
    double *h_u = new double[N];
    double *h_result = new double[N];

    // 初始化矩阵和向量
    initialize_matrices(h_D, h_G, h_u);

    // 进行和分解计算
    sum_factorization(h_D, h_G, h_u, h_result);

    // 输出结果
    for (int i = 0; i < N; ++i) {
        std::cout << h_result[i] << " ";
    }
    std::cout << std::endl;

    delete[] h_D;
    delete[] h_G;
    delete[] h_u;
    delete[] h_result;

    return 0;
}
```