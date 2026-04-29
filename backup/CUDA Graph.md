使用 CUDA 图（CUDA Graph）来加速向量加法操作的示例。
--》〉该示例展示了如何通过 CUDA 图将一系列 CUDA 操作打包成一个图形，然后在 GPU 上执行这个图形，从而减少内核启动的开销。
``` cuda
#include <cuda_runtime.h>
#include <iostream>

#define N 1000000  // 向量大小

__global__ void vectorAdd(const float* A, const float* B, float* C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 分配主机内存
    float *h_A = (float*)malloc(N * sizeof(float));
    float *h_B = (float*)malloc(N * sizeof(float));
    float *h_C = (float*)malloc(N * sizeof(float));

    // 初始化输入向量
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }

    // 分配设备内存
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, N * sizeof(float));
    cudaMalloc((void**)&d_B, N * sizeof(float));
    cudaMalloc((void**)&d_C, N * sizeof(float));

    // 创建 CUDA 图
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    // 将数据从主机传输到设备
    cudaMemcpyAsync(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice, stream);

    // 执行向量加法内核
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_A, d_B, d_C, N);

    // 将结果从设备传回主机
    cudaMemcpyAsync(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost, stream);

    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

    // 执行图
    cudaGraphLaunch(graphExec, stream);
    cudaStreamSynchronize(stream);

    // 验证结果
    for (int i = 0; i < N; ++i) {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
            std::cerr << "Result verification failed at element " << i << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    std::cout << "Test PASSED" << std::endl;

    // 释放资源
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    cudaStreamDestroy(stream);

    return 0;
}
```
代码说明：

	1.	内核函数 vectorAdd： 执行向量加法操作，将输入向量 A 和 B 的对应元素相加，结果存储在向量 C 中。
	2.	主函数 main：
	•	创建 CUDA 流 stream。
	•	分配主机和设备内存，并初始化输入向量。
	•	使用 cudaStreamBeginCapture 开始捕获流中的操作，创建 CUDA 图 graph。
	•	在捕获模式下，将数据从主机传输到设备，执行向量加法内核，并将结果从设备传回主机。
	•	使用 cudaStreamEndCapture 结束捕获，生成图对象 graph。
	•	实例化图，生成可执行图对象 graphExec。
	•	使用 cudaGraphLaunch 执行可执行图，并同步流以等待执行完成。
	•	验证计算结果的正确性。
	•	释放所有分配的资源，包括图对象、设备内存、主机内存和流。

注意事项：

	•	在捕获模式下，所有在流中提交的操作都会被记录到图中。
	•	实例化图会对其进行验证，并为后续的快速执行做好准备。
	•	执行图时，所有记录的操作将按照定义的依赖关系顺序执行。
	•	在实际应用中，通常会多次执行相同的图，以充分利用 CUDA 图的性能优势。

通过使用 CUDA 图，可以减少内核启动的开销，提高 GPU 的利用率，特别是在需要多次执行相同操作序列的场景中。