基于 ring buffer 和内存栅栏 __threadfence()：

核心数据结构（无锁 SPSC 环形队列）

``` cuda
// lock_free_queue.cuh

#pragma once
#include <cuda_runtime.h>

template <typename T, int CapacityPow2>
struct LockFreeQueueSPSC {
    // 要求容量是 2 的整数次幂，方便用位与取模
    static_assert((CapacityPow2 & (CapacityPow2 - 1)) == 0,
                  "Capacity must be power of 2.");

    // 环形缓冲区
    T buffer[CapacityPow2];

    // head：下一个要 pop 的位置索引
    // tail：下一个要 push 的位置索引
    // 约定：
    //   - 只有消费者线程写 head
    //   - 只有生产者线程写 tail
    //   - 双方都可以读对方的索引
    unsigned int head;
    unsigned int tail;

    __device__ void init() {
        head = 0;
        tail = 0;
    }

    // push：在队列尾部插入一个元素
    // 返回 true 表示成功，false 表示队列已满
    __device__ bool push(const T &value) {
        // 生产者线程

        // 本线程写 tail，无数据竞争
        unsigned int t = tail;
        // 读消费者的 head，需要保证不被编译器乱优化，cast 成 volatile 指针
        unsigned int h = *reinterpret_cast<volatile unsigned int *>(&head);

        // 如果 tail - head >= capacity 就满了
        if (t - h >= CapacityPow2) {
            return false; // full
        }

        // 写入数据到环形缓冲区
        buffer[t & (CapacityPow2 - 1)] = value;

        // 保证 buffer 写入在 tail 更新前对其他线程可见
        __threadfence();

        // 更新 tail，让消费者看到新元素
        *reinterpret_cast<volatile unsigned int *>(&tail) = t + 1;
        return true;
    }

    // pop：从队列头部取出一个元素
    // 返回 true 表示成功，false 表示队列为空
    __device__ bool pop(T &out) {
        // 消费者线程

        // 本线程写 head，无数据竞争
        unsigned int h = head;
        // 读生产者的 tail
        unsigned int t = *reinterpret_cast<volatile unsigned int *>(&tail);

        if (h == t) {
            return false; // empty
        }

        // 在读取数据前，确保我们看到的是生产者已经完成写入的数据
        __threadfence();

        out = buffer[h & (CapacityPow2 - 1)];

        // 再加一道 fence，确保 out 读完后再移动 head（比较保守的写法）
        __threadfence();

        *reinterpret_cast<volatile unsigned int *>(&head) = h + 1;
        return true;
    }

    // 当前队列元素数量（近似值，非强一致）
    __device__ unsigned int size() const {
        unsigned int h = *reinterpret_cast<const volatile unsigned int *>(&head);
        unsigned int t = *reinterpret_cast<const volatile unsigned int *>(&tail);
        return t - h;
    }

    __device__ bool empty() const {
        return size() == 0;
    }

    __device__ bool full() const {
        return size() >= CapacityPow2;
    }
};
```

example:

``` cuda
// main.cu

#include <cstdio>
#include "lock_free_queue.cuh"

constexpr int QUEUE_CAPACITY = 1024;  // 必须是 2 的幂
constexpr int N_ITEMS        = 1000;  // 需要传输的元素数量

__global__ void queue_kernel(int *output) {
    // 单个 block 里共享一个队列对象（放在共享内存里也行，这里简单放全局）
    __shared__ LockFreeQueueSPSC<int, QUEUE_CAPACITY> queue;

    if (threadIdx.x == 0) {
        // 生产者初始化队列
        queue.init();
    }

    __syncthreads(); // 确保队列初始化完毕

    if (threadIdx.x == 0) {
        // 生产者
        int produced = 0;
        while (produced < N_ITEMS) {
            if (queue.push(produced)) {
                produced++;
            }
            // 如果满了，会暂时 push 失败，就继续循环重试（自旋）
        }
    } else if (threadIdx.x == 1) {
        // 消费者
        int consumed = 0;
        while (consumed < N_ITEMS) {
            int v;
            if (queue.pop(v)) {
                output[consumed] = v;
                consumed++;
            }
            // 如果空了，就继续循环等待数据（自旋）
        }
    }
}

int main() {
    int *d_out = nullptr;
    int *h_out = new int[N_ITEMS];

    cudaMalloc(&d_out, N_ITEMS * sizeof(int));

    // 启动一个 block，至少两个线程：
    //   thread 0: producer
    //   thread 1: consumer
    queue_kernel<<<1, 2>>>(d_out);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, N_ITEMS * sizeof(int), cudaMemcpyDeviceToHost);

    // 简单检查：应该是 0..N_ITEMS-1
    bool ok = true;
    for (int i = 0; i < N_ITEMS; ++i) {
        if (h_out[i] != i) {
            printf("Mismatch at %d: got %d expected %d\n", i, h_out[i], i);
            ok = false;
            break;
        }
    }

    if (ok) {
        printf("Queue works! First 10 values: ");
        for (int i = 0; i < 10 && i < N_ITEMS; ++i) {
            printf("%d ", h_out[i]);
        }
        printf("\n");
    }

    cudaFree(d_out);
    delete[] h_out;
    return 0;
}
```
