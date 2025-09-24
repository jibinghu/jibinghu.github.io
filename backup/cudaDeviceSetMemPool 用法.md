cudaDeviceSetMemPool` 是 **NVIDIA 官方 CUDA Runtime API**（CUDA ≥ 11.2 引入），和 `cudaMallocAsync / cudaFreeAsync`、`cudaMemPoolCreate` 等属于同一套**异步内存分配与内存池**接口。头文件在 `cuda_runtime_api.h`，链接用常规的 `libcudart` 就行；你用的 CUDA 12.0 是支持的。

### 它是干嘛的？

* 每个 device 都有一个**当前（current）内存池**，`cudaMallocAsync` 会从这个池里拿内存、`cudaFreeAsync` 把内存归还到池里。
* `cudaDeviceSetMemPool(device, memPool)` 用来**把你创建的内存池设为该设备的当前池**。
* 如果不手动设置，也可以用**默认池**（可通过 `cudaDeviceGetDefaultMemPool` 获取并调属性）。

### 常见用法骨架

```cpp
// 1) 创建内存池（可选：也可以直接用默认池）
cudaMemPool_t pool;
cudaMemPoolProps props = {};
props.allocType = cudaMemAllocationTypePinned;   // 设备本地分配
props.handleTypes = cudaMemHandleTypeNone;
props.location.type = cudaMemLocationTypeDevice;
props.location.id   = 0; // device 0
cudaMemPoolCreate(&pool, &props);

// 可调一些属性，比如释放阈值（避免频繁还给驱动）
uint64_t threshold = 1ull << 30; // 1GB
cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold);

// 2) 设为当前设备的池
cudaDeviceSetMemPool(0, pool);

// 3) 在流中用异步分配/释放（从该池拿内存）
cudaStream_t s; cudaStreamCreate(&s);
void* d_ptr = nullptr;
size_t bytes = ...;
cudaMallocAsync(&d_ptr, bytes, s);
// ... kernel<<<..., s>>>(d_ptr) ...
cudaFreeAsync(d_ptr, s);
cudaStreamSynchronize(s);

// 4) 不用了可以销毁（确保没有在用）
cudaMemPoolDestroy(pool);
```

### 小贴士

* **驱动要求**：需要较新的驱动（一般 R460+）。CUDA 12.0 正常没问题。
* **混用规则**：可以和传统 `cudaMalloc/cudaFree` 同时存在，但**异步接口的释放/分配遵循流有序语义**，建议在同一条流上成对使用，避免额外同步。
* **默认池也能调优**：用 `cudaDeviceGetDefaultMemPool` 拿到默认池后，可用 `cudaMemPoolSetAttribute` 调 `ReleaseThreshold` 等，很多场景不用自建池也能吃到收益。
* **跨设备访问**：多 GPU 时如需共享，可用 `cudaMemPoolSetAccess` 配置访问权限。
* **为什么能加速**：它减少了频繁分配/释放带来的全局同步与碎片化开销，适合你现在 “`cudaMalloc` 占大头” 的场景。
