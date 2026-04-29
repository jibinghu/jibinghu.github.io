cudaDeviceSetMemPool` 是 **NVIDIA 官方 CUDA Runtime API**（CUDA ≥ 11.2 引入），和 `cudaMallocAsync / cudaFreeAsync`、`cudaMemPoolCreate` 等属于同一套**异步内存分配与内存池**接口。头文件在 `cuda_runtime_api.h`，链接用常规的 `libcudart` 就行；你用的 CUDA 12.0 是支持的。

### 它是干嘛的？

* 每个 device 都有一个**当前（current）内存池**，`cudaMallocAsync` 会从这个池里拿内存、`cudaFreeAsync` 把内存归还到池里。
* `cudaDeviceSetMemPool(device, memPool)` 用来**把你创建的内存池设为该设备的当前池**。
* 如果不手动设置，也可以用**默认池**（可通过 `cudaDeviceGetDefaultMemPool` 获取并调属性）。

默认内存池：

默认池是每个 device 都有的，由 CUDA Runtime 管理。\
cudaMallocAsync(ptr, size, stream) 会从**当前设备的“当前内存池（current mempool）”**拿内存；
如果你没有手动 cudaDeviceSetMemPool，那么“当前内存池”就是默认池。
若池里没有可复用块，运行时会向驱动申请新显存扩张池容量（受显存上限/碎片等限制）。
cudaFreeAsync(ptr, stream) 将内存归还到池里（而不是立刻归还给系统），便于后续复用，减少频繁分配/释放的开销与同步点。
何时把池里的闲置显存还给系统？由池属性控制，比如 cudaMemPoolAttrReleaseThreshold（阈值以下尽量保留、超过就尝试“修剪”归还）。你也可显式 cudaMemPoolTrimTo。

``` cpp
// 拿到当前设备的默认池
cudaMemPool_t defpool;
cudaDeviceGetDefaultMemPool(&defpool, /*device=*/0);

// 调高“保留阈值”，减少频繁还给系统导致的抖动
uint64_t threshold = 1ull << 30; // 1GB
cudaMemPoolSetAttribute(defpool, cudaMemPoolAttrReleaseThreshold, &threshold);

// 直接用 cudaMallocAsync / cudaFreeAsync（会走当前池；此处即默认池）
cudaStream_t s; cudaStreamCreate(&s);
void* p = nullptr;
cudaMallocAsync(&p, bytes, s);
// ... kernels on stream s ...
cudaFreeAsync(p, s);
cudaStreamSynchronize(s);
```

想显式从指定池分配，可用 cudaMallocFromPoolAsync；而 cudaMallocAsync 是“从当前池”分配的语义。
也可以自己 cudaMemPoolCreate 建池，然后 cudaDeviceSetMemPool(device, mypool) 把它设为“当前池”。
这些分配/释放都遵循**流有序（stream-ordered）**规则：在同一条流上不需要额外同步即可保证先分配后使用、先用后释放的正确性。
与传统 cudaMalloc/cudaFree 可并存，但为了减少隐式全局同步，建议同一逻辑链尽量用 async + pool 一套到底。

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


---


拿到 `cudaMemPool_t defpool` 之后，可以用 **cudaMemPoolGetAttribute / cudaMemPoolSetAttribute** 来查询或设置内存池属性（CUDA 11.2+ 的“按流顺序分配器”接口）。常用的有：

* `cudaMemPoolAttrReleaseThreshold`：释放阈值（`uint64_t`），达到阈值后空闲块会尽快还给驱动/系统。
* 运行时统计（只读）：`cudaMemPoolAttrReservedMemCurrent/High`、`cudaMemPoolAttrUsedMemCurrent/High`（当前/峰值已保留与已使用字节数）。
* 重用策略开关：`cudaMemPoolAttrReuseAllowOpportunistic`、`cudaMemPoolAttrReuseAllowInternalDependencies`、`cudaMemPoolAttrReuseFollowEventDependencies`（布尔值，控制空闲块如何被重用）。

> 官方文档要点：`cudaMallocAsync` 的分配**来自与该流所在设备关联的内存池**（默认是设备的默认池），可以通过 `cudaMemPoolGetAttribute` 获取池属性，`cudaDeviceGetAttribute(cudaDevAttrMemoryPoolsSupported)` 可查询设备是否支持该分配器。([[NVIDIA Docs](https://docs.nvidia.com/cuda/pdf/CUDA_Runtime_API.pdf)][1])

### 简单示例

```cpp
cudaMemPool_t defpool;
cudaDeviceGetDefaultMemPool(&defpool, /*device=*/0);  // 你现在的写法

// 1) 读释放阈值
uint64_t threshold = 0;
cudaMemPoolGetAttribute(defpool, cudaMemPoolAttrReleaseThreshold, &threshold);

// 2) 改释放阈值（比如 1GB）
uint64_t newThr = 1ull << 30;
cudaMemPoolSetAttribute(defpool, cudaMemPoolAttrReleaseThreshold, &newThr);

// 3) 读一些只读统计
size_t reserved_cur=0, reserved_high=0, used_cur=0, used_high=0;
cudaMemPoolGetAttribute(defpool, cudaMemPoolAttrReservedMemCurrent, &reserved_cur);
cudaMemPoolGetAttribute(defpool, cudaMemPoolAttrReservedMemHigh,    &reserved_high);
cudaMemPoolGetAttribute(defpool, cudaMemPoolAttrUsedMemCurrent,     &used_cur);
cudaMemPoolGetAttribute(defpool, cudaMemPoolAttrUsedMemHigh,        &used_high);

// 4) 查看重用策略（布尔）
int allowOpp=0, allowIntDep=0, followEvt=0;
cudaMemPoolGetAttribute(defpool, cudaMemPoolAttrReuseAllowOpportunistic,       &allowOpp);
cudaMemPoolGetAttribute(defpool, cudaMemPoolAttrReuseAllowInternalDependencies,&allowIntDep);
cudaMemPoolGetAttribute(defpool, cudaMemPoolAttrReuseFollowEventDependencies,  &followEvt);
```

小贴士：

* 读属性用 `cudaMemPoolGetAttribute(pool, attr, void* value)`；写属性用 `cudaMemPoolSetAttribute(...)`。
* 阈值太小会频繁把内存还回去，造成后续 `cudaMallocAsync` 又去向驱动要内存（变慢）；太大则可能导致占用增高。一般按峰值使用量留一点余量（比如峰值的 1–1.5 倍）较稳。
* 如果你需要不同设备/位置的默认池，新接口是 `cudaMemGetDefaultMemPool(cudaMemPool_t*, cudaMemLocation*, cudaMemAllocationType)`。([[NVIDIA Docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html)][2])


[1]: https://docs.nvidia.com/cuda/pdf/CUDA_Runtime_API.pdf "CUDA Runtime API"
[2]: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html "CUDA Runtime API :: CUDA Toolkit Documentation"