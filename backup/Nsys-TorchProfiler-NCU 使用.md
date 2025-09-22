主要参考：https://zhuanlan.zhihu.com/p/1945304372545291742

---

Blog 中有很多基础的知识，这里不做过多赘述。

要理解分布式工作负载中的 GPU 性能，需要考察模型算子与 GPU 设备的交互方式。 从宏观层面，可以将 GPU 操作分为三大类：

- 计算类 (COMP)
执行矩阵乘法等[数值计算](https://zhida.zhihu.com/search?content_id=262463237&content_type=Article&match_order=1&q=%E6%95%B0%E5%80%BC%E8%AE%A1%E7%AE%97&zhida_source=entity)
负责模型的所有数值运算处理
- 通信类 (COMM)
负责 GPU 设备间的[数据交换](https://zhida.zhihu.com/search?content_id=262463237&content_type=Article&match_order=1&q=%E6%95%B0%E6%8D%AE%E4%BA%A4%E6%8D%A2&zhida_source=entity)与同步
通常使用 NCCL 库（内核前缀为 “nccl”，如 NCCL_AllGather、NCCL_ReduceScatter、NCCL_AllReduce）
- 内存类 (MEM)
管理 GPU [内存分配](https://zhida.zhihu.com/search?content_id=262463237&content_type=Article&match_order=1&q=%E5%86%85%E5%AD%98%E5%88%86%E9%85%8D&zhida_source=entity)与释放
处理主机与设备间的数据传输：
Memcpy_H2D（主机到设备）
Memcpy_D2H（设备到主机）
Memcpy_D2D（设备到设备）
Memset…
现代 GPU（如 NVIDIA A100）支持多内核并发执行，可通过内核重叠技术缩短执行时间。常用实现方式是多 CUDA 流——不同 CUDA 流可以交错或并发运行，实现计算、通信和内存操作的重叠。

---

