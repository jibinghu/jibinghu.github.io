参考：

https://vllm.hyper.ai/docs/design-v1/prefix_caching/

https://mp.weixin.qq.com/s/RC8QV0Alyw0y8sEXAOOM3Q

---

前缀缓存 KV 缓存块是 LLM 推理中一种流行的优化技术，用于避免冗余的提示计算。核心思想很简单——我们缓存已处理请求的 KV 缓存块，当新请求到来时如果前缀与之前请求相同就重用这些块。由于前缀缓存几乎是无成本的且不会改变模型输出，它已被许多公共端点（如 OpenAI、Anthropic 等）和大多数开源 LLM 推理框架（如 SGLang）广泛采用。

## 一、回顾前缀缓存 + vLLM 的核心设计（基于官方 + 社区资料）

为了让下面的深入讲解更易追踪，先列一下主要参考资料／出处：

* vLLM 官方 “Prefix Caching” 设计文档（你最初给的链接）
* 社区 “图解 vLLM Automatic Prefix Cache (RadixAttention)”（cloud.tencent.com） ([[腾讯云](https://cloud.tencent.com/developer/article/2424704?utm_source=chatgpt.com)][1])
* 极术社区 “图解 vLLM 源码解析系列：Prefix Caching” ([[极术社区](https://aijishu.com/a/1060000000474505?utm_source=chatgpt.com)][2])
* AI 计算社区 “KVCacheManager 与 PrefixCaching” ([[极术社区](https://aijishu.com/a/1060000000513098?utm_source=chatgpt.com)][3])
* 51CTO “Refix Caching 详解” ([[51CTO](https://www.51cto.com/article/818034.html?utm_source=chatgpt.com)][4])
* 其他技术博客（如 “LLM 推理优化 — Prefix Caching” 在知乎 / CSDN 上的讨论） ([[知乎专栏](https://zhuanlan.zhihu.com/p/13499550235?utm_source=chatgpt.com)][5])
* “\[Prefill 优化] 图解 vLLM Prefix Prefill Triton Kernel” 讨论 kernel 层面的优化 ([[jintiankansha.com](https://www.jintiankansha.com/t/syiY5rhBTt?utm_source=chatgpt.com)][6])

下面分几个层面（概念／设计／实现细节／调度与性能／局限与变种）系统讲。

---

## 二、概念与目标：为什么要做前缀缓存？

在 LLM 推理中，一个请求的 prompt + 上下文（或历史对话）部分，需要先经过 Prefill 阶段，计算每个 token 的 Key / Value 向量（KV cache），供后续 decode 使用。如果没有任何缓存策略，那么每个新请求都要从头算一遍，即使这个请求的前缀（prompt 部分）与之前某个请求高度重叠。

**前缀缓存**（Prefix Caching）的目标就是：在多个请求之间 **共享公共前缀** 的 KV 计算，使得对于那些已有缓存的前缀部分，可以直接重用而无需重新计算，从而提升效率，尤其是 **首 token 的响应延迟** 可以显著降低。

这个思路在对话、few-shot 提示、采样并行、多请求共享前缀等场景中最为有效。

---

## 三、vLLM 的前缀缓存设计：总体结构与机制

下面逐层展开 vLLM 是如何把前缀缓存融入其 KV 管理、块管理器 (BlockManager) 与调度流程的。

### 3.1 物理块 / 逻辑块 / 哈希结构

在 vLLM 中，“块”（block）是缓存与管理 KV 的基本单位。一个 block 通常对应若干个 token（例如 block\_size = 16 token，也可以是其他设定）。块管理器负责把序列 token 划分成块，并为每个块维护它的 Key/Value 缓存。

每个物理块（PhysicalTokenBlock）有以下属性：

* `block_hash`：哈希值，用来代表这个块的前缀语义身份
* `num_hashed_tokens`：表示该块 hash 是建立在多少个 token 已经确定之后
* `ref_count`：当前有多少个请求（sequence）在共享此块
* `computed`：指示这个块是否已经被计算
* `last_accessed`：最近一次被访问的时间戳（用于 LRU 驱逐）
* 设备 / 存放位置等其他元数据 ([[极术社区](https://aijishu.com/a/1060000000474505?utm_source=chatgpt.com)][2])

此外，还有 “逻辑块”（LogicalTokenBlock）作为对请求（sequence）对应 token 范围的抽象映射。逻辑块映射到物理块（可能多个 sequence 共享同一个物理块） ([[极术社区](https://aijishu.com/a/1060000000474505?utm_source=chatgpt.com)][2])。

块管理器（BlockManager / BlockSpaceManagerV1）负责整体的块分配、缓存、释放、驱逐等流程。 ([[极术社区](https://aijishu.com/a/1060000000474505?utm_source=chatgpt.com)][2])

哈希机制是前缀缓存能否命中的关键。vLLM 为每个块计算一个 `block_hash`，这个 hash 涵盖：

* 父块的 hash（也就是前缀链路）
* 本块内部的 token IDs
* 额外的输入模态／参数差异（例如 LoRA 权重、图像输入的哈希） ([[腾讯云](https://cloud.tencent.com/developer/article/2424704?utm_source=chatgpt.com)][1])

这样子，当另一个请求在同一位置拥有相同前缀时（即这部分 token 序列与缓存块一致），就能通过哈希查表命中这个块。

### 3.2 块分配、追加、释放、驱逐：核心流程

下面按时间或操作顺序剖析各个流程。

#### 3.2.1 块分配 (allocate / allocate\_slots)

当一个新请求进入（或者已有请求在 Prefill 阶段）时，需要把 prompt token 拆分成若干块并分配物理块。分配流程大致如下：

1. **检查缓存命中**
   对于该请求的前缀 token 部分，块管理器会计算哈希，查找已有缓存块映射表（`cache_block_hash_to_block`），看是否有命中的物理块。若命中，则将该物理块的 `ref_count` 增 1，并把它从空闲队列中移除（因为这个块正在被使用，不应被驱逐） ([[极术社区](https://aijishu.com/a/1060000000513098?utm_source=chatgpt.com)][3])。

2. **分配新的块**
   对于没命中的后续部分，需要从自由块队列 (free queue) 中弹出空闲块来分配给请求。弹出时原则是从空闲队列头部取（即优先利用最“旧”的空闲块作为驱逐候选） ([[极术社区](https://aijishu.com/a/1060000000513098?utm_source=chatgpt.com)][3])。

3. **填满块 / 哈希写入映射表**
   当一个块被填满（即其槽位中对应 token 全部被计算）后，就可以给这个块计算最终哈希，并把它插入缓存映射表（`cache_block_hash_to_block`），使其他请求后续可以复用 ([[极术社区](https://aijishu.com/a/1060000000474505?utm_source=chatgpt.com)][2])。

vLLM v1 引入了 “append-only” 的策略：对于 decode 阶段追加 token 所产生的新块，虽然也会检查是否命中缓存，但在实际调度中 **不会重新替换已分配给这个请求的块**。即便命中，也不会改变请求当前的块结构。这样设计可以简化块管理、减少频繁重分配开销。 ([[极术社区](https://aijishu.com/a/1060000000513098?utm_source=chatgpt.com)][3])

社区也有指出：在 vLLM v0 版本中，如果检测命中，会释放当前块再复用已有；而 v1 改为 append-only，不做这样的替换。 ([[极术社区](https://aijishu.com/a/1060000000513098?utm_source=chatgpt.com)][3])

#### 3.2.2 追加 (append\_slots)

在 decode 阶段，每生成一个 token，就相当于是给这个请求追加一个新 token，需要为这个 token 分配新的 slot（如果当前块已满则需要新块）。在此过程中：

* 依然先做哈希检查，能命中就 reuse
* 若不能命中或块已满，就申请新的物理块
* 但正如上面所说，由于 append-only 策略，即便新块哈希命中，也不会替换旧块（已有块） ([[极术社区](https://aijishu.com/a/1060000000513098?utm_source=chatgpt.com)][3])

这种设计简化了块的一致性管理，但也牺牲了在 decode 阶段更高命中率的可能性。

#### 3.2.3 释放 (free) 和 引用计数 (ref\_count)

每当一个请求结束或者不再使用其关联块时：

* 对应物理块的 `ref_count` 减 1
* 如果 `ref_count` 变 0，则这个块可以被释放（加入自由块队列 tail） ([[极术社区](https://aijishu.com/a/1060000000513098?utm_source=chatgpt.com)][3])
* 释放入队时按照逆序（倒序地释放）加入空闲队列尾部，这样做是为了在驱逐时倾向保留那些最近可能还被复用的块，提高未来命中率 ([[极术社区](https://aijishu.com/a/1060000000513098?utm_source=chatgpt.com)][3])

#### 3.2.4 驱逐 (Evict / LRU)

在空闲块队列头部如果遇到一个 **已缓存块**（即存在于缓存映射表中的块），必须把它驱逐出缓存（从映射表中删除），以腾出块空间让新的块加入。

驱逐流程：

1. 弹出空闲队列头部块
2. 从缓存映射表中移除该块的 hash 映射
3. 将其视为普通空闲块（computed 标识清除、hash 值置空等） ([[极术社区](https://aijishu.com/a/1060000000474505?utm_source=chatgpt.com)][2])

驱逐策略本质是 LRU（最近最少使用）风格：空闲队列头部的块，通常是最久未使用的，所以优先被驱逐。

---

### 3.3 调度、Chunked Prefill & Kernel 优化

为了配合前缀缓存，vLLM 在调度与 kernel 优化层面也做了不少设计。

#### 3.3.1 Chunked Prefill（分块 Prefill）

在传统设计中，Prefill 阶段可能是一次性对整个 prompt 一次性做前缀计算（所有 token 一起做）。但在 vLLM（尤其 V1）中，为了更早地利用 prefix caching，也为了使调度更灵活，它支持 **将 prompt 拆成若干个 chunk**，分几次做 prefill。这样，即使 prompt 很长，也能逐块尝试命中前缀缓存。 ([[博客园](https://www.cnblogs.com/zackstang/p/19036108?utm_source=chatgpt.com)][7])

此外，这种分块 prefill 有利于混合调度（Prefill 与 Decode 并行）和资源调度灵活性。 ([[CSDN](https://blog.csdn.net/weixin_58753619/article/details/141611100?utm_source=chatgpt.com)][8])

#### 3.3.2 Prefix Prefill Kernel（Triton / CUDA 级别优化）

在 kernel 层面，也有专门针对 prefix 缓存命中的优化机制。文章 “\[Prefill 优化] 图解 vLLM Prefix Prefill Triton Kernel” 曾讲解其细节。核心点包括：

* 在 kernel 里做判断：有多少 token 在当前请求中 **命中了前缀缓存**，这些 token 对应的 KV 已经存在，不需要再做 attention／计算
* 对未命中部分才发起新的计算
* 使用 Tiling / 分块策略，使得 kernel 在不同 head size、不同 GPU 架构（支持 MQA/GQA）下都能正常跑
* 合并内存访问、减少冗余读取、管线化设计等 ([[jintiankansha.com](https://www.jintiankansha.com/t/syiY5rhBTt?utm_source=chatgpt.com)][6])

这样就能在最底层就绕过已有缓存部分的计算，减少 GPU 负荷。

#### 3.3.3 调度与 Continuous Batching（连续批处理）

前缀缓存的效果要和调度策略配合好，才能在整体系统层面体现性能提升。vLLM 的设计中：

* 它采用 token 级别调度（continuous batching）：即多个请求在每个时刻按当前 token 阶段并行处理，而不是一条请求处理完再处理下一条。这样可以充分利用 GPU 并行能力。 ([[博客园](https://www.cnblogs.com/zackstang/p/19036108?utm_source=chatgpt.com)][7])
* 调度器优先做 prefill，保证解码阶段有足够批量。 ([[CSDN](https://blog.csdn.net/weixin_58753619/article/details/141611100?utm_source=chatgpt.com)][8])
* 在 V1 中，不再明确区分 prefill / decode 阶段，而以 `{request_id: num_tokens}` 这种方式统一调度，使得前缀缓存、chunked prefill、decode 混合成为可能。 ([[博客园](https://www.cnblogs.com/zackstang/p/19036108?utm_source=chatgpt.com)][7])

整体来说，这种设计让前缀缓存不只是一个孤立的模块，而是被调度、内核优化、资源分配等在多个层面共同支撑。

---

## 四、深入剖析：一些关键细节与边界情况

下面我聚焦几个社区 / 源码分析里特别提到或容易被忽略的细节，帮你“挖得更深”。

### 4.1 append-only 策略的利弊

如前所述，vLLM v1 在 decode 阶段采用 append-only 策略：即便后续新块与已有缓存命中，也不会用缓存块替换已经分配给请求的块。这样做有以下优点：

* 简化块一致性管理、避免频繁重新绑定块
* 减少内存 / 结构调整的开销
* 避免在 decode 阶段反复做复杂判断、切换

但缺点也显著：

* 降低 decode 阶段缓存命中率：很多潜在可以命中的块因为早期分配的决定被“锁定”
* 对那些在 decode 阶段有很多重复前缀（例如输出生成有规律性）情形，无法充分利用缓存
* 增加了一定的缓存冗余（多个物理块或块引用可能语义相同） ([[极术社区](https://aijishu.com/a/1060000000513098?utm_source=chatgpt.com)][3])

社区文章就指出：在 vLLM v0 中，是允许替换的；而 v1 则为了工程简化，选择不做这种替换。 ([[极术社区](https://aijishu.com/a/1060000000513098?utm_source=chatgpt.com)][3])

### 4.2 最后一个块未满 / 部分填块的哈希问题

当一个请求处于 decode 阶段，最后的那个块可能并未填满（即没有完全被用满槽位）。在这种情况下，该块尚未确定最终的 token 哈希，因此**不宜立即将它插入缓存映射表**，因为未来 token 可能改变其内容和哈希。

vLLM 的做法是，用一个“临时 / 假的哈希”号标记它，只有在填满之后，才真正计算哈希并写入映射表。这样可以避免把不完整 / 临时状态的块被其他请求误用。 ([[极术社区](https://aijishu.com/a/1060000000474505?utm_source=chatgpt.com)][2])

这个机制说明：并非所有块都是随时可缓存，有些块在“确定身份”之前必须等待完整性。

### 4.3 释放入队的逆序策略与命中倾斜

在释放块（ref\_count = 0）时，vLLM 不是简单地把它加入空闲队列头部，而是按 **逆序（从后往前）** 加入队列尾部。这个设计背后的直觉是：越是最近释放的块，更可能被下一个紧接来的请求复用，因而让它在队列末尾（后面才被驱逐）能保留更久，从而提高命中率。 ([[极术社区](https://aijishu.com/a/1060000000513098?utm_source=chatgpt.com)][3])

这个细节虽看起来微妙，但对实际命中率有很大影响 —— 它是一种 “延后驱逐” 的启发式策略。

### 4.4 缓存冲突、并发与哈希安全

如任何哈希缓存设计，前缀缓存也面临冲突与一致性风险：

* **哈希冲突**：不同前缀被意外映射到同一个哈希值，从而错误地复用 KV。这在多租户 / 多模型 / 权重差异情况下尤为危险。vLLM 文档建议在这些场景下用更强哈希（如 SHA256）来降低冲突概率。 ([[腾讯云](https://cloud.tencent.com/developer/article/2424704?utm_source=chatgpt.com)][1])
* **并发 / 竞争**：多个请求并行做块分配、释放、命中检查时，需要保证数据结构的线程安全（映射表、空闲队列、ref\_count 更新等）。这些细节在源码中通常借锁 / 原子操作 / 并发安全结构来保障。社区一般没细节披露，但这是工程级挑战之一。
* **参数／模型不兼容**：即便 prompt 前缀相同，如果两次请求在模型权重、LoRA 插件、模态输入（如图像、提示器）等上有差别，那么即使前缀相同，也不能共享缓存。这就要求哈希 key 必须包含这些“变更维度”以保证安全。vLLM 的设计就是把这些差异纳入哈希计算中。 ([[腾讯云](https://cloud.tencent.com/developer/article/2424704?utm_source=chatgpt.com)][1])

---

## 五、前缀缓存的实际收益与性能表现

前缀缓存若设计得当，其性能收益可以很显著。下面是社区 / 博客中常见的性能对比和经验观察：

* 在对话场景中，后续轮次 prompt 常包含很多历史上下文，前缀缓存能显著减少 Prefill 阶段耗时。51CTO 文章就举例：第一次请求要完整构建 KV，第二次若共享长前缀，vLLM 可自动复用缓存，从而大幅减少重复计算 ([[51CTO](https://www.51cto.com/article/818034.html?utm_source=chatgpt.com)][4])
* 在一些 benchmark 中，Cached Attention（即前缀缓存 + 跳过重复计算）相比完全重新计算（Recomputed）在首 token 延迟和总体吞吐上都有明显优势。技术博客列出图表显示耗时下降。 ([[CSDN](https://blog.csdn.net/weixin_58753619/article/details/141611100?utm_source=chatgpt.com)][8])
* 在极术社区的源码解析文中提到，前缀缓存是 vLLM 作为一个推理引擎的“核心卖点”之一，因为它能在实际对话服务中把重复前缀计算的成本降下来。 ([[极术社区](https://aijishu.com/a/1060000000474505?utm_source=chatgpt.com)][2])
* 但是，在 decode 阶段缓存命中较少、append-only 限制的情况下，它的实际提升可能受限。也就是说，性能提升高度依赖于 **前缀重用率**。

总的来说，若系统中 prompt 重用率高、对话历史较长、多个请求共享前缀，那么前缀缓存能带来极大的加速；相反如果每个请求都截然不同，那么前缀缓存的作用就较弱。

---

append_only：

下面用一个简单的例子演示 append-only 策略在 decode 阶段的差异。
假设设置块大小 block_size = 4 token。
请求 1：Prompt = A, B, C, D, E, F， decode 输出 = G, H, I
那么整个 token 序列（prompt + output）是 A, B, C, D | E, F, G, H | I 被分成块（逻辑块）：
逻辑块 0: A, B, C, D、
逻辑块 1: E, F, G, H、
逻辑块 2: I（尚未满块）、
假设逻辑块 0 和 1 完整满块后被缓存（哈希插入缓存表）。
请求 2：用相同 prompt A, B, C, D, E, F， decode 输出也得出 G, H, I（例如用相同策略、无随机性）。
在 decode 阶段：
当处理到逻辑块 0、1 的那部分，vLLM 会做命中检查，发现可以复用已有缓存块 0 和块 1，于是为这些逻辑块直接映射到已有的物理块。
当进入逻辑块 2（token I）这个阶段，假设有一个已有的缓存块（哈希匹配）可供复用。如果是 v0 设计，可能会把这个逻辑块重新映射、释放已分配的新块、切换为缓存块；但在 v1 的设计里，即便命中，也 不会替换。它仍然让逻辑块 2 使用最初给它分配的新块。
因此，在 v1 的行为下，即使逻辑块 2 后来发现一个更优缓存块，也不会动态切换，那条请求的块结构就从头到尾固定了（append-only）。
这个例子也在 vLLM 文档里被提及，作为“重复块 / 冗余块”情景的示例：文档讲到请求 2 分析时可能出现一个与已有块重复的块（哈希一致）——在 v0 会替换、释放重复块，而在 v1 则因为 append-only 不做替换，形成冗余缓存。 


[1]: https://cloud.tencent.com/developer/article/2424704?utm_source=chatgpt.com "原理&图解vLLM Automatic Prefix Cache (RadixAttention)首 ..."
[2]: https://aijishu.com/a/1060000000474505?utm_source=chatgpt.com "图解大模型计算加速系列：vLLM源码解析3，Prefix Caching"
[3]: https://aijishu.com/a/1060000000513098?utm_source=chatgpt.com "图解Vllm V1系列6：KVCacheManager与PrefixCaching"
[4]: https://www.51cto.com/article/818034.html?utm_source=chatgpt.com "Refix Caching 详解：实现 KV Cache 的跨请求高效复用"
[5]: https://zhuanlan.zhihu.com/p/13499550235?utm_source=chatgpt.com "LLM推理优化 - Prefix Caching - 知乎"
[6]: https://www.jintiankansha.com/t/syiY5rhBTt?utm_source=chatgpt.com "[Prefill优化]图解vLLM Prefix Prefill Triton Kernel"
[7]: https://www.cnblogs.com/zackstang/p/19036108?utm_source=chatgpt.com "vLLM框架：LLM推理的高效机制 - ZacksTang - 博客园"
[8]: https://blog.csdn.net/weixin_58753619/article/details/141611100?utm_source=chatgpt.com "LLM推理加速3：推理优化总结Mooncake/AttentionStore ..."