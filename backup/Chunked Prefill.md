Chunked Prefill ：简单来说，它的实现是将一个很长的 Prompt 拆分成多个小的 "Chunk"（块），分多次迭代（Iteration）进入 GPU 进行计算，而不是一次性算完。

> 大模型推理中有两个阶段：Prefill（预填充/首词生成） 和 Decode（解码/后续生成）。Chunked Prefill（分块预填充），在学术界也常被称为 Sarathi-serve 或 Split-K Prefill，其核心目的是为了解决 Head-of-Line (HoL) Blocking（队头阻塞） 问题，并最大化 GPU 的计算/访存利用率。

以下是 Chunked Prefill 的底层实现逻辑，主要包含 调度策略 和 算子实现 两个层面：

1. 核心动机：为什么要拆？
在传统的 Continuous Batching（连续批处理）中：
- Prefill 阶段是 Compute-bound（计算密集型），因为是矩阵乘法（GEMM）。
- Decode 阶段是 Memory-bound（访存密集型），因为是矩阵向量乘法（GEMV）。

如果一个 Batch 里来了一个 32k 长度的 Prompt，传统的做法是一口气算完这 32k 的 Prefill。这会导致：
- 阻塞 Decode：正在进行 Decode 的短请求必须等待这个 32k Prefill 完成才能生成下一个 token，导致延迟骤增。
- 显存压力：一次性计算大 Prompt 需要巨大的中间激活显存（Activation Memory）。

2. 调度层实现 (Scheduler)：

调度器是实现 Chunked Prefill 的大脑。它的逻辑不再是“基于请求调度”，而是“基于 Token Budget（预算）调度”。

实现步骤：

- 设定 Budget：系统设定一个 max_num_batched_tokens（例如 512 或 2048）。这是 GPU 显存和计算能力能承受的最佳吞吐量。
- 混合队列（Mixed Batch）：优先放入所有的 Decode 请求（因为它们对延迟敏感，且计算量小）。计算 Decode 占用的 Token 数： $N_{decode}$ 。剩余预算： $R = Budget - N_{decode}$ 。
- 截断 Prefill：从等待队列中取出一个 Prefill 请求。如果该请求的 Prompt 长度  $L > R$ ，则只取前  $R$  个 token 进行计算。剩下的  $L - R$  个 token 留给下一个 Iteration 处理。状态保存：调度器必须标记该请求为 "Preempted" 或 "Chunking" 状态，确保下一轮调度时能接上。

伪代码：

``` python
def schedule(waiting_queue, running_queue, budget=2048):
    batch = Batch()
    current_tokens = 0
    
    # 1. 先满足 Decode 请求 (高优先级，低开销)
    for req in running_queue:
        if current_tokens < budget:
            batch.add_decode(req) # 贡献 1 个 token 计算量
            current_tokens += 1
            
    # 2. 用剩余预算处理 Prefill (填满 GPU)
    remaining_budget = budget - current_tokens
    
    if remaining_budget > 0 and not waiting_queue.empty():
        req = waiting_queue.peek()
        # 核心逻辑：只取一部分 Prompt
        num_tokens_to_process = min(req.remaining_prompt_len, remaining_budget)
        
        batch.add_prefill_chunk(req, start_idx=req.processed_len, 
                                length=num_tokens_to_process)
        
        req.processed_len += num_tokens_to_process
        
    return batch
```

3\. 算子与显存层实现 (Kernel & Memory)

这是最硬核的部分。将 Prompt 拆开后，Attention 的计算方式必须改变。

假设一个 Prompt 是 `[A, B, C, D]`，我们拆成两块：`Chunk1=[A, B]` 和 `Chunk2=[C, D]`。

#### 第一步：处理 Chunk 1 ([A, B])

  * **计算**：标准的 Self-Attention。
  * **KV Cache**：计算出的 $K_A, V_A, K_B, V_B$ 被写入 KV Cache 管理器（如 PagedAttention 的 Block 表中）。
  * **输出**：对于 Prefill 阶段，通常我们只关心最后一个 Token 的输出（用于预测下一个），但在 Chunked 模式下，如果这不是最后一块，我们甚至不需要输出 logits，只需要更新 KV Cache。

#### 第二步：处理 Chunk 2 ([C, D])

这是关键点。在计算 `[C, D]` 的 Attention 时，它们不仅要看自己（C看C，D看C和D），**还要看之前已经算好的 `[A, B]`**。

  * **Q (Query)**: 只有 `[C, D]` 的 Query 向量。
  * **K, V (Key/Value)**:
      * 当前计算出的 `[C, D]` 的 K, V。
      * **加上** 之前存在显存里的 `[A, B]` 的 K, V。
  * **FlashAttention 调用**：
    现在的 FlashAttention 接口（如 `flash_attn_varlen_func`）支持传入 `kv_cache` 指针。
      * `q`:  Shape `[2, Head, Dim]` (对应 C, D)
      * `k_cache`: 指向包含 A, B 的物理显存块。
      * `k_new`: Shape `[2, Head, Dim]` (对应 C, D)

**数学表达：**
对于 Chunk 2 的 Attention 输出：
$$\text{Attn}(Q_{chunk2}, K_{total}, V_{total}) = \text{Softmax}(\frac{Q_{chunk2} \cdot [K_{chunk1}; K_{chunk2}]^T}{\sqrt{d}}) \cdot [V_{chunk1}; V_{chunk2}]$$

这意味着底层的 Attention Kernel 必须支持\*\*“读取历史 KV Cache + 写入当前 KV Cache + 计算当前 Query”\*\* 的混合模式。vLLM 和 FlashInfer 都在算子层面做了这种适配。

4\. 优缺点分析 (Trade-off)

  * **优点**：
      * **极佳的 Latency 稳定性**：Decode 请求不会被长 Prefill 卡住，TTFT (Time To First Token) 变差了（对长 Prompt 而言），但 P99 Decode Latency 显著降低。
      * **计算与访存重叠 (Overlap)**：将 Compute-bound 的 Prefill 切碎，塞进 Memory-bound 的 Decode 缝隙里，充分利用了 GPU 的 Tensor Core 和显存带宽。
  * **缺点**：
      * **重复读取 KV**：处理 Chunk 2 时，必须重新从显存加载 Chunk 1 的 KV Cache。相比于一次性全算完（KV 都在片上 SRAM/L2 Cache 里），这增加了一些全局显存带宽开销。
      * **复杂性**：状态管理变得极其复杂（如果在 Chunk 中间系统崩溃了怎么办？调度器逻辑更难写）。

Q&A：

1. 但是对于每一个 sequence 来说，prefill 一定要在 decode 的前面的吧？(依赖性)

结论：是的，这是因果律（Causal Dependency），不可打破。
对于同一个 Request (Sequence) 而言，你必须先消化完 Prompt（Prefill），生成并存好之前所有 token 的 KV Cache，才能根据最后一个 token 的 hidden state 去预测生成第一个新 token（Decode 开始）。
- Chunked Prefill 不会改变这个顺序：它只是把“一口气吃完 Prompt”变成了“细嚼慢咽”。
- 但是，在整个 Batch 的层面上，Prefill 和 Decode 是可以**乱序混合（Interleaved）**的。即：Request A 的 Decode 步骤可以和 Request B 的 Prefill Chunk 步骤在同一个 Batch 中一起执行。

2. 另外 chunked prefill 我觉得是引入了额外的显存消耗的，就是为了TTFT 和 TTOF 的折中。因为如果优先新来的 prefill 的话，decode 就会很卡顿。但如果先来先服务的话，对于长 seq耗费时间很长，那么后来的 TTFT 就会很长。

关于“显存消耗”这一点，我们需要区分 容量 (Capacity) 和 带宽 (Bandwidth)。
- 显存容量 (Memory Capacity)：Chunked Prefill 实际上通常会降低峰值显存占用。传统 Full Prefill：处理长度 32k 的 Prompt，中间的 Activation（如 $QK^T$ 矩阵）非常大，容易 OOM。Chunked Prefill：每次只算 2k token，中间 Activation 很小。Trade-off：你说的“额外消耗”可能指调度器的元数据维护，或者 KV Cache 的碎片化，但这通常很小。
- 显存带宽 (Memory Bandwidth)：你是对的，这里有额外消耗。正如我们之前讨论的，计算 Chunk 2 时，需要重新从 HBM 读取 Chunk 1 的 KV Cache。这比数据都在 SRAM/L2 里的 Full Prefill 慢，浪费了带宽。

TTFT (Time To First Token) 和 TBT (Time Between Tokens, 也叫 Inter-Token Latency) 的博弈完全正确：

- FCFS (先来先服务, 无 Chunking)：

     - Decode 请求：被迫等待前面那个 32k Prompt 跑完。TBT 极差（卡顿感明显）。(prefill 插队)
     - 长 Prompt 请求：一口气跑完。TTFT 最优。

- Chunked Prefill：

     - Decode 请求：插队执行。TBT 极佳（丝般顺滑）。 (decode 插队)
     - 长 Prompt 请求：被迫分多次跑，还要被 Decode 插队。TTFT 变差。

结论：对于 Chat/Serving 场景，用户对“生成过程中卡顿”（TBT）的容忍度远低于“开始生成前多等一秒”（TTFT）。所以 Chunked Prefill 是用 TTFT 的劣化 换取 TBT 的稳定。

3. 现在 chunked prefill 将 prefill 的 Prompt 切分开，类似 decode 的 gemv，形成小的 gemm。首先将优先级给到 decode，也就是之前没算完的 decode；但是同时类似 vllm 的 pageattention，如果有空闲的显存 block，就留给 prefill。当然这也引入了一些 prefill 处理的策略。比如 prefill 的多个 chunk，前几个 chunk 块只需要记录 kv cache 即可，不需要计算得出最终的 probabilities 向量。而只需要保存最后一个 chunk 的 probabilities 向量即可。而 decode 因为是 gemv 的原因其实影响不大。

在 Prefill 阶段（无论是 Full 还是 Chunked）：假设 Prompt 是 [A, B, C, D]。我们计算 Attention 实际上是为了更新 A, B, C, D 的 KV Cache。Logits (Probability) 是为了预测 Next Token。

优化逻辑：

1. 中间 Chunks (如处理 [A, B] )：我们只需要它们的 $K, V$ 存入显存。我们完全不需要计算它们的 Logits，甚至不需要把 Hidden States 投影回词表维度（Unembedding Layer，通常是 Hidden_Size * Vocab_Size 的大矩阵乘，非常慢）。操作：Kernel 执行完 Attention 和 FFN 后，写入 KV Cache 直接返回，跳过最后的 Linear 层和 Softmax。
2. 最后一个 Chunk (如处理 [C, D] )：C：依然只需要 KV，不需要 Logits（因为我们已知下一个是 D）。D：只有这最后一个 Token 需要计算 Logits，用来采样生成第一个回复 E。为什么这在 Chunked Prefill 中很重要？因为 Decode 是高优先级的。如果一个 Prefill 被切成 10 份，前 9 份都省去了 Unembedding 和 Softmax 的计算，这节省了大量的 Compute 资源，让 GPU 能更快地释放出来去响应高优先级的 Decode 请求。

4. 在没有应用 chunked prefill 的时候，传统的 schedule 是采取 prefill 插队，而不是 prefill + decode 排队是吗？

传统 Schedule 的策略：是 "Prefill 插队" 还是 "排队"？
在没有 Chunked Prefill 的传统 Continuous Batching（如早期的 vLLM 或 TGI）中，策略通常是 First-Come-First-Serve (FCFS)，但表现出来的现象是 "Head-of-Line Blocking" (队头阻塞)。
并不是 Prefill 真的“插队”了： 通常调度器会维护一个 Waiting Queue（新来的）和 Running Queue（正在 Decode 的）。 当一轮 Decode 结束，显存有空余时，调度器会从 Waiting Queue 取出一个 Prefill 请求加入 Batch。
现象是“一粒老鼠屎坏了一锅粥”： 一旦这个 Prefill 请求进入了 GPU，因为它不可切分（假设是 32k 长度），GPU 就必须一口气算完这 32k tokens。
Decode 视角的感受：原本我也在车上（Running Queue），结果新上了一个胖子（Prefill），车门关死，开了 500ms 才到下一站。
结论：传统的调度策略通常是 "混合 Batch，但 Prefill 一旦上车就独占时间片"。如果显存够大，Prefill 和 Decode 是一起跑的；如果显存不够，调度器甚至会暂停 Decode（Preempt），腾出显存先让 Prefill 跑完（这种情况才是你说的 Prefill 插队/抢占）。
对比 Chunked Prefill： Chunked Prefill 是允许这个胖子“先伸进一只手”，算 50ms，然后大家一起处理 Decode，下一轮胖子再“伸进一只脚”。

5. 显卡对于大规模 GEMM 更效率更高？那是不是切分 chunk 也不能切分过小。

Chunk 不能切得太小。这涉及到 GPU 的 Roofline Model 和 Arithmetic Intensity (算术强度)。为什么 GEMM 效率高？矩阵乘法（GEMM）的计算量是 $O(N^3)$ ，访存量是 $O(N^2)$ 。随着 $N$ （在这里对应 Chunk Size）增大，计算/访存比变大，更能跑满 Tensor Core 的算力，这叫 Compute-bound。切太小会怎样？如果 Chunk Size 切到 1（极端情况），它就变成了 GEMV（矩阵向量乘），计算/访存比极低，受限于 HBM 带宽，GPU 利用率（SM Occupancy）会掉到 1% - 5%，这就叫 Memory-bound。这不仅没有加速，反而因为 Kernel Launch 的开销（Overhead）导致整体吞吐下降。最佳实践值：在业界实践（如 vLLM, DeepSeek, SGLang）中，Chunk Size 通常有一个 Min Threshold，一般是 128, 256 或 512。小于 128：效率急剧下降，得不偿失。512 - 2048：通常是 Sweet Spot，既能保持高 GEMM 效率，又不会让 Decode 等太久。

6. 但其实对于中间 chunk 和最后一个 chunk 的处理也是不得不做的，因为长 Prompt (未切分)的时候本来也就不需要映射到词表以及大矩阵的 Softmax 操作。

你的理解很透彻：这是本来就有的特性，但 Chunked Prefill 让它变得更"显式"了。

你是对的：即使在未切分的长 Prompt Prefill 中，我们本来也只需要计算最后一个 token 的 Logits。

流程对比：

Unchunked (Full) Prefill: 输入 [A, B, C, D] -> Transformer Layers -> 拿到 Hidden States [Ha, Hb, Hc, Hd] -> 只取 Hd -> lm_head -> Softmax -> Output. (中间的 Ha, Hb, Hc 在过了最后一层 Transformer 后就被丢弃了，根本不进 lm_head)

Chunked Prefill:

Iter 1 输入 [A, B] -> Transformer -> 存 KV Cache -> 直接 Return (完全不碰 lm_head)。

Iter 2 输入 [C, D] -> Transformer (读取 AB KV) -> 存 KV -> 只取 Hd -> lm_head -> Output.

这里的区别在于： 在 Chunked Prefill 的代码实现中，我们需要在 Kernel 层面 显式地告诉系统：“如果是中间的 Chunk，请在最后一层 Transformer 结束后直接退出，不要分配 Logits 的显存，也不要走后续流程”。

虽然从数学上讲和 Unchunked 是一样的，但在工程实现上，Chunked Prefill 需要更精细的状态机管理，确保中间 Chunk 不会触发任何多余的 Memory Allocation（比如误分配了一个 [Batch, Seq_Len, Vocab_Size] 的 logits tensor，那显存直接爆炸）。

7. 提到的第一点和 Paged Attention 的原理很像，只不过 Paged Attention 是在硬件层面实现，而 Chunked Prefill 是在算法层面实现。

关于 PagedAttention 和 Chunked Prefill 的关系

有一点需要纠正：**PagedAttention 并不是在硬件层面实现的，它依然是纯软件（Software/Kernel）层面的实现。**

理清 **PagedAttention**、**Chunked Prefill** 和 **Prefix Caching** 这三者的关系，它们其实是不同层面的东西：

* **PagedAttention (显存管理机制)**：
    * **核心原理**：模仿操作系统的 **Virtual Memory（虚拟内存）** 分页机制。它在软件层维护了一张“页表”，允许 KV Cache 在显存（HBM）中是不连续存储的。
    * **是不是硬件？** 不是。它是一套写在 CUDA Kernel 里的逻辑和 Host 端的管理代码。GPU 硬件本身（MMU）支持通用虚拟内存，但 PagedAttention 是在应用层专门为 Attention 算子写的一套逻辑。
    * **作用**：解决了显存碎片化问题。

* **Prefix Caching / Radix Attention (复用策略)**：
    * **核心原理**：基于 PagedAttention，给每个物理块（Block）打上指纹（Hash）。如果新来的 Prompt 前缀和之前的一样，直接引用显存里已有的块，不用重算。
    * **你的直觉是对的**：它确实和 PagedAttention 是一体的，有了 PagedAttention 的非连续存储能力，才有了高效复用的可能。

* **Chunked Prefill (调度执行策略)**：
    * **核心原理**：把计算任务切碎。
    * **与 PagedAttention 的关系**：Chunked Prefill **高度依赖** PagedAttention。
    * **为什么？** 想象一下，你处理 Chunk 1 产生了一些 KV，存到了显存的某些块里。然后 GPU 转头去处理别人的 Decode 任务了。等你回来处理 Chunk 2 时，你需要申请新的显存块。如果没有 PagedAttention，你可能要求显存必须连续，那这就很难插空；有了 PagedAttention，Chunk 2 的 KV 可以随便找个角落存，只要在页表里链上 Chunk 1 即可。

**总结修正**：
PagedAttention 是**地基**（解决怎么存），Chunked Prefill 是**施工队**（解决怎么算）。它们都是**软件/算法**层面的优化，运行在通用的 GPU 硬件上。

8. 什么是 RDMA？

为什么要 RDMA？（对比传统 TCP/IP）

假设要把 Prefill 节点 A 上的 KV Cache（比如 1GB 数据）传给 Decode 节点 B。

**传统 TCP/IP 方式（慢，累）：**
1.  **节点 A 应用层**：数据从 GPU 拷到 CPU 内存。
2.  **系统调用**：CPU 把数据从用户态（User Space）拷到内核态（Kernel Space）。
3.  **CPU 封装**：CPU 负责打 TCP 包，计算校验和。
4.  **网卡发送**：数据走网卡发出去。
5.  **节点 B 接收**：节点 B 的 CPU 收到中断，把数据解包，从内核态拷回用户态，再拷到 GPU。

**问题**：
* **CPU 负载高**：CPU 忙着搬砖（拷贝数据）和打包，没空做调度和推理逻辑。
* **延迟高**：多次内存拷贝（Context Switch）增加了巨大的 Latency。

**RDMA 方式（快，省）：**

1.  **Zero Copy (零拷贝)**：网卡（NIC）直接读取节点 A 的内存（甚至通过 GPUDirect RDMA 直接读 GPU 显存）。
2.  **Kernel Bypass (内核旁路)**：完全不经过 CPU，不需要操作系统内核参与，没有系统调用。
3.  **直接写入**：网卡直接把数据写入节点 B 的内存（或显存）。

**这就好比：**
* **TCP/IP**：你要给邻居送一箱苹果。你（CPU）先把苹果从仓库搬到客厅，打包，交给快递员。快递员送到邻居家，邻居（CPU）拆包，再搬到他家仓库。
* **RDMA**：你家仓库和邻居家仓库之间装了一个传送带。你按个按钮，苹果直接从你家仓库飞进他家仓库，**你和邻居（CPU）此时都在躺着看电视，完全不用动手。**

#### 2.2 RDMA 在 AI 中的应用

在 **PD 分离** 架构中，传输 KV Cache 的速度决定了生死的瞬间：

* **Prefill 节点** 算完 Prompt，生成了 KV Cache。
* **Decode 节点** 等着米下锅。
* 如果用 TCP/IP，传输这几百 MB 数据可能要几十毫秒，甚至比计算还慢，那 PD 分离就没意义了。
* 使用 **RDMA (通常基于 InfiniBand 或 RoCE 网络)**，传输延迟可以压到微秒级，且不消耗 CPU 资源，使得“跨机推理”像“本机推理”一样快。

---

补充知识：

1. Chunked Prefill 与 Radix Attention (Prefix Caching) 的天作之合

Chunked Prefill 不仅仅是为了防阻塞，它和 Prefix Caching（前缀缓存，即复用相同 System Prompt 的 KV Cache）结合时威力最大。
- 场景：多个请求都有相同的 10k 长度 System Prompt。
- 机制：如果没有 Chunking，新的请求可能要重算这 10k。但在 Chunked 机制下，调度器发现前 5 个 Chunk（共 10k）已经在显存里了，直接跳过计算，只把新的 User Prompt 作为第 6 个 Chunk 放入计算队列。

面试点：Chunked Prefill 让“细粒度的显存复用”成为可能。

2. 位置编码 (RoPE) 的处理陷阱

在写 Kernel 或看源码时要注意：Rotary Positional Embeddings (RoPE) 的计算依赖绝对位置。
问题：处理 Chunk 2（例如第 512-1024 个 token）时，传入 Kernel 的输入 tensor 索引是 0-512。
解决：必须显式传入一个 position_ids 张量。Kernel 计算 Attention Score 时，Q 和 K 的旋转角度必须基于 position_ids（512-1024），而不是基于当前 tensor 的局部索引（0-512）。如果这点搞错，模型会输出乱码。

3. Attention Mask 的复杂性

对于 FlashAttention 的调用，Chunked Prefill 需要处理 Block-Diagonal Mask（块对角掩码）或者更复杂的 Jagged Mask。
因为一个 Batch 里可能混合了 Request A 的 Chunk 2 和 Request B 的 Chunk 1。
算子必须确保 Request A 的 Q 只能看到 Request A 的 K（包括历史 KV），绝对不能看到 Request B 的数据。这比标准的 Causal Mask 实现要麻烦得多。

##（架构级替代方案）

Chunked Prefill 是在单卡/单节点内部解决资源争抢。如果我们把视角拉大到整个集群，有一个更宏大的替代/互补方案，也是目前大厂（如 Kimi/Moonshot, DeepSeek, OpenAI）的主流架构：

Prefill-Decode Disaggregation (PD 分离 / 分离式推理)
核心思想： 既然 Prefill 是 Compute-bound，Decode 是 Memory-bound，干脆把它们拆开，用不同的机器跑！
 
1. Prefill Instances (P 节点)：专门负责处理 Prompt。
     
     - 硬件：使用算力强的卡（如 H800），甚至不需要太大的 HBM（如果 Prompt 没那么长）。
     - 行为：一口气算完（Full Prefill），不搞 Chunking，利用率 100%。
     - 输出：将生成的 KV Cache 通过高速网络（RDMA/InfiniBand）传输给 D 节点。

2. Decode Instances (D 节点)：

     - 专门负责生成 Token。
     - 硬件：使用显存带宽大、容量大的卡（如 HBM3e），甚至可以用稍旧的卡（如 A800/A100），只要带宽够。
     - 行为：接收 P 节点传来的 KV Cache，直接开始 Decode。

