
## [TileRT](https://github.com/tile-ai/TileRT)

TileRT（tile-ai/TileRT）是 tile-ai 团队做的一个“面向极低延迟（ultra-low-latency）LLM 推理”的实验性运行时项目。它的核心定位不是追求吞吐（throughput）和大 batch 的效率，而是追求 **batch size 很小（甚至 1）时的极致响应速度**，把“单/少量请求的 token-per-output-token latency（TPOT）”压到更低。([[GitHub](https://github.com/tile-ai/TileRT)][1])

它想解决什么问题:

主流推理系统（面向线上服务）通常围绕 **高吞吐批处理**优化：尽量把更多请求凑成 batch、让算子以更大块（kernel/GEMM）去跑，从而提升 GPU 利用率。但在交互式应用（例如实时决策、长程 agent、交互式编程助手等）里，用户更在意的是“少量请求的响应时间”，这时批处理思路反而可能引入排队、同步与 pipeline 空泡。TileRT 明确把目标放在这类场景。([[GitHub](https://github.com/tile-ai/TileRT)][1])

核心思路：tile-level runtime（“算子切成 tile 任务，再由运行时重排/重叠”）

TileRT 的关键概念是 **tile-level runtime engine**：

* **编译器驱动的分解**：把 LLM 的算子拆成更细粒度的 “tile-level tasks”（你可以理解为比传统“一个算子一个大 kernel”更细的计算/搬运/通信任务粒度）。([[GitHub](https://github.com/tile-ai/TileRT)][1])
* **运行时重调度与重叠**：在多 GPU 场景下，运行时对 **compute / I/O / communication** 做更激进的重叠（overlap）和重排（reschedule），减少空转与等待，把硬件“空泡”尽量填满。([[GitHub](https://github.com/tile-ai/TileRT)][1])

此外，README 里也明确说这些编译器技术会并入 TileLang 和 TileScale 生态。([[GitHub](https://github.com/tile-ai/TileRT)][1])

项目当前状态与“你能不能直接用”

截至 README 所述的 preview 版本，TileRT 的可用形态更像“**特定硬件/特定模型/特定环境的演示型交付**”，而不是通用推理框架：

* **硬件**：当前预览构建主要面向 **8× NVIDIA B200** 的配置。([[GitHub](https://github.com/tile-ai/TileRT)][1])
* **环境**：推荐 Linux（Ubuntu 20.04+）与 Python 3.11–3.12，并且 PyTorch 需匹配 CUDA 12.8/12.9（与 B200 环境对应）。([[GitHub](https://github.com/tile-ai/TileRT)][1])
* **交付方式**：强烈建议用他们提供的 Docker 镜像（例如 `tileai/tilert:v0.1.0`），容器内再 `pip install tilert`。([[GitHub](https://github.com/tile-ai/TileRT)][1])

模型与权重

TileRT 目前的评测与示例围绕 DeepSeek-V3.2-Exp，并要求对原始权重做预处理；他们在 Hugging Face 提供了**预转换权重**供直接下载使用。([[GitHub](https://github.com/tile-ai/TileRT)][1])

性能信息怎么看

README 给了一个对比图，声称在 **8×B200、batch size=1、输入/输出 1K/1K** 的设置下，TileRT 在序列生成上显著快于 SGLang 与 vLLM（并注明了对比版本与 CUDA 版本）。([[GitHub](https://github.com/tile-ai/TileRT)][1])
需要注意：这是“特定硬件 + 特定模型 + 特定配置”的初步结果，更多硬件/模型/批量大小的覆盖属于其未来计划。([[GitHub](https://github.com/tile-ai/TileRT)][1])

你用这仓库能获得的“关键理解”

如果你是做推理系统/编译器/内核优化的，TileRT 最值得关注的是它在系统层面提出的方向：

1. **把算子执行从“粗粒度 kernel 序列”转为“tile 任务图”**（更细的调度单元）。([[GitHub](https://github.com/tile-ai/TileRT)][1])
2. **跨 compute / IO / 通信的极限重叠**，特别适合多卡低延迟推理（batch 小、同步更敏感）。([[GitHub](https://github.com/tile-ai/TileRT)][1])
3. **与 TileLang/TileScale 的关系**：TileRT 更像“运行时与系统侧落地”，TileLang/TileScale 更像“语言/编译器侧生产这些 tile 任务”。([[GitHub](https://github.com/tile-ai/TileRT)][1])

---

这是一个非常硬核的学习计划。针对你 HPC + CUDA 的背景，我建议跳过“从文档看起”的常规路线，直接采用 **“核心机制映射法”**。

你需要把这两个框架拆解为：**Python 控制平面 (调度/显存管理)** + **C++/CUDA 数据平面 (算子/执行)**。

以下是具体的“手术刀式”学习路径，旨在让你在面试中能讲出源码级的细节。

---

## 第一阶段：vLLM —— 工业界基准 (The Standard)

vLLM 的核心壁垒在于 **PagedAttention**（解决了显存碎片化）和 **Continuous Batching**（解决了动态长度请求的调度）。

#### 1. 核心概念与源码切入点
不要通读代码，vLLM 代码量太大了。只看这三个核心文件：

* **内存管理 (Block Manager)**
    * **目标**：理解逻辑块（Logical Block）如何映射到物理块（Physical Block）。这和你操作系统里的 `Page Table` 是一模一样的。
    * **源码位置**：`vllm/core/block_manager.py`
    * **看什么**：`BlockTable` 类。看它如何维护 `block_number` 的映射。
    * **面试考点**：如果有两个 Request 共享了前缀，引用计数（ref_count）是怎么变的？Copy-on-Write 机制在哪里实现的？

* **调度器 (Scheduler)**
    * **目标**：理解 Continuous Batching 是怎么塞数据的。
    * **源码位置**：`vllm/core/scheduler.py`
    * **看什么**：`_schedule` 函数。它怎么判断当前显存还够不够塞下一个 Token？
    * **面试考点**：vLLM 什么时候会发生 `Preemption` (抢占/换出)？当显存不足时，它是怎么把 KV Cache 从 GPU 换到 CPU 的 (Swap-out)？

* **执行器 (Model Executor)**
    * **目标**：Python 怎么调 C++ 算子。
    * **源码位置**：`vllm/model_executor/layers/attention/backends/flash_attn.py` (或者 xformers)
    * **看什么**：看它如何准备 `metadata`（比如 `block_tables`），然后传给 CUDA kernel。

#### 2. vLLM 的技术壁垒与创新
* **壁垒**：**PagedAttention Kernel**。
    * 去 `csrc/attention` 目录下看 CUDA 代码。你需要理解普通的 FlashAttention 是读连续内存，而 PagedAttention 是如何通过“间接寻址”（先查表，再读数据）来获取 Key/Value 的。这会增加寄存器压力和访存延迟，vLLM 是如何优化的？
* **最新创新**：
    * **Chunked Prefill**：以前 Prefill（第一阶段）和 Decode（第二阶段）是分开的。现在为了防抖动，把 Prefill 切成小块和 Decode 插空运行。
    * **FP8 / AWQ 支持**：看 `vllm/model_executor/layers/quantization`，了解量化算子怎么插入的。

---

## 第二阶段：SGLang —— 面向 Agent 的颠覆者 (The Challenger)

SGLang 的核心壁垒在于 **RadixAttention**（KV Cache 的树形复用）和 **Compiler-based Optimization**（计算图编译）。

#### 1. 核心概念与源码切入点
SGLang 是为了解决 vLLM 在多轮对话和复杂 Agent 场景下的“无记忆”痛点。

* **基数树缓存 (Radix Cache)** —— **这是 SGLang 的灵魂**
    * **目标**：理解它是如何自动把 Prompt 变成一棵树，并实现 LRU Eviction 的。
    * **源码位置**：`sglang/srt/managers/router/radix_cache.py`
    * **看什么**：
        * `insert()`: 新请求来了，怎么插到树里？
        * `match_prefix()`: 怎么找到最长的公共前缀？
    * **面试杀手锏**：对比 vLLM 的 Hash 匹配（基于 Block）和 SGLang 的 Radix 匹配（基于 Token 序列）。Radix 可以在任意位置 Fork，而 Hash 只能整块复用。

* **受限解码 (Structured Generation / FSM)**
    * **目标**：理解正则表达式怎么限制模型输出。
    * **源码位置**：`sglang/srt/constrained/` (基于 outlines 或 xgrammar)
    * **看什么**：它不是在 Python 层做 `if/else`，而是把 Regex 编译成了一个 **FSM (有限状态机)**。在采样的每一步，根据当前 State 掩盖掉不合法的 Token Logits。这几乎零开销。

* **SGLang Runtime (SRT)**
    * **看什么**：SGLang 其实复用了 vLLM 的很多底层 Kernel（比如 PagedAttention），但它把 **Scheduler** 换掉了。它不再是简单的 FCFS（先来先服务），而是**基于树的调度**（尽量让访问同一个树枝的请求一起跑，减少 Cache Miss）。

#### 2. SGLang 的技术壁垒与最新创新
* **壁垒**：**Frontend-Backend Co-design**。
    * 它定义了一种 DSL（领域特定语言）。你在前端写的 `fork`、`join` 会被编译成计算图，后端 Runtime 直接执行图，而不是像 Python 那样一句句发 HTTP 请求。这消除了 Python 解释器和网络通信的开销。
* **最新创新**：
    * **Jump Forward Decoding**：在树上做推测执行。
    * **Data Parallelism with Shared Prefix**：如果是多卡推理，大家共享前缀（System Prompt），SGLang 只需要传增量部分。

---

### 第三阶段：如何高效“跑通”源码（Action Plan）

不要光看，要 Debug。你的 HPC 背景让你很擅长 Trace。

**任务 1：日志埋点分析 vLLM 调度逻辑**
1.  启动 vLLM Server。
2.  在 `vllm/core/scheduler.py` 的 `_schedule` 函数里打 print，打印 `running_queue`, `waiting_queue`, `swapped_queue` 的长度。
3.  写个脚本，瞬间发 100 个请求。
4.  **观察**：什么时候 Waiting 变 Running？什么时候 Running 变 Swapped？这能让你彻底懂 Continuous Batching。

**任务 2：可视化 SGLang 的 Radix Tree**
1.  在 `sglang/srt/managers/router/radix_cache.py` 里，找个地方把 `self.tree` 打印出来（或者写个简单的递归打印函数）。
2.  跑一个 Few-shot 的 Agent Demo（先问 A，再基于 A 问 B）。
3.  **观察**：树是怎么长出来的？第二个请求是不是直接挂在了第一个请求的叶子节点上？

### 总结：面试话术构建

当你面试时，用这套逻辑来展示你的深度：

* **面试官**：“你对 vLLM 了解吗？”
* **你**：“了解。我深入看过它的 **Block Manager** 实现。它本质上是把操作系统虚拟内存的分页机制搬到了 GPU 显存管理上，解决了 KV Cache 的碎片化问题。但我发现它在处理 **Agent 多轮对话**时，KV Cache 的复用率不如 SGLang 高。”
* **面试官**：“哦？为什么？”
* **你**：“因为 vLLM 是基于 Block Hash 做缓存，粒度较粗且不支持树形回退。而 SGLang 的 **RadixAttention** 维护了一棵前缀树，对于 Agent 的 `Tree-of-Thought` 这种分叉搜索场景，它可以做到 **Zero-Overhead** 的上下文恢复。我目前的 C++ 项目也参考了类似的设计……”

**建议**：先花 3 天把 **vLLM 的 Scheduler** 看懂，再花 3 天看 **SGLang 的 RadixCache**。这两个看懂了，Infra 面试的 80% 难点你就通关了。