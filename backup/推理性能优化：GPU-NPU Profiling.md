https://zhuanlan.zhihu.com/p/1981436859470074335?share_code=bZtoKsWUIpcD&utm_psn=1981881255978096583

我直接照搬过来备份一下，老师写的还是很好，只不过真正看起来确实比较麻烦。

这篇 blog 主要是看 序列图如何分析，包括：

- profiling分析性能的案例。
- 多流并行在profiling上面是如何体现？
- 计算与通信掩盖如何分析？

我们会在其中穿插一些内容。

1. GPU场景

GPU + Qwen(Dense) + tracing场景。

首先用 vllm 收集 profiling：

``` python 
from vllm import LLM, SamplingParams
import torch
import torch.profiler as profiler
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


if __name__ == "__main__":
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
        record_shapes=True,          # 记录张量形状
        profile_memory=True,         # 记录内存分配/释放
        with_stack=False              # close 调用栈
        ) as prof:
        model_name = "/home/kaiyuan/models/Qwen2.5-7B-Instruct" 
        llm = LLM(model=model_name, dtype='float16', tensor_parallel_size=4)
        prompts = [
        "Hello, my name is",
        "The capital of France is",
        "The future of AI is",
        "Please introduce vLLM framework"
        ]
    # 设置采样参数
        sampling_params = SamplingParams(
        temperature=0.8,  # 控制生成文本的随机性，值越高越随机
        top_p=0.95,  # 控制采样范围，值越高生成文本越多样化
        max_tokens=50,  # 生成的最大 token 数量
        n=1
        )

        outputs = llm.generate(prompts, sampling_params)
    prof.export_chrome_trace("trace.json")
```

上述是 PyTorch Profiler，是在 pytorch 级别的 CPU 算子/ CUDA kernel 级别的分析，而 nsys/ncu/nvprof 是在 CUDA Driver 级别的分析，Linux perf/ftrace 等是在 cpu 级别的分析。

正常运行后，能够获得"trace.json"文件，直接拖拽到浏览器的[https://ui.perfetto.dev/](https://link.zhihu.com/?target=https%3A//ui.perfetto.dev/) （推荐）、如果是chrome浏览器可用chrome://tracing/

profiling导入之后，可找到主机端（python层）的执行时序、GPU端（stream）的执行时序：

<img width="1486" height="691" alt="Image" src="https://github.com/user-attachments/assets/7f7a81e2-4a2a-4d43-8dd0-b427ca432bde" />

先看一下python层执行情况。在profiling中找到序列图进行放大，我们可以定位一个完整的层的位置，比如用耗时较长的attention进行隔断，截取一段内容进行分析

<img width="1037" height="204" alt="Image" src="https://github.com/user-attachments/assets/5828889b-937f-41ab-a306-5ec78528643d" />

Qwen2.5 dense的模型主体是GQA+FFN，这些层的运算在profiling中可找到对应的位置。

<img width="1211" height="934" alt="Image" src="https://github.com/user-attachments/assets/d1583231-3be8-41d2-8e0d-abec6b66ae7c" />

通过放大profiling，找到GAQ在python中的时序图位置。有几个细节：

- 时序条的操作与操作之间存在空白，这并不代表执行不连续。

> Timeline 上 kernel 之间的空白不等价于 GPU 停机。很多时候 GPU 正在执行其他 stream 的 kernel、进行 DMA 拷贝、等待 sync、或者 profiler 没显示某些事件。只有当 SM 利用率下降到 0 时，才能判断 GPU 真正 idle。

- 在O linear计算完成后有个all reduce操作，这是因为开启TP并行，即OKV矩阵运算时的权重W进行了列切分、O矩阵运算采用W行切分，最后结果需要一个all reduce校正结果。

> 这里有些需要澄清的地方，在 TP(Tensor Parallel，相对于 DP/PP)  中，TP 发生在 大矩阵维度，不是在单个 head 内部。Multi-head attention 是模型设计上的逻辑维度切分；Tensor Parallel 是运行时为了利用多 GPU 资源而进行的权重张量物理切分。两者互不重叠，一个负责表示能力，一个负责计算并行性。
> QKV projection 的权重按列拆分后，每个 GPU 计算部分 heads；O projection 的权重按行拆分，因此每个 GPU 对最终 hidden_dim 输出贡献一部分；这些部分结果需要 All-Reduce(sum) 来恢复完整的 hidden_dim 输出。

<img width="1007" height="324" alt="Image" src="https://github.com/user-attachments/assets/f7d08787-de58-4a2f-bd99-8c380c44135c" />

KV共用一个线性层，所以后面有个split操作(一个大矩阵 concat 权重 qkv)；Q与K在算attention前需要进行RoPE运算。

<img width="1046" height="177" alt="Image" src="https://github.com/user-attachments/assets/7dbc621a-0e57-47e3-bd30-ccf69692605b" />

放大attention（图片中的unified attention）会看到许多细节操作，比如split、view、empty等，也能找到两个主要操作：KV cache保存、paged attention运算。

<img width="1440" height="244" alt="Image" src="https://github.com/user-attachments/assets/a345e47e-f1d9-4e2c-be06-ae45a1ed9a8a" />

FFN层是在attention计算之后，找到RMS Norm并进行放大，看到的细节：
- 向上投影Up和门控Gate是一个线性运算；
- 由于开启了TP并行，Up&Gate的权重列切、Down的权重行切，所以需要一个all reduce操作。

<img width="1182" height="310" alt="Image" src="https://github.com/user-attachments/assets/24535783-d7c6-4961-b827-cafb10e9441f" />

stream层的执行情况主要是反映GPU的kernel下发下去之后的执行逻辑，通过放大同样能够找到每个上层模块触发的底层操作。如下所示为FFN的线性运算的kernel时序。

<img width="1440" height="152" alt="Image" src="https://github.com/user-attachments/assets/d434d8c9-cf34-4868-8093-287c398a2f79" />

线性层的计算调用的是cutlass的kernel：

<img width="1440" height="390" alt="Image" src="https://github.com/user-attachments/assets/c62c0253-fe85-48f3-9459-52013a1aa62d" />

silu的计算kernel如下所示，可看到输入数据的类型。

<img width="1013" height="179" alt="Image" src="https://github.com/user-attachments/assets/b21abdac-976a-465a-8220-e578887bb3d8" />

通过测算可以捕获多个层的执行时间，如下所示捕获到一个GQA+FFN在python层的总执行耗时近七百多us。

<img width="843" height="473" alt="Image" src="https://github.com/user-attachments/assets/b50427cf-e6a7-4eed-84f1-9cd9d831fd64" />

2. NPU 场景

NPU是华为昇腾的芯片，搭配看profiling的专用工具是Insight，GPU有对应的Nsight。用专用工具的能够看到更加细粒度的时序信息。在Insight中导入一份Profiling后可以显示多个层的时序：

- Python：上层host侧的执行情况；
- CANN：算子API执行情况（类似CUDA层）；
- AscendHardware：是NPU芯片层的情况；

还有辅助的时序：通信层Communication、通算重叠度OverlapAnalysis、AI core Freq、HBM、LLC、QoS等。

<img width="1373" height="680" alt="Image" src="https://github.com/user-attachments/assets/b6b71fdb-15d2-4149-a470-5365db8827a7" />

模型时序分析：如果开了堆栈采集，能看到火焰图，类似py-spy的采集，内容比较详细：

<img width="1005" height="470" alt="Image" src="https://github.com/user-attachments/assets/faeac589-9f7f-4f8d-86b2-660cf37bb91b" />

当然也可以采集不带堆栈的，数据更接近实际数据。以DeepSeekV3的模型为例，一个MLA+MoE层运算的时序图如下：

python层的MoE与MLA之间有大段的to/iterm时序占用大量时间，实际这些是一些流的同步等待（与GPU例子不同）。
操作的名称大多以npu开头。

<img width="1440" height="396" alt="Image" src="https://github.com/user-attachments/assets/df64aa2f-62cc-467c-9c4f-bf9ccb864c31" />

可以找关键ops，比如attention运算。辅助能够找到一个MLA模块的如下图所示，类似地能够找到其它模块。

<img width="1440" height="420" alt="Image" src="https://github.com/user-attachments/assets/146860a2-e3b3-47a3-8e7a-f0da70a8b8ff" />

一般而言上层调用到底层执行会存在一定的时间滞后，这个可以通过选中具体的ops观测，如集群通信，从python层发起到CANN再到通信时序之间有个时间差异。

<img width="1440" height="409" alt="Image" src="https://github.com/user-attachments/assets/7326782d-1258-492e-bef3-38210966435d" />

单看NPU芯片上的流执行，会看到许多event wait，一般是流的操作之间排队使用片上资源：

<img width="1440" height="528" alt="Image" src="https://github.com/user-attachments/assets/faff349e-e8c6-454a-b426-a866dc530297" />

检查某个算子的计算时间和输入、输出shape：

<img width="1108" height="764" alt="Image" src="https://github.com/user-attachments/assets/b030b5ee-9b37-4406-8705-d911d60dcf5c" />

如果采集了多个step，还能看到step与step之间的主机下发/尾端处理时间，若没有多流掩盖硬件侧能找到一个较长的空闲（free）时间。

<img width="1355" height="673" alt="Image" src="https://github.com/user-attachments/assets/daf984ec-4159-437d-8448-c63b0656d1e6" />

当完成一些优化后，我们可以利用时序图比照优化前后的计算差异。

操作优化：比如，在MoE计算完成后如果有一个AllReduce操作，能够拆解为Reducescatter + allgather，并且allgather位置可以移动到比较靠后的位置。

<img width="1440" height="376" alt="Image" src="https://github.com/user-attachments/assets/6d8edd71-fe53-4627-b827-d61233cc8b92" />

重叠度优化：在硬件层观测时，可以看到一些优化手段带来的时序变化，这里以micro batch双流运算示例，能够看到两个stream分别执行了完整的MoE+MLA，只是时间上面有错位。

<img width="1440" height="328" alt="Image" src="https://github.com/user-attachments/assets/0995fdf1-7890-4c64-bf22-b0689a5951c3" />

这样带来的好处是计算通信掩盖比例得到提升，通过Overlap Analysis时序能够看到变化：

<img width="1355" height="452" alt="Image" src="https://github.com/user-attachments/assets/398f822d-b906-4df7-8563-e35fe983316d" />

Insight工具除了提供时序（timeline）分析，还可看更详细的内存占比、操作占比等分析，随着分析的进行，这些数据都能够比较好的帮助开发人员更好的挖掘一款硬件的潜力、或者调整模型算法的设计。




