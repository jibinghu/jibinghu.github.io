首先把老生常谈的 scale dot-product attention formula 拿出来：

$$
Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
$$

存储类型 | 全称 | 能否读写 | 典型延迟 | 典型用途 | 是否进入 GPU 性能模型
-- | -- | -- | -- | -- | --
SRAM | Static RAM | ✅ 读写 | 1–30 cycles | 寄存器、LDS、Shared、L1 | ✅ 核心
DRAM | Dynamic RAM | ✅ 读写 | 300–1000 cycles | 外部显存 | ✅ 核心
HBM | High Bandwidth DRAM | ✅ 读写 | 300–800 cycles | GPU 显存 | ✅ 核心（DRAM 的一种）
SROM / ROM | Static Read-Only Memory | ❌ 只读 | 快 | 存微码、固件 | ❌ 不讨论

<img width="2000" height="783" alt="Image" src="https://github.com/user-attachments/assets/4fc1ddc5-0adf-4d4e-90c8-6f9f1b8b7ea3" />

这个经典的 FA 图，左边就是多级显存的访存效率，右边的图说明了以 GPT-2为例，pyTorch 原生实现的 Attention 和 FlashAttention 进行 Kernel Fusion后的时间对比。

而中间的图说明了 FA 的计算架构。最关键的一点是，对于 Softmax 后生成的注意力分数进行 Kernel fuse，不去显式地实现注意力分数。这样在显存中就不会出现完整的分数的传输。

而在反向传播中，需要注意力分数的时候就进行重算，来避免当 sequence length 很大时的 $O(n^2)$ 的数据传输。


首先说明Naive 实现：

最朴素的 GPU 实现通常是三步、多个 CUDA kernel 或 GEMM：
1. Kernel/GEMM1: 计算
S = Q Kᵀ
- 用 `cublasGemmEx` 做一个 `(N×d) × (d×N) → (N×N)`
- 把 `S` 存到 HBM（显存）里
2. Kernel2: 对 S 行做 softmax（含减 max + exp + 归一化）
- 从 HBM 读 `S`
- 做 softmax
- 把 `P` 写回 HBM
3. Kernel/GEMM3: 计算
O = P V
- 再用 GEMM：`(N×N) × (N×d) → (N×d)`

对于多个 Kernel 可以想到的第一个思路就是 fuse，把 “S 计算 + softmax” 融合成一个大 kernel：块状加载 Q、K 到 shared memory，直接算 logits，立刻 softmax，减少一次写 S 的开销。

<img width="1000" height="491" alt="Image" src="https://github.com/user-attachments/assets/062546de-a6db-48f0-8453-bef803da565b" />

而 FA1 实现思路：完全不要显式存整个 S/P 矩阵到 HBM，改成“分块算 attention + 在线 softmax + 累积输出”。

最主要的是 Softmax 的实现，他每次要看整行的数据来进行归一化，所以要用递推公式保证数值稳定和正确：


3.2 在线 softmax（Online Softmax）：支持“分块 / 多次看到同一行”的 softmax  

普通 softmax 对一行 \( s_i \) 的公式：  

$$
p_{i,j} = \frac{e^{s_{i,j}}}{\sum_j e^{s_{i,j}}}
$$

但在 FA1 中，一行 \( s_i \) 的全部元素 \( s_{i,*} \) 是通过多个 K block 才“分段看到”的，  
所以要用一个递推公式来保持 **数值稳定性与结果正确性**。

<img width="431" height="83" alt="Image" src="https://github.com/user-attachments/assets/16b7d53f-e51b-481d-875e-83779167fe62" />

<img width="178" height="79" alt="Image" src="https://github.com/user-attachments/assets/6f787684-0a4b-483d-a8f4-907b4486a84d" />

对每一行 \( i \)，维护三个量：  

- \( m_i \)：当前见过的所有 logits 的最大值  
- \( l_i \)：当前见过的所有 logits 的  $\sum_j e^{s_{i,j} - m_i}$ 。
- \( acc_i \)：加权和  $\sum_j p_{i,j} \cdot v_j$ 。的累积  

当看到一个新的 block 的 logits $s_{i,j}^{(\text{block})}$ 时：

1）计算新的最大值：


$$
m_i^{new} = \max \left( m_i^{old}, \max_j s_{i,j}^{(\text{block})} \right)
$$


2）对旧的累积进行重标定（rescale）：


$$
l_i^{new}= e^{m_i^{old} - m_i^{new}} \cdot l_i^{old}+ \sum_j e^{ s_{i,j}^{(\text{block})} - m_i^{new} }
$$


3）输出加权和同样进行重标定并加上新块贡献：

$$
acc_i^{new}= e^{m_i^{old} - m_i^{new}} \cdot acc_i^{old}+ \sum_j e^{ s_{i,j}^{(\text{block})} - m_i^{new} } \cdot v_j
$$

循环完所有 block 之后：

$$
O_i = \frac{acc_i}{l_i}
$$

等价于下面的公式：

<img width="737" height="358" alt="Image" src="https://github.com/user-attachments/assets/18dd783f-c540-414a-9355-0d169b3c02fe" />

---

上述即 FA1 的原理实现，下面用 Triton 进行实现：




---

下面的代码是在串行合并，所以只有一个 pre_max -> cur_max 的更新，如果一旦变成并行分块，就会出现多个分块来进行分割的结构。

``` python
# 计算公式
# S = Q * (K.transpose(-1, -2))
# P = softmax(S) 
# O = P * V
import numpy as np

N = 4
d = 2

# 这里的 N 就是 seq_len，以一行作为示例
S = np.random.random(size=(1, N))
V = np.random.random(size=(N, d))

def tiled_softmax_then_matmul(S, V):
  # 分数
  acc = np.zeros(shape=(1, d)) # 到目前为止输出的加权和
  pre_max = float("-inf") # 到目前为止的最大值
  pre_sum = 0 # 到目前为止 logits 的指数和
  for i  in range(N): # 每个token，KV的列维度，为了简洁，这里把Q的行维度设为了1，因此没有了内循环
    s_i = S[:,i] # 每列S
    cur_max = max(pre_max, s_i) # 当前分块和之前分块一起的最大值
   # 将之间的 调整因子调整为减去 cur_sum
    pre_sum *= (np.exp(pre_max - cur_max)) # L10
    # 当前分块和之前分块一起的指数和
    cur_sum = pre_sum + np.exp(s_i - cur_max)
    # 当前分块的softmax结果
    score = np.exp(s_i - cur_max) / cur_sum # 到目前为止，当前块的 score 已经计算完毕，但是之前的 logits 还是以 pre_sum 作为底来计算的，所以还需要调整
    scale = pre_sum / cur_sum # 因为上一个分块的结果是基于当时的softmax中间sum组成的分母（presum），现在这个分块又得到了新的中间sum（cursum），所以需要更新：对上一个分块的结果acc做一个scale，保证结果的正确性
    acc *= scale # 进行更新之前的 logits 
    acc += score * V[i,] # scale后的中间结果加上当前分块的P * V = O
    # 更新
    pre_max = cur_max 
    pre_sum = cur_sum 
  return acc
```



---

补充，PCIe 和 NVlink 带宽：

互联方式 | 单向理论带宽 | 现实可达
-- | -- | --
PCIe 3.0 x16 | ~16 GB/s | ~12–13 GB/s
PCIe 4.0 x16 | ~32 GB/s | ~25–30 GB/s
PCIe 5.0 x16 | ~64 GB/s | ~50–60 GB/s
NVLink V2（V100） | ~50 GB/s/链路 | ~45 GB/s
NVLink V3（A100） | ~100 GB/s/链路 | ~90 GB/s
NVLink V4（H100） | ~150 GB/s/链路 | ~140 GB/s
GPU HBM 本地带宽 | 1–3 TB/s | ✅ 实测可达



