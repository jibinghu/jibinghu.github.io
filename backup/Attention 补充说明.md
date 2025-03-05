
---

kv cache 以及 casual mask 演示

![Image](https://github.com/user-attachments/assets/53697fd5-e62f-4a07-95d2-c33fa0265279)

MHA 的多头注意力是在权重矩阵阶段就切分的，同理，MQA/GQA的多头注意力的共享也是在权重矩阵阶段就共享的。

### 问题一：多头注意力（MHA）中的权重划分
在标准的**多头注意力（MHA）**中，权重矩阵 \( W_Q, W_K, W_V \) 的划分确实从矩阵级别就开始进行。具体来说：
1. **参数划分方式**：  
   假设模型的隐藏维度为 \( d_{\text{model}} \)，头数为 \( h \)，每个头的维度为 \( d_k = d_{\text{model}} / h \)。此时：
   • 每个头 \( i \) 的权重矩阵 \( W_Q^{(i)}, W_K^{(i)}, W_V^{(i)} \) 的维度为 \( d_{\text{model}} \times d_k \)，即 \[ [\text{hidden\_size}, \text{hidden\_size}/\text{num\_heads}] \]。
   • 多头注意力的核心实现是将输入 \( X \) 同时投影到 \( h \) 组独立的查询（Q）、键（K）、值（V）子空间，每组子空间对应一个注意力头。

2. **计算过程示例**：  
   输入 \( X \in \mathbb{R}^{n \times d_{\text{model}}} \)（序列长度 \( n \)）经过投影：
   \[
   Q_i = X W_Q^{(i)}, \quad K_i = X W_K^{(i)}, \quad V_i = X W_V^{(i)} \quad (\forall i \in \{1,2,...,h\})
   \]
   每个头独立计算注意力权重后，结果会拼接并通过线性层融合。

---

### 问题二：多查询注意力（MQA）的权重共享
在**多查询注意力（MQA）**中，共享的权重主要集中在键（K）和值（V）的投影矩阵上，具体如下：
1. **参数共享机制**：  
   • **K 和 V 共享**：所有注意力头共享同一组权重矩阵 \( W_K \) 和 \( W_V \)，即：
     \[
     K = X W_K, \quad V = X W_V \quad (\text{全局共享})
     \]
     这两个矩阵的维度仍为 \( d_{\text{model}} \times d_k \)，但与 MHA 不同，所有头复用相同的 K 和 V。
   • **Q 独立**：每个头保留独立的查询投影矩阵 \( W_Q^{(i)} \)，即每个头生成自己的 Q：
     \[
     Q_i = X W_Q^{(i)} \quad (\forall i \in \{1,2,...,h\})
     \]

2. **计算优势与影响**：  
   • **参数减少**：K 和 V 的参数量从 \( 2h \cdot d_{\text{model}} \cdot d_k \) 降至 \( 2 \cdot d_{\text{model}} \cdot d_k \)，显著节省内存。
   • **计算效率**：在解码阶段（如自回归生成），共享的 K 和 V 可避免重复计算，提升推理速度。例如，在生成第 \( t \) 个 token 时，历史 K 和 V 可缓存复用。

3. **维度匹配示例**：  
   假设 \( d_{\text{model}}=768 \)，头数 \( h=12 \)，则每个头的 \( d_k=64 \)：
   • MQA 的 \( W_K, W_V \) 维度为 \( 768 \times 64 \)，与 MHA 中每个头的 K/V 投影维度一致，但被所有头共享。
   • 每个头的 \( W_Q^{(i)} \) 仍为 \( 768 \times 64 \)，独立生成不同的 Q。

---

### 对比总结
| **特性**               | **多头注意力（MHA）**                          | **多查询注意力（MQA）**                      |
|------------------------|---------------------------------------------|------------------------------------------|
| **Q/K/V 投影**         | 每个头独立投影 Q, K, V                       | 每个头独立投影 Q，共享 K 和 V                 |
| **参数量**             | \( 3h \cdot d_{\text{model}} \cdot d_k \)   | \( (h + 2) \cdot d_{\text{model}} \cdot d_k \) |
| **适用场景**           | 对精度敏感的任务（如预训练）                   | 资源受限场景（如边缘设备推理、长序列生成）       |
| **注意力多样性**       | 高（独立 K/V 捕捉不同模式）                   | 较低（共享 K/V 可能限制多样性）                |

---

### 代码示例（PyTorch）
**MHA 实现片段**：
```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 每个头独立的 Q/K/V 投影矩阵
        self.W_Q = nn.Linear(d_model, d_model)  # 实际实现中会拆分为 h 个子矩阵
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        # 输入 x: [batch, seq_len, d_model]
        Q = self.W_Q(x)  # 拆分为 h 个 [batch, seq_len, d_k]
        K = self.W_K(x)
        V = self.W_V(x)
        # ... 后续计算注意力 ...
```

**MQA 实现片段**：
```python
class MultiQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 独立的 Q 投影，共享的 K/V 投影
        self.W_Q = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(num_heads)])
        self.W_K = nn.Linear(d_model, self.d_k)  # 全局共享
        self.W_V = nn.Linear(d_model, self.d_k)  # 全局共享
    
    def forward(self, x):
        # 输入 x: [batch, seq_len, d_model]
        K = self.W_K(x)  # 共享的 K
        V = self.W_V(x)  # 共享的 V
        Q_heads = [W_Q_i(x) for W_Q_i in self.W_Q]  # 每个头独立的 Q
        # ... 后续计算注意力 ...
```

---

### 深入思考
1. **MQA 的局限性**：  
   共享 K/V 可能导致模型无法充分捕捉输入的不同交互模式，尤其在需要细粒度语义区分的任务（如机器翻译）中，可能影响性能。可通过实验验证：在 T5 或 GPT 架构中，将 MHA 替换为 MQA 后，观察验证集损失和生成质量的变化。

2. **变体扩展**：  
   • **分组查询注意力（GQA）**：折中方案，将头分为 \( g \) 组，组内共享 K/V，平衡参数量与表达能力。
   • **动态投影**：根据输入内容动态生成 K/V 投影参数，缓解共享带来的信息损失。

---

文章已经写的非常好了，这里就不赘述了。

https://zhuanlan.zhihu.com/p/21799412936
