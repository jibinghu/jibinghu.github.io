多头注意力（MHA）和分组查询注意力（GQA）的伪代码实现：

---

### **1. 多头注意力（Multi-Head Attention, MHA）**
```python
def MultiHeadAttention(Q, K, V, num_heads, d_model):
    batch_size, seq_len = Q.size(0), Q.size(1)
    d_k = d_model // num_heads  # 每个头的维度
    
    # 1. 线性投影并分割头
    Q = linear(Q).view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
    K = linear(K).view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
    V = linear(V).view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
    
    # 2. 计算缩放点积注意力
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    attn = softmax(scores, dim=-1)
    context = torch.matmul(attn, V)
    
    # 3. 合并所有头并线性变换
    context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
    output = linear(context)
    return output
```

**关键步骤说明**：
1. 将输入 `Q/K/V` 分别投影到 `num_heads` 个独立的头。
2. 每个头计算独立的注意力权重。
3. 合并所有头的输出并通过线性层得到最终结果。

---

### **2. 分组查询注意力（Grouped-Query Attention, GQA）**
```python
def GroupedQueryAttention(Q, K, V, num_heads, num_groups, d_model):
    batch_size, seq_len = Q.size(0), Q.size(1)
    assert num_heads % num_groups == 0
    heads_per_group = num_heads // num_groups
    d_k = d_model // num_heads
    
    # 1. 投影 Q 到 h 个头，K/V 到 g 个头
    Q_proj = linear(Q).view(batch_size, seq_len, num_heads, d_k)  # [B, L, h, d_k]
    K_proj = linear(K).view(batch_size, seq_len, num_groups, d_k) # [B, L, g, d_k]
    V_proj = linear(V).view(batch_size, seq_len, num_groups, d_k) # [B, L, g, d_k]
    
    # 2. 扩展 K/V 以匹配每个组内的头数
    K_proj = K_proj.unsqueeze(2).expand(-1, -1, heads_per_group, -1, -1)  # [B, L, k, g, d_k]
    V_proj = V_proj.unsqueeze(2).expand(-1, -1, heads_per_group, -1, -1)  # [B, L, k, g, d_k]
    K_proj = K_proj.reshape(batch_size, seq_len, num_heads, d_k)          # [B, L, h, d_k]
    V_proj = V_proj.reshape(batch_size, seq_len, num_heads, d_k)          # [B, L, h, d_k]
    
    # 3. 调整维度并计算注意力
    Q_proj = Q_proj.transpose(1, 2)  # [B, h, L, d_k]
    K_proj = K_proj.transpose(1, 2)  # [B, h, L, d_k]
    V_proj = V_proj.transpose(1, 2)  # [B, h, L, d_k]
    
    scores = torch.matmul(Q_proj, K_proj.transpose(-2, -1)) / math.sqrt(d_k)
    attn = softmax(scores, dim=-1)
    context = torch.matmul(attn, V_proj)
    
    # 4. 合并输出
    context = context.transpose(1, 2).reshape(batch_size, seq_len, d_model)
    output = linear(context)
    return output
```

**关键步骤说明**：
1. 将 `Q` 投影到 `num_heads` 个查询头，`K/V` 投影到 `num_groups` 个键值头（`num_groups < num_heads`）。
2. 对 `K/V` 进行扩展，使每个查询组共享相同的键值头（例如，组内 `k` 个查询头共享 1 个键值头）。
3. 计算注意力时，组内的多个查询头复用同一组键值头，减少计算量。
4. 最终合并结果并通过线性层。

---

### **核心区别**：
| 特性               | MHA                          | GQA                          |
|--------------------|-----------------------------|------------------------------|
| **键值头数**        | 与查询头数相同（`num_heads`） | 少于查询头数（`num_groups`） |
| **计算复杂度**      | 较高（每个头独立计算）        | 较低（组内共享键值头）        |
| **应用场景**        | 标准 Transformer            | 大模型推理优化（如 LLaMA-2） | 

通过分组共享键值头，GQA 在保持表达能力的同时显著提升了推理效率。