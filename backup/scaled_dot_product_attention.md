# scaled_dot_product_attention
torch 中的 scaled_dot_product_attention 是 PyTorch 2.0 中引入的最优 Attention 接口之一，旨在通过硬件加速和优化的计算图，加速模型训练与推理。在 Transformer 结构中，Attention 操作是最核心的组件之一，而 Scaled Dot-Product Attention 是一种最常用的 Attention 机制。

Scaled Dot-Product Attention 解释

Scaled Dot-Product Attention 的计算公式为：

<img width="280" alt="image" src="https://github.com/user-attachments/assets/4bd010a1-1faf-488d-8b76-ef40bf9f099e">


其中：

	•	￼ 是查询（query）矩阵。
	•	￼ 是键（key）矩阵。
	•	￼ 是值（value）矩阵。
	•	￼ 是键矩阵的维度，用于缩放分数，避免分数过大。

torch.nn.functional.scaled_dot_product_attention

scaled_dot_product_attention 函数的主要优势在于硬件加速和高效计算。具体来说，它优化了以下几个方面：

	1.	并行计算：该函数在并行处理 GPU 和加速器时充分利用了硬件特性，可以直接在 GPU 上计算 QK 的转置，减少了手动分片和内存复制的开销。
	2.	缩放与 Softmax：缩放和 softmax 操作直接在 CUDA 内核中实现，提升了速度，同时减少了数值计算误差。
	3.	自动 Mask 支持：可以通过 attn_mask 参数传入掩码（mask），适用于自回归或语言建模等任务。
	4.	高效的内存管理：为了避免内存溢出，它会自动根据输入的大小和设备分配计算内存，确保计算稳定性。

示例代码

以下是使用 scaled_dot_product_attention 的简单示例：

import torch
import torch.nn.functional as F

# 假设输入的 Query、Key 和 Value 张量维度为 (batch_size, num_heads, seq_len, embed_dim)
Q = torch.randn(4, 8, 64, 64)  # 例如: batch_size=4, num_heads=8, seq_len=64, embed_dim=64
K = torch.randn(4, 8, 64, 64)
V = torch.randn(4, 8, 64, 64)

# 使用 scaled_dot_product_attention 计算 Attention 输出
attention_output = F.scaled_dot_product_attention(Q, K, V)
print(attention_output.shape)  # 输出形状: (4, 8, 64, 64)

注意事项

	•	Masking：attn_mask 可以用于掩盖序列中不需要关注的部分，例如填充位或者未来时间步。
	•	硬件要求：为了充分利用硬件加速，最好在支持的 GPU 或者加速器上运行，以获得最佳性能。

scaled_dot_product_attention 的设计旨在为 Transformer 模型提供更高效的计算支持，是当前 PyTorch 中实现高效 Attention 的推荐方法。

---

使用 pip install sageattention 后，
只需要在模型的推理脚本前加入以下三行代码即可：

图片
![image](https://github.com/user-attachments/assets/1b85c0c2-2ab4-4fd5-aaf6-34872e787fb5)


---

参考：

https://mp.weixin.qq.com/s/S1ZfDyg61pTXdyHiVN8SSA
