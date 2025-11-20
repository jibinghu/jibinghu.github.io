> Inductor 是 PyTorch 的默认后端编译器（backend），负责把 Python 的 PyTorch 计算图自动转换成高效的低层代码（CUDA kernel / Triton kernel / C++），从而加速模型推理和训练。

pytorch 执行有两种方式：

1. Eager Mode（默认） | 每行 Python 都即时执行 → 易用但慢
2. Compiled Mode（torch.compile） | 把整个模型图编译成更快的代码 → 快

`model = torch.compile(model)` 内部执行的流程：

``` css
Python model
    ↓ (TorchDynamo tracing)
Graph captured
    ↓
Inductor (optimize)
    ↓
Codegen (Triton/CUDA/CPU)
    ↓
Optimized executable model
```

即：

1. 图捕捉（Tracing）／动态转换 —— 由 TorchDynamo 完成。

2. 反向传播及其图捕捉（AOT Autograd） —— 捕捉前向 + 反向的计算图。

3. 后端代码生成及执行优化 —— 由 TorchInductor 实现，将中间表示(IR)转换为高效的机器/加速器代码。

举例：

``` python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4096, 4096)
        self.fc2 = nn.Linear(4096, 4096)

    def forward(self, x):
        return torch.relu(self.fc2(torch.relu(self.fc1(x))))

model = MLP().cuda()
opt_model = torch.compile(model)
```

inductor 执行过程：

1. 捕获图

两层 FC → ReLU → FC → ReLU 

2. 融合
将
`matmul + add + relu`
融合成：
`gemm → add bias → relu → store`
并生成如下的 triton kernel：
``` python
@triton.jit
def fused_linear_relu_kernel(X, W, B, Y, ...):
    ...
    acc = tl.dot(X_block, W_block)
    acc += bias
    acc = tl.maximum(acc, 0)
    tl.store(Y_ptr, acc)
```

编译结果缓存 = 只在“当前进程内”有效

---

https://zhuanlan.zhihu.com/p/595996564

这篇 blog 讲的很好，但是AI 编译是一门大学问，只能点到为止了。