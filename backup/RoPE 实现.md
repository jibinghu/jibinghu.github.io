RoPE (Rotary Positional Embeddings) 计算实例

## 1. 设定与参数

Head 维度： $d = 4$（偶数） 
RoPE 基数： $\theta = 10000$ 
当前位置： $t = 3$ 
输入向量（Query 或 Key）：

$$\mathbf{x} = [x_0, x_1, x_2, x_3] = [1.0,\ 2.0,\ 3.0,\ 4.0]$$

RoPE 将向量按相邻元素两两分组：

- Pair 0: $(x_0, x_1) = (1.0, 2.0)$
- Pair 1: $(x_2, x_3) = (3.0, 4.0)$

---

## 2. 数值计算流程

第一步：计算频率 $\omega_i$

公式：$\omega_i = \theta^{-2i/d}$，其中 $i$ 表示第几对（从 0 开始）。

Pair 0 ($i=0$):
- $$\omega_0 = 10000^{-0} = 1$$
Pair 1 ($i=1$):
- $$\omega_1 = 10000^{-2/4} = 10000^{-0.5} = \frac{1}{100} = 0.01$$

第二步：计算旋转角度 $\phi_{t,i}$

公式：$\phi_{t,i} = t \cdot \omega_i$（位置 $\times$ 频率）。

- Pair 0: $\phi_{3,0} = 3 \times 1 = 3$ (rad)
- Pair 1: $\phi_{3,1} = 3 \times 0.01 = 0.03$ (rad)

第三步：执行二维旋转对每一对 $(x_{2i}, x_{2i+1})$ 应用二维旋转矩阵：

$$
\begin{pmatrix} x'_{2i} \\ x'_{2i+1} \end{pmatrix} =
\begin{pmatrix} \cos\phi & -\sin\phi \\ \sin\phi & \cos\phi \end{pmatrix}
\begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}
$$

三角函数值准备：

$\phi=3$: $\cos(3) \approx -0.9900$, 
$\sin(3) \approx 0.1411$$\phi=0.03$: 
$\cos(0.03) \approx 0.9996$, 
$\sin(0.03) \approx 0.0300$

计算过程：

Pair 0 (大幅度旋转):

$$\begin{aligned}
x'_0 &= 1 \cdot (-0.9900) - 2 \cdot (0.1411) = -1.2722 \\
x'_1 &= 1 \cdot (0.1411) + 2 \cdot (-0.9900) = -1.8389
\end{aligned}$$

Pair 1 (微小幅度旋转):

$$\begin{aligned}
x'_2 &= 3 \cdot (0.9996) - 4 \cdot (0.0300) = 2.8787 \\
x'_3 &= 3 \cdot (0.0300) + 4 \cdot (0.9996) = 4.0882
\end{aligned}$$

最终结果

$$\mathbf{x}' \approx [-1.2722,\ -1.8389,\ 2.8787,\ 4.0882]$$


---

## 3. 核心补充：为什么这么做？

### 3.1 复数视角的直观理解

如果把每一对 $(x_{2i}, x_{2i+1})$ 看作复数平面上的一个复数 $z_i$，RoPE 的操作本质上就是乘以一个模长为 1 的复数：

$$\text{RoPE}(x, t) = x \cdot e^{i \theta_t}$$

模长不变：RoPE 不改变向量的模长（Norm），只改变方向。这使得 token 的语义信息（幅度）得以保留，只注入了位置信息（角度）。长短周期：$i=0$ (低维) 频率高，旋转极快，捕捉短期位置依赖。$i=d/2$ (高维) 频率低，旋转极慢，捕捉长期位置依赖。



### 3.2 相对位置特性的由来

这是 RoPE 的核心优势。当我们计算 Query ($q$ 在位置 $m$) 和 Key ($k$ 在位置 $n$) 的内积（Attention Score）时：

$$\langle R(q, m), R(k, n) \rangle = (q e^{im\omega}) \cdot (k e^{in\omega})^* = q k^* e^{i(m-n)\omega}$$

结果只与 $(m-n)$ 有关，即只取决于两个 token 的相对距离，而与它们的绝对位置 $m$ 或 $n$ 无关。这就是为什么 RoPE 拥有良好的外推性（Extrapolation）。

---

## 4. 工程实现补充：Cache 与 优化

在实际的大模型推理（如 Llama, Qwen）中，我们**不会**在每次计算时实时调用 `powf` 和 `sin/cos`，而是采用 **预计算（Precomputation/Caching）** 策略。

### 优化后的 C++ 伪代码 (模拟生产环境)

```cpp
// 预计算阶段 (在模型初始化时做一次)
// cos_sin_cache 大小为 [max_seq_len, head_dim]
void precompute_freqs_cis(float* cache, int max_len, int dim) {
    for (int t = 0; t < max_len; ++t) {
        for (int i = 0; i < dim / 2; ++i) {
            float freq = powf(10000.0f, -2.0f * i / dim);
            float phi = t * freq;
            // 存储格式通常为 [cos, sin]
            cache[(t * dim + 2*i) + 0] = cosf(phi);
            cache[(t * dim + 2*i) + 1] = sinf(phi);
        }
    }
}

// 推理阶段 kernel (直接查表)
// x: 输入向量
// cache: 预计算好的 cos/sin 表
// t: 当前 token 的位置
// d: head 维度
__device__ void apply_rope_cached(float* x, const float* cache, int t, int d) {
    // 获取当前位置 t 对应的 cache 指针偏移量
    const float* current_pos_cache = cache + t * d;

    for (int i = 0; i < d/2; ++i) {
        // 直接读取，无昂贵的 pow/cos/sin 计算
        float c = current_pos_cache[2*i];
        float s = current_pos_cache[2*i + 1];

        float x0 = x[2*i];
        float x1 = x[2*i + 1];

        // 旋转
        x[2*i]     = x0 * c - x1 * s;
        x[2*i + 1] = x0 * s + x1 * c;
    }
}

```

### 补充说明

1. **Cache Size**: 对于 Context Window 很大的模型，Cache 可能会很大，因此显存优化（如 FlashAttention）通常会在 kernel 内部融合 RoPE 计算，减少显存读写。
2. **内存布局**: Llama 官方实现中，为了利用 SIMD 指令，有时会将向量切分为 `[x_0...x_{d/2-1}]` 和 `[x_{d/2}...x_{d-1}]` 对应旋转（Rotate Half），而非相邻的 `[x_0, x_1]`。虽然数学上等价，但如果是手写算子对接权重，需注意这一**维度排列（Permutation）**差异。