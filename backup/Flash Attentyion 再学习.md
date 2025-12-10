上篇粗略介绍了 Flash Attention v1 和 v2 版本，接下来依托 `https://zhuanlan.zhihu.com/p/668888063?share_code=1qZngTA5I8sYX&utm_psn=1981879033806468061` 和 `https://www.bilibili.com/video/BV1zM4m1S7gg/?spm_id_from=333.337.search-card.all.click&vd_source=84db1f19043807302b768c2fc15ba091` 来实现更透彻的实现，俗话说不求甚解，但是在技术的实现上一定需要手把手复现一遍才能有更充分的话语权来解决类似问题：


原始的 Standard Self-Attention 的实现在不考虑 Mask 和 Scale 的情况下：

<img width="387" height="56" alt="Image" src="https://github.com/user-attachments/assets/1b56204e-bcb8-4397-814c-cbe6038ae3cd" />

在这种情况下，对于单个 Head 实现可以有 3-pass 的计算步骤：

<img width="447" height="163" alt="Image" src="https://github.com/user-attachments/assets/cc39c023-c192-4caa-86b0-ef615c8b9c5f" />

访存的HBM IO Access 复杂度是 $4Nd + 4N^2 = O(Nd + N^2)$ 。这样的话当 seq_len 很大的时候，访存的压力会急剧变大。

FlashAttention利用了Tiling(forward)+Recompute(backward)对Attention计算进行融合，特别是对于forward阶段的tiling，可以看做是对online-softmax技术的一种延伸。从结果上来看，相对于原始的3-pass算法，online-softmax是2-pass算法，而FlashAttention是1-pass算法。

首先的话 safe softmax，因为 fp16 的指数位是 5，偏移之后的最大的指数位能到 $2^{5}-1 $ ，再加上尾数位，大概能到 65500(肯定不准) 这个量级。先不探讨 BF16(其实也会发生下溢)，而这就导致溢出，所以对于原生的 softmax，先减去一个 max 值，来保证计算过程中不会发生数值溢出。

原始的 softmax：

<img width="722" height="156" alt="Image" src="https://github.com/user-attachments/assets/67a8ead1-f918-4ed2-b283-3c41ef6c7aeb" />

由于 $x_i - m <= 0$ ，safe softmax 也就相当于分子分母同时除以最大值 m，所以数值相等：

<img width="959" height="121" alt="Image" src="https://github.com/user-attachments/assets/2958ccdf-5219-4dd1-887a-84fba6803597" />

那么对于 safe softmax 有 3-pass 的算法，重复三轮 N 次循环(下图)：

1. 第一轮递归找到最大的 $m = x_i$ 。
2. 第二轮递归加每一个减去最大值的 $e^{x_i - m_N}$ 。注意，此时因为 $m_N$ 是依赖于上一个循环的，所以二者不能并行。
3. 第三轮循环再计算每一个 softmax 值，这里同样要依赖于上边的 $d_N$ 值，所以不能并行。

<img width="450" height="268" alt="Image" src="https://github.com/user-attachments/assets/cddc9a65-b89a-4d2c-9a81-c1fcc12d9c57" />

这里的 x 是维度为 (seq_len, seq_len) 的 logits，所以如果 SRAM 不能保存 $O(N^2)$ 的矩阵，就要去 HBM 访问 Q 和 K 三次，非常低效。

所以把上图中的三个式子 fusion，但是正如我们刚刚所说的，式子之间相互有依赖，所以我们不得不引入其他计算量来抵消当前的依赖：

<img width="466" height="70" alt="Image" src="https://github.com/user-attachments/assets/1e2a6a75-6949-4a47-9e0f-f3ada6fe8761" />

其中 $e^{m_{i-1} - m_i}$ 是分母累加的调整因子，具体见下图。

<img width="1130" height="649" alt="Image" src="https://github.com/user-attachments/assets/90f86ded-4e24-4746-99ff-b76c45de4e94" />

其实和下图是一样的意思：

<img width="340" height="239" alt="Image" src="https://github.com/user-attachments/assets/2031fb1c-d2f4-4050-9dad-1398b9bc27bb" />

这样的话三个式子之间的依赖都依靠引入调整因子这个多余的计算量解决了。













参考：

https://link.zhihu.com/?target=https%3A//courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf

