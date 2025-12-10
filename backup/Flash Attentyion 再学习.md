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

这样的话前两个式子之间的依赖都依靠引入调整因子这个多余的计算量解决了。

<img width="520" height="244" alt="Image" src="https://github.com/user-attachments/assets/c3ce6bf6-d57d-43bc-8d81-f62fe8040859" />

这里需要提醒，我们这里的 x 是 $QK^T$ ，维度为 (seq_len, seq_len)，我们避免了每次都需要将整个 $O(N^2)$ 的矩阵都从 HBM 中读写，而是分块来读取到共享内存中来实现。但是这里需要注意，我们在此只考虑了 softmax，对于 softmax 之前的 matmul 还需要考虑，而之前还有 $Q$ 和 $K^T$ 两个矩阵需要处理。

> 要么在算法中online计算，每次循环中去load一部分Q，K到片上内存，计算得到x。

这样我们就可以实现累加的并行了(通过分块和引入计算量来实现)，但是仍然需要得到 $d_N$ 之后才进行第二个循环(即分数的实现)，引入对 acc 的额外计算的话在事实上还是要再来一轮循环来调整做无用功，所以 softmax 事实上只能做到 2-pass 最优。但 Attention 可以做到 1-pass：

Flash Attention：

我们首先来看 Standard Self-Attention 是如何实现的：

<img width="663" height="293" alt="Image" src="https://github.com/user-attachments/assets/49aed06d-7246-4aae-8598-68db33418728" />

这是第一个循环， $Q 和 K^T$ matmul 得到 $x_i$ ，然后进行 online softmax (2-pass) 中的第一个循环，得到 softmax 的分子和分母；

<img width="729" height="144" alt="Image" src="https://github.com/user-attachments/assets/6add7bfc-46e2-4efb-a9b0-89550dbc1991" />

在第二个循环中根据 $d_N$ 得到最终的概率 (seq_len, seq_len)，注意这个时候因为 11 和 12 之间有依赖，最主要的是第二个循环依赖了 $m_N$ ，所以不能融合到第一个循环中。所以我们要像2-pass online softmax那样，找到 $o^'_i$ 与 $o^'_{i-1}$ 的不依赖于 $m_N$ 的递归关系。

1-pass Flash Attention：

首先定义：

<img width="214" height="63" alt="Image" src="https://github.com/user-attachments/assets/2e8e2a8d-c243-450b-ad06-c7b3b091afb7" />

可以看到他们还是依赖于 $m_N 和 d_N$ 的结束，但是可以用同样的 trick 来进行替代：

<img width="214" height="77" alt="Image" src="https://github.com/user-attachments/assets/87369abd-f3eb-4de4-b114-b0f1f1a52860" />

所以现在需要对概率也进行调整因子的调整以及加和。

<img width="564" height="240" alt="Image" src="https://github.com/user-attachments/assets/d5eb3d48-de39-4257-8b01-4c8f849fa837" />

如此之后，式子没有依赖地实现 1-pass Flash Attention 之后就可以 Tiling 来实现了：

<img width="545" height="153" alt="Image" src="https://github.com/user-attachments/assets/cdbf36d1-21e3-4f8d-9574-63f5ec34230a" />

每个 tile 内有 b 个 token，也就是 b 个 sel_len。接下来就以 tile 为单位进行计算：

<img width="634" height="331" alt="Image" src="https://github.com/user-attachments/assets/64994439-f18c-42fc-ae1d-59dd5d542847" />

注意，这里将 K 矩阵分成了多个块(实际上也可以切分 Q)，切分后的小块 load 到 SRAM 中，然后计算 $x_i$ ，接着进行剩余的计算。从算法逻辑上看，现在只需load Q,K,V一次，就能把Attention计算在kernel中全部完成。由3-pass的原始Self Attention，到1-pass 的FlashAttention，节省了S和P矩阵的显存，并且减少了Q,K的HBM IO Accesses。

<img width="380" height="445" alt="Image" src="https://github.com/user-attachments/assets/8fe1ed5f-de0b-4789-81b6-40fcf61ca88c" />

上图说明了 FlashAttention 在硬件上的计算方式。蓝色块代表位于 SRAM 中的块，而红色块对应第 i 行。L 表示序列长度，它可以非常大（例如 16k）；D 表示头维度，在 Transformer 中通常很小（例如 GPT3 的 128）；B 是可控的块大小。
值得注意的是，整体 SRAM 内存占用仅取决于 B 和 D，与 L 无关。因此，该算法可以扩展到长上下文而不会遇到内存问题（对于 H100 架构，GPU 共享内存很小，为 228kb/SM）。在计算过程中，我们从左到右遍历 KT 和 A 的块，从上到下遍历 V 的块，并相应地更新 m、d 和 O 的状态。

上图中，Q[i] 被放在 SRAM 中，在循环最外层；K_Tile和 V_Tile 被放在 SRAM 中。Q[i] 固定在最外层，对每个 K 的 tile 做乘法，得到一个 (1×B) 的 block logits；这个 block logits 立即参与在线 softmax，并与对应的 V tile (B×D) 融合相乘，累加到 (1×D) 的输出向量中；随着 K^T 和 V 沿着 L 方向逐 tile 滚动，最终完成一整行 O[i] 的计算。






参考：

https://link.zhihu.com/?target=https%3A//courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf

