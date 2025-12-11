上篇粗略介绍了 Flash Attention v1 和 v2 版本，接下来依托 `https://zhuanlan.zhihu.com/p/668888063?share_code=1qZngTA5I8sYX&utm_psn=1981879033806468061` 和 `https://www.bilibili.com/video/BV1zM4m1S7gg/?spm_id_from=333.337.search-card.all.click&vd_source=84db1f19043807302b768c2fc15ba091` 来实现更透彻的实现，俗话说不求甚解，但是在技术的实现上一定需要手把手复现一遍才能有更充分的话语权来解决类似问题：


## 原始的 Standard Self-Attention 的实现在不考虑 Mask 和 Scale 的情况下：

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

## Flash Attention v1：

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

<img width="1308" height="274" alt="Image" src="https://github.com/user-attachments/assets/4c11d3e6-398c-4217-bc17-63a0dc99096d" />

之前已经说过了 diag 左乘就是对矩阵的一行进行一致处理。

在 Flash Attention v2 中，就是将 online-softmax 分块的逻辑，将递归转成分块：

<img width="1163" height="375" alt="Image" src="https://github.com/user-attachments/assets/5ea2ba61-756c-4da2-91a2-acc402a27a2a" />

同样地，把 O 的计算考虑进来：

<img width="1047" height="105" alt="Image" src="https://github.com/user-attachments/assets/c23fdbc7-e8e4-4506-b8f6-ee09c5f80d6b" />

需要注意的是，FlashAttention的算法中有个Block Size的概念，也就是 $B_r$ 和 $B_c$ ：

<img width="621" height="134" alt="Image" src="https://github.com/user-attachments/assets/06754fdb-5062-4932-b56d-c9b1de9cc005" />

这样设置的目的是，为了确保SRAM能够放下所有Q, K, V的小块，其中 $M$ 就是系统可用的SRAM上限。那么，对于每一个Q 的分块 $Q_i, O_i$ ，以及K, V的分块 $K_j, V_j$ 需要的共享内存为：

$B_r$ 指的是一次从 Q 中读取的行数；
$B_c$ 指的是 K/V 方向的分块 Tile 的大小。

事实上，这里 $B_r$ 的大小要小于 $B_c$ 但是FA中将这部分四等分了。另外还有一个点：

<img width="248" height="86" alt="Image" src="https://github.com/user-attachments/assets/c5e7bc9b-b0c6-47b5-a2da-61320ccc9f27" />

> Br 是 tile 中一次处理的 Q 的行数（Br × d），而这些 Br 行必须能和 d 维 head 匹配到 GPU 的线程组织（warp）与 shared memory 访问模式中。
如果 Br > d，GPU 无法为这一 tile 建立合理的 2D thread mapping、寄存器排布和 shared memory swizzle，会浪费大量线程、产生 bank conflict、甚至无法形成 warp-level MMA 调度。

> 关于SRAM的认知，比如A100，我们常说，他的L1 Cache(SRAM)是192KB，这个值的颗粒度是SM，也就是每个SM都有192KB的SRAM，而A100有108个SM，因此，A100单卡上总共有20MB的SRAM。但是由于每个thread block只能被调度到一个SM上执行，SM之间的SRAM是不共享的。因此，实际算法设计时，考虑的是thread block的编程模型，要按照192KB去计算SRAM上能放的数据量。

FlashAttention backward pass最主要的优化就是：Recompute。对比Standard Self Attention，FlashAttention在前向不需要保留S和P矩阵，但是backward pass又需要S和P矩阵的值来计算梯度。那么怎么办呢？那自然就是就是和forward一样，利用Tiling技术，将Q,K,V分块load到SRAM，然后通过online recompute计算得到当前块的S和P值。具体到backward pass中计算逻辑就是：

<img width="1440" height="430" alt="Image" src="https://github.com/user-attachments/assets/8db3fa59-4ff3-4b59-864f-bfae46c71d43" />

另外，FA1 在反向阶段只有两种方式来实现：

1. 重新计算 S_blk（即 Q 与当前 K_block 再乘一次）
2. 利用 forward 中保留的标量 mᵢ 和 lᵢ 重建 softmax 值

由于我主要关注推理加速部分，所以关于训练我了解即可，总的来说引入了计算量，但是同样地被 IO 的减少所弥补。

## Flash Attention v2：

1. 减少大量非matmul的冗余计算，增加Tensor Cores运算比例
2. forward pass/backward pass均增加seqlen维度的并行，forward pass交替Q,K,V循环顺序
3. 更好的Warp Partitioning策略，避免Split-K（感觉这部分是为了故事完整加上的...）

传统地，回顾一遍 v1：

以K，V为外循环，Q为内循环。

j = 0，遍历 i：

<img width="1440" height="693" alt="Image" src="https://github.com/user-attachments/assets/6a88725c-324a-48e2-9eb0-37f3c8017ddc" />

j = 1，遍历 i：

<img width="1440" height="693" alt="Image" src="https://github.com/user-attachments/assets/ddab0769-cc77-41ed-8be9-e1cb2acc4e46" />

上图中实际上 O 在最后会累加，得到的只有三块 O。而且这里一定要注意区分 内循环和外循环，外循环就是固定了 K 和 V，内循环就是每行 Q 不断迭代。这个地方可能会有点晕，虽然按照流程来说 Q 是首先读取的，但实际中 QKV 是在一起读取的，所以可以将 Q 作为内循环来执行。

<img width="829" height="223" alt="Image" src="https://github.com/user-attachments/assets/417eae16-213b-43b2-b12b-5f8c9eeff9c6" />

上图中的讲解已经很清楚了， $O_{00}和O_{01}$ 是相关的数据，完全没必要再中间进行 HBM 的一次倒腾。所以我们就可以想到，把 Q 作为内循环转为 KV 作为内循环，这样就可以直接生成 $O_{00}和O_{01}$ ，而没必要像图中那样先将 $O_{00}, O_{10}, O_{20}$ 先生成出来没地方存，只能存到 HBM 中了。同时 Softmax 操作也是在 Row 维度的，所以这样循环会更方便。

<img width="1440" height="828" alt="Image" src="https://github.com/user-attachments/assets/6208028d-16e1-4358-8ab8-4743bf33915e" />

其实我们在 v1 的理解中就是以 v2 的角度来理解的(softmax 的实现)，v2 中把整个过程用 row 的方式实现了，所以不需要在最后进行 HBM 的额外存取。另外在第 10 行中可以看到并没有 $diag(l^{j}_i)$^{-1} $，这是为了减少非矩阵的计算。因为矩阵计算可以利用Tensor Cores加速，而不是用 CUDA Cores 来实现，加速比可以达到 16✖️。

<img width="822" height="170" alt="Image" src="https://github.com/user-attachments/assets/b4204976-f809-4522-906f-79d485bd9462" />

再者，v2 对cuda gemm层面优化：为什么相比于V1，V2在划分thread block时，要新增Q的seq_len维度上的划分呢？
先说结论，这样做的目的是尽量让SM打满。我们知道block是会被发去SM上执行的。以1块A100 GPU为例，它有108个SM，如果此时我们的block数量比较大（例如论文中所说>=80时），我们就认为GPU的计算资源得到了很好的利用。现在回到我们的输入数据上来，当batch_size和num_heads都比较大时，block也比较多，此时SM利用率比较高。但是如果我们的数据seq_len比较长，此时往往对应着较小的batch_size和num_heads，这是就会有SM在空转了。而为了解决这个问题，我们就可以引入在Q的seq_len上的划分。

v1 的 Thread Block：

<img width="1091" height="971" alt="Image" src="https://github.com/user-attachments/assets/e9a6c907-d4b3-4b50-b7d5-bb01a5e5c778" />

假设batch_size = 1，num_heads = 2，我们用不同的颜色来表示不同的head。我们知道在Multihead Attention中，各个head是可以独立进行计算的，在计算完毕后将结果拼接起来即可。所以我们将1个head划分给1个block，这样就能实现block间的并行计算，如此每个block只要在计算完毕后把结果写入自己所维护的O的对应位置即可。

v2 的 Thread Block：

<img width="1091" height="971" alt="Image" src="https://github.com/user-attachments/assets/36cd8d8d-2950-453a-8c5d-20e0dbc28ad0" />

现在我们继续假设batch_size = 1，num_heads = 2。与V1不同的是，我们在Q的seq_len维度上也做了切分，将其分成四份，即num_m_block = 4。所以现在我们共有1*2*4 = 8个block在跑。这些block之间的运算也是独立的，因为：

- head的计算是独立的，所以红色block和蓝色block互不干扰
- 采用Q做外循环，KV做内循环时，行与行之间的block是独立的，因此不同行的block互相不干扰。

每个block从Q上加载对应位置的切块，同时从KV上加载head0的切块，计算出自己所维护的那部分O，然后写入O的对应位置。

而在 前向和后向的过程中划分 Thread Block 的方式也会有所不同：

<img width="1440" height="825" alt="Image" src="https://github.com/user-attachments/assets/2c7f170e-86a1-444d-b905-eba363c07d3a" />

Warp 级别并行：

<img width="1280" height="518" alt="Image" src="https://github.com/user-attachments/assets/c6260ffd-6529-4171-b7ef-2356a654a4d3" />

上图中，左图表示 v1，右图表示 v2，不管是V1还是V2，在Ampere架构下，每个block内进一步被划分为4个warp，在Hopper架构下则是8个warp。

在左图（V1）中，每个warp都从shared memory上读取相同的Q块以及自己所负责计算的KV块。在V1中，每个warp只是计算出了列方向上的结果，这些列方向上的结果必须汇总起来，才能得到最终O矩阵行方向上的对应结果。所以每个warp需要把自己算出来的中间结果写到shared memory上，再由一个warp（例如warp1）进行统一的整合。所以各个warp间需要通讯、需要写中间结果，这就影响了计算效率。

在右图（V2）中，每个warp都从shared memory上读取相同的KV块以及自己所负责计算的Q块。在V2中，行方向上的计算是完全独立的，即每个warp把自己计算出的结果写到O的对应位置即可，warp间不需要再做通讯，通过这种方式提升了计算效率。不过这种warp并行方式在V2的BWD过程中就有缺陷了：由于bwd中dK和dV是在行方向上的AllReduce，所以这种切分方式会导致warp间需要通讯。








参考：

https://link.zhihu.com/?target=https%3A//courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf

https://zhuanlan.zhihu.com/p/691067658?share_code=1k9KBkbDbSt50&utm_psn=1982503349581521425