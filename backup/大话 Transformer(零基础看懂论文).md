
---

###### 由于课程实验要求以及专业学习关系，之前学过Transformer但仅局限于会用，这次深入探讨一下Transformer以便为大模型推理加速打个基础。

**Transformer 简介：**

Transformer模型的核心特点包括：

1. **自注意力机制（Self-Attention Mechanism）**：能够捕捉序列中各个位置之间的依赖关系，无论它们距离多远。
2. **位置编码（Positional Encoding）**：因为Transformer不使用RNN或CNN，因此需要通过位置编码来保留序列信息。
3. **多头注意力（Multi-Head Attention）**：使模型可以在不同的子空间中学习不同的特征表示。
4. **前馈神经网络（Feed-Forward Neural Network）**：应用于每个位置的独立全连接层。

Transformer模型的工作流程如下：

- **编码器**：输入序列通过若干层堆叠的编码器，每层包含一个自注意力机制和一个前馈神经网络。编码器输出是一个固定长度的表示，每个位置对应于输入序列中的一个位置。
- **解码器**：解码器也由若干层组成，每层包括一个自注意力机制、一个对编码器输出的注意力机制和一个前馈神经网络。解码器利用编码器的输出和自身的先前输出生成目标序列。

---

## 一、Transformer 原文解析

### 整体架构：

<div style="text-align: center;">
    <img src="https://img2024.cnblogs.com/blog/3358182/202406/3358182-20240616185042244-1546828406.jpg" width="390" height="350">
    <p>The Transformer 整体架构</p>
</div>

> Transformer 是 seq2seq 模型，分为Encoder和Decoder两大部分，如上图，Encoder部分是由6个相同的encoder组成，Decoder部分也是由6个相同的decoder组成，与encoder不同的是，每一个decoder都会接受最后一个encoder的输出。

<div style="text-align: center;">
    <img src="https://ar5iv.labs.arxiv.org/html/1706.03762/assets/Figures/ModalNet-21.png" width="300" height="430">
    <p>The Transformer - model architecture</p>
</div>

模型大致分为**Encoder(编码器)** 和 **Decoder(解码器)** 两个部分，分别对应上图中的左右两部分。
- 其中编码器由N个相同的层堆叠在一起(我们后面的实验取N=6)，每一层又有两个子层。第一个子层是一个**Multi-Head Attention(多头的自注意机制)**，第二个子层是一个简单的**Feed Forward(全连接前馈网络)**。两个子层都添加了一个残差连接+layer normalization的操作。
- 模型的解码器同样是堆叠了N个相同的层，不过和编码器中每层的结构稍有不同。对于解码器的每一层，除了编码器中的两个子层**Multi-Head Attention和Feed Forward**，解码器还包含一个子层**Masked Multi-Head Attention**，如图中所示每个子层同样也用了residual以及layer normalization。
- 模型的输入由**Input Embedding**和**Positional Encoding(位置编码)** 两部分组合而成，模型的输出由Decoder的输出简单的经过softmax得到。结合上图，我们对Transformer模型的结构做了个大致的梳理，只需要先有个初步的了解，下面对提及的每个模块进行详细介绍：

### 1. **模型输入(Word2Vec_Positional_Embedding)：**
    首先我们来看模型的输入是什么样的，先明确模型输入，后面的模块理解才会更直观。输入部分包含两个模块，Embedding 和 Positional Encoding。

<div style="text-align: center;">
    <img src="https://pic4.zhimg.com/80/v2-b26e66a4cf78ba13a6e2bae03ef877eb_1440w.webp" width="500" height="250">
    <p>输入部分</p>
</div>

首先，我们需要把输入的文字进行Embedding，每一个字（词）用一个向量表示，称为字向量，一个句子就可以用一个矩阵表示。然后把字向量加上位置信息得到Encoder的输入矩阵。
> 其实位置信息Positional Encoding是固定公式计算出来的，值不会改变，每次有数据来了直接加上Positional Encoding矩阵就行。

- 由于模型不包含递归和卷积，为了使模型利用序列的顺序，必须注入一些关于序列中标记的相对或绝对位置的信息。为此，我们在编码器和解码器堆栈底部的输入嵌入中添加了“位置编码”。位置编码 $d_{model}$ 与嵌入具有相同的维度，因此可以将两者相加。位置编码有很多选择，有学习的，也有固定的。

$\text{Positional Encoding计算公式：}$

 $\text{当 } i \text{ 为偶数时，} \quad PE_{pos, i} = \sin\left(\frac{pos}{10000^{\frac{2i}{d_{model}}}}\right)$
 $\text{当 } i \text{ 为奇数时，} \quad PE_{pos, i} = \cos\left(\frac{pos}{10000^{\frac{2i}{d_{model}}}}\right)$

pos：表示一句话中的第几个字。如图 $p^{23}$：表示第2个字（0开始计数）。
i：字向量的第i维度。如图 $p^{23}$：第3维度。i为偶数用sin函数（0开始计数），i为奇数用cos函数（0开始计数）。
$d_{model}$：字向量一共有多少维。上图每个字Embedding\_size等于4（代码中取512）。
例如：上图中 $p^{12}$ 表示“是”字的第2维度，对应的值就等于 $PE_{1,2} = \sin\left(\frac{1}{10000^{2 \cdot 2 / 4}}\right)$
例如：上图中 $p^{23}$ 表示“学”字的第3维度，对应的值就等于 $PE_{2,3} = \cos\left(\frac{2}{10000^{2 \cdot 3 / 4}}\right)$

> [!IMPORTANT]
> 关于位置公式的思考一：正弦和余弦函数的这种交替使用能有效地捕捉位置信息，同时避免因周期性导致的位置编码重复问题。具体来说，尽管正弦和余弦函数具有周期性，但由于不同维度的频率不同，这种周期性不会在所有维度上同时发生重叠。正弦函数和余弦函数在奇偶维度上的交替使用进一步减少了冲突的可能性。

> [!IMPORTANT]
> 关于位置公式的思考二：缩放因子中常数的选择：**1.** 10000作为缩放因子的选择，部分是基于经验和实验结果。在实际应用中，研究人员发现10000这个数值能在不同长度的序列上提供稳定和良好的性能。研究者们在开发Transformer模型时进行了大量实验，选择了这个值作为一个平衡点。**2.** 数值稳定性：选择一个太小的常数可能导致计算出的频率太高，太大又会导致数值精度受影响，导致数值不稳定，特别是在高维空间中。10000这个值足够大，能确保数值计算的稳定性，同时频率范围也适中，不会导致计算中的数值问题。

---

- 这里词向量Embedding的具体解释：

> [!NOTE]    
> 词向量的生成过程是通过训练模型来学习单词的分布式表示，使得这些向量可以在向量空间中反映单词之间的语义关系。将单词转换为词向量（word embedding）的过程是一种将离散的文本数据转化为连续的向量表示的技术，以便计算机可以理解和处理自然语言。具体的过程如下：

##### 1. **词汇表构建**
首先，构建一个包含所有要处理的单词的词汇表。词汇表中的每个单词都分配一个唯一的索引。

##### 2. **初始化词向量**
为每个单词初始化一个固定维度的向量（通常是随机初始化的），这些向量称为嵌入向量。初始向量的维度可以是50、100、300等，根据具体应用和计算资源来选择。

##### 3. **模型训练**
通过训练模型来调整这些初始向量，使它们能够捕捉单词之间的语义关系。常用的模型包括：

- **Word2Vec**：
  - **Skip-Gram**：目标是预测给定单词的上下文单词。
  - **CBOW（Continuous Bag of Words）**：目标是根据上下文预测中心词。
- **GloVe（Global Vectors for Word Representation）**：通过统计全局词共现矩阵来训练词向量。
- **FastText**：考虑单词的字符n-gram信息，有助于处理未见过的单词和拼写错误。

##### 4. **损失函数**
使用适当的损失函数来衡量模型的预测效果，并通过反向传播算法不断调整词向量以最小化损失。

##### 5. **词向量生成**
在训练过程中，模型不断调整词向量，使得语义相似的单词在向量空间中的距离更近。训练完成后，每个单词都对应一个固定的向量表示。

##### 具体例子

###### Word2Vec Skip-Gram模型

1. **输入**：中心词（target word）
2. **目标**：预测上下文词（context words）

例如，给定句子“the quick brown fox jumps over the lazy dog”，假设中心词是“quick”，上下文窗口大小为2，则上下文词是“the”和“brown”。

训练过程中模型会更新“quick”的向量，使其能够更好地预测“the”和“brown”。

###### GloVe模型

1. **输入**：词对共现矩阵
2. **目标**：最小化每个词对共现概率的误差

通过统计每对词在全局语料库中的共现次数，GloVe模型能够捕捉全局的词汇信息，并将这些信息转化为词向量。

##### 词向量应用

训练好的词向量可以用于各种NLP任务，例如文本分类、情感分析、命名实体识别等。此外，词向量还可以进行向量运算，例如：
- 向量(king) - 向量(man) + 向量(woman) ≈ 向量(queen)

这种运算反映了词向量捕捉到的词语之间的语义关系。

---        

### 2. 编码器层(注意力机制)：

> Transformer通过编码器对输入序列进行处理，最终得到了每个词的上下文敏感表示（contextualized representation）。这些表示不仅包含了每个词本身的信息，还包含了序列中其他词的信息，以及它们之间的依赖关系。具体来说，编码器的输出是一个与输入序列长度相同的序列，但每个位置的向量表示都已经综合了整个输入序列中的信息。

#### 1. Multi-Head Attention
    > 注意力函数可以描述为将查询和一组键值对映射到输出，其中查询、键、值和输出都是向量。输出计算为值的加权总和，其中分配给每个值的权重由查询与相应键的兼容性函数计算。
        为了方便理解，先解释单头注意力：

<div style="text-align: center;">
    <img src="https://pic1.zhimg.com/80/v2-6f99eb9935e048106d862d880c0f911c_1440w.webp" width="600" height="300">
    <p>Self Attention Mechanism_b_0</p>
</div>

输入部分输入含有位置信息的字向量 $a^i$，作为Self Attention Mechanism的输入。$a^i$ 可以理解成一句话中第 $i$ 个字的字向量。$a^i$ 会分别乘以 $W^Q, W^K, W^V$ 三个矩阵（矩阵乘法），得到 $q^i, k^i, v^i$。$q$ 可以理解成词的“查询”向量；$k$ 可以理解成词的“被查”向量；$v$ 可以理解成词的“内容”向量。下面就是计算每一个字的注意力信息。

<div style="text-align: center;">
    <img src="https://ar5iv.labs.arxiv.org/html/1706.03762/assets/Figures/ModalNet-19.png" width="200" height="400">
    <p>Scaled Dot-Product Attention</p>
</div>

例如，上图，我们在计算第0个字与句子中所有字的注意力信息。用 $q^0$ 分别乘以 $k^0, k^1, k^2, k^3$（对应坐标相乘相加），得到4个常数 $a_{00}, a_{01}, a_{02}, a_{03}$ 注意力值，再把4个数经过Softmax，得到第0个字与句子中所有字的注意力分数 $\hat{a}_{00}, \hat{a}_{01}, \hat{a}_{02}, \hat{a}_{03}$，它们和为1，最后再用注意力分数乘以对应的字信息 $v_0, v_1, v_2, v_3$，得到第0个字句子加权信息
$$b^0 = \hat{a}_{00} * v_0 + \hat{a}_{01} * v_1 + \hat{a}_{02} * v_2 + \hat{a}_{03} * v_3$$
实际就是矩阵相乘（下文）。

<div style="text-align: center;">
    <img src="https://pic1.zhimg.com/80/v2-5761c84bfcbe44ad24e80c792ad01368_1440w.webp" width="600" height="300">
    <p>Self Attention Mechanism_b_1</p>
</div>

如此进行，将第1、2……n个字的加权信息得出。

---

关于Q，K，V的解释：

1. 查询（Query，Q）
作用：查询向量用来表示我们要“查找”的信息。

举例：假设我们正在处理一个句子，并且我们希望了解当前词与句子中其他词的相关性。那么当前词的查询向量Q就是用来表示这个词的特定需求或视角。

2. 键（Key，K）
作用：键向量表示所有可能匹配的内容。

举例：在处理一个句子时，每个词都会有一个键向量K。这个向量表示这个词在句子中的某种特征或位置，类似于数据库中的索引，用于匹配查询向量。

3. 值（Value，V）
作用：值向量是与键向量对应的具体内容，它表示实际的信息。

举例：在句子中，每个词也会有一个值向量V。这个向量包含了这个词的实际内容信息，当查询匹配到键时，我们提取相应的值。

即：
- 查询向量Q：表示我们希望查找的内容或视角。
- 键向量K：表示所有可能的匹配特征，用于与查询向量进行匹配。
- 值向量V：表示实际的信息内容，在匹配后提取相关的信息。
通过这种机制，注意力机制能够有效地捕捉到序列中各部分之间的关系，并在需要时将注意力集中到相关部分，从而提高模型的表现。

---

**自注意力机制在计算机中的实现：**

上述自注意力计算过程是朴素过程，在计算机实现中由于计算机对矩阵处理的天然敏感性，可以将上述计算过程打包为矩阵计算的一系列过程，这里我直接将'引用文章'中的过程贴出来，详细过程可以去'引用一'中查看。

<div style="text-align: center;">
    <img src="https://pic4.zhimg.com/80/v2-aaff5c3e45c0af89e2f46673ae2f96cf_1440w.webp" width="600" height="300">
    <p>矩阵思想计算q，k，v</p>
</div>

如上图，我们直接用字向量 $a^i$ 矩阵，分布乘以 $W^Q, W^K, W^V$ 三个矩阵（矩阵乘法）形状[字向量长度，字向量长度]，得到 $q^i, k^i, v^i$ 三个矩阵。

<div style="text-align: center;">
    <img src="https://pic2.zhimg.com/80/v2-6e4378b21175aaea57585d6e0929e27d_1440w.webp" width="600" height="150">
    <p>计算注意力值</p>
</div>

再用 $q$ 矩阵成以 $k$ 矩阵得到注意力值 $a$ 矩阵。计算公式：$\alpha_{(i,j)} = \frac{qk^T}{\sqrt{d}}$。$k^T$: $k$ 的转置。$d$ 是输入的维度，上图矩阵 $d$ 等于4，下面代码 $d=512$。

> [!TIMPORTANT]
> 关于计算公式的思考：通过对维度 $\sqrt{d_k}$ 进行缩放，将点积值 $QK^T$ 缩放到一个合适的范围，使得Softmax 函数在计算时的输入值以及输出的梯度更合适，避免数值过大导致的数值不稳定性。

<div style="text-align: center;">
    <img src="https://pic1.zhimg.com/80/v2-8b191b05ff4e291536f56144ab1b0ce8_1440w.webp" width="450" height="150">
    <p>注意力分数</p>
</div>

矩阵 $a$ 再以每一行，经过Softmax计算出注意力分数矩阵 $\hat{a}_{ij}$（每一行的值加起来等于1）。公式：
$$\hat{a}_{(i,j)} = \frac{\alpha_{(i,j)}}{\sum_{j=0}^{s} \alpha_{(i,j)}}, \quad s = 3.$$

<div style="text-align: center;">
    <img src="https://pic4.zhimg.com/80/v2-4ebe5925fea4edd4e5cf3f9581430d37_1440w.webp" width="650" height="150">
    <p>注意力分数乘以V向量信息</p>
</div>

用注意力分数 $\hat{a}_{ij}$ 矩阵乘以 $v^j$ 矩阵得到输出 $b^i$ 矩阵，
$$b^0 = \hat{a}_{00} * v_0 + \hat{a}_{01} * v_1 + \hat{a}_{02} * v_2 + \hat{a}_{03} * v_3,$$
实际上就是注意力分数矩阵相乘和 $V$ 矩阵实际求加权和（这里有点绕）。以上就是完整计算注意力机制过程。

综合公式为： $$Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V $$

---

**多头注意力机制：**

<div style="text-align: center;">
    <img src="https://pic2.zhimg.com/80/v2-7eb8e13a290f42acecb00bce2db0091d_1440w.webp" width="700" height="320">
    <p>Multi-Head Attention</p>
</div>

> 论文中使用的是Multi-Head Attention，它与单头注意力不同的就是把原来的 $Q, K, V$ 三个大矩阵拆分成8个形状相同的小矩阵（在特征维度上拆分），也就是8头注意力。
> 
> 每一个小矩阵的形状 [句子字符的个数，子向量维度/8]，在下面代码中句子长度为5，子向量维度512，小矩阵是 [5, 64=512/8]。用上面3.2节相同的方式计算8个 $b$ 矩阵，然后把每一个head-Attention计算出来的 $b$ 拼在一起，作为输出。输入 $a$ 和输出 $b$ 的形状是相同的。为了方便画图，上图以2头注意力机制为例。整个流程从左到右。$X$ 表示矩阵相乘。$a$ 矩阵的shape：[句子长度，句子长度] 表示每两句话之间都有一个注意力分数，Concat：矩阵拼接。由于每个头的尺寸都缩小了，总计算成本与全维的单头注意力相似。

<div style="text-align: center;">
    <img src="https://ar5iv.labs.arxiv.org/html/1706.03762/assets/Figures/ModalNet-20.png" width="280" height="420">
    <p>Multi-Head Attention 多头注意力</p>
</div>

多头注意力机制通过引入多个“头”来扩展单头注意力机制，每个头独立地进行注意力计算，然后将结果合并。具体步骤如下：

1. **线性变换**：对输入的Q、K、V进行线性变换，生成多个查询、键和值。通常情况下，这些线性变换的参数是不同的，从而产生不同的Q、K、V。
   对于第i个头：$$ Q_i = QW_i^Q, \quad K_i = KW_i^K, \quad V_i = VW_i^V $$
   其中，\(W_i^Q\)、\(W_i^K\)、\(W_i^V\) 是线性变换的权重矩阵。
2. **计算注意力**：每个头独立地计算注意力：
  $$ \text{head}_i = \text{Attention}(Q_i, K_i, V_i) = \text{softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right)V_i $$
1. **合并头**：将所有头的输出拼接在一起：
$$ \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O $$
   其中，$ W^O $ 是输出的线性变换矩阵。

---

#### 2. **Add & Layer Normalization**

<div style="text-align: center;">
    <img src="https://pic4.zhimg.com/80/v2-12da19452d7b9b7be2ba8f4f3fa6ee5b_1440w.webp" width="420" height="300">
    <p>残差链接_层归一化</p>
</div>

现在解释蓝色圈起来的部分，Add是用了残差神经网络的思想，也就是把Multi-Head Attention的输入的 $a$ 矩阵直接加上Multi-Head Attention的输出 $b$ 矩阵（好处是可以让网络训练的更深）得到的和 $\bar{b}$ 矩阵，再在经过Layer normalization（归一化，作用加快训练速度，加速收敛）把 $\bar{b}$ 每一行（也就是每个句子）做归一为标准正态分布，最后得到 $\hat{b}$。均值和方差如下公式：

均值公式：$\mu_i = \frac{1}{s} \sum_{j=1}^{s} b_{ij}, s \text{ 为 } \bar{b}_i \text{ 的长度。}$
方差公式：$\sigma_i^2 = \frac{1}{s} \sum_{j=0}^{s} (b_{ij} - \mu_i)^2$
归一化公式：$\text{Layer Norm}(x) = \frac{b_{ij} - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}} \cdot \gamma + \beta$

这里的层归一化指的是将句子中所有字的同一维度进行归一；

<div style="text-align: center;">
    <img src="https://pic4.zhimg.com/80/v2-d5ed6f44d934f525cb1b0d638fbb6fe3_1440w.webp" width="620" height="200">
    <p>归一化</p>
</div>

> 在Transformer模型中，经过多头注意力机制得到的输出矩阵 \( b \) 会进一步通过残差连接（Residual Connection）和层归一化（Layer Normalization），以增强模型的训练效果和稳定性。下面详细解释这一过程。

**1. 残差连接（Residual Connection）**

残差连接是指将输入直接加到输出上，形成一个“跳跃连接”（skip connection）。对于多头注意力机制的输出矩阵 \( b \)，残差连接的公式为：
$$ \text{Output}_\text{residual} = \text{Input} + b $$

其中，$\text{Input}$是多头注意力机制的输入，即初始的词向量表示。

**2. 层归一化（Layer Normalization）**

层归一化是对输入的每一层进行标准化处理，以提高训练的稳定性和效率。与批归一化（Batch Normalization）不同，层归一化是在每一个时间步上进行归一化，而不是在整个批次上。

层归一化的公式为：
$$ \text{Output}_\text{layernorm} = \frac{\text{Output}_\text{residual} - \mu}{\sigma + \epsilon} \gamma + \beta $$

其中：
- $\mu$ 是输入的均值： $\mu = \frac{1}{d_{model}} \sum_{i=1}^{d_{model}} \text{Output}_\text{residual}^i $
- $\sigma$ 是输入的标准差： $\sigma = \sqrt{\frac{1}{d_{model}} \sum_{i=1}^{d_{model}} (\text{Output}_\text{residual}^i - \mu)^2}$
- $\epsilon$ 是一个很小的常数，用于防止除零。
- $\gamma$ 和 $\beta$ 是可学习的参数，用于对标准化后的值进行缩放和平移。

---

#### 3. **前馈神经网络（Feed Forward Neural Network, FFN）**

在Transformer编码器的结构中，每一个编码器层（Encoder Layer）由两个主要子层组成：多头注意力机制（Multi-Head Attention）和前馈神经网络（Feed Forward Neural Network, FFN）。在经过多头注意力机制和层归一化（Layer Normalization）后，编码器层会通过前馈神经网络来进一步处理特征。前馈神经网络的输出是经过多头注意力机制处理和层归一化后的词向量的进一步非线性变换。

### 前馈神经网络（FFN）

前馈神经网络通常包括两个线性变换和一个ReLU激活函数。公式如下：

$$ \text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2 $$

其中：
- $x$ 是输入的词向量。
- $W_1$ 和 $W_2$ 是可学习的权重矩阵。
- $b_1$ 和 $b_2$ 是偏置向量。
- $\max(0, \cdot)$ 表示ReLU激活函数。

###### 具体过程

1. **输入变换**：
   - 输入 \( x \) 经过第一个线性变换和ReLU激活函数：
    $$h = \max(0, xW_1 + b_1)$$
   - 这一步引入了非线性，允许模型学习到更复杂的特征表示。

2. **输出变换**：
   - 中间表示 \( h \) 经过第二个线性变换，得到最终输出：
    $$\text{FFN}(x) = hW_2 + b_2$$

###### 残差连接和层归一化

前馈神经网络的输出也会经过残差连接和层归一化：

1. **残差连接**：
   - 将前馈神经网络的输出与输入相加：
    $$y = x + \text{FFN}(x)$$

2. **层归一化**：
   - 对相加后的结果进行层归一化：
    $$\text{Output} = \text{LayerNorm}(y)$$

###### 输出结果的形式

最终的输出依然是词向量的表示，形状保持为 $[n, d_{model}]$，其中 $n$ 是序列长度，$d_{model}$ 是词向量的维度。这个输出可以被输入到下一个编码器层或者解码器中，作为进一步处理的输入。

---

### 3. 解码器层

> 在Transformer解码器的实现中，每个解码器由6个相同的层组成。除了每个编码器层中的两个子层外，解码器还插入了第三个子层，该子层对编码器堆栈的输出执行多头注意力机制。与编码器类似，每个子层都采用残差连接，然后进行层归一化。此外，解码器堆栈中的自注意力子层进行了修改，以防止关注后续位置。这种掩码机制，加上输出嵌入偏移一个位置的事实，确保了对位置 \(i\) 的预测只能依赖于小于 \(i\) 的已知输出。

##### 结构和流程

1. **自注意力机制（带掩码）**：
   - 解码器的第一个子层是多头自注意力机制，但增加了掩码，以确保模型在预测位置 \(i\) 的词时，只能看到位置 \(i\) 之前的词。掩码是一个上三角矩阵，使得注意力权重在计算时被限制在当前位置及其之前的词。

2. **编码器-解码器注意力机制**：
   - 解码器的第三个子层是**多头注意力机制，这个子层将解码器的当前输出作为查询（Q），编码器的输出作为键（K）和值（V）**，从而在解码过程中结合编码器的上下文信息。

3. **前馈网络**：
   - 解码器的第二个子层与编码器的第二个子层相同，是一个位置全连接的前馈网络，包含两个线性变换层和一个非线性激活函数。

4. **残差连接和层归一化**：
   - 每个子层都使用残差连接，将输入直接跳过子层并与子层的输出相加，然后进行层归一化。具体公式为：
     $$\text{LayerNorm}(x + \text{Sublayer}(x))$$
---
##### 1. 掩码自注意力机制

Masked Multi-Head Attention 是 Transformer 解码器中的一个关键组件，它与普通的多头自注意力机制类似，但增加了掩码机制（masking），以确保模型只能关注当前时间步之前的词，从而避免模型在训练时看到未来的信息。

以下是 Masked Multi-Head Attention 的具体实现步骤以及与普通多头自注意力机制的区别：

##### Multi-Head Attention 实现步骤

1. **输入嵌入与线性变换**：
   - 输入序列首先被嵌入为向量表示，然后通过三个不同的线性变换层，分别生成查询（Q）、键（K）和值（V）向量。假设输入序列长度为 $L$，每个词的嵌入维度为 $d_{model}$，则这一步生成的 Q、K、V 矩阵的维度均为 $L \times d_{model} $。

2. **计算注意力权重**：
   - 使用查询和键向量计算注意力权重矩阵。注意力权重矩阵的计算公式为：
     $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
     其中 $d_k$ 是键向量的维度，通常与查询向量的维度相同。

3. **应用掩码（仅限 Masked Multi-Head Attention）**：
   - 在计算注意力权重之前，对权重矩阵应用掩码（mask），确保模型只能关注当前时间步之前的词。掩码矩阵通常是一个上三角矩阵，其中上三角部分的值为负无穷（或大负数），其余部分为零，这样在 softmax 计算时，被掩盖的部分权重接近于零。
     $$ \text{MaskedAttention}(Q, K, V) = \text{softmax}\left(\frac{QK^T + \text{mask}}{\sqrt{d_k}}\right)V$$

4. **计算多头注意力**：
   - 将 Q、K、V 矩阵分为多个头（head），每个头分别计算注意力，然后将所有头的输出拼接在一起，通过一个线性变换层生成最终的多头注意力输出。假设有 \( h \) 个头，每个头的维度为 \( d_{head} = \frac{d_{model}}{h} \)，则各个头的计算过程为：
     $$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$
     其中 \( \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V) \)。

5. **输出处理**：
   - 将多头注意力的输出进行线性变换，并加入残差连接和层规范化。

##### Masked Multi-Head Attention 与 Multi-Head Attention 的区别

- **掩码机制**：
  - Masked Multi-Head Attention 在计算注意力权重之前应用了掩码矩阵，确保模型在预测下一个词时只能关注当前时间步之前的词。这在训练语言模型时非常重要，因为模型在预测序列中的每个词时不应该看到未来的词。
  - 普通的 Multi-Head Attention 没有这种限制，可以直接计算所有词之间的注意力权重。

- **使用场景**：
  - Masked Multi-Head Attention 主要用于解码器部分，用于生成序列时确保因果关系。
  - 普通的 Multi-Head Attention 可以用于编码器部分或解码器的编码-解码注意力部分，没有时间步的限制。

##### 代码示例

以下是一个简化版的 PyTorch 代码示例，展示了 Masked Multi-Head Attention 的实现：

```python
import torch
import torch.nn.functional as F
from torch import nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % num_heads == 0
        self.depth = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, v, k, q, mask):
        batch_size = q.size(0)

        q = self.split_heads(self.wq(q), batch_size)
        k = self.split_heads(self.wk(k), batch_size)
        v = self.split_heads(self.wv(v), batch_size)

        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = scaled_attention.permute(0, 2, 1, 3).contiguous()
        original_size_attention = scaled_attention.view(batch_size, -1, self.d_model)
        output = self.dense(original_size_attention)

        return output, attention_weights

    def scaled_dot_product_attention(self, q, k, v, mask):
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))
        dk = k.size()[-1]
        scaled_attention_logits = matmul_qk / torch.sqrt(torch.tensor(dk, dtype=torch.float32))

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = F.softmax(scaled_attention_logits, dim=-1)
        output = torch.matmul(attention_weights, v)

        return output, attention_weights

# Usage example
d_model = 512
num_heads = 8
batch_size = 64
seq_length = 50

# Dummy input
q = torch.rand(batch_size, seq_length, d_model)
k = torch.rand(batch_size, seq_length, d_model)
v = torch.rand(batch_size, seq_length, d_model)
mask = torch.tril(torch.ones(seq_length, seq_length))  # Lower triangular matrix

mha = MultiHeadAttention(d_model, num_heads)
output, attn_weights = mha(v, k, q, mask)
```

在这个实现中，掩码 `mask` 用于确保在计算注意力权重时只关注当前时间步之前的词。这使得模型在训练时能够正确地学习序列中的因果关系。

---

关于掩码矩阵 Mask Matrix：

掩码矩阵的构成是理解Masked Multi-Head Attention的关键。掩码矩阵用于屏蔽解码器自注意力机制中的未来位置，确保模型在预测下一个词时只能看到当前及之前的词。

##### 掩码矩阵的构成

掩码矩阵通常是一个上三角矩阵，其形状与注意力权重矩阵相同，为 $L \times L$，其中 $L$ 是输入序列的长度。具体构成如下：

1. **生成上三角矩阵**：
   - 该矩阵的对角线和下三角部分元素为0，上三角部分元素为1。表示形式如下：
     $$\text{mask}[i, j] = \begin{cases} 
     0 & \text{if } i \geq j \\
     1 & \text{if } i < j 
     \end{cases}$$
     例如，对于一个长度为4的序列，其掩码矩阵为：
     $$\text{mask} = \begin{bmatrix}
     0 & 1 & 1 & 1 \\
     0 & 0 & 1 & 1 \\
     0 & 0 & 0 & 1 \\
     0 & 0 & 0 & 0 
     \end{bmatrix}$$

2. **掩码应用**：
   - 在计算注意力权重时，将掩码矩阵添加到未归一化的注意力权重矩阵上。由于注意力权重矩阵中的值通常在对数空间中，为了有效地屏蔽未来的位置，掩码矩阵中的1会被转换成一个非常大的负数（通常是负无穷大 $-\infty$），以确保在softmax计算中这些位置的权重接近于0。

##### 掩码矩阵影响时间步的机制

1. **计算未归一化的注意力权重**：
   - 使用查询 $Q$ 和键 $K$ 计算未归一化的注意力权重：
     $$\text{scores} = \frac{QK^T}{\sqrt{d_k}}$$
   - 这个矩阵的形状为 $L \times L$。

2. **应用掩码**：
   - 将掩码矩阵添加到注意力权重矩阵上：
     $$\text{masked\_scores} = \text{scores} + \text{mask} \times (-\infty)$$
   - 由于掩码矩阵的上三角部分是1，因此这些位置的权重在加上一个非常大的负数后，变得非常小，在softmax计算时几乎等于0。

3. **计算归一化的注意力权重**：
   - 应用softmax函数得到归一化的注意力权重：
     $$\text{attention\_weights} = \text{softmax}(\text{masked\_scores})$$
   - 由于掩码的作用，模型只能关注当前及之前的时间步，未来的时间步被有效地屏蔽了。

##### 代码示例

以下是一个简化的代码示例，展示了如何构建和应用掩码矩阵：

```python
import torch
import torch.nn.functional as F

def create_mask(seq_length):
    mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1)
    return mask

def masked_attention(Q, K, V, mask):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    if mask is not None:
        scores = scores.masked_fill(mask == 1, float('-inf'))
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)
    return output, attention_weights

# Example usage
seq_length = 4
Q = torch.rand(seq_length, seq_length)
K = torch.rand(seq_length, seq_length)
V = torch.rand(seq_length, seq_length)
mask = create_mask(seq_length)

output, attn_weights = masked_attention(Q, K, V, mask)
```

在这个示例中，`create_mask` 函数生成了一个上三角掩码矩阵，`masked_attention` 函数在计算注意力权重时应用了该掩码矩阵，确保未来的时间步不会影响当前的预测。

##### 总结

掩码矩阵通过屏蔽未来位置，确保模型在预测当前位置时只能利用当前及之前的位置信息。这种机制在序列生成任务中至关重要，确保了模型的因果性。掩码矩阵与未归一化的注意力权重矩阵相加，使得未来位置的权重在softmax归一化后接近于0，从而有效地实现时间步的控制。

---
> [!IMPORTANT]
> 关于词向量的思考：
单纯的词向量训练（如Word2Vec、GloVe）能够在一定程度上捕捉语义和上下文信息，但在表达复杂的语言特性如文化背景、隐喻和情感等方面有其局限性。以下是详细的分析：

##### 词向量的优点和局限性

1. **优点**：
   - **语义相似性**：词向量通过在大规模语料上训练，可以捕捉到词与词之间的语义相似性。相似意义的词在向量空间中通常会靠近，例如“king”和“queen”之间的向量关系与“man”和“woman”之间的关系类似。
   - **上下文信息**：词向量能够捕捉到一定的上下文信息，特别是在基于上下文窗口的训练方法（如Word2Vec的Skip-gram和CBOW模型）中。词在相似上下文中出现的频率越高，其向量表示越相似。

2. **局限性**：
   - **缺乏上下文动态性**：传统的词向量方法无法动态地根据上下文改变词的表示。一个词在不同的上下文中可能有不同的含义，但其词向量是固定的。例如，“bank”在“金融机构”和“河岸”的不同语境中应该有不同的表示，但词向量模型无法区分这一点。
   - **文化背景和隐喻**：词向量难以捕捉深层次的文化背景和隐喻表达。这类信息通常依赖于更复杂的句法结构和语义关系，传统的词向量方法在这方面能力有限。
   - **情感和语气**：词向量对情感和语气的捕捉能力也有限。虽然某些情感词可能在向量空间中有所体现，但对整体句子的情感理解需要考虑更多的上下文和词间关系。

##### 进阶方法和模型

为了解决这些局限性，近年来发展了许多进阶方法和模型：

1. **上下文嵌入（Contextual Embeddings）**：
   - **ELMo**：通过双向LSTM在大规模语料上训练，生成的词向量能够根据上下文动态变化。一个词在不同句子中的表示会不同，能够更好地捕捉上下文信息。
   - **BERT**：基于Transformer的双向编码器表示，通过掩蔽语言模型（Masked Language Model）训练，能够更好地理解句子结构和上下文关系。BERT的词向量在不同的上下文中是动态变化的，极大地提升了对语义和上下文的理解能力。

2. **预训练语言模型**：
   - **GPT**：基于Transformer的生成预训练模型，通过大量文本的自回归训练，能够生成符合上下文的连贯文本，并捕捉复杂的语义和情感信息。
   - **T5**：将所有任务转换为文本生成任务，统一的模型架构使得其在多种NLP任务中表现出色。

3. **情感分析和情感词典**：
   - 使用专门的情感分析模型和情感词典，可以更精确地捕捉文本中的情感和语气。这些方法结合上下文嵌入技术，能够更好地理解句子的情感含义。

4. **跨文化语料和多语言模型**：
   - 通过多语言预训练模型（如mBERT、XLM-R），能够在跨语言和文化背景下训练模型，捕捉更广泛的文化背景和语义差异。

---

##### 2. Multi-Head Attention

Encoder 的 Multi-Head Attention 的结构和 Decoder 的 Multi-Head Attention 的结构也是一样的，只是 Decoder 的 Multi-Head Attention 的输入来自两部分，K，V 矩阵来自Encoder的输出，Q 矩阵来自 Masked Multi-Head Attention 的输出。


##### 3. Decoder 的输出预测结果

Decoder 的输出的形状[句子字的个数，字向量维度]。可以把最后的输出看成多分类任务，也就是预测字的概率。经过一个nn.Linear(字向量维度, 字典中字的个数)全连接层，再通过Softmax输出每一个字的概率。

---

对 BERT 的扩展：

BERT（Bidirectional Encoder Representations from Transformers）是一个基于Transformer的双向编码器表示模型，由Google的研究团队在2018年提出。它通过掩蔽语言模型（Masked Language Model，MLM）进行训练，从而能够更好地理解句子结构和上下文关系。以下是对BERT的具体解释：

##### BERT的核心思想

1. **双向编码器表示**：
   - 与传统的单向语言模型不同，BERT是双向的，它在处理每个词时，能够同时考虑其左侧和右侧的上下文。这种双向性使得BERT在理解词的意义时，可以参考整个句子的上下文，而不仅仅是词的前面或后面的部分。

2. **掩蔽语言模型（Masked Language Model，MLM）**：
   - 在训练过程中，BERT使用了一种称为掩蔽语言模型的方法。具体来说，它随机地掩盖（mask）句子中的一些词，然后要求模型根据上下文预测这些被掩盖的词。
   - 例如，对于句子“我爱[MASK]编程”，模型需要根据上下文预测出被掩盖的词“学习”。

3. **下一句预测（Next Sentence Prediction，NSP）**：
   - 除了MLM，BERT还使用了下一句预测任务来训练模型。这个任务是给定一对句子，模型需要判断第二个句子是否是第一个句子的自然后续句。
   - 这种任务帮助模型理解句子之间的关系，从而提升在句子级别的理解能力。

##### BERT的训练和结构

1. **训练数据**：
   - BERT在大规模的语料库（如Wikipedia和BooksCorpus）上进行预训练。这些语料库包含了大量的文本数据，为模型提供了丰富的上下文信息。

2. **模型结构**：
   - BERT的基础结构是一个多层的Transformer编码器。常见的BERT模型有两种尺寸：BERT-Base和BERT-Large。
     - **BERT-Base**：12层Transformer编码器，隐藏层大小为768，共有110M参数。
     - **BERT-Large**：24层Transformer编码器，隐藏层大小为1024，共有340M参数。

3. **输入表示**：
   - BERT的输入包括词嵌入、位置嵌入和段落嵌入。
     - **词嵌入**：将每个词转换为固定大小的向量表示。
     - **位置嵌入**：为每个词的位置添加位置信息，保留序列中的顺序信息。
     - **段落嵌入**：区分输入中的不同句子，特别是在NSP任务中，用于区分第一个句子和第二个句子。

##### BERT的应用与优点

1. **动态词向量**：
   - 由于BERT是双向的，并且在不同上下文中对词进行编码，它生成的词向量是动态的。这意味着同一个词在不同的句子中会有不同的向量表示，能够更准确地捕捉词的多义性和上下文依赖性。

2. **下游任务的微调**：
   - BERT在预训练之后，可以通过在特定任务上的微调来应用于各种NLP任务。只需在预训练模型的基础上添加一个简单的输出层，然后在任务数据上进行微调，就能取得很好的效果。
   - 常见的下游任务包括问答系统（如SQuAD）、文本分类（如情感分析）、命名实体识别和文本生成等。

3. **语义理解能力提升**：
   - 由于BERT能够同时考虑词的左右上下文，并且通过MLM和NSP任务进行预训练，它在理解句子的语义和上下文关系方面表现优异。
   - 在多种NLP基准测试中，BERT都取得了领先的成绩，显著提升了自然语言理解的效果。

##### 总结

BERT通过双向Transformer编码器和掩蔽语言模型训练，能够在不同上下文中动态生成词向量，极大地提升了模型对语义和上下文的理解能力。其预训练加微调的方式，使得BERT在各种NLP任务中都表现出色，成为了自然语言处理领域的一个重要里程碑。

---
   
REFERENCE：

<a href="https://zhuanlan.zhihu.com/p/403433120">1. 【Transformer】10分钟学会Transformer | Pytorch代码讲解 | 代码可运行</a>

<a href="https://arxiv.org/pdf/1706.03762">2. Attention Is All You Need.</a>