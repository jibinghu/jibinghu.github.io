https://zhuanlan.zhihu.com/p/1920946738270810330

知识点：

- BF16/FP16

	- 数据格式其实是我应该烂熟于心的，但是之前只对 float double int 等 IEEE754 标准下的进行学习，零散看过 bf16，但是没记住关键。另外浮点数还会有魔法数(在 awq 中)还值得记住。
	- FP16： 1 位符号位，5 位指数位，10 位尾数位
	- BF16： 1 位符号位，8 位指数位，7 位尾数位
	- FP32： 1 位符号位，8 位指数位，23 位尾数位

<img width="589" height="176" alt="Image" src="https://github.com/user-attachments/assets/354cf422-cdda-4bf9-9082-7fc4d926e27e" />

	- 总结下来就是 BGF16 用同样的存储位数，表示了更大的动态范围，但是损失了一定的精度表示。在 softmax、matmul、reduction 中 BF16 避免了溢出/下溢。比如 exp(12) = $1.63×10^5$ ，已经超过指数位 5 位可表示的最大范围 $6.55×10^4$ 。

<img width="316" height="40" alt="Image" src="https://github.com/user-attachments/assets/0ea7b30f-8d28-4037-9636-943024977d4c" />

	- 反向传播时也更适合 BF16 来避免梯度消失/爆炸
	- 所以训练阶段更多用 BF16
	- 而推理阶段在部分架构，尤其是 Tensore Core 中，FP16 的吞吐要比 BF16 快，而且没有训练时的一系列反向传播和误差计算等，所以影响比较小。


- cpu 和 gpu 的算子库

- cutlass
	- https://mp.weixin.qq.com/s?__biz=Mzg2ODk4MzE2MQ==&mid=2247484590&idx=1&sn=ce85ec7834824c82d208261dc3f2bc69&scene=21&poc_token=HHuqN2mj3D4wsQAeSiRD2Vl_rk5ki-CKUmGoeUnU

- xxxdnn primitive cache

- 项目里面xxx kernel的时候，你会如何判断该kernel是否还有优化空间？

- 在写你的xxx kernel时候，用到了哪些优化手段
	- 访存上，coalesced，消除bank conflict，xxx。计算上，尽可能减少分支，向量化计算，xxx

- workload应该不止一个，那么喂到该kernel的shape应该也会不一样？那这种情况下，你们如何针对不同的shape去开发并优化kernel？
	- 大致是针对不同shape采用略微不一样的优化手段，在这个过程中需要借助profiling工具来微调优化手段

- flash attention
	- flash attention v3主要的优化点是什么
	- https://mp.weixin.qq.com/s?__biz=Mzg2ODk4MzE2MQ==&mid=2247484698&idx=1&sn=f2059c5fb2e3d7ecba010646882000de&scene=21&poc_token=HGOrN2mjMB0VR7DX7KmcF4VebMdxPaqIqQnQWdfD
	- https://mp.weixin.qq.com/s?__biz=Mzg2ODk4MzE2MQ==&mid=2247484717&idx=1&sn=d73c31a47c49c247c9d0855cc45642e9&scene=21&poc_token=HG6rN2mjtyF38-UEeyZfakeIrwWcqXn8vip0ZhaJ

- 目前了解过哪些大模型推理引擎？
	- vllm/sglang 等

- bert和gpt模型的区别是什么
        - BERT = 基于 Transformer Encoder 的 双向理解模型
        - GPT = 基于 Transformer Decoder 的 单向生成模型

<img width="614" height="636" alt="Image" src="https://github.com/user-attachments/assets/b85a1a72-e3e4-4b43-b30d-d8c507adac35" />


- 再问点量化吧，Weight only int4量化和fp8量化，你觉得二者有什么应用上的区别
	- weight only int4更适合对显存要求苛刻的场景

- 为什么只量化weight，不量化activation？

- 动态量化和静态量化有听说过吗？区别是？

- 什么时候适合动态量化，什么时候适合静态量化？
