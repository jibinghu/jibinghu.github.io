https://zhuanlan.zhihu.com/p/1920946738270810330

知识点：

- BF16/FP16

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

- 再问点量化吧，Weight only int4量化和fp8量化，你觉得二者有什么应用上的区别
	- weight only int4更适合对显存要求苛刻的场景

- 为什么只量化weight，不量化activation？

- 动态量化和静态量化有听说过吗？区别是？

- 什么时候适合动态量化，什么时候适合静态量化？
