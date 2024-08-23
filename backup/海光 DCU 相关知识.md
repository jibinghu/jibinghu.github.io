> 目前还是在学习阶段，把之后可能时常需要用到的技术备忘在这里。
  当前在做的DCU相关需要用到hip和ROCm相关知识，但一直对dtk概念不清，在这里作以补充。

## 概要：

- CUDA(Compute Unified Device Architecture)是Nvidia推出的一种通用并行计算架构，包括CUDA[指令集架构](https://zhida.zhihu.com/search?q=%E6%8C%87%E4%BB%A4%E9%9B%86%E6%9E%B6%E6%9E%84)（ISA）和GPU内部的并行计算引擎

- 目前主流的路线主要有两种，第一种选择从芯片到计算平台库都全自研，比如华为基于自己的Ascend（昇腾）系列ASIC（application-specific integrated circuit）构建的CANN计算平台库以及[寒武纪](https://zhida.zhihu.com/search?q=%E5%AF%92%E6%AD%A6%E7%BA%AA)基于自家的MLU系列ASIC构建的Neuware；第二种则是选择自研+开源的路线，比如海光信息则是自研开发了DTK（DCU Toolkit）计算平台库，兼容开源的ROCm和适配自家自研的DCU，对标CUDA及GPU。由于兼容了ROCm开源计算平台库，进一步保证了海光DTK的通用性，再加上海光DCU加持，使得海光DTK发展成为较为成熟的生态环境。

![](https://pic2.zhimg.com/v2-21f93363f011e6a34437c622a727fa7d_r.jpg)

## ROCm：

- ROCm（Radeon Open Compute Platform）是AMD主导的一个开源计算平台库，Radeon是AMD GPU产品的品牌名，除ROCm之外，还有一系列ROCx的简称，如ROCr（ROC Runtime），ROCk（ROC kernel driver），ROCt（ROC Thunk）等。Windows的一家独大使得Linux的出现成为可能，iOS的封闭造就了Android的蓬勃发展，ROCm的[横空出世](https://zhida.zhihu.com/search?q=%E6%A8%AA%E7%A9%BA%E5%87%BA%E4%B8%96)才让CUDA拥有了真正的竞争对手。

![](https://pic4.zhimg.com/v2-47e773f774d86d3af79252a93fc253a7_r.jpg)

## 对比：

![image](https://github.com/user-attachments/assets/a534d2b8-08e7-42ec-af4f-5a7497dee5b4)

## DTK 架构对比：

![image](https://github.com/user-attachments/assets/a8181c9a-7297-4e35-af27-4410073da152)


## 链接：

<a href="https://link.zhihu.com/?target=https%3A//cancon.hpccube.com%3A65024/1/main">海光 dtk 社区]</a>
<a href="https://zhuanlan.zhihu.com/p/705584420">带你快速了解国产AI生态平台----海光DTK</a>