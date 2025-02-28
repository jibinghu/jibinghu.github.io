## 整体：

[浮点数精度](https://zhida.zhihu.com/search?content_id=246744417&content_type=Article&match_order=1&q=%E6%B5%AE%E7%82%B9%E6%95%B0%E7%B2%BE%E5%BA%A6&zhida_source=entity)：双精度（FP64）、单精度（FP32、TF32）、半精度（FP16、BF16）、8位精度（FP8）、4位精度（FP4、[NF4](https://zhida.zhihu.com/search?content_id=246744417&content_type=Article&match_order=1&q=NF4&zhida_source=entity)）

量化精度：[INT8](https://zhida.zhihu.com/search?content_id=246744417&content_type=Article&match_order=1&q=INT8&zhida_source=entity)、INT4 （也有INT3/INT5/INT6的）

## 精度：

在计算机中，浮点数存储方式，由由符号位（sign）、指数位（exponent）和小数位（fraction）三部分组成。符号位都是1位，指数位影响浮点数范围，小数位影响精度。

### FP精度

#### FP 64

FP64，是64位浮点数，由1位符号位，11位指数位和52位小数位组成。

但是FP8和FP4不是IEEE的标准格式。[IEEE标准](https://zhida.zhihu.com/search?content_id=246744417&content_type=Article&match_order=1&q=IEEE%E6%A0%87%E5%87%86&zhida_source=entity)：[https://en.wikipedia.org/wiki/IEEE_754](https://link.zhihu.com/?target=https%3A//en.wikipedia.org/wiki/IEEE_754)

FP8是2022年9月由多家芯片厂商定义的，论文地址：[https://arxiv.org/abs/2209.05433](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2209.05433)

FP4是2023年10月由某学术机构定义，论文地址：[https://arxiv.org/abs/2310.16836](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2310.16836)

FP8格式有两种变体，E4M3(4位指数和3位尾数)和E5M2(5位指数和2位尾数)

![Image](https://github.com/user-attachments/assets/50a43c07-155f-4c27-af6d-6906ff69502e)

### 特殊精度

#### TF 32

Tensor Float 32，英伟达针对机器学习设计的一种特殊的数值类型，用于替代FP32。首次在A100 GPU中支持。

由1个符号位，8位指数位（对齐FP32）和10位小数位（对齐FP16）组成，实际只有19位。在性能、范围和精度上实现了平衡。

![Image](https://github.com/user-attachments/assets/000b11c1-ea99-4071-9de0-12fec83d70d1)

python中查看是否支持：

``` python
import torch
//是否支持tf32
torch.backends.cuda.matmul.allow_tf32
//是否允许tf32，在PyTorch1.12及更高版本中默认为False
torch.backends.cudnn.allow_tf32
```
#### BF 32

Brain Float 16，由Google Brain提出，也是为了机器学习而设计。由1个符号位，8位指数位（和FP32一致）和7位小数位（低于FP16）组成。所以精度低于FP16，但是表示范围和FP32一致，和FP32之间很容易转换。

在 NVIDIA GPU 上，只有 Ampere 架构以及之后的GPU 才支持。

python中查看是否支持：

``` python
import transformers
transformers.utils.import_utils.is_torch_bf16_gpu_available()
```

#### NF 4

-bit NormalFloat，一种用于量化的特殊格式，于23年5月由华盛顿大学在QLoRA量化论文中提出，论文地址：[https://arxiv.org/abs/2305.14314](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2305.14314)

NF4是建立在分位数量化技术的基础之上的一种信息理论上最优的数据类型。把4位的数字归一化到均值为 0，标准差为 [-1,1] 的正态分布的固定期望值上，知道量化原理的应该就会理解。

FP精度和特殊精度加上，位数总结如下表：

![Image](https://github.com/user-attachments/assets/734d8d59-3d05-42f8-b6ed-0d19ba96914c)

---

多精度和混合精度

[多精度计算](https://zhida.zhihu.com/search?content_id=246744417&content_type=Article&match_order=1&q=%E5%A4%9A%E7%B2%BE%E5%BA%A6%E8%AE%A1%E7%AE%97&zhida_source=entity)，是指用不同精度进行计算，在需要使用高精度计算的部分使用双精度，其他部分使用半精度或单精度计算。

[混合精度计算](https://zhida.zhihu.com/search?content_id=246744417&content_type=Article&match_order=1&q=%E6%B7%B7%E5%90%88%E7%B2%BE%E5%BA%A6%E8%AE%A1%E7%AE%97&zhida_source=entity)，是在单个操作中使用不同的精度级别，从而在不牺牲精度的情况下实现计算效率，减少运行所需的内存、时间和功耗

---
https://zhuanlan.zhihu.com/p/713763703