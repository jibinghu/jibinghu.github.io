**PTX 的 `prmt` 指令**（byte **permute**）。它是 CUDA PTX 里的**按字节重排**指令，用来从两个 32 位寄存器里挑选并拼装 4 个字节，生成一个新的 32 位结果；很多场景下可替代多条移位/与/或操作（比如打包/解包、大小端转换、掩码生成、AES 等）。官方文档把它称为 *PRMT/byte permute*。([[NVIDIA Docs](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?utm_source=chatgpt.com)][1])

# 基本形式

```
prmt.b32 d, a, b, s;
```

* `a`、`b`：两个 32 位源寄存器（共 8 个源字节 a0..a3、b0..b3）。
* `s`：**选择器**（可以是立即数或寄存器）。它用 4 个 nibble（4×4bit）分别指定目标的 4 个字节该从哪个源字节来。
* `d`：目标 32 位寄存器。

# 选择器与语义（默认模式）

选择器的每个 **4 bit** 控制一个目标字节：

* 低 3 bit 选哪个源字节（0–3=来自 `a` 的字节，4–7=来自 `b` 的字节）。
* 最高位 **msb** 决定拷贝方式：msb=0 时**原样拷贝**；msb=1 时生成**按该字节符号位（bit7）扩展的掩码**（得到 `0x00` 或 `0xFF`），可用来快速做条件掩码。这个“默认模式”是 PRMT 的一大特色。([[Stack Overflow](https://stackoverflow.com/questions/60263413/when-is-the-default-variant-ptx-instruction-prmt-useful?utm_source=chatgpt.com)][2])

举例：

```
# 取 a 的字节2、a 的字节0、b 的字节3、b 的字节1 组成 d（全部 msb=0 原样拷贝）
# 选择器 s = 0x23E5 （从高到低 4 个 nibble：2,3,14,5）
prmt.b32 d, a, b, 0x23E5;
```

> nibble=0..3 表 a0..a3，4..7 表 b0..b3；E(=14) 的 msb=1 → 该目标字节会变成全 0x00/0xFF 的掩码（取决于被指向那个源字节的 bit7）。([[Stack Overflow](https://stackoverflow.com/questions/60263413/when-is-the-default-variant-ptx-instruction-prmt-useful?utm_source=chatgpt.com)][2])

# 特殊变体

PTX 还定义了若干变体/模式（文档里常见例如 `.b4e`、`.f4e`、`.rc8`、`.ecl`、`.ecr`、`.rc16` 等），用于更偏门的字节/半字节抽取、循环拼接等，默认不带后缀时就是上面的“默认模式”。这些模式用于实现字节级“漏斗移位”、位场抽取等技巧。([[Stack Overflow](https://stackoverflow.com/questions/60263413/when-is-the-default-variant-ptx-instruction-prmt-useful?utm_source=chatgpt.com)][2])

# 与 CUDA 内建函数

CUDA 提供了 `__byte_perm(x, y, s)` 内建函数，能在 C/C++ 里调用字节重排（底层映射到 PRMT/相关模式）。常用于从两个 32 位数里快速抽取并重排字节、做按 8 位倍数的移位拼接等。是否严格对应 PRMT 的“默认模式”取决于编译与架构实现细节，但总体就是“字节级重排”。([[NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/questions-about-byte-perm-x-y-s/17822?utm_source=chatgpt.com)][3])

# 典型用途

* **打包/解包**：把 4 个散落的字节组合成 32 位值，或反之。
* **大小端转换**：一次指令完成 0xAABBCCDD ↔ 0xDDCCBBAA。
* **掩码生成/条件合成**：利用 msb=1 的“符号扩展为 0x00/0xFF”特性。
* **按字节漏斗移位/旋转**：比多条移位与或更省。([[腾讯云](https://cloud.tencent.com/developer/ask/sof/114473886/answer/137015457?utm_source=chatgpt.com)][4])

# 性能提示

PRMT/`__byte_perm` 的吞吐量**依架构而异**（在某些 GPU 上很强，在另一些上与多条逻辑指令差不多甚至更慢）。实际用前建议在目标 GPU 上 **基准测试**。([[腾讯云](https://cloud.tencent.com/developer/ask/sof/114473886/answer/137015457?utm_source=chatgpt.com)][4])


[1]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?utm_source=chatgpt.com "1. Introduction — PTX ISA 9.0 documentation"
[2]: https://stackoverflow.com/questions/60263413/when-is-the-default-variant-ptx-instruction-prmt-useful?utm_source=chatgpt.com "When is the (default-variant) PTX instruction `prmt` useful?"
[3]: https://forums.developer.nvidia.com/t/questions-about-byte-perm-x-y-s/17822?utm_source=chatgpt.com "Questions about __byte_perm (x,y,s) - CUDA Programming and Performance ..."
[4]: https://cloud.tencent.com/developer/ask/sof/114473886/answer/137015457?utm_source=chatgpt.com "在CUDA中用SIMD实现位旋转算子-腾讯云开发者社区-腾讯云"