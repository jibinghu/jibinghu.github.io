
由于 gpt 给出的源和 `https://zhuanlan.zhihu.com/p/30652451322` 给出的 prmt 的说明不一致，在这里做个探讨：

参考： `https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?utm_source` -> `grep prmt`

prmt requires sm_20 or higher.
Permute bytes from register pair.

``` cpp
prmt.b32{.mode}  d, a, b, c;
.mode = { .f4e, .b4e, .rc8, .ecl, .ecr, .rc16 };
```

实际作用都是一样的：Pick four arbitrary bytes from two 32-bit registers, and reassemble them into a 32-bit destination register.
但是对于索引寄存器 c 的使用，在 nvidia doc里：

在没有指定mode 的情况下，c寄存器是 4*4 的：

<img width="1064" height="54" alt="Image" src="https://github.com/user-attachments/assets/7c9e0d5d-ad8a-45b8-8190-736bc7332f82" />

目前看起来的说法 gpt 给的是正确的，最下面的说明：

>  Least Significant Bit Substitution 最低有效位
> 最高有效位(Most Significant Bit)

<img width="1093" height="81" alt="Image" src="https://github.com/user-attachments/assets/00f2137f-29bf-4022-ba3b-19708b3f26c0" />

再参考下：https://zhuanlan.zhihu.com/p/660630414 以及 https://zhuanlan.zhihu.com/p/657070837

文章里讲的更详细些：

> 对于目标寄存器中的每个字节，定义了一个 4 位选择器。选择值的 3 个 低位lsb 指定应将 8 个源字节中的哪一个移至目标中位置。 msb 定义是否应直接复制原始字节值，或者是否应复制符号（即，是否进行符号扩展）；msb=0表示直接复制原始的bit值，msb=1，则表示进行符号扩展。为简单起见，这里只关注PRMT指令的通用形式。（事实上，这个指令还有f2e、b4e、rc8等特殊模型）

下面的代码和注释说明比较清楚了：

``` cpp
// --------------------------- Notes from NVIDIA FasterTransformer ------------------------------
// This converter is meant to be used with data interleaved in a 32-bit register where the even 
// elements are in the low bits and the odd elements are in the high bits of the register. In 
// addition, it assumes elements were originally signed and had a bias of 2**(b-1) added (where 
// b is the number of bits in the type) to make all numbers unsigned. This converter will 
// uninterleave the data and subtract the bias while converting to the result type.
// --------------------------- Notes from NVIDIA FasterTransformer ------------------------------
// -------------------------------------- Notes from Personal -----------------------------------
// 个人理解：
// 假设保存好的uint8量化权重，在内存中，是交织(interleaved)后的布局，偶数索引的元素保存在低bits，奇数索引的元素
// 保存在高bits，也就是原始在内存中的布局（右侧为低字节）{e3,e2,e1,e0} 交织为 {e3,e1,e2,e0}. 这应该是为了更好
// 地利用硬件的特性获得更好的性能。另外，也假设保存好uint8权重是已经 + 2**(b-1)的了，即128，已经是unsigned数值。
// 因此，反量化函数，需要完成几个事，即：反量化、解交织 和 减128恢复原值大小。
template<>
struct FastInterleavedAndBiasedNumericArrayConverter<half_t, uint8_t, 4> {
    using result_type = Array<half_t, 4>;
    using source_type = Array<uint8_t, 4>;
    CUTLASS_DEVICE
    static result_type convert(source_type const& source)
    {   
        result_type result; // Array<half_t, 4>  32x2 bits
        // 注意，这里的h实际上指向了一块大小为32x2bits的连续内存，只是为了方便后续的
        // 操作，reinterpret为uint32_t，即h[0]代表低32bits，h[1]代表高32bits
        uint32_t*      h   = reinterpret_cast<uint32_t*>(&result);
        uint32_t const i8s = reinterpret_cast<uint32_t const&>(source);
        // 字节选择器，虽然是uint32_t，但实际只有低16bits有值
        // byte selector: [0][101] [0][010] [0][101] [0][000]
        static constexpr uint32_t mask_for_elt_01     = 0x5250;   
        // byte selector: [0][101] [0][011] [0][101] [0][001] 
        static constexpr uint32_t mask_for_elt_23     = 0x5351;    
        // pack {b, a}成{{b7, b6, b5, b4},{b3, b2, b1, b0}}
        // {b, a} = {{0x64, 0x64, 0x64, 0x64}, {b3, b2, b1, b0}}
        // 由于原始在内存中的布局（右侧为低字节）{e3,e2,e1,e0} 已经交织为 
        // {e3,e1,e2,e0}所以{b, a}在内存中实际的值排布为：
        // {b, a} = {start_byte_for_fp16, i8s} = 
        // {{0x64, 0x64, 0x64, 0x64}, {e3, e1, e2, e0}}
        static constexpr uint32_t start_byte_for_fp16 = 0x64646464;  
        // mask_for_elt_01就是选择器，根据选择器和{b,a}，我们可以的到h[0]的值
        // mask_for_elt_01 -> [0][101] [0][010] [0][101] [0][000]
        // mask_for_elt_01 ->   d.b3     d.b2     d.b1     d.b0
        // mask_for_elt_01 ->   5        2        5        0
        // mask_for_elt_01 ->   0x64     e1       0x64     e0
        //            h[0] ->   0x64[e1]64[e0]
        asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(h[0]) : "r"(i8s), "n"(start_byte_for_fp16), "n"(mask_for_elt_01));
        // mask_for_elt_23就是选择器，根据选择器和{b,a}，我们可以的到h[1]的值
        // mask_for_elt_23 -> [0][101] [0][011] [0][101] [0][001]
        // mask_for_elt_23 ->   d.b3     d.b2     d.b1     d.b0
        // mask_for_elt_23 ->   5        3        5        1
        // mask_for_elt_23 ->   0x64     e3       0x64     e2
        //            h[1] ->   0x64[e3]64[e2]
        asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(h[1]) : "r"(i8s), "n"(start_byte_for_fp16), "n"(mask_for_elt_23));
        // 需要注意的是h[1]h[0]保存的值，已经是解交织后的排布了，即 {e3,e2,e1,e0}
        // NOTE: ei = ei_ori + 128


        // 把 int8 转成 fp16
        // Lastly, we subtract 1152 from our constructed number using fp16 math to get our signed integer as fp16.
        static constexpr uint32_t I8s_TO_F16s_MAGIC_NUM = 0x64806480;
        // h[0] ->   0x[64[e1]][64[e0]]   -   0x[6480][6480]
        // h[0] ->   0x([64[e1]] - [6480]) ([64[e0]] - [6480])
        // h[0] ->   0x[e1_ori][e0_ori]
        asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[0]) : "r"(h[0]), "r"(I8s_TO_F16s_MAGIC_NUM));
        // h[1] ->   0x[64[e3]][64[e2]]   -   0x[6480][6480]
        // h[1] ->   0x([64[e3]] - [6480]) ([64[e2]] - [6480])
        // h[1] ->   0x[e3_ori][e2_ori]
        asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[1]) : "r"(h[1]), "r"(I8s_TO_F16s_MAGIC_NUM));
        // 最终，获得量化权重的FP16表示，并且完成解交织
        // h[1]h[0](右侧为低字节)解交织为 FP16 arr {e3_ori_f16, e2_ori_f16, e1_ori_f16, e0_ori_f16} 
        // arr[0] = e0_ori_f16, arr[1] = e1_ori_f16, arr[2] = e2_ori_f16, arr[3] = e3_ori_f16
        return result;
    }
    CUTLASS_DEVICE result_type operator()(source_type const& s) { return convert(s); }
};
```

---

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