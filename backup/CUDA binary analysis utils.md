cuobjdump ：
cuobjdump 是 NVIDIA 提供的一个工具，用于提取和显示 CUDA 二进制文件（即 CUDA 应用程序的可执行文件）中的信，可以用来分析cubin文件和host文件。它可以显示以下内容：
1.  内核信息 ：显示 CUDA 内核的名称、内核的 PTX（并行线程执行）代码等。
2.  内存分配 ：显示 CUDA 程序中使用的全局内存、常量内存和共享内存的信息。
3.  设备代码 ：提取并显示设备代码，即运行在 GPU 上的代码。
4.  其他信息 ：例如 CUDA 版本信息、编译器选项等。
使用 cuobjdump的常见命令：
cuobjdump -x <filename>
其中 <filename>是 CUDA 可执行文件的名称。
![image](https://github.com/user-attachments/assets/646f2e1e-8b23-4682-8844-b4f8003f5e67)

nvdisasm：
nvdisasm 是 NVIDIA 提供的一个工具，用于反汇编 CUDA 二进制文件。它将 CUDA 应用程序的二进制代码反汇编为人类可读的汇编代码。这对于理解 CUDA 程序的低级实现细节、调试和性能优化非常有用。nvdisasm 可以显示 CUDA 程序中每个指令的详细信息，包括寄存器使用、指令操作码等。
    nvcc -o cudatest cudatest.cu -gencode=arch=compute_80,code=\"sm_80,compute_80\" --keep
使用 nvdisasm 的常见命令：
nvdisasm <filename>
其中 `<filename>` 是 CUDA 可执行文件的名称。

### 总结

- cuobjdump 主要用于提取和显示 CUDA 可执行文件中的各种信息，如内核、内存分配和设备代码。
- nvdisasm 则用于反汇编 CUDA 可执行文件，将二进制代码转换为汇编代码，以便进行低级别的分析和优化。

这些工具对于 CUDA 程序的开发和优化都是非常有用的，可以帮助开发人员更好地理解和改进 CUDA 程序的性能。