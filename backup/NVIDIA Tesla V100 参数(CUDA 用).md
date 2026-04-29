<html><head></head><body>

<p>NVIDIA Tesla V100（Volta 架构，Compute Capability = 7.0）的比较全面的硬件／资源参数汇总：</p>
<hr>

<h2>基本架构参数</h2>
<table border="1" cellpadding="4" cellspacing="0">
  <tr><th>项目</th><th>数值 / 描述</th></tr>
  <tr><td>架构 / 代号</td><td>Volta / GV100 :contentReference[oaicite:0]{index=0}</td></tr>
  <tr><td>完整 GV100 芯片的 SM 总数</td><td>84 个 SM :contentReference[oaicite:1]{index=1}</td></tr>
  <tr><td>在实际 V100 上启用的 SM 数目</td><td>80 个 SM（部分禁用以提高良品率）:contentReference[oaicite:2]{index=2}</td></tr>
  <tr><td>每 SM 的计算单元组成</td>
    <td>64 个 FP32 核心 + 32 个 FP64 核心 + 64 个 INT32 核心 + 8 个 Tensor Core :contentReference[oaicite:3]{index=3}</td></tr>
  <tr><td>各种核之间并行执行能力</td><td>FP32 和 INT32 指令可并行发射（因它们有独立资源）:contentReference[oaicite:4]{index=4}</td></tr>
  <tr><td>GPU 全局 L2 缓存</td><td>6,144 KB（6 MB）:contentReference[oaicite:5]{index=5}</td></tr>
  <tr><td>HBM2 显存带宽（峰值）</td><td>约 900 GB/s（标准版本）:contentReference[oaicite:6]{index=6}</td></tr>
  <tr><td>显存容量</td><td>通常 16 GB 或 32 GB HBM2（不同型号）:contentReference[oaicite:7]{index=7}</td></tr>
</table>

<hr>

<h2>SM 级资源 / 线程 / 寄存器 / 共享内存 限制</h2>
<table border="1" cellpadding="4" cellspacing="0">
  <tr><th>资源类型</th><th>限制 / 说明</th></tr>
  <tr><td>最大并发 warp 数</td><td>64 个 warp（即 64 × 32 = 2048 线程）:contentReference[oaicite:8]{index=8}</td></tr>
  <tr><td>最大并发线程数／SM</td><td>2,048 线程 :contentReference[oaicite:9]{index=9}</td></tr>
  <tr><td>最大并发线程块（resident blocks）／SM</td><td>32 个线程块 :contentReference[oaicite:10]{index=10}</td></tr>
  <tr><td>线程块最大线程数</td><td>1,024 线程 :contentReference[oaicite:11]{index=11}</td></tr>
  <tr><td>寄存器总数（32-bit 寄存器）／SM</td><td>65,536 个（即 64K）:contentReference[oaicite:12]{index=12}</td></tr>
  <tr><td>每线程最大寄存器数</td><td>255 个 32-bit 寄存器 :contentReference[oaicite:13]{index=13}</td></tr>
  <tr><td>线程块可使用的最大寄存器总数</td><td>最高可达 65,536 个寄存器（即一个 block 理论上可动用 SM 全部寄存器）:contentReference[oaicite:14]{index=14}</td></tr>
  <tr><td>共享内存 + L1 缓存 合并空间</td><td>每 SM 最多可配置 96 KB（共享内存 + L1 混合）:contentReference[oaicite:15]{index=15}</td></tr>
  <tr><td>单个线程块可用共享内存上限</td><td>理论上可使用全部 96 KB（如果该 block 占用全部共享内存）:contentReference[oaicite:16]{index=16}</td></tr>
  <tr><td>默认静态共享内存限制</td><td>静态分配通常受限为 48 KB，若超过则需动态分配机制 :contentReference[oaicite:17]{index=17}</td></tr>
  <tr><td>SM 分区 / 内部结构</td><td>每 SM 被划分为 4 个 partition（每个 partition 有自己的调度器 / 指令缓存 L0 / 部分资源）:contentReference[oaicite:18]{index=18}</td></tr>
  <tr><td>每个 partition 的寄存器文件</td><td>每个 partition 有约 64 KB 的寄存器空间（即 SM 的寄存器空间被划分给 4 个分区）:contentReference[oaicite:19]{index=19}</td></tr>
</table>

<hr>

<h2>不同精度 / Tensor Core 吞吐能力</h2>
<table border="1" cellpadding="4" cellspacing="0">
  <tr><th>精度 / 类型</th><th>理论峰值算力 / 吞吐</th><th>说明 / 备注</th></tr>
  <tr><td>FP64（双精度）</td><td>约 7.8 TFLOPS（在 SXM / GV100 版本）:contentReference[oaicite:20]{index=20}</td><td>每 SM 有 32 个 FP64 核心</td></tr>
  <tr><td>FP32（单精度）</td><td>约 15.7 TFLOPS（Boost 时钟下）:contentReference[oaicite:21]{index=21}</td><td>标准 CUDA 核心用于标量浮点运算</td></tr>
  <tr><td>Tensor / 混合精度（FP16 输入 + FP32 累加）</td><td>125 TFLOPS（混合精度 Tensor 运算）:contentReference[oaicite:22]{index=22}</td><td>全卡共有 640 个 Tensor Core（每 SM 8 个）:contentReference[oaicite:23]{index=23}</td></tr>
  <tr><td>普通 FP16（非 Tensor 运算）</td><td>理论上可接近 FP32 的 2×，但不常用作主路径</td><td>Volta 架构主要通过 Tensor Core 加速矩阵运算，对标量 FP16 支持有限</td></tr>
</table>

<p>关于 Tensor Core 的设计细节：</p>
<ul>
  <li>每个 SM 拥有 8 个 Tensor Core → 全卡共 640 个 :contentReference[oaicite:24]{index=24}</li>
  <li>每个 Tensor Core 在一个时钟周期内可执行一次 4×4 矩阵乘加 (即 D = A×B + C) :contentReference[oaicite:25]{index=25}</li>
  <li>单个 Tensor Core 每周期可执行 64 个 FMA 操作（乘加对） → 每 SM 的 8 个 Tensor Core 共 512 FMA（或 1024 个浮点运算） :contentReference[oaicite:26]{index=26}</li>
  <li>理论上，整卡的 Tensor 运算峰值为 125 TFLOPS 混合精度 :contentReference[oaicite:27]{index=27}</li>
  <li>在实际测得的性能中，使用 cuBLAS 等优化库时可达 ~83 TFLOPS 混合精度水平（比理论峰值低一些）:contentReference[oaicite:28]{index=28}</li>
</ul>

<hr>

<h2>子结构 / 缓存 / 内存 / 带宽 等</h2>
<table border="1" cellpadding="4" cellspacing="0">
  <tr><th>子系统</th><th>参数 / 大小 / 吞吐</th><th>说明 / 备注</th></tr>
  <tr><td>L2 缓存</td><td>6,144 KB（6 MB）:contentReference[oaicite:29]{index=29}</td><td>整卡共享，用于跨 SM 缓存加速</td></tr>
  <tr><td>共享内存 + L1 混合空间</td><td>每 SM 最多可配置 96 KB :contentReference[oaicite:30]{index=30}</td><td>Volta 架构中共享内存与 L1 缓存资源是可配置混合使用的</td></tr>
  <tr><td>静态共享内存（__shared__）</td><td>通常上限 48 KB，超过部分需动态分配</td><td>编译器 / 运行时可能根据资源优化做调整 :contentReference[oaicite:31]{index=31}</td></tr>
  <tr><td>访存带宽（HBM2）</td><td>约 900 GB/s（标准版本）:contentReference[oaicite:32]{index=32}</td><td>部分 V100S / SXM 型号带宽更高（如 1134 GB/s）:contentReference[oaicite:33]{index=33}</td></tr>
  <tr><td>显存容量</td><td>16 GB 或 32 GB HBM2（具体型号不同）:contentReference[oaicite:34]{index=34}</td><td>支持 ECC（错误检测与修正）</td></tr>
  <tr><td>NVLink / 接口带宽</td><td>在 NVLink 型号下可达高带宽互联，PCIe 型号受 PCIe 总线限制 :contentReference[oaicite:35]{index=35}</td><td>NVLink 型号互联带宽可达数百 GB/s</td></tr>
  <tr><td>L0 / 指令缓存 / 调度单元</td><td>每 SM 划分为多个 partition（通常 4 个），每个有自己的指令缓存（L0）和调度单元 :contentReference[oaicite:36]{index=36}</td><td>此结构可减少资源冲突、提高并行度</td></tr>
  <tr><td>线程调度 / 独立线程执行</td><td>Volta 支持 Independent Thread Scheduling，warp 内线程可相对独立调度 / 同步（`__syncwarp()`）:contentReference[oaicite:37]{index=37}</td><td>增加了调度灵活性</td></tr>
</table>

<hr>

<h2>限制 / 资源约束（回顾 + 补充）</h2>
<ul>
  <li>每 SM 最大线程数：2,048（即 64 个 warp）:contentReference[oaicite:38]{index=38}</li>
  <li>每 SM 最大并发线程块数：32 个 block :contentReference[oaicite:39]{index=39}</li>
  <li>每线程块（block）最大线程数：1,024 线程 :contentReference[oaicite:40]{index=40}</li>
  <li>每线程最大寄存器数：255 个（32-bit）:contentReference[oaicite:41]{index=41}</li>
  <li>每 SM 寄存器总量：65,536 个（32-bit）:contentReference[oaicite:42]{index=42}</li>
  <li>每线程块可使用的最大寄存器总数：理论上可动用 SM 全部寄存器（65,536 个）:contentReference[oaicite:43]{index=43}</li>
  <li>线程块的最大维度（CUDA API 限制）：通常单 block 最多 1024 线程，各维度不超过 1024 等（具体依 CUDA 版本）</li>
  <li>共享内存 / L1 空间总量可配置（如前所述）</li>
</ul>

</body></html>
