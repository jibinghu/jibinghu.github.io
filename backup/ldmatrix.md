
<hr>
<h1>是什么</h1>
<p><code inline="">ldmatrix.sync</code> 是 <strong>按 warp 协作</strong>、从 <strong>shared memory</strong> 中一次性读取 <strong>8×8 的矩阵 tile</strong> 到寄存器的指令，用来给 Tensor Core 的矩阵乘（<code inline="">mma.sync</code> / <code inline="">mma.sp.sync</code> 等）准备操作数 <strong>fragment</strong>。<br>
它是 <strong>同步（sync）且 warp 级别</strong> 的：同一 warp 的 32 个线程必须一起执行该指令。</p>
<hr>
<h1>典型语法</h1>
<pre><code class="language-ptx">ldmatrix.sync.aligned.m8n8.x{1|2|4}[.trans].shared.b{16|8} {dst_list}, [addr];
</code></pre>
<ul>
<li>
<p><code inline="">aligned</code>：要求基地址满足对齐（通常 ≥16B，对 b16 的 8×8 刚好 16B/行）。</p>
</li>
<li>
<p><code inline="">m8n8</code>：一次处理的 tile 尺寸是 8×8。</p>
</li>
<li>
<p><code inline="">x1|x2|x4</code>：<strong>一次加载几个 8×8 tile 的 fragment</strong>（返回 1/2/4 份寄存器结果）。</p>
<ul>
<li>
<p>常见：<code inline="">x4</code> 给 A 操作数，<code inline="">x2</code> 给 B 操作数（与后续 <code inline="">mma.sync.m16n8k16</code> 等形状匹配）。</p>
</li>
</ul>
</li>
<li>
<p><code inline="">trans</code>：在加载时对 8×8 <strong>做转置</strong>（常用于 B 操作数的列主布局需求）。</p>
</li>
<li>
<p><code inline="">shared</code>：数据源必须在 shared memory。</p>
</li>
<li>
<p><code inline="">b16|b8</code>：元素宽度（半精度/BF16/INT16 等用 <code inline="">b16</code>；INT8 等用 <code inline="">b8</code>）。</p>
</li>
<li>
<p><code inline="">{dst_list}</code>：目的寄存器列表。</p>
<ul>
<li>
<p><code inline="">b16.x1</code>：每个线程产出 1 个 32-bit 寄存器（里面打包 2 个 16-bit 元素）；</p>
</li>
<li>
<p><code inline="">b16.x2</code>：每线程 2 个寄存器；</p>
</li>
<li>
<p><code inline="">b16.x4</code>：每线程 4 个寄存器。<br>
<code inline="">b8</code> 时每寄存器打包更多元素（4 个 8-bit）。</p>
</li>
</ul>
</li>
<li>
<p><code inline="">[addr]</code>：<strong>每个线程提供一个地址寄存器</strong>（指向 shared 内的一段），硬件按既定模式把 8×8 tile 的每行/列分配给不同线程寄存器。</p>
</li>
</ul>
<blockquote>
<p>小结：<code inline="">x</code> 的倍数越大，一次为后续 <code inline="">mma</code> 准备的 fragment 越多，也越省指令。</p>
</blockquote>
<hr>
<h1>执行与线程/寄存器映射（直观理解）</h1>
<ul>
<li>
<p>warp 内 32 线程被划分成 <strong>4 组（每组 8 线程）</strong>，每组 8 条 lane 提供 8 条起始地址，合起来 <strong>装载一个 8×8</strong>。</p>
</li>
<li>
<p><code inline="">x1</code>：warp 等价于并行处理 <strong>4 个 8×8</strong>（每 8 线程一组，各拿一个 tile）；</p>
</li>
<li>
<p><code inline="">x2/x4</code>：在同一次指令中为同一目的操作数加载 <strong>2/4 份 fragment</strong> 到更多寄存器，便于后续更大的 <code inline="">mma</code> 形状直接使用。</p>
</li>
<li>
<p><code inline="">.trans</code> 会在装载到寄存器时做 8×8 的转置，使寄存器布局直接符合 <code inline="">mma.sync</code> 所需（例如 <code inline="">row.col</code> 变体里 A 用非转置、B 常用转置）。</p>
</li>
</ul>
<blockquote>
<p>具体 lane 到元素的映射由硬件定义，你只需按官方推荐的“行指针计算方式”把每 lane 的 <code inline="">[addr]</code> 算对即可。</p>
</blockquote>
<hr>
<h1>与 <code inline="">mma.sync</code> 的常用组合（示例）</h1>
<pre><code class="language-cpp">// 假设 As、Bs 是 shared memory 中准备好的 tile
uint32_t a0,a1,a2,a3;  // A fragment (b16, x4)
uint32_t b0,b1;        // B fragment (b16, x2)

uint32_t a_addr = __cvta_generic_to_shared(As_lane_ptr);
uint32_t b_addr = __cvta_generic_to_shared(Bs_lane_ptr);

// A：不转置，x4
asm volatile(
  "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
  : "=r"(a0),"=r"(a1),"=r"(a2),"=r"(a3)
  : "r"(a_addr));

// B：转置，x2（常见搭配）
asm volatile(
  "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
  : "=r"(b0),"=r"(b1)
  : "r"(b_addr));

// 然后喂给 Tensor Core 做 16x8x16 的 MMA（示意）
uint32_t d0,d1,d2,d3;   // 累加结果片段
asm volatile(
  "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f32 "
  "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
  : "+r"(d0),"+r"(d1),"+r"(d2),"+r"(d3)
  :  "r"(a0),"r"(a1),"r"(a2),"r"(a3), "r"(b0),"r"(b1));
</code></pre>
<hr>
<h1>常见要点与坑</h1>
<ol>
<li>
<p><strong>地址必须是 shared 空间地址</strong>：用 <code inline="">__cvta_generic_to_shared()</code> 或 PTX <code inline="">cvta.to.shared.u32</code> 转换。</p>
</li>
<li>
<p><strong>对齐</strong>：确保 tile 起始地址 16B 对齐（<code inline="">aligned</code>）。shared 数组建议 <code inline="">__align__(128)</code>。</p>
</li>
<li>
<p><strong>所有 32 线程都要参与</strong>（warp 同步语义），不要在半 warp 里执行。</p>
</li>
<li>
<p><strong><code inline="">.trans</code> 用在恰当的操作数</strong>（通常 B 操作数需要）。</p>
</li>
<li>
<p><strong>Bank 冲突</strong>：规划好 shared 内的行跨距（stride），很多实现用 <code inline="">(+padding)</code> 避免 32-bank 冲突。</p>
</li>
<li>
<p><strong>配套形状</strong>：<code inline="">x4/x2</code> 选择应与目标 <code inline="">mma.sync</code> 的 <code inline="">m*n*k</code> 形状匹配，否则需要额外重排/指令。</p>
</li>
<li>
<p><strong>架构要求</strong>：<code inline="">ldmatrix</code> 一般要求 <strong>SM 7.5+</strong>（Turing 及以后）；<code inline="">b8</code> 相关在 Ampere+ 更常见。</p>
</li>
</ol>
<hr>
<h1>什么时候用</h1>
<ul>
<li>
<p>你已经把 A、B 的 tile 从 global 拷到 shared（常配 <code inline="">cp.async</code>）；</p>
</li>
<li>
<p>需要用 Tensor Core 做半精度/BF16/INT8 的 GEMM/卷积；</p>
</li>
<li>
<p>希望 <strong>一次指令</strong> 把 8×8 的片段按 <strong>硬件喜欢的布局</strong> 放进寄存器，减少显式重排。</p>
</li>
</ul>
<hr>
<h1>快速对照表</h1>

变体 | 元素类型 | 每线程寄存器数 | 说明
-- | -- | -- | --
m8n8.x1.shared.b16 | 16-bit | 1 | 加载 1 份 8×8 fragment
m8n8.x2.shared.b16 | 16-bit | 2 | 加载 2 份（常用于 B）
m8n8.x4.shared.b16 | 16-bit | 4 | 加载 4 份（常用于 A）
… .trans … | – | – | 对 8×8 做转置装载

