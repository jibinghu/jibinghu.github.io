<img width="1398" height="1870" alt="Image" src="https://github.com/user-attachments/assets/0ecad7b4-9570-4c32-b09d-6317afd382f5" />

如上图 Hopper 的 GPU 架构，每个 SM 中有多个warp scheduler，每个 scheduler 每个周期可以发送一个访存指令或一个计算指令。在上图中，对于不同的数据类型有不同的计算单元，另外还有 Tensor Core，每个 SM 最下侧是访存单元以及一个 SFU，特殊函数单元用于 sin/cos/tanh 等函数的硬件实现。

> 从上图可以看出来，每个 SM 中有 4 个 warp scheduler，事实上从 volta 到 blackwell 都是每个 SM 中有 4 个 warp scheduler。每个 block 在运行时会分配到一个 SM 中，每个 SM  的 Shared Memory 是在 block 内同步de，另外可以看到Nvidia在 SM 内 L1 Cache 和 Shared Memory 是在一块的，所以如果数据的局部性足够好，是可以将 L1 Cache 的效果发挥好的，即使不用 Shared Memory。但是对比 AMD 的 Rocm 架构：L1 Cache 和 Shared Memory 是在物理上是分开的，所以一定要用 Shared Memory 才能提高程序效果。

在 warp-unit 中需要dispatch 大于 32 个线程(1个 warp) 才能实现延迟隐藏，GPU 能高效运行的本质就是通过频繁低开销地切换 warp 来实现延迟隐藏。

<img width="580" height="211" alt="Image" src="https://github.com/user-attachments/assets/f69f82a9-3cc3-47cd-b098-84c1ea2ea72d" />

Cycle0： WARP0: INST0
Cycle1： WARP0: INST1
Cycle2: WARP0: INST2
Cycle3： WARP1: INST0
Cycle4： WARP1: INST1
Cycle5： WARP1: INST2
Cycle6： WARP0: INST3

发射7条指令，只需要6个cycle。如果不做de-sechdule，就需要9个cycle。

当 GPU 检测到某个指令会特别长，而且后续的指令会被当前指令 block 的时候，就会主动发起 warp deschedule，来切另外一个 warp 来运行。这个切换的开销非常小，因为我们每个寄存器都有自己的物理寄存器空间，不需要额外做 context 切换，只要不发生 Icache miss 的话，一个 cycle 内就可以切换成功。warp scheduler 这个硬件就是做这个操作用的。

实际在架构设计中切换的时机是精心设计的，可以由 ISA 中单独设置的位域（参考 cpp 中的 yield()函数）来由软件主动发起切换，也有可能是发生 cache miss，也有可能是特定指令（LD 指令通常会发生切换，ST 一般不会，由于有 write buffer），也可能是 FU 的 slot 已经满了或是遇到了特定的 fence。这些设计每家厂家都不尽相同。

在 GPU 中，每个 warp 内是锁步的(SIMT)。但是对于一个 SM来说并不是，一个 sm 中的warp 有可能分别处在算数执行或等待内存或挂起状态。每个 warp/线程是有自己的寄存器的，所以warp切换的时候不需要上下文切换，在一个 cycle 内就可以切换完成。

ISA 层面的显式 hint（如 YIELD）->在 NVIDIA SASS / PTX 中：某些指令可以携带 yield hint，告诉 scheduler：“这条指令之后，我可能不是最佳候选”。

data cache miss 和 instruction cache miss：

项目 | D-Cache miss | I-Cache miss
-- | -- | --
影响范围 | 单个 warp | 可能是整个 SM
scheduler 能否换 warp | 可以 | 往往不行
是否容易被 latency hiding | 是 | 很难

> I-Cache 在 SM 内是共享前端资源

``` shell
Instruction Fetch
   ↓
I-Cache (per SM)
   ↓
Decode / Dispatch
   ↓
Warp Scheduler
```

当 I-Cache miss 发生时：

- 前端无法继续提供指令
- scheduler 没有“可发射指令”
- 即使 SM 里有几十个 warp
- 也可能 全部一起等

关注 icache：

```
用 profiler 证实，而不是猜

关注指标：
sm__inst_executed
sm__icache_requests
sm__icache_misses
前端 stall reason（Nsight Compute）
```

指令一致性只保证在“同一个 warp 内部”，不保证在同一个 SM 内，更不保证在不同 SM 之间。

I-Cache 是 per-SM 的，而不是全 GPU 共享的。

> 当 kernel 的“指令工作集（instruction working set）”，超过单个 SM 的 I-Cache 容量，且 warp 的 PC 访问模式缺乏局部性（非顺序、分散跳转），就会产生频繁的 I-Cache miss，进而导致 SM 前端停顿（front-end stall），使得 warp scheduler 即使有大量 warp 也无法发射指令，最终显著影响执行效率。

data cache：

在 GPU 中没办法类似 CPU 的 MOESI 协议保持 cache coherence，在主流 NVIDIA / AMD GPU 架构中：L1 cache 默认是 incoherent 的（跨 SM 不一致），只有 L2 cache 才是全局一致的。

GPU 本来就规范分块处理数据，真的需要跨 SM 协调数据一致性时，可以使用原子操作/memory fence() 来实现同步。所以对于 data cache 采用 lazy write-out：所谓 lazy write-through 和 write-through 的区别便是 lazy write-through 会等一个 memory barrier 或者计时器定时才会写回，来避免一些琐碎小数据占据总线带宽。

> 回顾：MOESI 协议：在多个 cache 可能缓存同一内存地址时，保证“读到的值”和“写入的值”在逻辑上是正确的。

状态 | 含义
-- | --
M (Modified) | 只在本 cache 中，且已修改，内存是旧的
O (Owned) | 可被多个 cache 共享，但当前 cache 负责回写
E (Exclusive) | 只在本 cache 中，未修改
S (Shared) | 多个 cache 共享，未修改
I (Invalid) | 无效

在cuda里，需要通过memory_barrier来确保L1的一致性，最简单的办法就是__syncthreads(),它做线程同步的同时会插一个memory barrier，cuda也有别的原语能实现效果，不过我做cuda比较少啦，一般GPU公司都会有自己的memory fence指令。

基于以上的出发点，也就理解了为什么会有呢么多的memory layout，什么linear，什么tiled linear，什么twiddle， 都是为了让访存可以空间上连续起来，最后多个线程的访存命令打包成一个transaction发给总线来最大程度上地节省总线带宽。

在 Memory ISA 设计中，load/store 指令往往可以携带缓存策略提示，用于影响数据进入哪一级 cache、是否参与替换以及替换优先级。在 GPU 上，为了提升频繁访问数据的命中率，既可以通过显式的持久化缓存机制提高 cache line 的保留优先级，也可以通过只读加载指令将数据引导至独立的只读缓存路径，从而避免与普通数据访问竞争。CUDA 中的 __ldg() 属于后者，其提升命中率的原因在于缓存分流而非提高 cache 替换优先级。

这里重点讲讲broadcast，什么是broadcast？broadcast就是当你多个线程去读同一个地址（极端一点，32个线程读同一个地址），你不需要连续发32个transaction，而是发一个transaction，从总线读回来之后呢，通过专门设计的broadcast电路，来共享这个数据。所谓broadcast电路，其实就是一路数据线分裂成32路再加一点控制电路，没什么特殊的。对于const 内存，一般都会有broadcast电路。

附一个访存示例：

``` cuda
__global__ void loadtest(volatile float* data){
    float m;
    m = data[blockIdx.x * blockDim.x + threadIdx.x];
}
```

``` ptx
.visible .entry _Z8loadtestPVf(
	.param .u64 _Z8loadtestPVf_param_0
)
{
	.reg .f32 	%f<2>;
	.reg .b32 	%r<5>;
	.reg .b64 	%rd<5>;

	ld.param.u64 	%rd1, [_Z8loadtestPVf_param_0];
	cvta.to.global.u64 	%rd2, %rd1;
	mov.u32 	%r1, %ctaid.x;
	mov.u32 	%r2, %ntid.x;
	mov.u32 	%r3, %tid.x;
	mad.lo.s32 	%r4, %r1, %r2, %r3;
	mul.wide.u32 	%rd3, %r4, 4;
	add.s64 	%rd4, %rd2, %rd3;
	ld.volatile.global.f32 	%f1, [%rd4];
	ret;
}
```

ld.param.u64 %rd1, [_Z8loadtestPVf_param_0]; 这一步是从param加载参数到寄存器里。这个param字段是什么意思，就是kernel的传参。一般硬件里会有一个模块在运行kernel之前把参数传到特定的位置。

cvta.to.global.u64 如果你狠狠RTFM过的话，你会知道这个指令会返回一个global的地址。呢为什么要有这一步呢，是因为一开始传进来的rd1是一个general的指针。这一步不一定触发MMU（MMU设计的时候一般会留一个touch语义，就是只蹭蹭（x），不用真的发起transaction，主要目的是更新TLB，让TLB去prefetch）。

再后面依托全是计算索引，我们无需在乎。

ld.volatile.global.f32 %f1, [%rd4]; 直到运行这一行，我们算正式地运行了ld。其中global就是指从global的DDR里读，volatile是指编译器不可以优化这一条指令（删除或重排）。重排这里不解释了，在CPU的设计中也常提到指令重排，作为基本知识不再赘述。







