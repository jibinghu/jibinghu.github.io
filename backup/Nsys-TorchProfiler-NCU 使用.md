主要参考：https://zhuanlan.zhihu.com/p/1945304372545291742

---

Blog 中有很多基础的知识，这里不做过多赘述。

要理解分布式工作负载中的 GPU 性能，需要考察模型算子与 GPU 设备的交互方式。 从宏观层面，可以将 GPU 操作分为三大类：

- 计算类 (COMP)
执行矩阵乘法等[数值计算](https://zhida.zhihu.com/search?content_id=262463237&content_type=Article&match_order=1&q=%E6%95%B0%E5%80%BC%E8%AE%A1%E7%AE%97&zhida_source=entity)
负责模型的所有数值运算处理
- 通信类 (COMM)
负责 GPU 设备间的[数据交换](https://zhida.zhihu.com/search?content_id=262463237&content_type=Article&match_order=1&q=%E6%95%B0%E6%8D%AE%E4%BA%A4%E6%8D%A2&zhida_source=entity)与同步
通常使用 NCCL 库（内核前缀为 “nccl”，如 NCCL_AllGather、NCCL_ReduceScatter、NCCL_AllReduce）
- 内存类 (MEM)
管理 GPU [内存分配](https://zhida.zhihu.com/search?content_id=262463237&content_type=Article&match_order=1&q=%E5%86%85%E5%AD%98%E5%88%86%E9%85%8D&zhida_source=entity)与释放
处理主机与设备间的数据传输：
Memcpy_H2D（主机到设备）
Memcpy_D2H（设备到主机）
Memcpy_D2D（设备到设备）
Memset…
现代 GPU（如 NVIDIA A100）支持多内核并发执行，可通过内核重叠技术缩短执行时间。常用实现方式是多 CUDA 流——不同 CUDA 流可以交错或并发运行，实现计算、通信和内存操作的重叠。

---


Nsight System -> nsys 使用：

Nsight Systems (nsys) 作为 CUDA Toolkit 的标准组件之一，随其一同安装。 其可执行文件路径为 ${CUDA_HOME}/bin/nsys。 nsys 的功能集与支持的 API 追踪范围随 CUDA 版本迭代而更新，建议使用较新版本以获得更全面的分析能力。

原理：

nsys 采用动态插桩（Dynamic Instrumentation）技术。它通过劫持链接器调用（Linker Interception），在程序运行时为关键的库函数（如 CUDA API、OS API）动态插入性能采集探针（Probes）。 此机制无需修改或重新编译[源代码](https://zhida.zhihu.com/search?content_id=262463237&content_type=Article&match_order=1&q=%E6%BA%90%E4%BB%A3%E7%A0%81&zhida_source=entity)，即可非侵入式地捕获函数调用、执行时间等[系统级](https://zhida.zhihu.com/search?content_id=262463237&content_type=Article&match_order=1&q=%E7%B3%BB%E7%BB%9F%E7%BA%A7&zhida_source=entity)事件。

用法

> nsys 的核心功能通过其命令行接口暴露，尤其适用于服务器环境。
> nsys 的主要子命令是 profile，用于执行并分析目标应用程序。

``` bash
nsys profile [options] <application> [application_args]
[options]: 控制分析行为的选项，如追踪内容、采样频率等。
<application>: 待分析的可执行文件或脚本。
[application_args]: 传递给应用程序的参数。
```

执行后，nsys 会生成一个 .nsys-rep 报告文件，该文件可由 Nsight Systems GUI 进行[可视化]分析。 其核心参数如下：


参数 | 功能描述 | 示例
-- | -- | --
--output=<filename> | 指定输出报告的文件名。 | -o my_report
--force-overwrite | 如果报告文件已存在，则强制覆盖。 | -f
--trace=<apis> | 指定要追踪的 API 集合，以逗号分隔。关键 API 包括 cuda, nvtx, osrt, mpi, cublas, cudnn。 | -t cuda,nvtx,osrt
--sample=<type> | 控制 CPU IP 采样方式 (process-tree, system-wide, none)。此选项会覆盖 --cpuctxsw 的设置。 | -s process-tree
--cpuctxsw=<scope> | 记录 CPU 上下文切换 (process-tree, system-wide, none)。仅在 --sample=none 时可独立设置。 | --cpuctxsw=none
--delay=<seconds> | 延迟指定秒数后开始性能采集。 | -y 10
--duration=<seconds> | 从采集开始计时，持续指定秒数后停止。 | -d 30
--capture-range=<type> | 指定用于控制采集开始 / 停止的机制。关键值为 nvtx。 | -c nvtx
--nvtx-capture=<range> | 当 -c nvtx 被指定时，此选项用于定义具体的 NVTX 范围 (Range) 名称。 | -p "training_phase"
--gpu-metrics-devices | 采集指定 GPU 的核心性能指标。注意：可能引入额外开销。更适合用于定性分析而非精确的性能测量。 | --gpu-metrics-devices=all

示例：

``` bash
# 默认配置已覆盖大部分通用场景
nsys profile -o report_default.nsys-rep ./my_app
# nsys profile 默认等价于 --trace=cuda,opengl,nvtx,osrt --sample=process-tree (Linux)

# 对于包含 MPI 的多节点应用，需显式添加 MPI 追踪
nsys profile \
    --trace=cuda,nvtx,osrt \
    --sample=process-tree \
    --output=report_overview.nsys-rep \
    ./my_app
# Windows 上 osrt 应该换成 wddm
```

``` bash
# 示例：假设瓶颈已定位在 GPU Kernel，关闭 CPU 相关采集以减少干扰
nsys profile \
    --trace=cuda,nvtx \
    --sample=none \
    --cpuctxsw=none \
    --output=report_gpu_focused.nsys-rep \
    ./my_app
```

对于长时间运行或具有明显初始化阶段的应用，仅分析其核心计算部分。

``` bash
# 方案 A：基于时间窗口的采集 
nsys profile --delay=60 --duration=120 -o report_window.nsys-rep ./my_app

# 方案 B：基于 NVTX [触发器](https://zhida.zhihu.com/search?content_id=262463237&content_type=Article&match_order=1&q=%E8%A7%A6%E5%8F%91%E5%99%A8&zhida_source=entity)的采集 
nsys profile --capture-range=nvtx --nvtx-capture=「training_phase」 --output=report_nvtx.nsys-rep ./my_app
```

Warning!!!
> ❗内存占用：分析复杂或长时间运行的应用会生成巨大的报告文件，打开和分析时需要消耗大量主机内存（建议 > 64GB）。
> ❗CUDA 版本兼容性：新版本的 CUDA Toolkit 通常伴随功能更强的 nsys，支持更丰富的 API 追踪和 GPU 指标。


---

时间线构图：

CUDA HW - GPU 硬件活动
CUDA HW 直接展示了你的 NVIDIA GPU 硬件真正在做什么，是 nsys 最重要的一行。它不受软件 API 调用的延迟影响，反映的是 GPU 芯片内部计算单元的实际工作状态。


CUDA HW 行通常包含多个子轨道（sub-track），代表了 GPU 上不同类型的工作队列或引擎：

<!-- Failed to upload "image.png" -->

Kernels (计算核心)：蓝色的条块，每个代表一个 CUDA Kernel 的执行
条块密集且连续: GPU 计算核心高度饱和，理想状态
条块之间有明显空隙: 性能问题信号，GPU 计算核心在等待
条块很短，数量极多: “小 Kernel 问题”，效率不高
Memory (内存操作)：绿色的条块，代表 GPU 与显存之间的数据传输
频繁且大的绿色条块: 表明程序有大量的数据传输
理想状态: 数据拷贝应与计算重叠

CUDA API - CUDA [应用程序接口](https://zhida.zhihu.com/search?content_id=262463237&content_type=Article&match_order=1&q=%E5%BA%94%E7%94%A8%E7%A8%8B%E5%BA%8F%E6%8E%A5%E5%8F%A3&zhida_source=entity)调用
记录了程序（运行在 CPU 上）何时调用了 CUDA 相关的函数。这些是 CPU 向 GPU [驱动程序](https://zhida.zhihu.com/search?content_id=262463237&content_type=Article&match_order=1&q=%E9%A9%B1%E5%8A%A8%E7%A8%8B%E5%BA%8F&zhida_source=entity)发出的”指令”。 通常显示为各种不同颜色的条块，每个都对应一个 CUDA API 调用：

cudaLaunchKernel - 启动计算核心
cudaMemcpy - 拷贝内存
cudaStreamSynchronize - 等待 GPU 完成任务 关键观察点：长条块的 cudaStreamSynchronize 意味着 CPU 必须停下来等待 GPU 完成队列里所有的任务，这会严重影响性能。

<img width="1357" height="291" alt="Image" src="https://github.com/user-attachments/assets/a5c1ecf0-7084-4e12-8c71-be0d05d90568" />

通常显示为各种不同颜色的条块，每个都对应一个 CUDA API 调用，例如 cudaLaunchKernel (启动一个计算核心), cudaMemcpy (拷贝内存), cudaStreamSynchronize (等待 GPU 完成任务)。

CUDA API 行显示的是 CPU 的行为，而不是 GPU 的行为。CPU 调用 cudaLaunchKernel 只是把一个任务“扔”进了 GPU 的指令队列，GPU 可能稍后才会真正执行它。即异步执行。
长条块的 cudaStreamSynchronize: 这是一个强烈的“同步点”信号，意味着 CPU 必须停下来，等待 GPU 完成队列里所有的任务。频繁或长时间的同步会严重影响性能，因为它破坏了 CPU 和 GPU 的并行性。
通过对比 CUDA API 和 CUDA HW，你可以分析 API 调用到硬件实际执行之间的延迟。


NVTX

> NVTX (NVIDIA Tools Extension) - 自定义应用标记

NVTX (NVIDIA Tools Extension) 允许[开发者](https://zhida.zhihu.com/search?content_id=262463237&content_type=Article&match_order=1&q=%E5%BC%80%E5%8F%91%E8%80%85&zhida_source=entity)在代码中插入自定义的范围 (Ranges) 和标记 (Markers)，是关联代码逻辑与时间线活动的关键桥梁。 如下方例子在训练过程中 前向传播 (forward)，反向传播 (backward)，优化器更新 (opt)使用 nvxt 进行标记。 简单的用@nvxt 装饰器语法装饰函数，或者用 with nvtx 进行管理上下文管理。

<img width="633" height="162" alt="Image" src="https://github.com/user-attachments/assets/7e54fd79-fa19-45d6-90f9-d647d1b7ee22" />

<img width="1440" height="109" alt="Image" src="https://github.com/user-attachments/assets/105e0d5f-3c2f-4786-a7d5-5f282a2790b3" />

<img width="678" height="234" alt="Image" src="https://github.com/user-attachments/assets/75311d0c-ed46-488f-b8bb-94fe5018513e" />


> 一些公式化的场景时间线解读：
“阶梯状”的 CUDA HW: 代表 GPU 在等待 CPU 提交任务，可能是 CPU 瓶颈或 API 调用开销。
密集的短 Kernel 条块: 小核函数过多，是算子融合的绝佳机会。
CUDA HW 中的大段空白: 严重的数据供给不足或同步等待。
cudaMemcpy 与 Kernel 执行串行: 没有实现计算与传输的重叠。


---

示例 1 GPU 空闲–数据集供给不足

代码举例：

``` python
# 导入所有需要的库
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
# 导入 transforms 模块，用于数据增强
from torchvision import transforms 
import time
import nvtx

# --- 全局配置参数 ---

# 设置计算设备
device = 「cuda」 if torch.cuda.is_available() else 「cpu」

# 设置训练的总轮数
num_epoch = 1

# 设置批次大小
batch_size = 128

# 设置数据加载器工作进程数量
# 切换这里的数值 (0, 2, 4 等) 并用 nsys 观察差异
num_workers = 32 


# --- 定义神经网络模型 ---

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, 5), nn.ReLU(),
            nn.Conv2d(32, 32, 5), nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(),
            nn.Conv2d(64, 64, 3), nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Flatten(1),
            nn.Linear(576, 10)
        )

    @nvtx.annotate(「forward」, color=「blue」)
    def forward(self, x):
        return self.layers(x)

# --- 定义训练和测试的单步函数 ---

def train_step(dataloader, model, loss_fn, opt):
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        # 1. 前向传播
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # 2. 反向传播和优化
        with nvtx.annotate(「backward」, color=「red」):
            loss.backward()
        
        with nvtx.annotate(「opt」, color=「green」):
            opt.step()
            opt.zero_grad()

def test_step(dataloader, model, loss_fn, opt):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0.0, 0.0
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y)
            correct += (pred.argmax(1) == y).type(torch.float).sum()
    
    test_loss /= num_batches
    correct /= size
    print(f『Accuracy: {(100*correct):.1f}%, AVG loss: {test_loss:.8f} \n』)

# --- 制造 CPU 瓶颈的自定义 Transform ---
class SlowTransform:
    「」「一个用于演示的慢变换，模拟复杂的 CPU 预处理」「」
    def __call__(self, img):
        # 模拟一个复杂的、CPU 密集型的图像处理过程
        # 注意：在真实项目中不要使用 time.sleep
        time.sleep(0.001) 
        return img

# --- 主执行流程函数 ---

def Pipe():
    # --- 1. 数据准备 ---
    
    # 核心：定义一个 CPU 密集的转换组合 (Data Augmentation)
    heavy_transforms = transforms.Compose([
        transforms.RandomRotation(15),       # 随机旋转 (CPU 密集)
        transforms.RandomCrop(28, padding=4), # 随机裁剪 (CPU 密集)
        transforms.ToTensor(),              # 转换为张量
        SlowTransform()                     # 自定义的慢变换
    ])

    # 下载并加载训练数据集，应用 CPU 密集型变换
    training_data = datasets.MNIST(
        root=「data」, train=True, download=True, transform=heavy_transforms,
    )

    # 下载并加载测试数据集
    test_data = datasets.MNIST(
        root=「data」, train=False, download=True, transform=transforms.ToTensor(),
    )

    total_samples = len(training_data)
    # 使用一半数据进行训练
    training_samples, _ = random_split(training_data, [total_samples//2, total_samples-total_samples//2])
    
    # 创建训练数据加载器 (开启 pin_memory 可以加速数据从 CPU 到 GPU 的传输)
    train_dataloader = DataLoader(
        training_samples, batch_size=batch_size, num_workers=num_workers, pin_memory=True
    )

    # 创建测试数据加载器
    test_dataloader = DataLoader(
        test_data, batch_size=batch_size, num_workers=num_workers, pin_memory=True
    )

    # 打印数据形状
    for X, y in test_dataloader:
        print(f「Shape of X [N, C, H, W]: {X.shape}」) 
        print(f「Shape of y: {y.shape} {y.dtype}」) 
        break 

    # --- 2. 模型、损失函数和优化器初始化 ---
    model = Model().to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    # --- 3. 训练循环 ---

    # 预热阶段：处理前几个批次以消除启动开销，这对性能分析很重要
    print(「Starting warmup...」)
    for _ in range(5): 
        # 确保数据加载器可以提供数据
        try:
            X, y = next(iter(train_dataloader))
        except StopIteration:
            break

        X, y = X.to(device), y.to(device)
        opt.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        opt.step()
    
    print(「Warmup finished. Starting profiling...」)
    torch.cuda.synchronize() # 确保预热结束

    for epoch in range(num_epoch):
        print(f「Epoch {epoch+1}\n-------------------------------」)
        now = time.time()
        
        # 训练步骤
        train_step(dataloader=train_dataloader, model=model, loss_fn=loss_fn, opt=opt)
        
        # 确保 GPU 所有任务完成
        torch.cuda.synchronize() 
        epoch_duration = time.time() - now
        
        print(f「Epoch duration: {epoch_duration:.2f} s」)
        print(f「avg batch train speed: {(len(train_dataloader)/epoch_duration):.2f} it/s」)
        
        now = time.time()
        test_step(dataloader=test_dataloader, model=model, loss_fn=loss_fn, opt=opt)
        torch.cuda.synchronize()
        test_duration = time.time() - now
        print(f「teststep speed: {(len(test_dataloader)/test_duration):.2f} it/s」)

if __name__ == 「__main__」:
    Pipe()
```

>  num_worker 的作用：

在深度学习训练中，GPU 训练速度往往比 CPU 读取和处理数据要快很多。如果数据预处理很复杂（比如你代码里的 RandomRotation、RandomCrop、以及 SlowTransform），CPU 处理会成为瓶颈，导致 GPU 等待数据，浪费算力。
num_workers 的作用就是开启多个 CPU 子进程，并行地从磁盘读取数据、执行 transform、然后把 batch 准备好，这样能更好地“喂饱”GPU，减少 GPU 等待。

常见设置：
num_workers=0
单进程加载（主进程负责读数据）。好处是简单，不会有进程间通信问题，但速度慢。
num_workers>0
开启多个子进程并行加载。每个 worker 会分担一部分 batch 的数据准备工作，一般会显著提升吞吐。
如果你的数据增强和预处理很轻量，num_workers=2~4 就够了。
如果预处理很重（比如图像变换很多、你这里还模拟了 time.sleep），可以设得大一些（如 8、16、甚至 32），不过太大可能会导致 上下文切换开销过大 或 内存不足。

---

示例 2 内存传输瓶颈–gpu 通信瓶颈

内存瓶颈主要分为两种形态：

带宽受限 (Bandwidth-Bound)：当程序需要读写的数据总量巨大时，性能会受到 GPU 显存（DRAM）或 PCIe 总线峰值传输速率的限制。
延迟受限 (Latency-Bound)：当程序进行大量微小、碎片化的内存操作时，性能会受到每一次操作的固定开销的限制。




