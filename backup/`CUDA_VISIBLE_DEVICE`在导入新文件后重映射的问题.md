打印出来的 CUDA_VISIBLE_DEVICES 是 "2,3"，但依然报错 RuntimeError: CUDA error: invalid device ordinal

可能的原因及解决方法：

1. CUDA_VISIBLE_DEVICES 和 GPU 设备编号的映射问题

当你设置 CUDA_VISIBLE_DEVICES=2,3 时，系统实际上将只识别设备 cuda:2 和 cuda:3。然而，在内部 PyTorch 中的设备编号会从 0 开始，表示的是可用的设备。也就是说，设置了 CUDA_VISIBLE_DEVICES="2,3" 后，实际可用的设备会变为：
	- cuda:0 实际上对应的是物理设备 cuda:2
	- cuda:1 实际上对应的是物理设备 cuda:3

这意味着如果你在代码中使用 self.embedding_model.to("cuda:3")，这实际上是访问物理设备 cuda:1（而不是设备 cuda:3），而 cuda:3 不再是有效的设备，因为它已经被映射到 cuda:1。

解决方法：
将代码中的 self.embedding_model.to("cuda:3") 改为 self.embedding_model.to("cuda:1")，即把 cuda:3 映射到 cuda:1。

2. 确认可用的设备编号

可以通过以下方式打印出 PyTorch 检测到的设备编号，并确保你使用的设备编号在可用范围内：

import torch
print(torch.cuda.device_count())  # 输出设备数量
for i in range(torch.cuda.device_count()):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")

这会帮助你确认系统中实际可用的 GPU 数量和设备名称。如果你发现只有 2 张 GPU 可用（编号为 cuda:0 和 cuda:1），而 cuda:2 或 cuda:3 不存在，那么就会导致报错。

3. 检查是否在多进程/多线程环境中运行

如果你在多进程或者多线程环境中使用 CUDA_VISIBLE_DEVICES，有时候环境变量可能不会正确传递给子进程或线程。这可能导致子进程无法访问指定的 CUDA 设备。

解决方法：
	- 确保你在每个进程或线程中都设置了正确的 CUDA_VISIBLE_DEVICES。
	- 如果使用多进程，确保每个进程都能读取到正确的环境变量。你可以在每个进程中再次打印 os.environ.get("CUDA_VISIBLE_DEVICES") 来验证。

4. 显式设置设备

在你的代码中，如果 CUDA_VISIBLE_DEVICES 环境变量设置正确，但仍然出错，可以尝试显式地设置 PyTorch 的 CUDA 设备，例如：
``` python
import torch

# 显式设置当前设备
device = torch.device("cuda:0")  # 或 "cuda:1"
self.embedding_model.to(device)
```
这将直接选择一个有效的设备，而不依赖于 CUDA_VISIBLE_DEVICES。

5. 检查是否存在 CUDA 驱动或版本不匹配问题

有时，尽管显卡驱动和 PyTorch 版本匹配，仍然可能发生兼容性问题。你可以通过以下命令检查 CUDA 和 PyTorch 的版本是否匹配：
``` python
import torch
print(torch.version.cuda)  # 输出 PyTorch 使用的 CUDA 版本
print(torch.cuda.is_available())  # 检查 CUDA 是否可用
```
确保 PyTorch 和 CUDA 驱动程序版本兼容。不同的 PyTorch 版本可能对 CUDA 有不同的要求。

综上所述：
	1. 调整设备编号：由于你设置了 CUDA_VISIBLE_DEVICES="2,3"，PyTorch 会将设备 cuda:2 映射为 cuda:0，cuda:3 映射为 cuda:1。因此，将代码中的 cuda:3 改为 cuda:1 可能解决问题。
	2. 打印并验证可用设备：通过 torch.cuda.device_count() 确认系统上实际可用的 GPU 数量，并检查是否存在设备号不匹配的情况。
	3. 确保多进程环境中的环境变量传递：确保在多进程或多线程环境中正确传递 CUDA_VISIBLE_DEVICES 环境变量。
	4. 显式设置设备：显式设置 torch.device("cuda:0") 等，以避免依赖 CUDA_VISIBLE_DEVICES 可能引发的问题。
