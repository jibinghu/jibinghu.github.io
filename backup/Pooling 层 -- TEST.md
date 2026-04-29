基于 Torch 的脚本：
``` python
import torch
import torch.nn as nn

# Create a sample tensor (2D)
input_tensor = torch.tensor([[1., 2., 3., 4.],
                             [5., 6., 7., 8.],
                             [9., 10., 11., 12.],
                             [13., 14., 15., 16.]])

# Reshape the tensor to 1x1x4x4 (as expected by pooling layers for 2D input)
input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)

# Max Pooling with padding
max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
max_pooled_output = max_pool(input_tensor)

# Average Pooling with padding
# avg_pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
# avg_pooled_output = avg_pool(input_tensor)

print("Input Tensor:\n", input_tensor)
print("\nMax Pooled Output with Padding:\n", max_pooled_output)
print("\nAverage Pooled Output with Padding:\n", avg_pooled_output)
```

---

### 4 * 4 的 Pooling 层是比较典型的，所以进行多方面测试：

- **kernel_size=2, stride=2, padding=0：**
``` bash
Max Pooled Output with Padding:
 tensor([[[[ 6.,  7.,  8.],
          [10., 11., 12.],
          [14., 15., 16.]]]])
```
这部分还是没有异议的，当Stride=1时从开头进行Pooling；
- **kernel_size=2, stride=2, padding=0：**
``` bash
 tensor([[[[ 6.,  8.],
          [14., 16.]]]])
```
当Stride足够覆盖一次时，也是没有异议的。
- **kernel_size=2, stride=3, padding=0：**
``` bash
 tensor([[[[6.]]]])
```
### 含有 Padding 操作(pad should be at most half of effective kernel size)：
- **kernel_size=2, stride=2, padding=2：**
``` bash
 tensor([[[[ 1.,  2.,  3.,  4.,  4.],
          [ 5.,  6.,  7.,  8.,  8.],
          [ 9., 10., 11., 12., 12.],
          [13., 14., 15., 16., 16.],
          [13., 14., 15., 16., 16.]]]])
```
- **kernel_size=2, stride=2, padding=2：**
``` bash
 tensor([[[[ 1.,  3.,  4.],
          [ 9., 11., 12.],
          [13., 15., 16.]]]])
```
- **kernel_size=2, stride=3, padding=2：**
``` bash
 tensor([[[[ 1.,  4.],
          [13., 16.]]]])
```