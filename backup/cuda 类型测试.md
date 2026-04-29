btw -> 看不到 nvcc时的解决办法：

``` cuda
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
```

类型：

``` cpp
#include <iostream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>


int main(){
        std::cout << "以字节为单位：" << std::endl;
        std::cout << "sizeof(uint4)： " << sizeof(uint4) << std::endl;
        std::cout << "sizeof(int4)： " << sizeof(int4) << std::endl;
        std::cout << "sizeof(int)： " << sizeof(int) << std::endl;
        std::cout << "sizeof(__half)： " << sizeof(__half) << std::endl;
}
```

输出：

> 以字节为单位：
> sizeof(uint4)： 16
> sizeof(int4)： 16
> sizeof(int)： 4
> sizeof(__half)： 2

构造方式：CUDA 提供 make_int4(...) 和 make_uint4(...)

<vector_type.h>(被<cuda_runtime.h>包含)的数据类型 int4 和 uint4 实质上都是结构体：

``` cpp
// int4
typedef struct __device_builtin__ {
    int x, y, z, w;
} int4;

// uint4
typedef struct __device_builtin__ {
    unsigned int x, y, z, w;
} uint4;
```
