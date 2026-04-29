Fortran 调用 C 代码主要有 现代标准方式（推荐） 和 传统旧式方式（不推荐）。

目前最通用、最标准的方法是利用 Fortran 2003 标准引入的 ISO_C_BINDING 模块。这使得 Fortran 和 C 之间的数据类型映射和函数调用变得标准化且可移植。以下是详细的几种方式及核心注意事项：

1. 使用 ISO_C_BINDING (标准推荐方式)

这是目前开发中最正规的方式。通过 `use, intrinsic :: iso_c_binding` 模块，你可以明确指定 C 语言对应的类型（如 c_int, c_double, c_ptr 等），并使用 bind(c) 属性来声明接口。

核心步骤：

1. 定义 C 函数。
2. 在 Fortran 中编写 Interface 块：使用 bind(c) 绑定函数名。
3. 使用 value 属性：C 语言默认是传值 (Pass by Value)，而 Fortran 默认是传址 (Pass by Reference)。如果 C 函数参数不是指针，Fortran 端必须加 value 属性。

示例代码
C 代码 (`math_ops.c`)：

``` cpp
#include <stdio.h>// 一个简单的加法函数int add_in_c(int a, int b) {

    return a + b;

}// 修改数组内容的函数void modify_array(double* arr, int n) {

    for(int i = 0; i < n; i++) {

        arr[i] = arr[i] * 2.0;

    }

}
```

Fortran 代码 (main.f90)：

``` fortran
program main

    use, intrinsic :: iso_c_binding
    implicit none

    interface
        ! 1. 对应 int add_in_c(int a, int b)
        function add_in_c(a, b) result(res) bind(c, name="add_in_c")
            import :: c_int
            implicit none
            integer(c_int), value, intent(in) :: a, b  ! 注意 value 属性
            integer(c_int) :: res
        end function add_in_c

        ! 2. 对应 void modify_array(double* arr, int n)
        subroutine modify_array(arr, n) bind(c, name="modify_array")
            import :: c_double, c_int
            implicit none
            real(c_double), intent(inout) :: arr(*) ! 传递数组指针/引用
            integer(c_int), value, intent(in) :: n
        end subroutine modify_array
    end interface

    integer(c_int) :: x, y, sum
    real(c_double) :: my_array(5) = [1.0, 2.0, 3.0, 4.0, 5.0]

    ! 调用 C 函数
    x = 10
    y = 20
    sum = add_in_c(x, y)
    print *, "C Result:", sum

    ! 调用处理数组的 C 函数
    call modify_array(my_array, 5)
    print *, "Modified Array:", my_arrayend program main
```

编译与链接：

``` bash
gcc -c math_ops.c
gfortran -c main.f90
gfortran main.o math_ops.o -o app
./app
```

2. 使用 C 指针 (C_PTR 和 C_F_POINTER)

当你需要处理动态内存分配，或者 C 语言返回一个指针给 Fortran 时，需要使用 type(c_ptr)。
- 场景：C 分配内存，Fortran 使用。
- 方法：C 返回 void* 或 double*，Fortran 接收为 type(c_ptr)，然后使用 call c_f_pointer(cptr, fptr, shape) 将其转换为 Fortran 指针。

3. 传统/旧式方法 (不再推荐)

在 Fortran 2003 之前，开发者通常依赖编译器的特定行为（Name Mangling）。
- 原理：Fortran 编译器通常会在函数名后加下划线（例如 mysub 变成 mysub_）。C 代码必须手动匹配这个名字。

C 代码写法：

``` c
// 必须手动加下划线以匹配 gfortran 的默认行为void old_style_func_(int *a) { // Fortran 默认传指针，所以 C 这边必须是指针
    *a = *a + 1;
}
```

缺点：不可移植（不同编译器加下划线的规则不同），类型安全性差，字符串处理极其麻烦。

4. 关键注意事项 (Pitfalls)

A. 内存布局 (Column-major vs Row-major)
Fortran 数组是列优先 (Column-major) 存储的，而 C 语言数组是行优先 (Row-major) 存储的。
影响：如果你在 C 中像 arr[i][j] 这样访问由 Fortran 传入的二维数组，你会发现矩阵是转置的。
解决：在 C 代码中交换索引访问，或者在 Fortran 端传递转置后的数组。

B. 字符串处理
这是最容易出错的地方。
C 语言：字符串以空字符 \0 结尾。
Fortran：字符串是固定长度的，没有 \0，通常用空格填充。

解决：
Fortran 传给 C：需要显式添加 c_null_char，例如 type(c_ptr) :: c_str = c_loc(f_str // c_null_char)。
C 传给 Fortran：Fortran 需解析直到遇到 \0。
C. 布尔值
Fortran 的 LOGICAL 和 C 的 int (或 _Bool) 内部表示可能不同。建议在接口中传递 integer(c_int)，然后在 Fortran 内部根据 0 或 1 转换为 LOGICAL。

---

类似这种方式的混合编程就是单纯的链接是吗：

``` fortran
!    subroutine trid_cpp_c_sw(delpc, delp, ptc, pt, u,v, w, uc,vc, ua,va, wc,  &
!                             ut, vt, dt2, hydrostatic, dord4)
!       real, intent(INOUT), dimension(isd:ied,  jsd:jed+1):: u, vc
!       real, intent(INOUT), dimension(isd:ied+1,jsd:jed  ):: v, uc
!       real, intent(INOUT), dimension(isd:ied, jsd:jed):: delp,  pt,  ua, va, w
!       real, intent(OUT  ), dimension(isd:ied, jsd:jed):: delpc, ptc, ut, vt, wc
!       real,    intent(IN) :: dt2
!       logical, intent(IN) :: hydrostatic
!       logical, intent(IN) :: dord4

!       call trid_c_sw(sw_corner, se_corner, ne_corner, nw_corner, hydrostatic, dord4,         &
!                      delpc, delp, ptc, pt, u, v, w, uc, vc, ua, va, wc, ut, vt, dt2,         &
!                      cosa_u, cosa_v, cosa_s, rsin_u, rsin_v, rsin2, dx, dy, sina_u, sina_v,  &
!                      rarea, rarea_c, dxc, dyc, fC, rdxc, rdyc, )

!    end subroutine trid_cpp_c_sw
```

简单来说，**是的，本质上就是“单纯的链接”**。

但为了让这个链接能正常工作且程序不崩溃，**必须在代码层面严格遵守一套“隐形契约”**。
您提供的代码片段看起来非常像是一个 **Fortran Wrapper（包装层）**，它的作用通常是把 Fortran 的数据结构整理好，然后把内存地址（指针）“扔”给底层的 C/C++ 函数。

我们可以从以下三个层面来深度解析这种混合编程方式：

1. 物理层面：确实是“单纯的链接”

在编译过程中，操作确实非常简单直接：
1. **编译 Fortran**：编译器（如 `gfortran`）把您的 `.f90` 代码编译成二进制目标文件（`.o`），其中 `call trid_c_sw` 会生成一个查找符号（Symbol）的指令（通常是 `trid_c_sw_`，带下划线）。
2. **编译 C/C++**：编译器（如 `gcc`）把 C 代码编译成 `.o` 文件，生成一个函数入口符号。
3. **链接**：链接器（`ld`）把这两个 `.o` 文件拼在一起。只要 Fortran 找的符号名和 C 提供的符号名能对上，链接就成功了。

2. 逻辑层面：不仅是链接，更是“内存地址的传递”

您展示的这段代码非常典型（看起来很像 FV3 动力核心或者是某种偏微分方程求解器的代码）。在这个 `call` 语句背后，发生的事情主要是 **“传址” (Pass by Reference)**。
看看您传入的变量：

```fortran
real, intent(INOUT), dimension(isd:ied, jsd:jed+1):: u
logical, intent(IN) :: hydrostatic

```

当 Fortran 执行 `call trid_c_sw(..., u, ...)` 时，它实际传给 C 语言的是：
* **数组 `u**`：传过去的是数组首元素的**内存地址**（指针）。C 语言那边对应的必须是 `double* u` 或 `float* u`。
* **逻辑值 `hydrostatic**`：传过去的也是**地址**（指针），指向一个存储布尔值的内存块。C 语言那边必须用 `int*` 或 `bool*` 来接收。

3. 这段代码的潜在风险（为什么不能“随便链接”）

虽然原理是链接，但如果这是旧式写法（没有使用 `bind(c)`），这种方式非常脆弱。您需要特别注意以下几点，否则程序算出全是乱码或者直接 Segfault：

A. 符号名匹配 (Name Mangling)
如果 `trid_c_sw` 是 C 代码，它必须写成：

```c
// C 代码必须手动加下划线来迎合 Fortran
void trid_c_sw_(double* sw_corner, ..., double* u, ..., int* hydrostatic, ...) { 
    // ...
}
```

或者，如果它是 C++ 代码，还必须加上 `extern "C"` 来防止 C++ 改变函数名：

```cpp
extern "C" {
    void trid_c_sw_(...);
}
```

B. 数组的“形状”丢失

Fortran 传给 C 的只是一个首地址指针。

* C 语言**不知道** `u` 是一个 `(isd:ied, jsd:jed+1)` 的二维数组。
* C 语言**不知道**数组的边界 `isd` 或 `ied` 是多少。
* **结论**：您必须像代码里那样，手动把 `isd, ied, jsd` 等维度信息作为整数参数传给 C，C 才能正确地计算索引（例如 `u[j * stride + i]`）。

C. 逻辑值的坑

Fortran 的 `logical` 并不是标准的 `int`。
* 有的编译器 `true` 是 1，`false` 是 0。
* 有的编译器 `true` 是 -1。
* **这种直接链接的方式，要求 C 语言那边非常清楚 Fortran 编译器的内部表示，否则判断 `if(*hydrostatic)` 可能会出错。**