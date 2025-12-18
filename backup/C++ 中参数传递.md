对取地址 $、解引用 * 等详细回顾：

一、形参与实参：

形参（parameter）：函数声明/定义中写的参数变量。
实参（argument）：调用函数时传进去的表达式。

``` cpp
void f(int x);     // x 是形参
int a = 3;
f(a);              // a 是实参
f(3);              // 3 也是实参
```

在 c++ 中，传参的方式可以分为三类：

1. 按值传递：`T x`
2. 按引用传递：`T& x` / `const T$& x`
3. 按指针传递：`T *x` / `const T *x`

根本区别是：调用者能否影响调用者的对象，以及是否会发生拷贝/移动。

二、 `&` 的含义：

1. & 在“类型里”：引用类型

`T&`：左值引用（可修改绑定对象）
`const T&`：只读引用（避免拷贝，且可绑定临时对象）
`T&&`：右值引用（用于移动语义/完美转发）

``` cpp
void g(int& x) { x = 10; }        // 修改调用者变量
void h(const int& x) { /*只读*/ } // 不改调用者
```

2. & 在“表达式里”：取地址运算符

`&obj` 得到 T* 指针

``` cpp
int a = 3;
int* p = &a;  // &a 是取地址，类型是 int*
```

在这种情况下的输出：
``` shell
p: 0x102bf0000
*p: 3
```

三、`*` 的含义

1. `*` 在“类型里”：指针类型

`T*`：指向 T 的指针
`T**`：指向“指针(T*)”的指针（俗称二级指针）

``` cpp
int a = 1;
int* p = &a;    // p 是指针
int** pp = &p;  // pp 是二级指针
```

2. `*` 在“表达式里”：解引用运算符

- *p 访问 p 指向的对象

``` cpp
int a = 1;
int* p = &a;
*p = 2;  // 修改 a
```

const 放在哪：const T* vs T* const（非常高频）

- 规则：const 修饰其左侧最近的类型（无左侧则修饰右侧）
- const T* p：指向 const T 的指针（不能通过 p 改 T）
- T* const p：const 指针（p 不能改指向，但能改 T）
- const T* const p：都不能改

``` cpp
int a=1,b=2;

const int* p1 = &a; // *p1 只读，p1 可指向 b
p1 = &b;            // ok
//*p1 = 3;          // error

int* const p2 = &a; // p2 固定指向 a，但可以改 a
*p2 = 3;            // ok
//p2 = &b;          // error
```

常见“标准陷阱”与判定
8.1 T& 不能绑定临时值
``` cpp
void f(int& x);
f(3);      // error：3 是临时量
```
8.2 const T& 可以绑定临时值（延长临时生命周期到引用作用域）
``` cpp
void g(const std::string& s);
g("abc");  // ok：会构造临时 std::string 绑定到 const&
```
8.3 指针实参要么是指针，要么取地址
``` cpp
void h(int* p);

int a=1;
h(&a);     // ok
h(a);      // error：a 不是 int*
```
8.4 二级指针需要“指针变量的地址”
``` cpp
void alloc(int** pp);

int* p=nullptr;
alloc(&p);     // ok
//alloc(p);    // error：p 是 int*，不是 int**
```

在工程代码中，绝大多数情况下不应该用 T** 表示二维数组。