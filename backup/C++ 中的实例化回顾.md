### 普通实例化

在 C++ 中，struct 和 class 在实例化时的语法是相同的。两者在实例化时都可以使用以下方式：
``` cpp
// 使用 class 定义和实例化
class Solution {
public:
    void printMessage() {
        std::cout << "Hello from Solution!" << std::endl;
    }
};

Solution solu; // 实例化一个 Solution 类型的对象

// 使用 struct 定义和实例化
struct Data {
    int value;
    void printValue() {
        std::cout << "Value: " << value << std::endl;
    }
};

Data dataInstance; // 实例化一个 Data 类型的对象
```

**区别**

- 访问权限：class 中的成员默认是 private，而 struct 中的成员默认是 public。
- 用法场景：struct 通常用于定义简单的数据结构，而 class 则多用于定义复杂的类和对象，包含更多功能和封装。

**实例化方式**

- 无论是 class 还是 struct，你可以直接使用 类型名 对象名; 的方式来实例化对象。
- 它们在实例化和使用时语法完全一致。

### new 关键字实例化

在 C++ 中，无论是 class 还是 struct，都可以使用 new 关键字进行动态内存分配。new 关键字会在堆上分配内存并返回对象的指针。

#### 使用 new 关键字实例化 class 和 struct

**class 的实例化：**

``` cpp
class Solution {
public:
    void printMessage() {
        std::cout << "Hello from Solution!" << std::endl;
    }
};

// 使用 new 关键字实例化 Solution
Solution* solu = new Solution(); // 返回的是指向 Solution 的指针

// 使用指针调用成员函数
solu->printMessage();

// 释放内存
delete solu;
```

**struct 的实例化：**

``` cpp
struct Data {
    int value;
    void printValue() {
        std::cout << "Value: " << value << std::endl;
    }
};

// 使用 new 关键字实例化 Data
Data* dataInstance = new Data(); // 返回的是指向 Data 的指针

// 使用指针调用成员函数和访问成员变量
dataInstance->value = 42;
dataInstance->printValue();

// 释放内存
delete dataInstance;
```

**说明**

- new 返回的类型：new 返回的是指向对象的指针，类型为 ClassName* 或 StructName*。
- 内存管理：使用 new 关键字在堆上分配内存后，需要使用 delete 释放内存以防止内存泄漏。
- 访问成员：使用 . 访问成员是针对对象，而使用 -> 是针对指针。

**使用 new 的优势**

- 动态分配：在运行时分配内存，可以灵活处理不确定数量的对象。
- 对象生存期：在堆上分配的对象在手动调用 delete 前一直存在，而在栈上分配的对象在超出作用域后会自动销毁。

**注意事项**

- 使用 new 时，确保在不需要对象时使用 delete 释放内存，否则会导致内存泄漏。
- 在现代 C++（C++11 及更高版本）中，建议使用智能指针（如 std::unique_ptr 或 std::shared_ptr）来自动管理动态内存，减少手动 delete 的风险：

``` cpp
#include <memory>

std::unique_ptr<Solution> solu = std::make_unique<Solution>();
solu->printMessage(); // 无需手动 delete，智能指针会自动释放内存
```

使用 new 和 delete 可以让你更好地控制对象的生命周期和内存管理，但使用智能指针更安全，推荐在现代 C++ 中使用智能指针来替代原生指针。

---

###注意和 Python 中的实例化写法作区分：

在 Python 中，实例化 class 和 struct（在 Python 中没有特定的 struct 关键字，一般使用 class 来表示结构体）是非常简单和直观的，直接通过调用类的构造函数即可。Python 不像 C++ 那样有 new 关键字，因为它的内存管理由 Python 解释器自动处理。

#### Python 中的类实例化

**在 Python 中，实例化类的语法是：**

``` python
class Solution:
    def __init__(self):
        print("Solution instance created")
    
    def print_message(self):
        print("Hello from Solution!")

# 实例化一个 Solution 对象
solu = Solution()  # 直接调用类名即可
solu.print_message()  # 调用实例方法
```

**Python 中的内存管理**

- 内存管理：Python 使用内存自动管理和垃圾回收机制，因此你不需要像 C++ 那样手动使用 delete。当一个对象没有引用时，Python 的垃圾回收器会自动释放内存。
- 无 new 关键字：在 Python 中，实例化对象不使用 new 关键字。直接调用类名 ClassName() 会调用类的构造函数 __init__()，并返回一个新实例。

**Python 中的“结构体”**

在 Python 中，没有类似于 C++ 的 struct，但可以通过 class 定义简单的数据结构，或者使用 namedtuple 或 dataclass 来实现轻量级的结构体。

**使用 class：**

``` python
class Data:
    def __init__(self, value):
        self.value = value

data_instance = Data(42)
print(data_instance.value)  # 输出 42
```

**使用 namedtuple：**

``` python
from collections import namedtuple

Data = namedtuple('Data', ['value'])
data_instance = Data(42)
print(data_instance.value)  # 输出 42
```

**使用 dataclass（Python 3.7+）：**

``` python
from dataclasses import dataclass

@dataclass
class Data:
    value: int

data_instance = Data(42)
print(data_instance.value)  # 输出 42
```

**总结**

- 在 Python 中，实例化对象时直接使用 ClassName()。
- Python 的内存管理是自动的，不需要手动使用 delete。
- 可以用 class 来定义结构体，也可以使用 namedtuple 或 dataclass 提供更轻量级的实现。