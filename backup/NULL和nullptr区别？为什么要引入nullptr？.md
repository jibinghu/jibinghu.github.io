### 区别

#### 1. 类型不同
- **`NULL`**：在 C++ 里，`NULL` 通常被定义为 `0` 或者 `(void*)0`。在 C 语言中，`NULL` 一般被定义为 `(void*)0`，但在 C++ 里，因为 `void*` 不能隐式转换为其他指针类型，所以 `NULL` 常被定义成 `0`。本质上，`NULL` 是一个整数常量。
- **`nullptr`**：`nullptr` 是 C++11 引入的一个新关键字，它的类型是 `std::nullptr_t`。`std::nullptr_t` 可以隐式转换为任意指针类型，这让它在语义上更明确地表示空指针。

#### 2. 函数重载解析不同
- **`NULL`**：当用于函数重载时，由于 `NULL` 本质是整数 `0`，在调用重载函数时可能会引发混淆。比如有接受整数和指针作为参数的重载函数，使用 `NULL` 调用时，可能会调用接受整数参数的函数。
- **`nullptr`**：`nullptr` 会精确匹配指针类型的参数，不会出现类似 `NULL` 的混淆问题。

### 引入 `nullptr` 的原因

#### 1. 消除二义性
在函数重载的场景下，使用 `NULL` 可能会导致函数调用的二义性。下面是一个示例代码：
```cpp
#include <iostream>

// 接受整数参数的函数
void func(int num) {
    std::cout << "Called func(int): " << num << std::endl;
}

// 接受指针参数的函数
void func(int* ptr) {
    if (ptr == nullptr) {
        std::cout << "Called func(int*): nullptr" << std::endl;
    } else {
        std::cout << "Called func(int*): " << *ptr << std::endl;
    }
}

int main() {
    // 使用 NULL 调用 func
    func(NULL);

    // 使用 nullptr 调用 func
    func(nullptr);

    return 0;
}
```
在上述代码中，当使用 `NULL` 调用 `func` 时，由于 `NULL` 被当作整数 `0`，所以会调用 `func(int)` 函数。而使用 `nullptr` 调用时，会调用 `func(int*)` 函数，避免了二义性。

#### 2. 语义更清晰
`nullptr` 明确表示空指针，相比 `NULL` 更能准确传达代码的意图。在阅读和维护代码时，`nullptr` 让代码的语义更加清晰易懂。

#### 3. 类型安全
`nullptr` 具有明确的类型 `std::nullptr_t`，只能用于指针相关的操作，而 `NULL` 作为整数常量，可能会被错误地用于非指针的上下文中。使用 `nullptr` 可以提高代码的类型安全性。 