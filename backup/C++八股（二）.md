---
在 C++ 里，`static` 关键字有三种主要作用，下面为你详细介绍：

### 1. 修饰局部变量
当 `static` 用于修饰局部变量时，该变量的生命周期会延长至整个程序运行期间，而非局限于所在函数的执行期间。这意味着该变量只会被初始化一次，后续再调用这个函数时，这个变量会保留上一次调用结束时的值。

**示例代码**：
```cpp
#include <iostream>

void testStaticLocal() {
    static int count = 0;
    std::cout << "Count: " << count << std::endl;
    count++;
}

int main() {
    testStaticLocal();
    testStaticLocal();
    testStaticLocal();
    return 0;
}
```
**解释**：
在 `testStaticLocal` 函数里，`count` 被定义为静态局部变量。首次调用该函数时，`count` 初始化为 0 并输出，之后 `count` 自增为 1。再次调用此函数，`count` 不会重新初始化，而是保留上一次的值 1，输出后再自增为 2，以此类推。

### 2. 修饰全局变量和函数
当 `static` 用于修饰全局变量或函数时，会限制它们的作用域仅在定义它们的文件内，这就避免了在其他文件中通过 `extern` 关键字来引用它们。

**示例代码**：
- **file1.cpp**
```cpp
#include <iostream>
// 静态全局变量
static int globalStaticVar = 10;

// 静态函数
static void staticFunction() {
    std::cout << "This is a static function." << std::endl;
}

void testGlobalStatic() {
    std::cout << "Global static variable: " << globalStaticVar << std::endl;
    staticFunction();
}
```
- **main.cpp**
```cpp
#include <iostream>
// 调用 file1.cpp 中的函数
extern void testGlobalStatic();

int main() {
    testGlobalStatic();
    return 0;
}
```
**解释**：
在 `file1.cpp` 中，`globalStaticVar` 是静态全局变量，`staticFunction` 是静态函数，它们的作用域仅限于 `file1.cpp` 文件。在 `main.cpp` 里，无法直接访问 `globalStaticVar` 和 `staticFunction`，只能通过调用 `testGlobalStatic` 函数来间接使用它们。

### 3. 修饰类的成员变量和成员函数
当 `static` 用于修饰类的成员变量时，该变量会被所有类的对象共享，而非每个对象都有一份副本。当 `static` 用于修饰类的成员函数时，该函数不依赖于类的对象，可以直接通过类名来调用。

**示例代码**：
```cpp
#include <iostream>

class MyClass {
public:
    // 静态成员变量
    static int staticMemberVar;

    // 静态成员函数
    static void staticMemberFunction() {
        std::cout << "Static member variable: " << staticMemberVar << std::endl;
    }
};

// 初始化静态成员变量
int MyClass::staticMemberVar = 20;

int main() {
    // 通过类名调用静态成员函数
    MyClass::staticMemberFunction();

    // 修改静态成员变量的值
    MyClass::staticMemberVar = 30;

    // 再次调用静态成员函数
    MyClass::staticMemberFunction();

    return 0;
}
```
**解释**：
在 `MyClass` 类中，`staticMemberVar` 是静态成员变量，`staticMemberFunction` 是静态成员函数。`staticMemberVar` 被所有 `MyClass` 对象共享，`staticMemberFunction` 不依赖于类的对象，可以直接通过类名调用。在 `main` 函数中，我们通过类名调用 `staticMemberFunction` 来访问和修改 `staticMemberVar` 的值。 