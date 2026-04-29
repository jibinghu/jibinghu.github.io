## final和override的作用？final为什么能提高代码执行效率？

> override：保证在派生类中声明的重载函数，与基类的虚函数有相同的签名，作用就是用于编译期代码检查。
final：阻止类的进一步派生和虚函数的进一步重写，同时也是一种名为去虚拟化的优化技巧，相当于把运行期多态转换为了编译期多态，提高了执行效率。


### `final` 和 `override` 的作用：

1. **`final`**：
   - 在 C++11 引入了 `final` 关键字，它有两个主要作用：
     - **防止继承**：当 `final` 修饰一个类时，表示这个类不能被继承。例如：
       ```cpp
       class Base {
           // 基类
       };

       class Derived final : public Base { // 错误，不能从 `Derived` 类继承
       };
       ```
     - **防止重写**：当 `final` 修饰一个虚函数时，表示该函数不能在派生类中被重写。例如：
       ```cpp
       class Base {
       public:
           virtual void func() final { // 不能被重写
               std::cout << "Base func" << std::endl;
           }
       };

       class Derived : public Base {
       public:
           // 错误：无法重写 `func` 因为它被标记为 `final`
           void func() override {
               std::cout << "Derived func" << std::endl;
           }
       };
       ```

2. **`override`**：
   - `override` 是用于显式标明某个函数是重写基类的虚函数。这样做可以让编译器检查这个函数是否确实重写了一个虚函数。
   - 如果函数没有正确重写基类中的虚函数（比如拼写错误、签名不匹配等），编译器会报错。
     ```cpp
     class Base {
     public:
         virtual void func() { std::cout << "Base func" << std::endl; }
     };

     class Derived : public Base {
     public:
         void func() override { // 明确表示重写了 `Base` 类的 `func` 方法
             std::cout << "Derived func" << std::endl;
         }
     };
     ```

### 为什么 `final` 能提高代码执行效率？

1. **避免虚函数查找**：
   - `final` 表示该类或虚函数不允许被继承或重写。这样编译器在生成代码时，可以使用更高效的方式来调用函数，而不需要进行虚函数查找。
   - 对于一个类中的虚函数，编译器通常会使用虚函数表（VTable）来动态查找对应的函数。而 `final` 标记的类或函数，编译器知道它不会再被继承或重写，因此可以直接调用函数，而不需要查找虚函数表，从而提高了运行时效率。

2. **优化机会**：
   - 对于 `final` 类，编译器可以进行更激进的优化，因为它知道该类无法被扩展。
   - 对于 `final` 虚函数，编译器可以直接调用函数而不使用虚表查找。也就是说，函数调用会变成普通的静态函数调用，从而去除了虚拟调用的开销。

### 示例代码：

```cpp
#include <iostream>

class Base {
public:
    virtual void func() final { // `final` 防止重写
        std::cout << "Base func" << std::endl;
    }
};

class Derived : public Base {
public:
    // 错误：不能重写 `func` 因为它是 `final`
    void func() override {
        std::cout << "Derived func" << std::endl;
    }
};

int main() {
    Base* obj = new Base();
    obj->func(); // 调用 Base 类的 func，虚函数调用会被优化为静态调用

    delete obj;
    return 0;
}
```

总结：`final` 可以减少继承和重写的开销，使得编译器可以进行更多优化，从而提高代码执行效率。

---

在 C++ 中，虚函数的实现依赖于 **虚函数表（VTable）** 和 **虚函数指针（VPtr）** 机制。这个机制是 C++ 支持多态性和动态绑定的关键。

### 虚函数在内存中的实现：

1. **虚函数表（VTable）**：
   - 每个含有虚函数的类都会有一个虚函数表（VTable）。虚函数表是一个指向类的虚函数的指针数组。它包含了该类所有虚函数的地址。每个类只有一张虚函数表，除非它有虚函数被重写。
   - 如果一个类没有虚函数，编译器就不会创建虚函数表。

2. **虚函数指针（VPtr）**：
   - 每个对象实例都有一个指向虚函数表的指针（VPtr）。这个指针通常作为对象的一个隐藏成员存储在对象的内存布局中。每个对象实例会有自己的 VPtr，指向其类的虚函数表。

### 内存布局：

- 假设我们有一个类 `Base`，它含有一个虚函数 `func`。每个 `Base` 对象都含有一个虚函数指针（VPtr），指向该类的虚函数表（VTable）。如果类 `Derived` 继承自 `Base` 并重写了 `func`，则 `Derived` 对象也会包含自己的 VPtr，但该 VPtr 会指向 `Derived` 类的虚函数表。

### 示例：
```cpp
#include <iostream>

class Base {
public:
    virtual void func() {
        std::cout << "Base func" << std::endl;
    }
};

class Derived : public Base {
public:
    void func() override {
        std::cout << "Derived func" << std::endl;
    }
};

int main() {
    Base* base = new Derived();
    base->func();  // 调用 Derived::func，使用虚函数机制
    delete base;
    return 0;
}
```

### 内存分析：

1. **虚函数表（VTable）**：
   - `Base` 类的虚函数表会包含指向 `Base::func` 的指针。
   - `Derived` 类的虚函数表会包含指向 `Derived::func` 的指针（即使 `Derived` 类重写了 `func`，它会覆盖 `Base` 类的虚函数）。

2. **虚函数指针（VPtr）**：
   - 对于 `Base` 类型的指针 `base` 指向 `Derived` 类对象时，`base` 对象的 VPtr 会指向 `Derived` 类的虚函数表，而不是 `Base` 类的虚函数表。
   - 这样，当调用 `base->func()` 时，实际调用的是 `Derived::func`，而不是 `Base::func`，这是因为 `base` 的 VPtr 指向的是 `Derived` 类的虚函数表。

### 内存结构示意：

假设我们有如下类结构：

```cpp
class Base {
public:
    virtual void func() {
        std::cout << "Base func" << std::endl;
    }
};

class Derived : public Base {
public:
    void func() override {
        std::cout << "Derived func" << std::endl;
    }
};
```

- `Base` 类的虚函数表（VTable）：
  - `Base::func` 的地址。
- `Derived` 类的虚函数表（VTable）：
  - `Derived::func` 的地址。

对于一个 `Derived` 对象：

- 它的内存结构可能是：
  1. **VPtr**（指向 `Derived` 类的虚函数表）。
  2. **成员变量**（如果有的话）。

在 `main` 函数中，创建了一个 `Base*` 类型的指针 `base`，指向一个 `Derived` 对象。在调用 `base->func()` 时，实际执行的是 `Derived::func()`，而不是 `Base::func()`，因为 `base` 的 VPtr 指向的是 `Derived` 类的虚函数表。

### 为什么使用虚函数表：

1. **多态性**：虚函数表允许动态绑定，支持在运行时决定调用哪个函数。即使你使用基类指针或引用，程序也会根据对象的实际类型来调用正确的函数。
2. **性能代价**：虽然虚函数提供了多态性，但它们引入了一定的性能开销，因为每次调用虚函数时，都需要通过虚函数表进行查找。

### 虚函数的内存开销：

- **VTable**：每个含有虚函数的类需要一个虚函数表。虚函数表的大小通常等于类中虚函数的个数，每个虚函数的表项通常是一个指向该虚函数的指针。
- **VPtr**：每个包含虚函数的对象会有一个指向虚函数表的指针，通常这个指针是对象内存布局的一部分。

### 结论：

虚函数的内存实现依赖于虚函数表和虚函数指针机制。虚函数表是一个包含虚函数地址的数组，每个对象会有一个指向该表的指针。通过这种方式，C++ 实现了多态性和动态绑定。虽然引入了一些内存开销和执行时的查找开销，但它使得 C++ 可以支持运行时的多态性特性。

---

### 虚函数的定义和作用：

**虚函数**是 C++ 中的一种成员函数，它允许在派生类中重写（覆盖）基类中的函数，并通过基类指针或引用来调用派生类的函数。这种特性是 C++ 支持 **多态性**（polymorphism）的核心之一。

#### 关键点：
- **虚函数**是基类中的函数，通常使用 `virtual` 关键字标记。
- **动态绑定**：虚函数调用是在运行时根据对象的实际类型决定的，而不是编译时根据指针或引用的类型决定。
- 虚函数允许派生类对基类的函数进行 **重写**。

### 为什么需要虚函数？

C++ 中的多态性是通过虚函数实现的。通过虚函数，程序可以根据对象的实际类型（而非指针或引用的类型）来调用合适的函数。这使得我们可以用相同的接口（基类指针或引用）操作不同类型的对象。

### 语法示例：

```cpp
#include <iostream>

class Base {
public:
    // 声明虚函数
    virtual void func() {
        std::cout << "Base class function" << std::endl;
    }

    virtual ~Base() {}  // 虚析构函数
};

class Derived : public Base {
public:
    // 重写基类的虚函数
    void func() override {
        std::cout << "Derived class function" << std::endl;
    }
};

int main() {
    Base* basePtr = new Derived();  // 基类指针指向派生类对象
    basePtr->func();  // 调用的是 Derived 类的 func

    delete basePtr;  // 动态分配内存需要手动释放
    return 0;
}
```

### 输出：
```
Derived class function
```

### 解释：

1. **虚函数的声明**：
   - 在 `Base` 类中，`func` 函数被声明为虚函数，意味着我们可以在派生类中重写它。
   
2. **动态绑定**：
   - 在 `main` 函数中，`basePtr` 是一个指向 `Base` 类的指针，但它指向的是 `Derived` 类的对象。
   - 当我们调用 `basePtr->func()` 时，实际上调用的是 `Derived` 类中重写的 `func()` 函数，而不是 `Base` 类中的 `func()`。
   - 这种通过指针或引用来决定调用哪个函数的行为叫做 **动态绑定**（或称为 **后期绑定**）。
   
3. **多态性**：
   - 通过虚函数，我们可以在运行时根据对象的实际类型来选择调用哪个函数。这就是 **多态性** 的核心，它使得程序可以处理不同类型的对象，而不需要显式地检查对象的类型。

### 为什么使用虚函数？

1. **实现多态性**：
   - 虚函数使得在使用基类指针或引用时，可以调用派生类中重写的函数。这样，我们可以写出更加通用和灵活的代码。例如，在处理不同类型的对象时，可以通过基类接口统一进行操作，而不关心对象的具体类型。

2. **允许派生类扩展和修改基类的行为**：
   - 通过虚函数，派生类可以修改基类中方法的行为（即重写基类的虚函数）。这使得我们能够通过继承和重写的方式，定制对象的行为。

### 虚函数的内存开销：

- 每个类（如果它有虚函数）都需要一张 **虚函数表（VTable）**，它是一个包含指向虚函数的指针的数组。虚函数表存储了该类的虚函数的地址。
- 每个对象实例（如果是含有虚函数的类）会包含一个指向虚函数表的指针（即虚函数指针，VPtr）。这个指针指向类的虚函数表。
- 这样，每次调用虚函数时，程序会通过虚函数指针查找虚函数表，并在运行时决定调用哪个版本的函数。

### 虚函数的总结：

1. **定义**：虚函数是基类中的函数，用 `virtual` 关键字声明，允许在派生类中被重写。
2. **动态绑定**：虚函数支持动态绑定，即在运行时根据对象的实际类型来选择调用哪个版本的函数。
3. **多态性**：虚函数是实现 C++ 中多态性的关键。它使得基类指针或引用可以用于操作不同类型的对象，并自动调用正确的函数。
4. **内存开销**：虚函数机制需要虚函数表和虚函数指针，这会带来一些内存开销和运行时性能损耗。

虚函数是面向对象编程中实现多态性、继承和扩展行为的基础工具，使得代码更具通用性和灵活性。