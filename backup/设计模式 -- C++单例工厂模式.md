高屋建瓴：设计模式是针对「对象与对象之间如何协作」这一层面的抽象经验总结。

---

首先将 `static` 修饰说明：

## Static 关键字

`static` 关键字的本质不是静态，而是：

- 限制可见性；
- 延长生命周期
- 提升抽象层次

修饰对象 | 核心作用
-- | --
类成员变量 | 类级共享变量，变量属于类所有，不属于某个对象
类成员函数 | 类级函数，无 this，常被用于单例模式。不能是 virtual，不能直接访问非 static 成员。
全局变量 | 编译单元私有，相对于`extern`，只在编译单元内可见，static 使全局变量具有 internal linkage
局部变量 | 保持状态，生命周期在整个程序内，但是作用域只在函数内部，相当于延长了生命周期，重新调用函数时还可以继续使用变量。不同于普通局部变量的一次函数调用的生命周期。static 局部变量是 “函数私有的全局变量”，可以用作函数内缓存/统计调用次数等。
普通函数 | 文件私有，类似全局变量，只在编译单元内可见，类似于在编译单元外隐藏了函数名



---

## 设计模式：

GoF 将 23 种模式分为 三大类，这是“标准划分”：

``` cpp
23 种设计模式
├─ 创建型（Creational）5 种
├─ 结构型（Structural）7 种
└─ 行为型（Behavioral）11 种
```

一、创建型模式：关注对象如何被创建

1. Singleton 单例模式

目的：保证某个类在系统中只有一个实例，并提供全局访问点。
场景：内存池(内存的分配器)、日志管理、配置管理、GPU 上下文等。

2. Factory 工厂模式

目的：定义创建对象的接口，由子类决定实例化哪个类。

3. Abstract Factory 抽象工厂

目的：创建一整族相互关联的对象，而不指定具体类。

4. Builder 建造者

目的：将复杂对象的构建过程与表示分离。

5. Prototype 原型

目的：通过拷贝已有对象来创建新对象。


其余的结构性、行为型在工作中后续再补充。

二、 单例工厂模式

单例工厂模式 = 两个模式的组合：

1. Singleton（单例）：保证某个“核心对象”在进程中只有一个实例
2. Factory（工厂）：
- 对外提供统一的创建 / 获取接口
- 屏蔽具体类型与构造细节

单例模式必不可少需要用 `static` 来实现，以 `allocator` 来举例：

Allocator 类：

``` cpp
class CPUDeviceAllocator : public DeviceAllocator {
public:
  CPUDeviceAllocator() {
    // 初始化资源池、统计信息等
  }

  void* allocate(size_t bytes) override {
    return std::malloc(bytes);
  }

  void release(void* ptr) override {
    std::free(ptr);
  }

  ~CPUDeviceAllocator() override {
    // 释放全局资源（如有）
  }
};
```

单例工厂：

``` cpp
class CPUDeviceAllocatorFactory {
public:
  static std::shared_ptr<CPUDeviceAllocator> get_instance() {
    static std::shared_ptr<CPUDeviceAllocator> instance =
        std::make_shared<CPUDeviceAllocator>();
    return instance;
  }

private:
  CPUDeviceAllocatorFactory() = delete;
};
```

代码解释：

`CPUDeviceAllocatorFactory` 向外部提供一个唯一的 `CPUDeviceAllocator` 实例，这个类不需要被实例化，只需要作为一个命名空间和创建入口。

`static std::shared_ptr<CPUDeviceAllocator> get_instance()` 这个 `static` 成员函数：不依赖类实例，没有 this 指针，可以直接通过类名调用：`CPUDeviceAllocatorFactory::get_instance();` 。 

上述代码第一个 static 构造一个静态成员函数，第二个 static 声明一个静态成员变量，构成单例工厂模式。
