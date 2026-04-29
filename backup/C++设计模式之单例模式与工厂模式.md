1. 单例模式（Singleton Pattern）

单例模式是一种设计模式，保证在应用程序的生命周期内，一个类只有一个实例，并且提供一个全局访问点来获取这个实例。它常用于需要共享资源（如数据库连接、配置对象等）的场景。

单例模式的关键点：

	•	唯一性：只能创建该类的一个实例。
	•	全局访问点：通过类的静态方法来访问这个唯一的实例。
	•	延迟初始化：只有当第一次请求实例时，才会创建它。

C++ 实现单例模式

class Singleton {
public:
    // 获取唯一实例的静态方法
    static Singleton& GetInstance() {
        static Singleton instance;  // 懒汉式，调用时才创建
        return instance;
    }

    // 删除拷贝构造函数和赋值操作符，防止拷贝
    Singleton(const Singleton&) = delete;
    Singleton& operator=(const Singleton&) = delete;

    // 其他业务逻辑方法
    void DoSomething() {
        std::cout << "Doing something in Singleton instance." << std::endl;
    }

private:
    // 私有构造函数，外部无法直接创建实例
    Singleton() {}
};

单例模式的工作原理：

	1.	私有构造函数：通过将构造函数设为私有，禁止外部代码直接创建该类的实例。
	2.	静态方法：通过一个静态方法 GetInstance() 提供访问类实例的全局接口。
	3.	懒汉式：实例是在第一次调用 GetInstance() 时才创建的（懒汉式），这保证了只有在需要时才会分配内存。
	4.	禁止拷贝和赋值：为了防止通过拷贝构造或赋值创建新的实例，将拷贝构造函数和赋值操作符设为 delete。

单例模式的应用场景：

	•	全局配置类：应用程序中的配置对象，只需要一个实例。
	•	日志系统：日志管理类，全局共享日志资源。
	•	资源管理类：如数据库连接池、线程池等，需要全局唯一且共享的资源。

2. 工厂模式（Factory Pattern）

工厂模式是一种创建型设计模式，用于定义一个接口用于创建对象，而将具体的对象创建过程延迟到子类或其他类中。它的核心思想是将对象的实例化与使用者分离，使得创建代码更加灵活、可扩展。

工厂模式的关键点：

	•	封装对象创建逻辑：将对象的创建封装在工厂类中，使用者只需要知道工厂的接口，而不关心具体对象的创建过程。
	•	解耦：工厂模式可以让客户端代码与具体的类解耦，客户端只需要知道工厂的接口，无需关心具体类的实现细节。
	•	灵活可扩展：通过扩展新的具体工厂，能够很方便地引入新的对象类型，而不影响已有代码。

工厂模式的基本类型：

	1.	简单工厂模式（Simple Factory Pattern）：工厂类根据传入的参数决定实例化哪种类。
	2.	工厂方法模式（Factory Method Pattern）：定义一个工厂接口，将具体的实例化工作交由不同的子类实现。
	3.	抽象工厂模式（Abstract Factory Pattern）：提供一个接口，创建一系列相关的或依赖的对象，而无需指定具体类。

C++ 实现简单工厂模式

// 产品抽象类
class Product {
public:
    virtual void Use() = 0;
    virtual ~Product() = default;
};

// 具体产品类A
class ConcreteProductA : public Product {
public:
    void Use() override {
        std::cout << "Using Product A" << std::endl;
    }
};

// 具体产品类B
class ConcreteProductB : public Product {
public:
    void Use() override {
        std::cout << "Using Product B" << std::endl;
    }
};

// 工厂类
class SimpleFactory {
public:
    // 创建产品的静态方法
    static std::shared_ptr<Product> CreateProduct(const std::string& type) {
        if (type == "A") {
            return std::make_shared<ConcreteProductA>();
        } else if (type == "B") {
            return std::make_shared<ConcreteProductB>();
        }
        return nullptr;
    }
};

工厂模式的工作原理：

	1.	抽象产品类：定义了产品类的接口或抽象类，具体的产品类将继承并实现该接口。
	2.	工厂类：包含一个或多个创建产品的静态方法（或接口），根据参数来创建具体的产品对象。
	3.	客户端代码：通过调用工厂类的方法来创建具体的产品对象，并且只依赖于工厂接口，解耦了客户端与具体的产品类。

工厂模式的应用场景：

	•	对象创建复杂或依赖于配置：当对象的创建需要复杂的逻辑时，将其封装到工厂类中更具灵活性。
	•	需要灵活扩展：如果经常需要新增产品类型，工厂模式能方便地扩展产品，而无需修改客户端代码。
	•	日志框架、数据库连接：根据不同的配置创建不同的数据库连接对象或日志系统。

3. 单例模式和工厂模式的对比与结合

	•	单例模式确保某个类在系统中只有一个实例，并且可以全局访问。
	•	工厂模式用于封装创建对象的逻辑，提供一种更灵活、可扩展的对象创建方式。

两者可以结合使用，例如一个工厂类本身可以是单例模式，这样整个系统中只有一个工厂对象负责创建产品对象。

单例和工厂结合的例子：

class ProductFactory {
public:
    static ProductFactory& GetInstance() {
        static ProductFactory instance;
        return instance;
    }

    std::shared_ptr<Product> CreateProduct(const std::string& type) {
        if (type == "A") {
            return std::make_shared<ConcreteProductA>();
        } else if (type == "B") {
            return std::make_shared<ConcreteProductB>();
        }
        return nullptr;
    }

    // 禁止拷贝和赋值
    ProductFactory(const ProductFactory&) = delete;
    ProductFactory& operator=(const ProductFactory&) = delete;

private:
    ProductFactory() {}  // 私有化构造函数
};

总结：

	•	单例模式：保证一个类有且仅有一个实例，适用于共享资源的场景。
	•	工厂模式：封装对象创建逻辑，允许灵活创建不同的对象，适用于需要动态实例化对象的场景。