## C++ 17 结构化绑定

stl 的 map 容器很多读者应该都很熟悉，map 容器提供了一个 **insert** 方法，我们用该方法向 map 中插入元素，但是应该很少有人记得 **insert** 方法的返回值是什么类型，让我们来看一下 C++98/03 提供的 **insert** 方法的签名：

```
std::pair<iterator,bool> insert( const value_type& value );
```

这里我们仅关心其返回值，这个返回值是一个 **std::pair** 类型，由于 map 中的元素的 key 不允许重复，所以如果 insert 方法调用成功，T1 是被成功插入到 map 中的元素的迭代器，T2 的类型为 bool，此时其值为 true（表示插入成功）；如果 insert 由于 key 重复，T1 是造成 insert 插入失败、已经存在于 map 中的元素的迭代器，此时 T2 的值为 false（表示插入失败）。

在 C++98/03 标准中我们可以使用 **std::pair** 的 **first** 和 **second** 属性来分别引用 T1 和 T2 的值。如下面的我们熟悉的代码所示：

```
#include <iostream>
#include <string>
#include <map>

int main()
{
    std::map<std::string, int> cities;
    cities["beijing"]   = 0;
    cities["shanghai"]  = 1;
    cities["shenzhen"]  = 2;
    cities["guangzhou"] = 3;

    //for (const auto& [key, value] : m)
    //{
    //    std::cout << key << ": " << value << std::endl;
    //}

    //这一行在 C++11 之前写法实在太麻烦了，
    //std::pair<std::map<std::string, int>::iterator, int> insertResult = cities.insert(std::pair<std::string, int>("shanghai", 2));
    //C++ 11中我们写成：
    auto insertResult = cities.insert(std::pair<std::string, int>("shanghai", 2));

    std::cout << "Is insertion successful ? " << (insertResult.second ? "true" : "false") 
              << ", element key: " << insertResult.first->first << ", value: " << insertResult.first->second << std::endl;

    return 0;
}
```

代码 **19** 行实在太啰嗦了，我们使用 auto 关键字让编译器自动推导类型。

**std::pair** 一般只能表示两个元素，C++11 标准中引入了 **std::tuple** 类型，有了这个类型，我们就可以放任意个元素了，原来需要定义成结构体的 POD 对象我们可以直接使用 **std::tuple** 表示，例如下面表示用户信息的结构体：

```
struct UserInfo
{
    std::string username;
    std::string password;
    int         gender;
    int         age;
    std::string address;
};

int main()
{
    UserInfo userInfo = { "Tom", "123456", 0, 25, "Pudong Street" };
    std::string username = userInfo.username;
    std::string password = userInfo.password;
    int gender = userInfo.gender;
    int age = userInfo.age;
    std::string address = userInfo.address;

    return 0;
}
```

我们不再需要定义 struct UserInfo 这样的对象，可以直接使用 **std::tuple** 表示：

```
int main()
{    
    std::tuple<std::string, std::string, int, int, std::string> userInfo("Tom", "123456", 0, 25, "Pudong Street");

    std::string username = std::get<0>(userInfo);
    std::string password = std::get<1>(userInfo);
    int gender = std::get<2>(userInfo);
    int age = std::get<3>(userInfo);
    std::string address = std::get<4>(userInfo);

    return 0;
}
```

从 **std::tuple** 中获取对应位置的元素，我们使用 **std::get** ，其中 N 是元素的序号（从 0 开始）。

与定义结构体相比，通过 **std::pair** 的 **first** 和 **second** 还是 **std::tuple** 的 **std::get** 方法来获取元素子属性，这些代码都是非常难以维护的，其根本原因是 **first** 和 **second** 这样的命名不能做到见名知意。

C++17 引入的**结构化绑定**（Structured Binding ）将我们从这类代码中解放出来。**结构化绑定**使用语法如下：

```
auto [a, b, c, ...] = expression;
auto [a, b, c, ...] { expression };
auto [a, b, c, ...] ( expression );
```

右边的 **expression** 可以是一个函数调用、花括号表达式或者支持结构化绑定的某个类型的变量。例如：

```
//形式1
auto [iterator, inserted] = someMap.insert(...);
//形式2
double myArray[3] = { 1.0, 2.0, 3.0 };
auto [a, b, c] = myArray;
//形式3
struct Point
{
    double x;
    double y;
};
Point myPoint(10.0, 20.0);
auto [myX, myY] = myPoint;
```

这样，我们可以给用于绑定到目标的变量名（语法中的 **a**、**b**、**c**）起一个有意义的名字。

需要注意的是，绑定名称 **a**、**b**、**c** 是绑定目标的一份拷贝，当绑定类型不是基础数据类型时，如果你的本意不是想要得到绑定目标的副本，为了避免拷贝带来的不必要开销，建议使用引用，如果不需要修改绑定目标建议使用 const 引用。示例如下：

```
double myArray[3] = { 1.0, 2.0, 3.0 };
auto& [a, b, c] = myArray;
//形式3
struct Point
{
    double x;
    double y;
};
Point myPoint(10.0, 20.0);
const auto& [myX, myY] = myPoint;
```

**结构化绑定**（Structured Binding ）是 C++17 引入的一个非常好用的语法特性。有了这种语法，在遍历像 map 这样的容器时，我们可以使用更简洁和清晰的代码去遍历这些容器了：

```
std::map<std::string, int> cities;
cities["beijing"] = 0;
cities["shanghai"] = 1;
cities["shenzhen"] = 2;
cities["guangzhou"] = 3;

for (const auto& [cityName, cityNumber] : cities)
{
    std::cout << cityName << ": " << cityNumber << std::endl;
}
```

上述代码中 **cityName** 和 **cityNumber** 可以更好地反映出这个 map 容器的元素内容。

我们再来看一个例子，某 WebSocket 网络库（https://github.com/uNetworking/uWebSockets）中有如下代码：

```
std::pair<int, bool> uncork(const char *src = nullptr, int length = 0, bool optionally = false) {
        LoopData *loopData = getLoopData();

        if (loopData->corkedSocket == this) {
            loopData->corkedSocket = nullptr;

            if (loopData->corkOffset) {
                /* Corked data is already accounted for via its write call */
                auto [written, failed] = write(loopData->corkBuffer, loopData->corkOffset, false, length);
                loopData->corkOffset = 0;

                if (failed) {
                    /* We do not need to care for buffering here, write does that */
                    return {0, true};
                }
            }

            /* We should only return with new writes, not things written to cork already */
            return write(src, length, optionally, 0);
        } else {
            /* We are not even corked! */
            return {0, false};
        }
    }
```

代码的第 **9** 行 **write** 函数返回类型是 **std::pair**，被绑定到 **[written, failed]** 这两个变量中去。前者在写入成功的情况下表示实际写入的字节数，后者表示是否写入成功。

```
std::pair<int, bool> write(const char *src, int length, bool optionally = false, int nextLength = 0) {
    //具体实现省略...
}
```

**结构化绑定的限制**

结构化绑定不能使用 **constexpr** 修饰或被申明为 static，例如：

```
//正常编译
auto [first, second] = std::pair<int, int>(1, 2);
//无法编译通过
//constexpr auto [first, second] = std::pair<int, int>(1, 2);
//无法编译通过
//static auto [first, second] = std::pair<int, int>(1, 2);
```

注意：有些编译器也不支持在 lamda 表达式捕获列表中使用结构化绑定语法。

---
`auto& [_, op]` 是 C++17 引入的一种**结构化绑定**语法。它的目的是解构（解包）容器中的元素，直接访问这些元素的组成部分。我们来详细解释：

### 结构化绑定语法：
在 C++17 中，结构化绑定允许你直接将一个复合类型（如 `std::pair` 或 `std::tuple`）的成员解构成单独的变量。例如，在迭代一个 `std::map` 时，通常我们会得到 `std::pair<const Key, Value>` 类型的元素，使用结构化绑定，可以把这对键值对直接解构为两个独立的变量。

### 在代码中的含义：
```cpp
for (const auto& [_, op] : next_ops)
```
- **`next_ops`** 是一个包含键值对的容器（比如 `std::map` 或 `std::unordered_map`）。在这个例子中，每个元素是一个类似于 `std::pair<Key, Value>` 的类型。
- **`[_, op]`** 通过结构化绑定来解构每个元素：
  - `_` 代表容器中的键（key），但是我们不关心键的具体值，所以用 `_` 来忽略它。
  - `op` 代表容器中的值（value），即 `next_ops` 容器中的第二个元素，它指向后继操作节点（`RuntimeOperator` 对象）。
  
### `auto&` 的作用：
- **`auto&`** 表示通过引用方式绑定这个解构后的值，这样可以避免拷贝对象。因为我们不需要修改元素，所以使用了 `const auto&`，确保该引用是常量引用，不允许修改。

### 结构化绑定的好处：
在没有结构化绑定之前，我们可能需要这样写：
```cpp
for (const auto& pair : next_ops) {
    const auto& op = pair.second;
    // 使用 op
}
```
使用结构化绑定后，可以直接解构 `pair`，使代码更加简洁和易读：

```cpp
for (const auto& [_, op] : next_ops) {
    // 使用 op
}
```
这种写法避免了显式地访问 `pair.first` 和 `pair.second`，使代码更简洁。

### 总结：
`auto& [_, op]` 是 C++17 中的结构化绑定语法，用来直接解构键值对。`_` 忽略了键，而 `op` 代表值（即每个操作节点）。