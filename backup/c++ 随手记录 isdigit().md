在 C++ 中，`isdigit()` 是一个用于判断字符是否为十进制数字（即 `'0'` 到 `'9'`）的标准函数。它定义在头文件 `<cctype>` 中，其原型如下：([[blog.csdn.net](https://blog.csdn.net/Guo___Liang/article/details/121981604?utm_source=chatgpt.com)][1], [[geeksforgeeks.org](https://www.geeksforgeeks.org/isdigit-in-cpp/?utm_source=chatgpt.com)][2])

```cpp
int isdigit(int ch);
```



---

### 🧠 函数说明

* **参数**：`ch` 是要检查的字符，通常应确保其值可以表示为 `unsigned char` 或等于 `EOF`，否则行为未定义。
* **返回值**：如果 `ch` 是十进制数字字符，则返回非零值（通常为 `1`）；否则返回 `0`。
* **注意事项**：为了避免未定义行为，建议在传递 `char` 类型的变量时，先将其转换为 `unsigned char` 类型。([[apiref.com](https://www.apiref.com/cpp-zh/cpp/string/byte/isdigit.html?utm_source=chatgpt.com)][3], [[learn.microsoft.com](https://learn.microsoft.com/zh-cn/cpp/c-runtime-library/reference/isdigit-iswdigit-isdigit-l-iswdigit-l?view=msvc-170&utm_source=chatgpt.com)][4])

---

### ✅ 示例代码

以下是一个使用 `isdigit()` 函数的示例，判断输入的字符是否为数字：([[blog.csdn.net](https://blog.csdn.net/Guo___Liang/article/details/121981604?utm_source=chatgpt.com)][1])

```cpp
#include <iostream>
#include <cctype>

int main() {
    char ch = '5';

    if (std::isdigit(static_cast<unsigned char>(ch))) {
        std::cout << ch << " 是数字字符。" << std::endl;
    } else {
        std::cout << ch << " 不是数字字符。" << std::endl;
    }

    return 0;
}
```



输出：

```
5 是数字字符。
```



---

### 🔄 应用示例：统计字符串中的数字字符个数

```cpp
#include <iostream>
#include <string>
#include <algorithm>
#include <cctype>

int main() {
    std::string str = "C++20 于 2020 年发布。";
    int count = std::count_if(str.begin(), str.end(), [](unsigned char c) {
        return std::isdigit(c);
    });

    std::cout << "字符串中包含 " << count << " 个数字字符。" << std::endl;
    return 0;
}
```



输出：

```
字符串中包含 4 个数字字符。
```



---

### ⚠️ 常见注意事项

* **参数类型**：`isdigit()` 接受 `int` 类型的参数，但应确保该值可以表示为 `unsigned char` 或等于 `EOF`，否则可能导致未定义行为。
* **字符与数字的区别**：`isdigit()` 用于检查字符是否为数字字符，而不是判断整数值是否为数字。例如，`isdigit(52)` 实际上检查的是 ASCII 值为 52 的字符，即 `'4'` 是否为数字字符。
* **宽字符支持**：对于宽字符（`wchar_t`），可以使用 `iswdigit()` 函数，定义在 `<cwctype>` 头文件中。([[stackoverflow.com](https://stackoverflow.com/questions/35391765/correct-way-of-using-isdigit-function-in-c?utm_source=chatgpt.com)][5], [[geeksforgeeks.org](https://www.geeksforgeeks.org/isdigit-in-cpp/?utm_source=chatgpt.com)][2])
