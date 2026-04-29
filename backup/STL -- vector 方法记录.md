std::vector 提供了许多有用的成员函数，可以方便地操作和访问元素。以下是一些常用的 std::vector 方法：

1. 访问元素的方法

- front()：返回向量的第一个元素。
- back()：返回向量的最后一个元素。
- at(index)：返回指定索引 index 处的元素（带边界检查），如果 index 超出范围，会抛出 std::out_of_range 异常。
- operator[]：使用方括号 [] 访问元素（不进行边界检查），如 vec[index]。

2. 容量相关的方法

- size()：返回向量中的元素数量。
- empty()：判断向量是否为空。如果为空，则返回 true。
- capacity()：返回当前向量的容量，即在不进行内存重新分配的情况下，可以容纳的最大元素数。
- reserve(n)：将向量的容量至少增加到 n，以减少插入大量元素时的重新分配开销。
- shrink_to_fit()：将 capacity 减小到和 size 一致，以释放多余的内存。

3. 修改元素的方法

- push_back(value)：在向量末尾添加一个元素 value。
- pop_back()：移除向量末尾的一个元素。
- insert(position, value)：在 position 位置插入一个元素 value。position 是一个迭代器。
- erase(position)：移除 position 位置的元素，position 是一个迭代器。
- erase(first, last)：移除从 first 到 last 范围的元素，first 和 last 都是迭代器。
- clear()：清空向量中的所有元素，使其 size 变为 0。
- resize(n)：将向量的大小调整为 n，如果当前大小大于 n，则多余的元素被删除；如果小于 n，则插入默认值。

4. 迭代器相关的方法

- begin()：返回指向第一个元素的迭代器。
- end()：返回指向末尾后一个位置的迭代器（不指向实际元素）。
- rbegin()：返回指向最后一个元素的反向迭代器。
- rend()：返回指向第一个元素前一个位置的反向迭代器。

5. 其他方法

- assign(count, value)：用 count 个 value 来填充向量，可以用于重置向量的内容。
- emplace_back(args...)：在向量末尾构造一个元素，参数 args... 被传递给该元素的构造函数。
- swap(other)：与另一个向量 other 交换内容，通常用于优化性能。

示例代码
``` cpp
#include <iostream>
#include <vector>

int main() {
    std::vector<int> vec = {1, 2, 3, 4, 5};

    // 使用 front() 和 back()
    std::cout << "First element: " << vec.front() << std::endl;
    std::cout << "Last element: " << vec.back() << std::endl;

    // 使用 size() 和 capacity()
    std::cout << "Size: " << vec.size() << ", Capacity: " << vec.capacity() << std::endl;

    // 使用 push_back() 和 pop_back()
    vec.push_back(6);
    std::cout << "After push_back: ";
    for (int v : vec) std::cout << v << " ";
    std::cout << std::endl;

    vec.pop_back();
    std::cout << "After pop_back: ";
    for (int v : vec) std::cout << v << " ";
    std::cout << std::endl;

    // 使用 insert() 和 erase()
    vec.insert(vec.begin() + 1, 10);
    std::cout << "After insert: ";
    for (int v : vec) std::cout << v << " ";
    std::cout << std::endl;

    vec.erase(vec.begin() + 1);
    std::cout << "After erase: ";
    for (int v : vec) std::cout << v << " ";
    std::cout << std::endl;

    // 使用 clear()
    vec.clear();
    std::cout << "After clear, Size: " << vec.size() << std::endl;

    return 0;
}
```
这些方法可以帮助更方便、灵活地操作 std::vector。