Lambda函数是C++11引入的一种简洁的匿名函数，用于临时定义一次性的小函数。Lambda函数的核心特性是可以直接在代码中声明，而不需要具名的函数名，这使得它们在需要快速定义短小功能时非常方便，例如作为标准库算法的参数、回调函数、闭包等。

1. 基本语法

Lambda函数的一般形式如下：

[capture](parameters) -> return_type { body }

	•	[capture]：用于捕获外部变量。
	•	(parameters)：用于定义参数列表（与普通函数类似）。
	•	-> return_type：指定返回类型（通常可以省略，编译器会自动推导）。
	•	{ body }：Lambda函数的函数体。

2. 各部分的用法

2.1 捕获列表 [capture]

捕获列表用于指定可以在Lambda中使用的外部变量。常见的捕获方式包括：

	•	按值捕获 [x]：Lambda中使用外部变量的副本，不影响原变量。
	•	按引用捕获 [&x]：按引用捕获外部变量，Lambda对变量的更改会影响外部变量。
	•	全部按值捕获 [=]：Lambda中可以按值访问所有在作用域内的变量。
	•	全部按引用捕获 [&]：Lambda中可以按引用访问所有在作用域内的变量。
	•	混合捕获 [=, &y] 或 [&, x]：指定部分按值，部分按引用捕获。

示例：

int x = 10;
auto lambda_val = [x]() { return x * 2; };  // 按值捕获x
auto lambda_ref = [&x]() { x *= 2; };       // 按引用捕获x
lambda_ref();

2.2 参数列表 (parameters)

参数列表和普通函数的参数一样，用于指定Lambda函数的输入参数。参数可以是基本类型、对象、指针、引用等。

示例：

auto add = [](int a, int b) { return a + b; };
int result = add(3, 4);  // result = 7

2.3 返回类型 -> return_type

Lambda函数可以显式指定返回类型，通常在返回类型不易推导或需要强制类型转换时使用。编译器会自动推导返回类型，省略-> return_type也可以工作。

示例：

auto divide = [](int a, int b) -> double { return (double)a / b; };
double result = divide(5, 2);  // result = 2.5

2.4 函数体 { body }

Lambda函数体可以是任何合法的C++代码，可以包含变量声明、控制语句等。

3. Lambda函数的常见用法

3.1 用作标准库算法的回调函数

Lambda函数常用于标准库算法（如 std::for_each、std::sort、std::find_if）的回调函数。

std::vector<int> numbers = {3, 1, 4, 1, 5, 9};
std::sort(numbers.begin(), numbers.end(), [](int a, int b) { return a > b; }); // 降序排列

3.2 用于事件回调或异步操作

Lambda函数可以作为事件处理程序或异步操作的回调函数，因为它们无需命名。

void doAsyncTask(std::function<void()> callback) {
    // 异步任务完成后调用回调
    callback();
}
doAsyncTask([]() { std::cout << "Task completed!" << std::endl; });

3.3 捕获外部变量的闭包

Lambda函数的捕获机制使得它们可以保存外部变量的状态，形成闭包。

int count = 0;
auto counter = [&count]() { return ++count; };
std::cout << counter(); // 输出1
std::cout << counter(); // 输出2

3.4 递归 Lambda 表达式

Lambda 表达式通常不能直接调用自身，但可以使用 std::function 将其包装，以实现递归调用。

std::function<int(int)> factorial = [&](int n) {
    return n <= 1 ? 1 : n * factorial(n - 1);
};
int result = factorial(5);  // result = 120

3.5 在多线程中使用Lambda

Lambda函数常用于创建线程时的任务定义。

#include <thread>
std::thread t([]() {
    std::cout << "Running in a separate thread" << std::endl;
});
t.join();

4. 示例总结

以下是一个使用Lambda的综合示例：

#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> numbers = {1, 2, 3, 4, 5};

    // 1. 使用Lambda输出所有元素
    std::for_each(numbers.begin(), numbers.end(), [](int n) { std::cout << n << " "; });
    std::cout << std::endl;

    // 2. 捕获变量并修改
    int factor = 2;
    std::for_each(numbers.begin(), numbers.end(), [&factor](int &n) { n *= factor; });

    // 3. 带返回类型的Lambda求和
    auto sum = [](int a, int b) -> int { return a + b; };
    int result = sum(10, 5);

    std::cout << "Sum: " << result << std::endl;
    return 0;
}

总结

Lambda函数的用法灵活，支持快速定义一次性函数、闭包和回调，可以按值或引用捕获外部变量，适用于多线程和异步编程。