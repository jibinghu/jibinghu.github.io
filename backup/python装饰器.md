在 Python 中，@ 语法通常用于 装饰器（decorator）。装饰器是一种特殊的语法，用于在不修改函数或类代码的情况下，扩展其功能。以下是对装饰器的一些基础介绍，帮助你快速入门：

1. 什么是装饰器？

装饰器本质上是一个 高阶函数，它接受一个函数或类作为输入，返回一个增强功能后的新函数或类。

使用 @decorator_name 的语法糖（syntactic sugar），你可以轻松地将装饰器应用于函数或类，而无需手动调用装饰器函数。

2. 装饰器的基本结构

以下是一个简单的装饰器示例：
``` python
# 定义装饰器函数
def my_decorator(func):
    def wrapper():
        print("这是在函数执行前添加的代码")
        func()  # 执行原函数
        print("这是在函数执行后添加的代码")
    return wrapper

# 使用装饰器（手动调用）
def my_function():
    print("这是原始函数的代码")

decorated_function = my_decorator(my_function)
decorated_function()

# 输出：
# 这是在函数执行前添加的代码
# 这是原始函数的代码
# 这是在函数执行后添加的代码
```

3. 使用 @ 语法糖

使用 @ 可以直接将装饰器绑定到目标函数，而无需手动调用：

``` python
# 定义装饰器
def my_decorator(func):
    def wrapper():
        print("这是在函数执行前添加的代码")
        func()
        print("这是在函数执行后添加的代码")
    return wrapper

# 使用 @ 语法糖
@my_decorator
def my_function():
    print("这是原始函数的代码")

my_function()

# 输出：
# 这是在函数执行前添加的代码
# 这是原始函数的代码
# 这是在函数执行后添加的代码
```

4. 带参数的装饰器

如果目标函数需要参数，可以调整装饰器来支持：
``` python
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print("开始执行函数...")
        result = func(*args, **kwargs)  # 支持传递参数
        print("函数执行完毕")
        return result
    return wrapper

@my_decorator
def add(a, b):
    return a + b

print(add(3, 4))

# 输出：
# 开始执行函数...
# 函数执行完毕
# 7
```
5. 多个装饰器叠加

可以叠加多个装饰器，从上到下依次应用：
``` python
def decorator1(func):
    def wrapper():
        print("装饰器1: 开始")
        func()
        print("装饰器1: 结束")
    return wrapper

def decorator2(func):
    def wrapper():
        print("装饰器2: 开始")
        func()
        print("装饰器2: 结束")
    return wrapper

@decorator1
@decorator2
def my_function():
    print("这是函数的主体内容")

my_function()

# 输出：
# 装饰器1: 开始
# 装饰器2: 开始
# 这是函数的主体内容
# 装饰器2: 结束
# 装饰器1: 结束
```
6. 应用场景

装饰器非常灵活，常见应用场景包括：
	1.	日志记录：在函数执行前后打印日志。
	2.	性能计时：记录函数执行时间。
	3.	权限校验：在执行函数前检查用户权限。
	4.	缓存：在多次调用时返回缓存结果。
	5.	事务管理：如数据库的事务处理。

7. 进阶：带参数的装饰器

如果装饰器本身需要参数，可以再嵌套一层函数：
``` python
def repeat(times):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(times):
                func(*args, **kwargs)
        return wrapper
    return decorator

@repeat(3)
def say_hello():
    print("你好！")

say_hello()

# 输出：
# 你好！
# 你好！
# 你好！
```
总结

装饰器是 Python 中非常强大的功能，可以在不修改原函数代码的情况下扩展功能。其核心思想是：
	1.	将一个函数作为输入。
	2.	在函数执行前后添加额外功能。
	3.	返回一个增强后的函数。

如果你想深入了解或有具体的使用需求，可以随时告诉我，我可以结合实际案例详细讲解！