题目：
---

Python实现一个QPS性能测试工具，具体要求如下： 1、 QPS定义：request/second 表示服务每秒处理请求数的性能 2、 要求N个Client同时请求http服务 3、 N个Client并发请求要求使用协程、线程或进程 4、 N个Client并发模拟真实场景，动态调整，通过一定时间间隔初步增加并发量，例如从0,4,8,16（此要求可选） 补充： 5、 可以使用代码补全工具，可以上网查看语法，不能使用copilot直接检索答案 6、代码逻辑尽量模块化编程。

---
> 要实现一个 QPS 性能测试工具，我们可以使用 Python 的 asyncio 库来模拟并发请求，配合 aiohttp 进行 HTTP 请求。下面是一个模块化的实现：
步骤和思路：
	1.	使用 asyncio 来管理协程，实现并发请求。
	2.	使用 aiohttp 发送 HTTP 请求。
	3.	模拟多个客户端进行并发请求。
	4.	动态调整并发量，逐步增加客户端数量（0, 4, 8, 16…）。
	5.	在一定时间间隔内记录请求数量，计算 QPS。

代码：

``` python

import asyncio
import aiohttp
import time
from typing import List

# 记录每秒请求数
class QPSMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.request_count = 0

    def increment(self):
        self.request_count += 1

    def calculate_qps(self) -> float:
        elapsed_time = time.time() - self.start_time
        if elapsed_time == 0:
            return 0.0
        return self.request_count / elapsed_time

# 定义单个客户端任务
async def send_request(session: aiohttp.ClientSession, url: str, monitor: QPSMonitor):
    try:
        async with session.get(url) as response:
            await response.text()  # 读取返回内容
            monitor.increment()  # 请求成功，计数
    except Exception as e:
        print(f"Request failed: {e}")

# 启动多个客户端并发请求
async def start_clients(url: str, client_count: int, monitor: QPSMonitor, interval: int):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for _ in range(client_count):
            tasks.append(send_request(session, url, monitor))
        
        # 执行并发请求
        await asyncio.gather(*tasks)
        
        # 计算并打印当前的QPS
        print(f"Current QPS (after {interval} seconds): {monitor.calculate_qps()}")

# 模拟动态增加并发量
async def dynamic_load_test(url: str, max_clients: int, step: int, interval: int):
    monitor = QPSMonitor()
    current_clients = 0
    
    while current_clients <= max_clients:
        print(f"Starting test with {current_clients} clients...")
        await start_clients(url, current_clients, monitor, interval)
        
        # 每次增加客户端数量
        current_clients += step
        await asyncio.sleep(interval)  # 模拟每隔一段时间增加并发量

# 运行性能测试
def run_performance_test(url: str, max_clients: int = 16, step: int = 4, interval: int = 2):
    asyncio.run(dynamic_load_test(url, max_clients, step, interval))

if __name__ == "__main__":
    url = "http://example.com"  # 替换为实际测试的 HTTP 服务地址
    run_performance_test(url)
```

