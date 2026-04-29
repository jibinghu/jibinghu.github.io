from collections import deque 是用于导入 Python 标准库 collections 中的 deque（双端队列）的语句。deque 是一种高效的双端队列数据结构，支持在队列的两端快速地添加和删除元素。deque 的性能比列表在两端添加和删除更高，因此在实现队列、栈、滑动窗口、广度优先搜索等场景中非常实用。

基本代码示例和解释

以下是一些基本的 deque 操作的代码示例：
``` python
from collections import deque

# 初始化一个双端队列
dq = deque()

# 在队尾添加元素
dq.append('A')
dq.append('B')

# 在队头添加元素
dq.appendleft('C')

# 查看队列内容
print(dq)  # 输出：deque(['C', 'A', 'B'])

# 从队尾移除元素
dq.pop()  # 输出：'B'

# 从队头移除元素
dq.popleft()  # 输出：'C'

# 查看剩余的队列内容
print(dq)  # 输出：deque(['A'])
```
常用操作

deque 提供了许多常用的方法，适用于不同的应用场景：

	1.	添加元素：
	•	append(x): 在队尾添加元素 x。
	•	appendleft(x): 在队头添加元素 x。
	2.	删除元素：
	•	pop(): 从队尾移除并返回一个元素。
	•	popleft(): 从队头移除并返回一个元素。
	3.	扩展队列：
	•	extend(iterable): 将一个可迭代对象的所有元素添加到队尾。
	•	extendleft(iterable): 将一个可迭代对象的所有元素添加到队头（元素顺序会被反转）。
	4.	旋转队列：
	•	rotate(n): 将队列旋转 n 步。如果 n 是正数，队列的尾部元素会移到头部；如果是负数，则头部元素会移到尾部。
	5.	其他操作：
	•	clear(): 清空队列。
	•	count(x): 统计队列中值为 x 的元素数量。

deque 的扩展用法

deque 的灵活性使得它可以用于多种数据结构和算法的实现。以下是一些常见的扩展用法：

1. 实现栈

deque 可以用作栈（后进先出，LIFO），因为 append() 和 pop() 操作在队尾添加和删除元素，与栈的操作一致。
``` python
stack = deque()
stack.append(1)
stack.append(2)
print(stack.pop())  # 输出：2
print(stack.pop())  # 输出：1
```
2. 实现队列

deque 本身就是一种双端队列，可以直接用于队列（先进先出，FIFO）操作，通过 append() 添加到队尾，popleft() 从队头取出。
``` python
queue = deque()
queue.append(1)
queue.append(2)
print(queue.popleft())  # 输出：1
print(queue.popleft())  # 输出：2
```
3. 滑动窗口

deque 可以用于实现滑动窗口，适合处理一段序列中每次固定长度的窗口数据（例如移动平均数）。deque 的 maxlen 参数可以设置最大长度，超出长度时会自动移除最旧的元素。
``` python
def moving_average(sequence, n):
    dq = deque(maxlen=n)
    result = []
    for num in sequence:
        dq.append(num)
        result.append(sum(dq) / len(dq))
    return result

print(moving_average([1, 2, 3, 4, 5], 3))  # 输出：[1.0, 1.5, 2.0, 3.0, 4.0]
```
4. 广度优先搜索（BFS）

在图或树结构的广度优先搜索（BFS）中，deque 非常适合用作队列，因为可以快速地从队头取出节点并在队尾添加新节点。
``` python
def bfs(graph, start):
    queue = deque([start])
    visited = set([start])
    while queue:
        node = queue.popleft()
        print(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

# 示例图（邻接表）
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

bfs(graph, 'A')
```
5. 实现 Kahn 拓扑排序算法

在有向无环图（DAG）的拓扑排序中，deque 可用于存储入度为 0 的节点，并逐步从队列中取出节点处理，直到所有节点都处理完毕。Kahn 算法的实现如下：
``` python
def kahn_topological_sort(graph):
    in_degree = {node: 0 for node in graph}
    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] += 1

    queue = deque([node for node in graph if in_degree[node] == 0])
    topological_order = []

    while queue:
        node = queue.popleft()
        topological_order.append(node)
        
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(topological_order) == len(graph):
        return topological_order
    else:
        return "Graph has a cycle, topological sorting is not possible."

# 示例图
graph = {
    "起床": ["洗脸", "刷牙"],
    "洗脸": ["早餐"],
    "刷牙": ["早餐"],
    "早餐": []
}

print(kahn_topological_sort(graph))
```
6. 双向访问需求

如果你需要在队列两端快速地进行插入和删除操作，deque 可以实现这一需求。例如，解决一些窗口最大值或最小值问题时，deque 可以方便地从两端操作数据。

总结

	•	deque 是一种双端队列数据结构，适合在队列两端快速添加和删除元素。
	•	常用操作包括 append、appendleft、pop、popleft、rotate 等。
	•	deque 适合用在实现栈、队列、滑动窗口、广度优先搜索等算法中。
	•	其高效的双端操作让它在处理需要频繁从两端操作数据的场景中表现出色。