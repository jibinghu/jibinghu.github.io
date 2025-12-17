对象生命周期（RAII）与智能指针所有权：

``` cpp
TEST(test_buffer, allocate) {
  using namespace base;
  auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
  {
    Buffer buffer(32, alloc);
    ASSERT_NE(buffer.ptr(), nullptr);
    LOG(INFO) << "HERE1";
  }
  LOG(INFO) << "HERE2";
}

TEST(test_buffer, allocate2) {
  using namespace base;
  auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
  std::shared_ptr<Buffer> buffer;
  { buffer = std::make_shared<Buffer>(32, alloc); }
  LOG(INFO) << "HERE";
  ASSERT_NE(buffer->ptr(), nullptr);
}
```

对于第一个测试：

- `Buffer buffer(32, alloc);` 是栈上对象（automatic storage duration）。
- 离开花括号作用域 } 时，buffer 立即析构（确定性析构）。
- 因此如果 Buffer 的析构函数负责 free()/delete/cudaFree 等释放动作，那么：
     - 释放发生在 HERE1 之后、HERE2 之前。

用日志可以非常清楚地“卡点”析构时机：
- HERE1 打印时：对象仍存活
- 退出内层 {}：析构执行（若析构里也有日志，你会看到它出现在 HERE1 与 HERE2 之间）
- HERE2 打印时：对象已经销毁，资源应该已经释放

这类写法非常适合验证 RAII：作用域结束资源必须释放。

---

而对于第二个测试：

`make_shared<Buffer>` 在 堆上创建 Buffer 对象。

- buffer 是一个 std::shared_ptr<Buffer>，持有对象的共享所有权。
- 内层 {} 结束时，并不会析构 Buffer，因为：
     - 对象的引用计数仍然至少为 1（外层的 buffer 仍持有它）。
- Buffer 的析构会发生在 最后一个 shared_ptr 被销毁或 reset 的时刻：

本例中通常是测试函数结束时 buffer 离开作用域才析构（或你显式 buffer.reset()）。

因此：

在 LOG(INFO) << "HERE"; 和 ASSERT_NE(buffer->ptr(), nullptr); 时，

> 这里有个点非常重要，buffer 这个共享指针在大括号开始之前就已经定义好了，但是在大括号内分配了一个堆对象使得 buffer 指向这个堆对象，所以大括号结束之后，buffer 还是会指向申请的堆对象的。

> 这里要说明一点，在 c++ 中只有有名字的对象才是变量，才有作用域。

- Buffer 一定还活着，内存也还没释放。
- 这段测试并没有验证“离开某个小作用域就释放资源”，它验证的是：
- 通过 shared_ptr 持有对象时，对象能跨作用域存活且可访问。










