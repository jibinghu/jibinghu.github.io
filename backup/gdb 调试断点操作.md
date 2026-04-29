在 GDB 调试过程中，设置断点并在断点处停下来后，你可以执行一系列操作来检查、修改程序的状态，或继续调试。以下是一些常用操作指令：

### 查看变量和内存
- **查看当前栈帧的局部变量**：
  ```plaintext
  info locals
  ```

- **查看特定变量的值**：
  ```plaintext
  print variable_name
  ```

- **查看特定内存地址的值**：
  ```plaintext
  x /nfu address
  ```
  - `n`：显示的单元数
  - `f`：格式（如 `d` 表示十进制，`x` 表示十六进制）
  - `u`：单位（如 `b` 表示字节，`w` 表示字）

### 修改变量和内存
- **修改变量的值**：
  ```plaintext
  set variable_name = new_value
  ```

- **修改内存的值**：
  ```plaintext
  set {type} address = new_value
  ```
  例如，修改一个 `int` 类型的值：
  ```plaintext
  set {int} 0x7fffffffe1b0 = 10
  ```

### 堆栈操作
- **查看当前堆栈帧信息**：
  ```plaintext
  info frame
  ```

- **查看调用栈**：
  ```plaintext
  backtrace
  ```
  或者简写：
  ```plaintext
  bt
  ```

- **切换到某个特定的栈帧**：
  ```plaintext
  frame frame_number
  ```

- **查看特定栈帧的局部变量**：
  ```plaintext
  info locals
  ```

### 设置和管理断点
- **设置断点**：
  ```plaintext
  break function_name
  ```
  或者在特定行号设置断点：
  ```plaintext
  break filename:line_number
  ```

- **列出所有断点**：
  ```plaintext
  info breakpoints
  ```

- **启用/禁用断点**：
  ```plaintext
  enable breakpoint_number  
  disable breakpoint_number
  ```

- **删除断点**：
  ```plaintext
  delete breakpoint_number
  ```

### 继续执行
- **继续执行程序**：
  ```plaintext
  continue
  ```
  或者简写：
  ```plaintext
  c
  ```

- **单步执行程序（逐行执行）**：
  ```plaintext
  step
  ```
  或者简写：
  ```plaintext
  s
  ```

- **单步执行，不进入函数调用（逐过程执行）**：
  ```plaintext
  next
  ```
  或者简写：
  ```plaintext
  n
  ```

### 条件断点
- **为断点设置条件**：
  ```plaintext
  condition breakpoint_number condition_expression
  ```

### 程序运行状态检查
- **检查程序是否已结束**：
  ```plaintext
  info program
  ```

### 堆栈回溯与变量观察
- **获取程序的执行历史**：
  ```plaintext
  info registers
  ```

- **查看指定寄存器的值**：
  ```plaintext
  print $register_name
  ```
