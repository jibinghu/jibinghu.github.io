在使用 **gdb** (GNU Debugger) 进行调试时，有一些非常实用的命令，可以帮助你更高效地进行程序的调试。以下是一些常用的 **gdb** 命令：

### 1. **基本命令**
- `run` (或 `r`)：启动程序并开始调试。
- `quit` (或 `q`)：退出 **gdb**。
- `help`：显示帮助信息。
- `continue` (或 `c`)：继续程序执行，直到下一个断点。
- `step` (或 `s`)：单步进入函数，执行当前行，并进入函数调用内部。
- `next` (或 `n`)：单步执行，但如果当前行有函数调用，则不进入该函数。
- `finish`：继续执行直到当前函数返回。
- `break <location>` (或 `b <location>`)：设置断点。`<location>` 可以是函数名、文件名行号等。
- `delete`：删除当前所有断点，或者通过指定断点编号删除特定断点。
- `info break`：查看当前所有的断点及其状态。

### 2. **查看和打印变量**
- `print <expression>` (或 `p <expression>`)：打印当前变量的值或表达式的值。
- `display <expression>`：在每次停下来时自动显示某个表达式的值。
- `info locals`：查看当前函数中所有局部变量的值。
- `info args`：查看当前函数的参数值。
- `info registers`：查看所有寄存器的值。

### 3. **调试信息**
- `list` (或 `l`)：列出源代码，默认会显示当前停留的函数上下文。
- `backtrace` (或 `bt`)：查看函数调用栈，显示从程序开始到当前停留点的调用路径。
- `frame <n>`：切换到指定的调用栈帧（`n` 是栈帧的编号）。
- `up`：向上移动到上一级调用栈。
- `down`：向下移动到下一级调用栈。
- `info source`：显示当前源文件的相关信息。

### 4. **内存和堆栈**
- `x /<n><format> <address>`：查看内存内容。`<n>` 是读取多少单元，`<format>` 是输出格式（例如：`x/10x` 打印 10 个十六进制数）。
- `set <variable> = <value>`：修改变量的值。

### 5. **条件断点**
- `break <location> if <condition>`：设置一个条件断点，只有满足特定条件时，程序才会停下来。
- `condition <breakpoint-number> <condition>`：为已有的断点添加条件。

### 6. **程序运行控制**
- `start`：启动程序，并在主函数的第一行暂停。
- `stepi`：逐条指令单步执行（适用于汇编代码调试）。
- `nexti`：逐条指令执行，跳过函数调用。
- `until <location>`：继续运行直到指定的位置。

### 7. **调试共享库**
- `set solib-search-path <path>`：设置共享库的搜索路径。
- `info sharedlibrary`：列出当前加载的所有共享库。

### 8. **其它**
- `watch <expression>`：设置观察点，当表达式的值发生变化时停下来。
- `catch <event>`：设置事件监视器，例如：`catch throw` 监视 C++ 异常抛出。
- `target`：用于指定远程调试或调试目标，通常用于嵌入式开发。
- `set pagination off`：禁用分页，使得输出不分页显示。

### 9. **调试多线程程序**
- `info threads`：查看当前所有线程的信息。
- `thread <n>`：切换到指定线程进行调试。
- `thread apply all <command>`：对所有线程执行指定命令。

--- 

### 附加：

进入 gdb （GNU 调试器）有多种方式，主要取决于你的目标和程序的调试方式。以下是几种常见的进入 gdb 的方式：

1. 从命令行启动 gdb 调试已编译的程序

最常见的方式是直接在终端中启动 gdb，并加载要调试的程序。

gdb <executable-file>

	•	<executable-file> 是你要调试的程序的可执行文件路径。
	•	例如，如果你有一个名为 a.out 的程序，可以使用以下命令：

gdb a.out

启动 gdb 后，程序会暂停在程序的入口处，你可以使用 run 或 r 命令来启动程序。

2. 通过调试时指定命令行参数

如果你需要给程序传递命令行参数，可以在进入 gdb 后通过 set args 命令指定参数。例如：

gdb a.out
(gdb) set args arg1 arg2
(gdb) run

这会将 arg1 和 arg2 作为命令行参数传递给 a.out。

3. 直接从命令行运行程序并进入 gdb

你也可以直接在命令行中使用 gdb 启动程序并直接进入调试模式，无需手动进入 gdb 后再输入 run 命令：

gdb --args <executable-file> <program-args>

例如：

gdb --args ./a.out arg1 arg2

此命令会启动 gdb 并加载 a.out 可执行文件，并传递命令行参数 arg1 和 arg2。

4. 在运行时附加到已运行的程序

如果你已经有一个正在运行的程序，并希望在其运行时附加调试，可以使用 gdb 附加到该进程：

gdb attach <pid>

	•	<pid> 是你要调试的进程的进程 ID。
	•	你可以通过命令 ps aux | grep <program-name> 查找正在运行的进程的 PID。

5. 远程调试

如果你需要调试远程主机上的程序，可以使用 gdb 的远程调试功能。通常这涉及到在目标机器上运行 gdbserver，然后在本地机器上通过 gdb 连接。
	1.	在目标机器上启动 gdbserver：

gdbserver <host>:<port> <executable-file>

	2.	在本地机器上使用 gdb 连接到远程目标：

gdb <executable-file>
(gdb) target remote <host>:<port>

6. 使用 core dump 文件调试

如果程序崩溃并生成了 core dump 文件，你可以通过 gdb 来加载这个文件进行分析。假设 core 文件和程序名是 a.out：

gdb a.out core

这将加载崩溃时的内存状态，并允许你查看崩溃原因。

7. 在启动时加载特定的符号文件

如果你没有源代码但有调试符号文件（.debug 文件），可以通过以下命令启动 gdb 并加载符号文件：

gdb <executable-file> <path-to-symbol-file>

例如：

gdb myprogram /path/to/myprogram.debug

8. 在启动时执行脚本

你还可以在启动 gdb 时指定一个脚本，自动执行一系列调试命令。例如：

gdb -x script.gdb <executable-file>

其中，script.gdb 是一个包含调试命令的脚本文件。这样可以在启动时自动设置断点、变量、条件等。

---

gdb 的 -p 选项是用于 附加到一个正在运行的进程 进行调试的方式。使用 -p 参数，你可以将 gdb 连接到一个指定的进程 ID (PID)，从而在该进程正在运行时对其进行调试。

gdb -p <pid> 命令

gdb -p <pid> 用于将 gdb 附加到一个已经运行的进程。通过这个命令，gdb 会连接到指定的进程，并允许你在不中断进程执行的情况下进行调试。

使用方法
	1.	首先，找出你想要调试的进程 ID (PID)，可以使用命令如 ps 或 top 来列出正在运行的进程及其 PID。例如：

ps aux | grep <your_program>


	2.	然后使用 gdb -p 命令附加到该进程。例如，假设 PID 是 12345：

gdb -p 12345

这会启动 gdb 并将其附加到 PID 为 12345 的进程上。

附加过程中的常见行为：
	•	进程暂停：当你附加到一个正在运行的进程时，进程会被暂停（即 gdb 会暂停进程的执行）。你需要手动使用 continue 命令继续执行进程。
	•	调试操作：在附加到进程后，你可以像调试本地程序一样设置断点、查看变量、跟踪调用栈等。
	•	信号处理：在一些系统中，附加进程时会发送信号给该进程（通常是 SIGSTOP），并且在开始调试之前需要恢复进程。

示例

假设你有一个名为 my_program 的程序正在运行，并且它的 PID 是 12345：
	1.	启动 gdb 并附加到该进程：

gdb -p 12345


	2.	一旦 gdb 连接到进程，进程会被暂停。你可以使用以下命令来调试：
	•	continue (或 c)：继续程序的执行。
	•	backtrace (或 bt)：查看调用栈。
	•	info locals：查看局部变量。
	3.	在调试完成后，退出 gdb 使用命令 quit，如果需要，可以继续让进程执行。

退出和恢复
	•	当你通过 gdb 附加到进程并完成调试时，你可以使用 quit 命令退出 gdb。如果你希望在退出时让程序继续运行，可以使用：

(gdb) detach

这会让 gdb 从进程中分离，并允许该进程继续运行。

适用场景

gdb -p 特别适用于以下场景：
	•	你希望调试一个已经在生产环境中运行的进程。
	•	你想调试一个崩溃的进程或在死锁状态下的进程。
	•	在远程调试时，目标程序已在目标主机上启动并运行。