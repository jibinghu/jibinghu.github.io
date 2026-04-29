gdb 调试的常用命令：
	1.	启动调试：
	•	gdb ./your_program：启动 gdb 并加载程序。
	2.	设置断点：
	•	break main：在 main 函数入口处设置断点。
	•	break filename.c:line_number：在特定的行号设置断点。
	•	break function_name：在特定函数入口设置断点。
	3.	运行程序：
	•	run 或 r：开始运行程序。
	4.	单步执行：
	•	step 或 s：单步进入到函数内部。
	•	next 或 n：单步执行，但不进入函数内部。
	5.	继续执行：
	•	continue 或 c：继续执行直到下一个断点。
	6.	查看变量值：
	•	print variable_name：查看变量的当前值。
	•	display variable_name：每次程序暂停时自动显示该变量的值。
	7.	查看堆栈信息：
	•	backtrace 或 bt：查看当前堆栈跟踪。
	•	frame 或 f：查看当前帧的详细信息。
	8.	修改变量值：
	•	set variable_name=value：设置变量的值。
	9.	退出调试：
	•	quit 或 q：退出 gdb 调试。
	10.	查看源代码：
	•	list 或 l：查看当前源代码。
	•	list filename.c:line_number：查看指定文件和行号的代码。
	11.	查看程序状态：
	•	info locals：查看当前函数的局部变量。
	•	info registers：查看所有寄存器的状态。
	12.	断点管理：
	•	delete breakpoint_number：删除指定编号的断点。
	•	disable breakpoint_number：禁用指定的断点。
	•	enable breakpoint_number：启用指定的断点。
	13.	程序中断：
	•	Ctrl + C：在程序运行过程中中断执行。

