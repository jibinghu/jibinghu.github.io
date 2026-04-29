mkdir -p 是 Linux 和类 Unix 系统中用于创建目录的命令，具体解释如下：

命令语法

mkdir -p [路径/目录名称]

参数解释

	•	mkdir：用于创建一个或多个目录。
	•	-p：表示 “parents”，有两个主要功能：
	1.	递归创建目录：
	•	如果指定路径中间的父目录不存在，会自动创建所有需要的父目录。
	•	比如，创建路径 /a/b/c，如果 /a 和 /a/b 不存在，它会自动先创建这些父目录。
	2.	避免报错：
	•	如果指定目录已存在，不会报错，而是静默处理。

示例

1. 递归创建多级目录

mkdir -p /home/user/projects/myproject

	•	如果 /home/user/projects/ 和 /home/user/projects/myproject/ 不存在，会自动创建它们。
	•	如果已存在，不会报错。

2. 避免已存在目录的错误

mkdir -p /home/user/existing_dir

	•	如果目录 /home/user/existing_dir 已存在，这条命令不会报错。

3. 无 -p 的效果

如果不加 -p 参数：

mkdir /home/user/projects/myproject

	•	如果中间的父目录 /home/user/projects/ 不存在，会报错：No such file or directory。

常用场景

	•	创建复杂的多层目录结构。
	•	在脚本中使用，避免因为目录已存在而导致脚本中断。

总结

mkdir -p 是一种安全、高效的方式来创建目录，特别是在需要递归创建或避免重复创建的情况下。