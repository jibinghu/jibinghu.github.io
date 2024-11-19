可以使用 tar 命令将 .tar.gz 文件解压到指定的文件夹。以下是具体步骤：

1. 基本语法

tar -xzvf 文件名.tar.gz -C 目标文件夹路径

2. 参数解释

- -x：解压文件。
- -z：表示文件是经过 gzip 压缩的。
- -v：显示解压的过程（可选，显示解压进度信息）。
- -f：指定文件。
- -C：指定解压到的目标目录。

3. 示例

假设：
- .tar.gz 文件路径是 /home/user/file.tar.gz。
- 解压目标文件夹路径是 /home/user/destination/。

运行以下命令：

`tar -xzvf /home/user/file.tar.gz -C /home/user/destination/`

4. 注意事项

- 确保目标文件夹存在：如果目标文件夹 /home/user/destination/ 不存在，需要先创建：

`mkdir -p /home/user/destination/`


- 权限问题：如果你没有权限操作目标文件夹，可能需要加上 sudo：

`sudo tar -xzvf /home/user/file.tar.gz -C /home/user/destination/`

通过以上步骤，.tar.gz 文件将被解压到指定的文件夹中。