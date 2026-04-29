1. 使用 pkill 命令

pkill 允许你通过进程名来杀死所有与该名称匹配的进程，快捷而高效。

`pkill -9 fms.x`

	•	-9 表示强制终止进程（即发送 SIGKILL 信号）。
	•	fms.x 是你要终止的进程名称。

2. 使用 killall 命令

killall 也是一个通过进程名终止进程的命令，功能和 pkill 类似。

`killall -9 fms.x`

这两个命令会直接终止所有名为 fms.x 的进程，无需逐个列出 PID，特别适用于大量进程。

---

`ps` 是一个非常有用的命令，用于查看当前系统中的进程。以下是一些常用的 `ps` 命令及其示例，帮助你更好地理解和使用它。

1. **显示所有进程**
   - **命令**: `ps aux`
   - **解释**: 列出所有用户的所有进程（包括其他用户的进程），显示详细信息。
   - **示例输出**:
     ```
     USER       PID  %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND
     user      5646  0.2  0.1 3182456 32724 ?       Rl   2025   0:42 ./fms.x
     root      1567  0.0  0.2 2496752 20436 ?       Ss   10:12   0:01 systemd
     ```

2. **显示当前用户的进程**
   - **命令**: `ps u`
   - **解释**: 显示当前用户的进程，简洁的信息。
   - **示例输出**:
     ```
     USER       PID  %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND
     user      5646  0.2  0.1 3182456 32724 ?       Rl   2025   0:42 ./fms.x
     ```

3. **显示某个进程的详细信息**
   - **命令**: `ps -fp <PID>`
   - **解释**: 显示指定进程（根据 PID）的详细信息。
   - **示例**: `ps -fp 5646`
   - **输出**:
     ```
     UID        PID  PPID  C STIME TTY      TIME CMD
     user      5646  5645  0 2025 pts/0    00:00:42 ./fms.x
     ```

4. **显示进程树**
   - **命令**: `ps --forest`
   - **解释**: 以树形结构显示进程，显示父进程与子进程的关系。
   - **示例输出**:
     ```
     PID TTY      STAT   TIME COMMAND
     5645 pts/0    S+     0:00 /bin/bash
     └─5646 pts/0    Rl     0:42 ./fms.x
     ```

5. **显示所有进程并根据某列排序**
   - **命令**: `ps aux --sort=-%cpu`
   - **解释**: 显示所有进程，并按 CPU 使用率降序排序。
   - **示例输出**:
     ```
     USER       PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND
     user      5646  45.2  0.2 3182456 32468 ?       Rl   2025 412:29 ./fms.x
     user      2345  12.1  0.1 1023404 13928 ?       R    2025  56:12 my_process
     ```

6. **显示特定进程名的进程**
   - **命令**: `ps aux | grep <进程名>`
   - **解释**: 显示与特定进程名匹配的所有进程。
   - **示例**: `ps aux | grep fms.x`
   - **输出**:
     ```
     user      5646  45.2  0.2 3182456 32468 ?       Rl   2025 412:29 ./fms.x
     user      2345  12.1  0.1 1023404 13928 ?       R    2025  56:12 my_process
     ```

7. **显示进程的详细状态**
   - **命令**: `ps -eo pid,ppid,stat,command`
   - **解释**: 显示进程 ID（PID）、父进程 ID（PPID）、进程状态（STAT）和命令（COMMAND）。
   - **示例输出**:
     ```
     PID   PPID  STAT  COMMAND
     5646  5645  Rl    ./fms.x
     1567  1     Ss    systemd
     ```

8. **显示特定用户的所有进程**
   - **命令**: `ps -u <用户名>`
   - **解释**: 显示指定用户的所有进程。
   - **示例**: `ps -u user`
   - **输出**:
     ```
     USER       PID  %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND
     user      5646  0.2  0.1 3182456 32724 ?       Rl   2025   0:42 ./fms.x
     ```

9. **显示进程详细信息并使用特定格式**
   - **命令**: `ps -eo pid,uid,user,stime,etime,cmd`
   - **解释**: 显示 PID、UID、用户名、启动时间、运行时间、命令。
   - **示例输出**:
     ```
     PID  UID USER     STIME  ELAPSED  CMD
     5646 1000 user     2025   412:29   ./fms.x
     ```

10. **限制显示进程数量**
    - **命令**: `ps aux --sort=-%cpu | head -n 10`
    - **解释**: 显示 CPU 使用率前 10 名的进程。
    - **示例输出**:
      ```
      USER       PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND
      user      5646  45.2  0.2 3182456 32468 ?       Rl   2025 412:29 ./fms.x
      ```

11. **显示当前终端上的所有进程**
    - **命令**: `ps -t <终端>`
    - **解释**: 显示指定终端的所有进程。可以使用 `tty` 命令查看当前终端。
    - **示例**: `ps -t pts/0`
    - **输出**:
      ```
      PID  TTY      STAT   TIME COMMAND
      5646 pts/0    Rl     0:42 ./fms.x
      ```

这些命令和选项可以帮助你灵活地查看系统中的进程状态、进程树、资源占用等信息，针对不同的需求灵活使用 `ps` 命令。希望这些配置和示例能帮助你更高效地使用 `ps`！