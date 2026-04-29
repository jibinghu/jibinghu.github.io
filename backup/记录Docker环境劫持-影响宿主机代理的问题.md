``` bash
User@ubuntu:~$ wget www.baidu.com
--2025-05-28 15:15:03--  http://www.baidu.com/
Connecting to 172.26.1.45:3128... failed: No route to host.
```
在使用docker发现路由不可达，ping跳板机也遇到问题：
``` bash 
User@ubuntu:~$ ping 172.26.1.45
PING 172.26.1.45 (172.26.1.45) 56(84) bytes of data.
From 172.26.0.1 icmp_seq=1 Destination Host Unreachable
```
且服务器内有很多Docker Container运行历史时，要考虑是不是Docker占用了网络。
> 可以使用`route -n`查看目前的路由情况。

解决：
- Docker 在运行过程中会创建许多网络，包括默认的桥接网络、自定义网络等。随着时间的推移，一些网络可能会因为容器的删除而变得不再使用。这些未使用的网络会占用系统资源，docker network prune 命令可以帮助清理这些未使用的网络。