### 拉取镜像：
`docker pull mmtnrw/freeswitch:latest`

### 启动
运行镜像：
- 基本命令是下面一行，但是显然还需要端口映射等信息
> docker run -d --name 89M_freeswitch mmtnrw/freeswitch:latest

``` bash

docker 中所有 freeswitch 安装目录：

/ # find ./ -name "freeswitch"
./var/lib/freeswitch
./var/run/freeswitch
./var/log/freeswitch
./etc/freeswitch
./usr/bin/freeswitch
./usr/share/freeswitch
./usr/lib/freeswitch

mod 安装目录：/usr/lib/freeswitch/mod

```