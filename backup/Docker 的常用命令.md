以下是一些 Docker 的常用命令，涵盖了从容器管理、镜像操作到网络管理等各个方面：

1. 容器管理命令
	•	启动一个容器
docker run <镜像名>
启动并运行一个新的容器。如果镜像不存在，则会自动拉取。
	•	启动容器并在后台运行
docker run -d <镜像名>
-d 参数让容器在后台运行。
	•	查看正在运行的容器
docker ps
查看当前正在运行的容器。
	•	查看所有容器（包括停止的容器）
docker ps -a
	•	停止容器
docker stop <容器ID或容器名>
停止一个正在运行的容器。
	•	启动已经停止的容器
docker start <容器ID或容器名>
	•	重启容器
docker restart <容器ID或容器名>
	•	删除容器
docker rm <容器ID或容器名>
删除已停止的容器。
	•	查看容器的日志
docker logs <容器ID或容器名>
	•	进入容器内
docker exec -it <容器ID或容器名> bash
使用 bash 进入容器内部。

2. 镜像管理命令
	•	查看本地镜像
docker images
列出所有本地存在的镜像。
	•	拉取镜像
docker pull <镜像名>
从 Docker Hub 或其他仓库拉取镜像。
	•	构建镜像
docker build -t <镜像名>:<标签> <路径>
从指定路径（通常是 Dockerfile 所在目录）构建镜像。
	•	删除镜像
docker rmi <镜像ID或镜像名>
删除一个镜像。

3. 网络管理命令
	•	查看 Docker 网络
docker network ls
列出所有 Docker 网络。
	•	创建 Docker 网络
docker network create <网络名>
创建一个新的网络。
	•	连接容器到网络
docker network connect <网络名> <容器名或容器ID>
将一个容器连接到指定的网络。
	•	断开容器与网络的连接
docker network disconnect <网络名> <容器名或容器ID>
断开容器与指定网络的连接。

4. 容器数据卷管理命令
	•	查看本地卷
docker volume ls
列出所有 Docker 卷。
	•	创建卷
docker volume create <卷名>
创建一个新的数据卷。
	•	删除卷
docker volume rm <卷名>
删除一个数据卷。
	•	查看卷的详细信息
docker volume inspect <卷名>
获取卷的详细信息，如挂载位置。

5. 容器镜像操作
	•	导出容器为镜像
docker export <容器ID或容器名> > <镜像.tar>
将容器导出为 tar 包。
	•	导入 tar 包为镜像
docker import <镜像.tar>
从 tar 包导入镜像。

6. Docker 配置和其他命令
	•	查看 Docker 版本
docker version
显示 Docker 客户端和服务器的版本信息。
	•	查看 Docker 系统信息
docker info
显示 Docker 的系统信息（包括镜像、容器数量、存储驱动等）。
	•	检查容器状态和诊断
docker inspect <容器ID或容器名>
返回一个 JSON 格式的容器详细信息。

7. Docker Compose 命令（如果使用 Compose 管理多个容器）
	•	启动 Compose 项目
docker-compose up
启动并创建 Compose 文件中的所有服务容器。
	•	后台启动 Compose 项目
docker-compose up -d
在后台启动服务。
	•	停止 Compose 项目
docker-compose down
停止并删除 Compose 项目中的容器和网络。
	•	查看 Compose 项目日志
docker-compose logs
查看 Compose 项目中各个服务的日志。

这些命令涵盖了 Docker 的常见操作，可以帮助你高效地管理容器、镜像和网络等。如果你需要更深入的了解或其他特定命令，Docker 的[官方文档](https://docs.docker.com/)提供了详细的解释。