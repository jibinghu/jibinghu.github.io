# Docker详解及其使用方法

Docker 是一个开源的应用容器化平台，提供一种轻量级虚拟化解决方案，用于打包、分发和运行应用程序及其依赖环境。它的核心是通过容器（Container）技术，让开发者能够在任何地方运行应用程序，而不必担心环境配置差异。

---

## 一、Docker 的核心概念

1. **镜像 (Image)**  
   - 类似于操作系统的快照或模板。  
   - 它是一个只读的、包含了应用程序运行所需的环境、依赖项和配置的文件系统。  
   - 镜像可以用来启动容器。

2. **容器 (Container)**  
   - 基于镜像创建的实例，类似于虚拟机，但更加轻量。  
   - 容器是独立的，可以运行应用程序，并与主机系统隔离。

3. **Dockerfile**  
   - 用于定义镜像构建过程的脚本文件。  
   - 通过描述一系列命令，生成一个镜像。

4. **Docker Engine**  
   - Docker 的运行时环境，包括 Docker 守护进程（Daemon）和客户端。  
   - 它负责镜像管理、容器运行等核心功能。

5. **Docker Hub**  
   - 官方的镜像仓库。  
   - 用户可以从中下载或上传自己的镜像。

---

## 二、Docker 的特点

1. **轻量级**  
   - 容器共享主机操作系统的内核，不需要为每个容器启动一个独立的操作系统，资源占用小。

2. **快速启动**  
   - 容器可以在几秒钟内启动，而虚拟机通常需要更长时间。

3. **跨平台一致性**  
   - 通过镜像，确保开发、测试和生产环境一致。

4. **模块化**  
   - 通过容器将应用程序分解成多个独立的服务模块，方便管理和扩展。

---

## 三、Docker 的安装与配置

### 1. 在 Linux 系统上安装 Docker

以 Ubuntu 为例：

```bash
# 更新系统包
sudo apt-get update

# 安装依赖
sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common

# 添加 Docker 的官方 GPG 密钥
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# 添加 Docker 官方软件库
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# 安装 Docker CE（社区版）
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io

# 验证安装
sudo docker version

2. 配置非 root 用户使用 Docker

sudo usermod -aG docker $USER
newgrp docker

四、Docker 的基本使用方法

1. 镜像管理
	•	查看本地镜像：

docker images

	•	下载镜像：

docker pull <镜像名>:<标签>
# 示例：下载最新版的 Ubuntu 镜像
docker pull ubuntu:latest

	•	删除镜像：

docker rmi <镜像ID或名称>

2. 容器管理
	•	启动容器：

docker run -it --name <容器名> <镜像名>
# 示例：启动一个 Ubuntu 容器
docker run -it --name my-ubuntu ubuntu

	•	列出运行中的容器：

docker ps

	•	列出所有容器（包括已停止）：

docker ps -a

	•	停止容器：

docker stop <容器ID或名称>

	•	删除容器：

docker rm <容器ID或名称>

	•	进入容器：

docker exec -it <容器ID或名称> bash

3. 构建镜像
	•	创建一个 Dockerfile 文件：

# 示例内容
FROM ubuntu:latest
RUN apt-get update && apt-get install -y nginx
CMD ["nginx", "-g", "daemon off;"]

	•	构建镜像：

docker build -t <镜像名>:<标签> .
# 示例：
docker build -t my-nginx:1.0 .

4. 网络管理
	•	列出 Docker 网络：

docker network ls

	•	创建网络：

docker network create <网络名>

	•	将容器连接到指定网络：

docker network connect <网络名> <容器名>

五、Docker 的应用场景
	1.	开发与测试环境
	•	通过镜像快速创建一致的开发和测试环境。
	2.	微服务架构
	•	将应用拆分为多个容器，分别部署。
	3.	CI/CD 管道
	•	利用 Docker 集成到持续集成和部署流程中。
	4.	分布式系统
	•	结合 Docker Compose 和 Kubernetes 进行容器编排。

六、Docker 在 Linux 系统中的表现
	•	Docker 在 Linux 系统上运行时，依赖于 Linux 内核的 cgroups 和 namespaces，通过隔离和限制资源，运行一个或多个容器。
	•	在文件系统中，Docker 的容器和镜像数据通常存储在 /var/lib/docker 下。容器类似于系统中的一个独立的进程，它并不是一个传统意义上的目录（如 direction），而是运行环境中的一个逻辑单位，使用底层存储引擎（如 overlay2）实现文件共享和分层管理。

如果你需要详细实践某个场景或了解更高级的使用方法，可以告诉我！

