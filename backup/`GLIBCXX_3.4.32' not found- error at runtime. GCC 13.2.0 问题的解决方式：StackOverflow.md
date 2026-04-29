> the /lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.32' is for gcc13, so we need to update this file.`

0. 查看当前 GLIBCXX 版本
首先通过命令`strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX`
- strings：这是一个 Linux 命令，用于提取二进制文件或库文件中可打印的字符串。通常，这些字符串包括库中的函数名、符号、版本号等。
1. 添加 PAP源
`sudo add-apt-repository ppa:ubuntu-toolchain-r/test`:
> 这个命令添加了一个新的软件包源 (ppa:ubuntu-toolchain-r/test)，该源提供了更新版本的 GCC 和相关库，包括 libstdc++。
- add-apt-repository：这是用于在 Ubuntu 中添加新的软件包源的命令。Ubuntu 使用软件源（也称为 PPA, Personal Package Archive）来存储和管理软件包。
- ppa:ubuntu-toolchain-r/test：这是一个特定的 PPA 源，专门用于测试版和更新版的工具链（如 GCC、G++ 和 libstdc++ 库）。这个 PPA 提供了比官方源更新的版本，可以用来升级工具链中的相关包。
2. 更新软件包列表：
`sudo apt-get update`:
> 更新本地的包索引列表，以确保能够从新的 PPA 源中获取最新的库版本。

3. 升级 libstdc++6：
`sudo apt-get install --only-upgrade libstdc++6`:
> 升级现有的 libstdc++6 库;

---

通过 `ppa:ubuntu-toolchain-r/test` 添加源的原因是因为 **PPA（Personal Package Archive，个人包存档）** 是 Ubuntu 特有的一种软件分发机制。它使得第三方开发者能够很容易地将自己维护的软件包发布给 Ubuntu 用户。

### 1. **PPA 的工作原理**：
PPA 是由 Ubuntu Launchpad 平台托管的一个专用存档，用于提供由个人、组织或开源项目维护的软件包。用户可以通过 `add-apt-repository` 命令轻松添加 PPA，并从中获取和安装更新的软件包。每个 PPA 都有一个唯一的标识符，像 `ppa:ubuntu-toolchain-r/test` 就是这种标识符。该 PPA 源提供了更新版本的 GCC 和相关工具。

当你添加一个 PPA 时，以下事情会发生：
- 该 PPA 的软件包列表（即源）会被添加到系统的 `sources.list.d` 目录中。
- 软件包管理工具（如 `apt`）可以从该源中下载并安装软件包。
- `apt-get update` 会更新你的系统软件包索引，确保系统知道该 PPA 中的可用软件包版本。

在执行 `add-apt-repository ppa:ubuntu-toolchain-r/test` 时，系统会自动添加该 PPA 的源到 `/etc/apt/sources.list.d/` 中，并从 Launchpad 下载其包列表。

### 2. **与 PyPI 源的区别**：
`PyPI`（Python Package Index）和 Ubuntu 的 PPA 源在本质上是不同的东西：
- **PyPI** 是一个专门的 Python 软件包索引库，它存放和分发基于 Python 的项目。用户通过 `pip` 工具下载和管理 Python 包。这些包通常是 Python 脚本和模块，不涉及系统级的编译或依赖。
- **PPA** 是为 **系统级软件包管理** 服务的，涉及操作系统中的核心库、二进制文件和系统工具。通过 `apt` 来管理的软件包往往包含编译后的二进制文件，并可能有复杂的依赖关系。

当你添加 PyPI 源时，你是告诉 `pip` 工具去哪里下载 Python 包。而 PPA 是告诉 Ubuntu 系统通过 `apt` 去哪里获取特定的系统级软件包。由于 PPA 是专为 Ubuntu 和基于 Debian 的系统设计的，所以使用简单的 PPA 标识符（例如 `ppa:ubuntu-toolchain-r/test`）是方便用户的快捷方式。

### 3. **为什么 PPA 不使用网址？**
PPA 是通过 Ubuntu 的 `Launchpad` 平台托管的，因此它有简化的格式：`ppa:<user>/<ppa-name>`。这不仅方便了用户（不需要记住复杂的 URL），也使得 PPA 源的管理更方便和统一。在后台，`ppa:ubuntu-toolchain-r/test` 实际上会被解析为一个标准的 URL，例如：
```
http://ppa.launchpad.net/ubuntu-toolchain-r/test/ubuntu
```
系统会自动完成这个解析过程，所以你不需要手动添加 URL。

相比之下，PyPI 本质上是一个全球通用的 Python 软件包仓库，而不是与某个特定 Linux 发行版紧密集成的工具，因此它的源是通过标准 URL（如 `https://pypi.org`）来指定的。

总结来说，PPA 是 Ubuntu 的一部分，提供了一种更简便的方式来管理和分发软件包，而不像 Python 的 `pip` 那样依赖标准网址，这使得 PPA 的管理更轻量化和灵活。