./bootstrap.sh 和 ./configure、make 以及 CMake 和 make 都是常见的自动化构建工具，但它们的工作方式和使用场景有所不同。我们可以从以下几个方面来比较它们：

1. 工具类型
	•	./bootstrap.sh 和 ./configure + make
	•	./bootstrap.sh 和 ./configure 是传统的 GNU Autotools 系统的一部分，通常用于管理和构建开源项目。
	•	./bootstrap.sh：这是一个脚本，通常用于初始化项目的构建系统，生成所需的配置文件（如 configure 脚本）。它的作用类似于启动一个新的构建过程，生成适合当前平台的构建配置。
	•	./configure：这个脚本会检查系统环境、依赖库、编译器等，然后生成适合的 Makefile（或其他构建文件）。它会进行一些平台相关的配置，确保构建环境正确。
	•	make：这是一个构建工具，利用 Makefile 文件定义的规则来编译项目。make 会自动处理依赖关系，决定编译和链接过程。
	•	CMake 和 make
	•	CMake 是一个跨平台的构建工具，它提供了一种描述构建过程的方式，可以生成多种平台（如 Unix、Windows、Xcode 等）上使用的构建系统文件。
	•	make：同样是一个构建工具，用于根据 CMake 生成的 Makefile 来编译和构建项目。

2. 生成的构建文件
	•	GNU Autotools (bootstrap.sh + ./configure + make)
	•	./bootstrap.sh 会运行并生成一个 configure 脚本，后者会检查系统环境、依赖等。
	•	./configure 脚本生成一个 Makefile，这个 Makefile 包含了如何编译和安装程序的具体指令。
	•	make 基于 Makefile 执行编译过程，最终构建程序。
	•	CMake + make
	•	CMake 使用 CMakeLists.txt 文件定义项目的构建过程。CMake 是一个跨平台构建工具，可以生成不同平台的构建文件（如 Makefile，Xcode 工程文件，Visual Studio 项目文件等）。
	•	make 基于 CMake 生成的 Makefile 文件来执行编译和构建。

3. 平台支持
	•	GNU Autotools
	•	主要适用于 UNIX-like 系统，如 Linux 和 macOS，但也支持 Windows 系统（通过 MinGW 或 Cygwin 等工具）。
	•	在 Windows 上使用时，可能需要额外配置工具链或环境，GNU Autotools 不像 CMake 那样原生支持多平台。
	•	CMake
	•	CMake 是一个跨平台的工具，支持 Windows、Linux、macOS 和其他平台。它的设计目的是简化跨平台开发，因此它生成的构建文件可以适应各种平台和构建工具（例如生成 Visual Studio 的 .sln 文件或 Xcode 工程文件等）。

4. 依赖管理
	•	GNU Autotools
	•	./configure 脚本会检查系统上已安装的依赖库和工具，如果缺少依赖项，通常会在配置过程中报错。
	•	依赖管理和检查大多数是通过手动配置和在 Makefile 中定义的路径来进行的。
	•	CMake
	•	CMake 提供了更强大的依赖管理功能，能够自动检测系统中的依赖库、头文件，并根据需要配置编译选项。
	•	还支持与 find_package、ExternalProject 等模块集成，自动管理第三方库。

5. 可扩展性和灵活性
	•	GNU Autotools
	•	GNU Autotools（包括 ./configure）系统在处理复杂的构建需求时可能需要较为繁琐的配置。对于一些特殊的项目，编写 Makefile.am 和 configure.ac 文件的方式可能较为复杂和繁琐。
	•	对于较老的项目和很多经典的开源项目来说，GNU Autotools 是一种常见的选择。
	•	CMake
	•	CMake 更加灵活，能够生成多种平台的构建文件。它不仅支持传统的 make，还可以生成 Visual Studio、Xcode、Ninja 等构建系统的项目文件。
	•	CMake 的语法和配置方式相对现代、易于扩展，因此在跨平台开发和大项目中表现得更加高效和方便。

6. 配置过程的差异
	•	GNU Autotools（./bootstrap.sh + ./configure）
	•	这个过程通常会执行一系列检查（如检查编译器、依赖库等），生成一个 Makefile 文件并进行配置。配置过程可能会比较慢，特别是对旧版库或系统特有的依赖。
	•	配置文件和构建脚本通常是静态的（基于项目的需求手动编写）。
	•	CMake
	•	CMake 会生成平台相关的构建文件，例如生成 Makefile 或 Visual Studio 工程文件。它会自动检测并生成必要的构建文件，适应不同平台和构建工具。
	•	CMake 的配置过程相对更加灵活和快速，尤其是在需要支持多平台或跨平台项目时。

7. 构建过程
	•	GNU Autotools（./configure + make）
	•	一旦配置完成，make 将根据 Makefile 执行构建过程。make 会根据文件之间的依赖关系（在 Makefile 中定义）来决定需要重新构建的部分。
	•	CMake + make
	•	make 也基于 CMake 生成的 Makefile 执行构建，但 CMake 会先自动生成对应平台的构建文件，因此 make 的使用更为通用，且支持更广泛的平台和构建工具。

总结：主要差异

特性	GNU Autotools (./bootstrap.sh + ./configure + make)	CMake + make
平台支持	主要适用于 UNIX-like 系统，支持 Windows 需要额外工具	跨平台，支持 Windows、Linux、macOS 等多种平台
构建文件生成	生成 Makefile，通常只能用于 make 构建	生成多个平台的构建文件（如 Makefile，Visual Studio 工程文件等）
配置复杂度	./configure 脚本检查系统环境并生成 Makefile	CMakeLists.txt 提供跨平台配置，支持依赖自动检测
依赖管理	手动配置，依赖库检查较为繁琐	强大的依赖管理和自动化功能，支持查找第三方库
灵活性和扩展性	灵活性较差，适用于简单项目，处理复杂项目时繁琐	更高的灵活性，支持更复杂的项目和跨平台构建

	•	GNU Autotools 适合传统的 UNIX-like 系统，很多经典的开源项目使用它，但配置和扩展性相对较差。
	•	CMake 更加现代化，支持跨平台开发和灵活的依赖管理，适合需要在不同平台和构建工具上工作的项目。