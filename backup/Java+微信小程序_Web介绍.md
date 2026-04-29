> 前言：由于课程需要，简单地对 Java 相关框架以及Java Web相关知识做简单地学习，以备他用。

- [Spring Boot](#Spring_Boot)
- [微信小程序开发](#Weixin)

#### Spring Boot

> Spring Boot 是由 Pivotal Team 开发的基于 Spring 框架的项目，它简化了基于 Spring 应用的开发过程。Spring Boot 的设计目标是简化新 Spring 应用的创建，并提供一组开箱即用的功能来开发微服务和独立的生产级 Spring 应用。

在介绍Spring Boot前，有必要了解Spring Framework 来对其Spring 家族有宏观上的了解：

> [! IMPORTANT]
> Spring 是一个庞大的开源框架家族，提供了一系列用于构建现代 Java 应用的工具和库。Spring 框架家族涵盖了从核心容器到数据访问、Web 开发、微服务等多个领域。提供了一个全面且灵活的框架生态系统，涵盖了从核心容器到数据访问、Web 开发、微服务、批处理和企业集成等多个领域。通过这些模块，开发人员可以构建各种类型的应用，从简单的 Web 应用到复杂的分布式系统，Spring 家族的强大和灵活性使其成为 Java 开发的主流选择。

Spring 家族包含多个项目和子项目，每个项目都有特定的功能和用途。主要包括：

1. **Spring Framework**：核心框架，提供依赖注入、AOP、数据访问、事务管理等功能。
2. **Spring Boot**：简化 Spring 应用的创建、配置和部署，提供自动配置和起步依赖。
3. **Spring Data**：提供数据访问的抽象层，支持多种数据存储技术（如 JPA、MongoDB、Redis）。
4. **Spring Security**：提供强大的认证和授权功能。
5. **Spring Cloud**：构建分布式系统和微服务的工具集。
6. **Spring Batch**：用于批处理应用开发。
7. **Spring Integration**：用于企业集成模式的实现。
8. **Spring WebFlux**：用于构建响应式 Web 应用的框架。
9. **Spring AMQP**：用于与 AMQP 消息代理（如 RabbitMQ）集成。

<img src="https://img2024.cnblogs.com/blog/3358182/202406/3358182-20240603215551084-1910605694.png" weight="500" height="300">

从上图中可以看出，这里罗列了 Spring 框架的七大核心技术体系，分别是微服务架构、响应式编程、云原生、Web 应用、Serverless 架构、事件驱动以及批处理。当然，这些技术体系各自独立但也有一定交集，例如微服务架构往往会与基于 Spring Cloud 的云原生技术结合在一起使用，而微服务架构的构建过程也需要依赖于能够提供 RESTful 风格的 Web 应用程序等。

在日常开发过程中，如果构建单块 Web 服务，可以采用 Spring Boot。如果想要开发微服务架构，那么就需要使用基于 Spring Boot 的 Spring Cloud，而 Spring Cloud 同样内置了基于 Spring Cloud Stream 的事件驱动架构。

当然，所有我们现在能看到的 Spring 家族技术体系都是在 Spring Framework 基础上逐步演进而来的。在介绍上述技术体系之前，我们先简单了解下 Spring Framework 的整体架构，如下图所示：

<img src="https://img-blog.csdnimg.cn/img_convert/cc7b7242484095926dd1a621f5079af3.png" weight="500" height="300">

图中最上面的两个框就是构建应用程序所需要的最核心的两大功能组件，也是我们日常开发中最常用的组件，即数据访问和 Web 服务。这两大部分功能组件中包含的内容非常多，而且充分体现了 Spring Framework 的集成性，也就是说，框架内部整合了业界主流的数据库驱动、消息中间件、ORM 框架等各种工具，开发人员可以根据需要灵活地替换和调整自己想要使用的工具。



Spring Boot 构建在 Spring Framework 基础之上，是新一代的 Web 应用程序开发框架。我们可以通过下面这张图来了解 Spring Boot 的全貌：

<img src="https://img-blog.csdnimg.cn/img_convert/0dddbf1e138885722677e9089bed0036.png" weight="500" height="300">

通过浏览 Spring 的官方网站，我们可以看到 Spring Boot 已经成为 Spring 中顶级的子项目。自 2014 年 4 月发布 1.0.0 版本以来，Spring Boot 俨然已经发展为 Java EE 领域开发 Web 应用程序的首选框架。

🥮：使用 Spring Boot 开发一个 RESTful风格 的 HTTP 端点所需要做的编码工作，如下所示：

```java
@SpringBootApplication
@RestController
public class DemoApplication {
 
    @GetMapping("/helloworld")
	public String hello() { 
	    return "Hello World!";
	}
 
	public static void main(String[] args) {
        SpringApplication.run(DemoApplication.class, args);
    }
}
```

这是一个经典的“Hello World”程序，而且使用 Spring Boot 来构建这样一个支持 RESTful 风格的 Web 应用程序只需要几秒钟。一旦创建了一个 Spring Boot 应用程序，并添加类似上面的 DemoApplication 类，我们就可以启动 Spring Boot 内置的 Web 服务器并监听 8080 端口，剩余的一切工作 Spring Boot 都帮你自动完成。

> Spring Boot 并没有像以前使用 Spring MVC 一样需要指定一大堆关于 HTTP 请求和响应的 XML 配置，Spring Boot应用大量第三方库几乎可以零配置的使用。事实上，Spring Boot 的运行过程同样还是依赖于 Spring MVC，但是它把原本需要开发人员指定的各种配置项设置了默认值，并内置在了运行时环境中，例如默认的服务器端口就是 8080，如果我们不需要对这些配置项有定制化需求，就可以不做任何的处理，采用既定的开发约定即可。这就是 Spring Boot 所倡导的约定优于配置（Convention over Configuration）设计理念。
---
**Spring Boot的主要优点是让我们更加专注于实际开发工作，而非环境配置。以下是Spring Boot的一些核心功能：**

**自动配置**：Spring Boot自动配置意味着它能自动为你的应用程序添加对第三方库的支持。例如，如果你在类路径下添加了Spring Web MVC, Spring Boot会自动配置模板引擎、静态资源支持等。

**嵌入式服务器**：Spring Boot带有像Tomcat或Jetty这样的嵌入式Servlet容器，开发者无需额外部署war文件即可启动应用。

**监控应用**：Spring Boot Actuator模块提供了许多服务，如检查应用状态、审计、追踪等功能。

**微服务**：Spring Boot是构建微服务架构的基础，它能快速地创建独立运行的应用。

**可独立运行的Spring项目**：Spring Boot可以以jar包的形式独立运行。

**简化的Maven配置**：Spring提供推荐的基础 POM 文件来简化Maven 配置。

---

##### Spring Boot 与前后端的关系

1. **后端开发**：Spring Boot 本质上是一个后端框架，用于构建基于 Java 的 Web 应用和微服务。它提供了强大的 REST API 开发支持，使得后端开发变得更加高效和便捷。
2. **前后端分离**：在现代 Web 开发中，前后端分离是一种常见的架构模式。前端通常由单页应用框架（如 React、Angular、Vue.js）构建，通过 REST API 与后端进行通信。Spring Boot 非常适合这种架构模式，因为它能够轻松地创建 RESTful 服务，为前端提供数据和业务逻辑支持。
3. **集成前端资源**：尽管前后端分离是趋势，但在某些情况下，后端项目可能仍需要提供静态资源（如 HTML、CSS、JavaScript）。Spring Boot 允许你将这些前端资源打包到应用中，并通过内嵌的 Web 服务器提供服务。

### Spring Framework
Spring Framework 是一个全面的应用框架，提供了多种功能模块，包括依赖注入、面向切面编程、数据访问、事务管理、Web MVC 框架等。
1. **依赖注入（Dependency Injection）**：通过 IoC 容器管理对象的创建和依赖关系。
2. **面向切面编程（AOP）**：提供横切关注点（如日志记录、事务管理等）的支持。
3. **数据访问**：支持 JDBC、JPA、ORM（如 Hibernate）等多种数据访问技术。
4. **事务管理**：提供声明式和编程式事务管理。
5. **Web 框架**：提供 Spring MVC，用于构建 Web 应用。

### Spring MVC

**Spring MVC** 是 Spring Framework 的一个子项目，用于构建基于模型-视图-控制器（MVC）架构的 Web 应用。Spring MVC 的主要特性包括：

- **请求处理**：通过控制器（Controller）处理 HTTP 请求。
- **模型和视图**：支持多种视图技术，如 JSP、Thymeleaf、Freemarker 等。
- **表单处理**：提供表单绑定、验证和数据转换的支持。
- **REST 支持**：方便地构建 RESTful Web 服务。

### Spring Boot

**Spring Boot** 是 Spring 的一个子项目，用于简化 Spring 应用的创建、配置和部署。Spring Boot 的主要特性包括：

- **自动配置**：根据项目的依赖和配置自动配置 Spring 应用，减少手动配置的复杂性。
- **起步依赖**：提供一组预配置的依赖，方便快速引入常用功能模块。
- **嵌入式服务器**：支持将应用打包成独立的 JAR 文件，并内嵌 Tomcat、Jetty 或 Undertow 服务器。
- **生产就绪功能**：提供监控、度量、健康检查等生产级特性。
- **CLI 工具**：支持使用 Spring Boot CLI 通过 Groovy 脚本快速创建应用。

### 微信小程序开发

> 基本结构:
> - **视图层**来实现页面结构，由WXML和WXSS编写，并由组件进行展示。
> - **逻辑层**来实现后台功能，由JavaScript编写，逻辑层将数据进行处理后发送给视图层，同时接受视图层的事件反馈。

**小程序的核心技术主要是三个：**

- *页面布局：WXML，类似HTML*
- *页面样式：WXSS，几乎就是CSS(某些不支持，某些进行了增强，但是基本是一致的)*
- *页面脚本：JavaScript+WXS(WeixinScript)*

**小程序架构：**

**逻辑层 App Service**
> 小程序开发框架的逻辑层使用 JavaScript 引擎为小程序提供开发 JavaScript 代码的运行环境以及微信小程序的特有功能。
逻辑层将数据进行处理后发送给视图层，同时接受视图层的事件反馈。
开发者写的所有代码最终将会打包成一份 JavaScript 文件，并在小程序启动的时候运行，直到小程序销毁。这一行为类似 ServiceWorker，所以逻辑层也称之为 App Service。
在 JavaScript 的基础上，微信增加了一些功能，以方便小程序的开发：
增加 App 和 Page 方法，进行程序注册和页面注册。
增加 getApp 和 getCurrentPages 方法，分别用来获取 App 实例和当前页面栈。
提供丰富的 API，如微信用户数据，扫一扫，支付等微信特有能力。
提供模块化能力，每个页面有独立的作用域。
注意：小程序框架的逻辑层并非运行在浏览器中，因此 JavaScript 在 web 中一些能力都无法使用，如 window，document 等。

**视图层 View**
> 框架的视图层由 WXML 与 WXSS 编写，由组件来进行展示。
    将逻辑层的数据反映成视图，同时将视图层的事件发送给逻辑层。
    WXML(WeiXin Markup language) 用于描述页面的结构。
    WXS(WeiXin Script) 是小程序的一套脚本语言，结合 WXML，可以构建出页面的结构。
    WXSS(WeiXin Style Sheet) 用于描述页面的样式。
    组件(Component)是视图的基本组成单元。

微信小程序基本文件结构和项目目录：

```
    .
    ├── app.js     # 小程序的逻辑文件
    ├── app.json   # 小程序的配置文件
    ├── app.wxss   # 全局公共样式文件
    ├── pages      # 存放小程序的各个页面
    │   ├── index  # index页面
    │   │   ├── index.js     # 页面逻辑
    │   │   ├── index.wxml   # 页面结构
    │   │   └── index.wxss   # 页面样式表
    │   └── logs   # logs页面
    │       ├── logs.js      # 页面逻辑
    │       ├── logs.json    # 页面配置
    │       ├── logs.wxml    # 页面结构
    │       └── logs.wxss    # 页面样式表
    ├── project.config.json
    └── utils
        └── util.js
```
<img src="https://img2024.cnblogs.com/blog/3358182/202406/3358182-20240603223012713-830660686.png" width="800" height="150">
即：


<img src="https://img-blog.csdnimg.cn/direct/46d9d8e0269840559af74d7cb078d6ca.png" width="170" height="250">

<img src="https://img-blog.csdnimg.cn/7985c981a0904d8b92e66df8a9eca4cb.png" width="350" height="300">

REFERENCE(THANKS FOR)：

<a href="https://blog.csdn.net/weixin_45857341/article/details/136006501">家族生态：如何正确理解 Spring 家族的技术体系？</a>

<a href="https://blog.csdn.net/FMC_WBL/article/details/135162094">Springboot是什么？Springboot详解！入门介绍</a>
