UniMRCP 是一个开源项目，旨在跨平台实现媒体资源控制协议（MRCP），符合 IETF 的 RFC6787（MRCPv2）和 RFC4463（MRCPv1）规范。该项目集成了 SIP、RTSP、SDP、MRCPv2 以及 RTP/RTCP 协议栈，为集成者提供一致的协议接口。 ([GitHub](https://github.com/unispeech/unimrcp/blob/master/docs/mainpage.docs))

### 项目架构：

UniMRCP 的架构由多个模块和库组成，主要包括：
#### 库（Libraries）：
•	apr-toolkit：基于 APR 和 APR-util 库构建的工具集，提供任务抽象、日志记录等功能。
	•	mpf：媒体处理框架。
	•	mrcp：MRCP 基础实现，包括消息、解析器和资源。
	•	mrcpv2-transport：MRCPv2 传输层的实现。
	•	mrcp-signaling：抽象的 MRCP 信令（会话管理）接口。
	•	mrcp-engine：抽象的资源引擎接口。
	•	mrcp-client：基于抽象信令接口的 MRCP 客户端栈实现。
	•	mrcp-server：基于抽象信令和引擎接口的 MRCP 服务器栈实现。
	•	uni-rtsp：MRCPv1 所需的最小 RTSP 栈实现。
#### 模块（Modules）：
•	mrcp-sofiasip：使用 SofiaSIP 库实现的抽象信令接口。
	•	mrcp-unirtsp：使用 UniRTSP 库实现的抽象信令接口。
#### 插件（Plugins）：
•	demo-synth：模拟语音合成的 TTS 插件。
	•	demo-recog：模拟语音识别的 ASR 插件。
	•	demo-verif：模拟说话人验证的 SVI 插件。
	•	mrcp-recorder：录音插件的实现。
#### 平台（Platforms）：
•	libunimrcpclient：基于底层 mrcp-client 库，使用 mrcp-sofiasip 和 mrcp-unirtsp 模块构建的 UniMRCP 客户端栈。
	•	libunimrcpserver：基于底层 mrcp-server 库，使用 mrcp-sofiasip 和 mrcp-unirtsp 模块构建的 UniMRCP 服务器栈。
	•	unimrcpclient：基于 UniMRCP 客户端栈的示例 C 应用程序。
	•	umc：基于 UniMRCP 客户端栈的示例 C++ 应用程序。
	•	unimrcpserver：UniMRCP 服务器应用程序。

### 主要特性：
•	协议支持： 完全支持 MRCPv1 和 MRCPv2 协议。
	•	资源类型： 支持语音合成（TTS）、语音识别（ASR）、说话人验证（SVI）和录音等资源。
	•	跨平台： 兼容多种操作系统，包括 Windows 和 Linux。
	•	编程语言： 主要使用 C 和 C++ 语言编写。
	•	开源许可： 采用 Apache 2.0 许可证。

### 项目资源：
•	官方网站： http://www.unimrcp.org/
	•	下载地址： http://www.unimrcp.org/downloads
	•	文档： http://www.unimrcp.org/documentation
	•	代码仓库： https://github.com/unispeech/unimrcp
	•	问题跟踪： https://github.com/unispeech/unimrcp/issues
	•	讨论组： https://groups.google.com/group/unimrcp

通过上述模块和库的协同工作，UniMRCP 提供了一个完整的 MRCP 实现，方便开发者在不同平台上集成和使用 MRCP 协议。