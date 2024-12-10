<img width="372" alt="image" src="https://github.com/user-attachments/assets/1ab89dce-ffb9-452e-a3bb-2dbaca537d6b">

---

1. autoload_configs 的位置

在 FreeSWITCH 的安装目录中，autoload_configs 文件夹通常位于配置目录下：

默认路径：

/usr/local/freeswitch/conf/autoload_configs/

如果找不到：
	•	如果你是从源代码编译安装的 FreeSWITCH，则安装目录通常为 /usr/local/freeswitch。
	•	如果使用的是包管理工具安装，配置目录可能在 /etc/freeswitch/autoload_configs。

在该目录中，你可以找到模块加载相关的配置文件，如 ai.conf.xml、modules.conf.xml 等。

2. 如何利用 mod_ai 实现接入自有 ASR

mod_ai 是 FreeSWITCH 提供的模块，用于集成 AI 服务（包括 ASR 和 TTS）。下面是实现接入自有 ASR 的步骤：

(1) 加载 mod_ai

在 autoload_configs/modules.conf.xml 文件中，确保 mod_ai 已被加载：

<load module="mod_ai"/>

(2) 配置 ai.conf.xml

找到 ai.conf.xml 文件（位于 autoload_configs 中），按照以下步骤修改以接入你的 ASR 引擎。

示例配置：

<configuration name="ai.conf" description="AI Integration">
  <profiles>
    <profile name="custom_asr">
      <param name="type" value="asr"/> <!-- 声明为 ASR -->
      <param name="server" value="http://your-asr-server.com/api"/> <!-- ASR 接口地址 -->
      <param name="apikey" value="your-api-key"/> <!-- 可选：用于认证 -->
      <param name="language" value="en-US"/> <!-- 设置语言 -->
      <param name="timeout" value="5000"/> <!-- 超时时间，单位毫秒 -->
    </profile>
  </profiles>
</configuration>

参数说明：
	•	type: 必须设置为 asr。
	•	server: 指向你的 ASR 接口地址。
	•	apikey: 如果你的 ASR 接口需要认证，可以添加。
	•	language: 设置默认语言（如 zh-CN、en-US）。
	•	timeout: 请求的超时时间。

(3) 配置 Dialplan 使用 ASR

在 dialplan 配置文件（如 conf/dialplan/default.xml）中，设置 ASR 相关的处理逻辑。

示例配置：

<extension name="asr_test">
  <condition field="destination_number" expression="^1234$">
    <!-- 选择使用的 ASR profile -->
    <action application="set" data="ai_profile=custom_asr"/>
    
    <!-- 开始语音识别 -->
    <action application="play_and_detect_speech" data="/path/to/prompt.wav detect=asr,10"/>
    
    <!-- 输出识别结果到日志 -->
    <action application="log" data="ASR Result: ${speech_text}"/>
  </condition>
</extension>

注意：
	•	ai_profile 对应 ai.conf.xml 中的 profile 名称。
	•	play_and_detect_speech 播放提示音并进行语音识别，识别结果保存在 ${speech_text}。

(4) 测试
	1.	重启 FreeSWITCH 服务：

systemctl restart freeswitch


	2.	测试呼叫：
呼叫 1234（示例号码）并验证 ASR 是否正常工作。
	3.	查看日志：
检查 FreeSWITCH 的日志，确认识别结果：

tail -f /var/log/freeswitch/freeswitch.log

(5) 自有 ASR 接口要求

确保你的 ASR 接口满足以下要求：
	•	接收音频流或文件（常见格式如 PCM、WAV）。
	•	返回 JSON 格式的识别结果，包含以下字段：

{
  "text": "识别结果文本",
  "confidence": 0.95
}



(6) 若需自定义音频处理

mod_ai 的默认行为可能不满足需求，你可以修改音频编码格式或实现自定义音频流传输。具体可以通过 FreeSWITCH 的 mod_sofia 捕获 RTP 流，并将其转发到你的 ASR 服务。

如果需要进一步指导，可以详细说明你的 ASR 接口和需求，我可以提供更具体的建议和代码示例！