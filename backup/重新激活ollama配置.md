在配置完 /etc/systemd/system/ollama.service 文件后，可以通过以下步骤激活并启动该服务：

1. 重新加载 systemd 配置

首先，需要重新加载 systemd 的配置，以便 systemd 识别新的服务文件：

sudo systemctl daemon-reload

2. 启动服务

重新加载配置后，可以使用以下命令启动 ollama 服务：

sudo systemctl start ollama.service

3. 设置开机自启动（可选）

如果希望在系统启动时自动启动 ollama 服务，可以启用自启动：

sudo systemctl enable ollama.service

4. 检查服务状态

可以使用以下命令检查服务是否启动成功：

sudo systemctl status ollama.service

这将显示 ollama 服务的当前状态和日志。如果服务启动正常，你应该会看到类似于“active (running)”的状态。

5. 查看服务日志（可选）

如果服务没有正常启动，可以查看其日志以进行故障排查：

journalctl -u ollama.service

这将显示 ollama 服务的运行日志，有助于发现可能的错误原因。