在 Ubuntu 服务器上禁止密码登录、只允许公钥登录，可以按照以下步骤操作：

---

## 1. 修改 SSH 配置文件

编辑 `/etc/ssh/sshd_config` 文件：

```bash
sudo nano /etc/ssh/sshd_config
```

找到并修改（或新增）以下配置：

```ini
# 禁止密码登录
PasswordAuthentication no

# 确保启用公钥认证
PubkeyAuthentication yes

# 可选：禁止 root 用户远程登录（推荐）
PermitRootLogin no
```

---

## 2. 重启 SSH 服务

保存后，重启 SSH 服务使配置生效：

```bash
sudo systemctl restart ssh
```

---

## 3. 确保公钥已配置

在你要登录的用户主目录下，确认公钥文件存在：

```bash
~/.ssh/authorized_keys
```

权限要求：

```bash
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
```

---

## 4. 测试连接

在关闭现有会话前，**建议新开一个终端测试**：

```bash
ssh user@your_server_ip
```

确认能正常用公钥登录后，再退出原会话。

---

## 5. 注意事项

* 如果远程是云服务器（如阿里云、AWS、GCP），要先在控制台添加/上传公钥，否则一旦禁用密码可能会完全失去访问权限。
* 建议在修改前保留一个备用的终端会话，防止误操作导致无法登录。
