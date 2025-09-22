``` bash
# 如果已有key，忽略第一行命令
ssh-keygen -t rsa -b 2048 -C "<comment>" # comment可以是当前电脑的备注，方便你识别自己的设备
# 随后可以按默认配置一路回车，直至成功生成key
# 跳转到密钥文件所在目录
cd C:\Users\<username>\.ssh
# 查看并复制id_rsa.pub
type id_rsa.pub | clip
```