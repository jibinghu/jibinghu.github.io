https://zhuanlan.zhihu.com/p/677622575?share_code=12FzRI3tKPLIq&utm_psn=1903096199495541413

```
# 查到了如下两个命令可以一定程度上发现被隐藏的进程
[sysdig](https://zhida.zhihu.com/search?content_id=238712996&content_type=Article&match_order=1&q=sysdig&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NDY2ODY4NDAsInEiOiJzeXNkaWciLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyMzg3MTI5OTYsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.39UtTm5OJOKMHHlRPn7w_E3wI1xANgN0lnMfqACshcg&zhida_source=entity) -c topprocs_cpu # 该命令可以输出cpu占用的排行，经测试可以显示出被隐藏的进程
[unhide](https://zhida.zhihu.com/search?content_id=238712996&content_type=Article&match_order=1&q=unhide&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NDY2ODY4NDAsInEiOiJ1bmhpZGUiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyMzg3MTI5OTYsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.8zSX_tfrCuLHaKwRMEosYQchj0ARiIl8O0zDM2IhI_o&zhida_source=entity) proc # 自助搜索隐藏进程，linux系统中一切皆文件，proc目录下保存的就是所有正在运行程序的进程ID，即PID
```
以上两个命令需要额外安装，不是系统自带命令

```
# sysdig可用如下命令安装，curl是类似wget的下载命令，这里下载的是一个bash脚本，下载后再通过bash执行安装
curl -s https://s3.amazonaws.com/download.draios.com/stable/install-sysdig | bash
# unhide 可以通过如下命令进行安装
yum install unhide # 中毒的是[Centos7](https://zhida.zhihu.com/search?content_id=238712996&content_type=Article&match_order=1&q=Centos7&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NDY2ODY4NDAsInEiOiJDZW50b3M3IiwiemhpZGFfc291cmNlIjoiZW50aXR5IiwiY29udGVudF9pZCI6MjM4NzEyOTk2LCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.CnLDI7ucXXE5wALn1HfCdZ-pDPAyK2zd9quRPdnTWB0&zhida_source=entity)系统，所以用的是yum
```

```
# 至于为什么不要第一时间用kill删掉病毒进程，这是一个惨痛教训，后面会提到。
# 首先我们用如下命令查询进行是如何被运行的
systemctl status 60818 # 60818为病毒的PID
# 这里的输出结果PID目录下environ文件内容是类似的
```

```
# 所以这里我先通过如下命令将该服务停掉
journalctl -u mdcheck-4838751a # 终止之前可以用该命令查看服务运行状态
systemctl stop mdcheck-4838751a # 终止该挖矿服务
systemctl disable mdcheck-4838751a # 终止该挖矿服务的开机自启
```

```
# 挖矿病毒都会有定时的网络发送信息，所以可用如下命令查看在病毒运行期间，是否存在异常IP地址
netstat -natp
```

```
# 知道这些后，我们可以用防火墙firewalld或iptables对这些IP进行封禁
iptables -I INPUT -s IP -j DROP
firewall-cmd --permanent --add-rich-rule='rule family="ipv4" source address="IP" reject'
# 其中IP字段替换为真实的黑客IP
```

```
systemctl enable iptables # 开机自启
systemctl start iptables # 打开服务
```

```
chattr -iR /etc/dns # 解除改目录的误删保护，R为递归执行，但是文件夹下的病毒都被隐藏，所以这里的R其实没有起作用
```

```
# 一方面是ssh远程登录没有禁用root的远程登录，此时需要在/etc/ssh/sshd_config中添加
PermitRootLogin no
# 保存后，对ssh服务进行重启
systemctl restart sshd
```

