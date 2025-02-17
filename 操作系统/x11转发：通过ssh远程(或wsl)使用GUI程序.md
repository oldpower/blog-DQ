## x11转发：通过ssh远程(或wsl)使用GUI程序

通过x11转发来实现远程查看图形界面，服务器端默认为Linux。

客户端分为Mac和Linux两种情况。

### 1、服务器端
修改服务器端的ssh设置：sudo vim /etc/ssh/ssh_config 
```bash
sudo vim /etc/ssh/sshd_config
```

找到以下内容，取消注释，如果找不到的话直接在下面加上这几行即可：
```bash
X11Forwarding yes
X11DisplayOffset 10
X11UseLocalhost yes
```
服务器端修改完成后需要执行命令重启sshd服务 ：

```bash
sudo systemctl restart sshd.service
```

### 2、客户端
`Linux：`
修改客户端的ssh设置：sudo vim /etc/ssh/ssh_config 
```bash
sudo vim /etc/ssh/ssh_config
```

添加或取消注释以下三行：
```bash
ForwardAgent yes
ForwardX11 yes
ForwardX11Trusted yes
```

重启客户端的ssh服务：
```bash
sudo systemctl restart ssh.service
```

添加-X参数连接服务器：
```bash
ssh -X user@ip
```
---
**Mac：**
在Mac上使用x11转发需要下载Xquartz，直接去官网下载dmg文件安装即可。安装好之后可以直接打开终端，连接远程服务器。要使用x11转发服务，需要在连接时加上-X参数：

