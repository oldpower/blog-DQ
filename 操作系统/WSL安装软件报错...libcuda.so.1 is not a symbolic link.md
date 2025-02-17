## WSL安装软件报错/sbin/ldconfig.real: /usr/lib/wsl/lib/libcuda.so.1 is not a symbolic link

WSL安装软件报错/sbin/ldconfig.real: /usr/lib/wsl/lib/libcuda.so.1 is not a symbolic link

### 原因
/usr/lib/wsl/lib/目录下都是文件而不是链接，且该目录只读，需要在其他目录操作

### 解决
给/user/lib/wsl/lib目录下所有文件创建链接
```bash
cd /usr/lib/wsl
sudo mkdir lib2
sudo ln -s lib/* lib2
```

更改wsl配置文件 ，将 /usr/lib/wsl/lib 改为 /usr/lib/wsl/lib2
```bash
sudo vim /etc/ld.so.conf.d/ld.wsl.conf
```

测试修改是否生效
```bash
sudo ldconfig
```

### 永久修改 

修改/etc/wsl.conf
```bash
sudo cat >> /etc/wsl.conf << EOF
[automount]
ldconfig = fasle
EOF 
```
