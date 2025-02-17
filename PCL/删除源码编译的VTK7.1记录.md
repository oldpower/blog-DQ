## 删除源码编译的VTK7.1记录

### 1. 找到 VTK 的安装路径
在编译安装 VTK 时，你可能使用了 cmake 命令，并指定了安装路径。通常情况下，安装路径可能是 /usr/local，但也可能是其他自定义路径。通过查看 CMakeCache.txt 文件来确定安装路径。在编译目录中找到 CMakeCache.txt，然后搜索 CMAKE_INSTALL_PREFIX 这个变量，它指明了安装路径。

```bash
cat CMakeCache.txt | grep CMAKE_INSTALL_PREFIX
```

### 2. 删除安装文件
根据安装路径，删除 VTK 7.1 的相关文件和目录:
```bash
sudo rm -rf /usr/local/include/vtk-7.1

sudo rm -rf /usr/local/lib/cmake/vtk-7.1

sudo rm -rf /usr/local/lib/vtk-7.1
sudo rm -rf /usr/local/bin/vtk*7.1*

sudo rm -rf /usr/local/lib/libvtk*
sudo rm -rf /usr/local/bin/vtk*
```
请注意，/usr/local/lib 目录下可能包含一些库文件，确保这些库文件确实是 VTK 7.1 的，然后再删除。 


### 3. 删除配置文件
有时安装程序会在 /etc/ld.so.conf.d/ 目录下生成配置文件，需要检查并删除这些文件:
```bash
sudo rm /etc/ld.so.conf.d/vtk-7.1.conf
```

### 4. 清理缓存
删除相关文件后，更新动态链接库缓存：
```bash
sudo ldconfig
```

### 5. 删除编译目录
如果你还保留有编译 VTK 7.1 的源码目录，可以将其删除:
```bash
rm -rf /path/to/vtk-7.1-build        
```
### 7. 验证删除
通过以下命令验证 VTK 7.1 是否已经被删除：
```bash
ldconfig -p | grep vtk
#或者尝试导入 VTK 模块：
python -c "import vtk"
```

