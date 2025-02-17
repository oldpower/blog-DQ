​
## <center>LunarVim安装</center>
LunarVim以其丰富的功能和灵活的定制性，迅速在Nvim用户中流行开来。它不仅提供了一套完善的默认配置，还允许用户根据自己的需求进行深度定制。无论是自动补全、内置终端、文件浏览器，还是模糊查找、LSP支持、代码检测、格式化和调试，LunarVim都能轻松应对。

### 1、安装LunarVim
#### 方式1
这种方式是先安装Neovim，然后将LunarVim仓库拉取到~/.config/nvim，并使用LunarVim的init.lua初始化配置文件。
在开始之前，请确保你已经安装了Neovim。接下来，使用Git克隆LunarVim仓库：
```bash
git clone --depth=1 https://github.com/LunarVim/LunarVim.git ~/.config/nvim
```
在终端中运行以下命令以初始化你的配置，这将创建一个符号链接到init.lua文件，作为Neovim的配置入口。
```bash
bash
cd ~/.config/nvim
ln -sf init.lua ~/.config/nvim/init.vim
nvim
```
自动安装依赖，打开Neovim后，输入以下命令安装所有必要的依赖：
```bash
:LuarocksInstall
```
关闭并重新启动Neovim，现在你应该有了预配置的LunarVim环境。 
```bash
nvim
```
#### 方式2
[按照LunarVim社区文档进行安装](https://www.lunarvim.org/zh-Hans/docs/1.2/installation)

Installation 

Prerequisites​

 - Make sure you have installed the latest version of Neovim v0.8.0+.
 - Have git, make, pip, python npm, node and cargo installed on your system.
 - Resolve EACCES permissions when installing packages globally to avoid error when installing packages with npm.
 - PowerShell 7+ (for Windows)

Release​

(Neovim 0.8.0)

No alarms and No surprises:

```bash
LV_BRANCH='release-1.2/neovim-0.8' bash <(curl -s https://raw.githubusercontent.com/lunarvim/lunarvim/fc6873809934917b470bff1b072171879899a36b/utils/installer/install.sh)
```

### 2、其它 
个人感觉使用LunarVim版本release-1.2/neovim-0.8比较好用。
![在这里插入图片描述](access/001.png#pic_center)

**参考**

[Linux上配置LunarVim：快速初始化Neovim，让你的文本编辑更加清爽和强大-腾讯云开发者社区-腾讯云](https://cloud.tencent.com/developer/article/2215919)



​