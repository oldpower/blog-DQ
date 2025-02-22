## Linux查看磁盘使用情况

⏰ 时效性说明（2025年02月22日更新）
以下命令在最新 Linux 发行版（如 Ubuntu 25.04、CentOS Stream 9）中均适用，涵盖传统命令与高效工具。

📊 Linux 查看磁盘使用情况命令大全

### 1. 快速概览磁盘空间

**命令df（Disk Free）**

```bash
# 基础用法（显示单位为GB/MB，易读）
df -h

# 查看特定文件系统（如 ext4）
df -hT | grep ext4

# 仅显示磁盘使用率超过80%的分区（预警排查）
df -h | awk 'NR==1 || $5 >= "80%"'
```


**输出关键列：**

 - Filesystem：磁盘分区
 - Use%：使用百分比
 - Mounted on：挂载点

### 2. 深度分析目录占用

**命令 du（Disk Usage）**

```bash
# 查看当前目录总占用（-s 汇总，-h 易读单位）
du -sh /path/to/directory

# 列出目录下所有子目录大小（按从大到小排序）
du -h --max-depth=1 /var | sort -hr

# 快速定位大文件（显示前10大文件）
sudo du -a / | sort -n -r | head -n 10
```

**参数说明**：

 - --max-depth=1：仅显示一级子目录
 - sort -hr：按人类可读数值逆序排序

### 3. 交互式可视化工具

**工具 ncdu（需安装）**

```bash
# 安装（Debian/Ubuntu）
sudo apt install ncdu

# 扫描并分析指定目录（如根目录）
sudo ncdu /

# 快捷键说明：
# ↑↓ 导航 | d 删除文件 | g 切换显示单位 | q 退出
```

优势：直观展示目录树结构，支持实时删除文件释放空间。

### 4. 查看块设备与分区

**命令 lsblk**

```bash
# 显示所有块设备及挂载点（-f 包含文件系统类型）
lsblk -f
```

**输出示例**:

```text
Copy Code
NAME   FSTYPE   SIZE MOUNTPOINT  
sda            500G  
├─sda1 ext4    512M /boot  
└─sda2 btrfs   499G /  
```
---

🚀 场景化选择建议

| 需求 | 推荐命令 |
| ---- | ---- |
|快速检查磁盘剩余空间|	df -h|
|定位大文件/目录|	du -sh * \| sort -hr|
|交互式清理磁盘|	ncdu|
|排查分区挂载异常|	lsblk -f 或 df -Th|
