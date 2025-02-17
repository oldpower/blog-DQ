## <center>Linux 环境下编译安装 OpenCV 4.8.x</center>

在 **Linux 环境下编译安装 OpenCV 4.8.x** 需要安装一系列依赖库。以下是详细的步骤说明，并附上每个依赖库的作用解释。

---

### **1. 环境准备**
#### **操作系统**
- 推荐使用 **Ubuntu 20.04/22.04** 或其他基于 Debian 的 Linux 发行版。

#### **编译器**
- **GCC 9 或更高版本**（默认已安装）
- **CMake 3.5.1 或更高版本**（用于配置和生成构建文件）

#### **Python（可选）**
- **Python 3.6 或更高版本**
- **NumPy**（用于 Python 绑定）

---

### **2. 安装依赖库**
以下是 OpenCV 4.8.x 编译所需的依赖库及其作用：

#### **2.1 基本编译工具**
```bash
sudo apt update
sudo apt install build-essential cmake git pkg-config
```
- **build-essential**：包含 GCC 编译器和基本的开发工具（如 make）。
- **cmake**：用于配置和生成 OpenCV 的构建文件。
- **git**：用于下载 OpenCV 源码。
- **pkg-config**：用于管理编译时的库路径和链接选项。

#### **2.2 图像编解码库**
```bash
sudo apt install libjpeg-dev libpng-dev libtiff-dev libopenjp2-7-dev
```
- **libjpeg-dev**：JPEG 图像格式支持。
- **libpng-dev**：PNG 图像格式支持。
- **libtiff-dev**：TIFF 图像格式支持。
- **libopenjp2-7-dev**：JPEG 2000 图像格式支持。

#### **2.3 视频编解码库**
```bash
sudo apt install libavcodec-dev libavformat-dev libswscale-dev libavutil-dev
```
- **libavcodec-dev**：视频编解码支持（FFmpeg 的一部分）。
- **libavformat-dev**：视频容器格式支持（FFmpeg 的一部分）。
- **libswscale-dev**：视频缩放和颜色空间转换支持（FFmpeg 的一部分）。
- **libavutil-dev**：FFmpeg 的工具库，提供通用功能。

#### **2.4 GUI 支持**
```bash
sudo apt install libgtk-3-dev
```
- **libgtk-3-dev**：GTK 图形界面库，用于 OpenCV 的窗口显示和用户交互。

#### **2.5 线性代数库**
```bash
sudo apt install libopenblas-dev libatlas-base-dev liblapack-dev gfortran
```
- **libopenblas-dev**：高性能线性代数库，用于矩阵运算。
- **libatlas-base-dev**：优化的线性代数库。
- **liblapack-dev**：线性代数库，用于高级数学运算。
- **gfortran**：Fortran 编译器，用于编译某些数学库。

#### **2.6 多线程支持**
```bash
sudo apt install libtbb2 libtbb-dev
```
- **libtbb-dev**：Intel TBB（Threading Building Blocks）库，用于多线程并行计算。

#### **2.7 GStreamer 支持**
```bash
sudo apt install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
```
- **libgstreamer1.0-dev**：GStreamer 多媒体框架，用于视频流处理。
- **libgstreamer-plugins-base1.0-dev**：GStreamer 基础插件。

#### **2.8 Python 绑定支持(可选)**
```bash
sudo apt install python3-dev python3-numpy
```
- **python3-dev**：Python 3 开发头文件和库。
- **python3-numpy**：NumPy 库，用于 Python 绑定的矩阵运算。

---

### **3. 下载 OpenCV 源码**
```bash
# 下载 OpenCV 和 OpenCV Contrib 源码
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git

# 切换到 4.8.x 版本
cd opencv
git checkout 4.8.x
cd ../opencv_contrib
git checkout 4.8.x
```

---

### **4. 配置 CMake**
```bash
# 创建构建目录
cd ../opencv
mkdir build
cd build

# 配置 CMake
cmake -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
      -D OPENCV_ENABLE_NONFREE=ON \
      -D BUILD_opencv_python2=OFF \  # 禁用 Python 2 绑定
      -D BUILD_opencv_python3=OFF \  # 禁用 Python 3 绑定
      -D BUILD_opencv_python_bindings_generator=OFF \  # 禁用 Python 绑定生成器
```


#### **可选配置**
- **启用 CUDA 支持**：
  ```bash
  -D WITH_CUDA=ON \
  -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
  -D CUDA_ARCH_BIN="7.5" \  # 根据 GPU 架构设置
  -D WITH_CUDNN=ON \
  ```
- **启用 OpenCL 支持**：
  ```bash
  -D WITH_OPENCL=ON \
  ```
- **启用 VTK 支持**：
  ```bash
  -D WITH_VTK=ON \
  -D VTK_DIR=/path/to/vtk/build \  # 指定 VTK 安装路径
  ```
- **启用 Python 支持**：
  ```bash
  -D BUILD_opencv_python3=ON \
  -D PYTHON3_EXECUTABLE=$(which python3) \
  -D PYTHON3_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
  -D PYTHON3_LIBRARY=$(python3 -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
  -D PYTHON3_NUMPY_INCLUDE_DIRS=$(python3 -c "import numpy; print(numpy.get_include())") 
  ```
---

### **5. 编译和安装**
```bash
# 编译（根据 CPU 核心数调整 -j 参数）
make -j$(nproc)

# 安装
sudo make install
```

---

### **6. 验证安装**
#### **C++ 验证**
```cpp
#include <opencv2/core.hpp>
#include <iostream>

int main() {
    std::cout << "OpenCV version: " << cv::getVersionString() << std::endl;
    return 0;
}
```
编译并运行：
```bash
g++ -o test_opencv test_opencv.cpp `pkg-config --cflags --libs opencv4`
./test_opencv
```

#### **Python 验证**
```python
import cv2
print("OpenCV version:", cv2.__version__)
```

---

### **7. 常见问题**
1. **缺少依赖库**：
   - 根据 CMake 输出的错误信息安装缺失的依赖库。
2. **CUDA 支持问题**：
   - 确保已安装 CUDA Toolkit 并正确配置环境变量。
3. **Python 绑定问题**：
   - 确保 Python 和 NumPy 已正确安装，并在 CMake 中正确配置路径。
