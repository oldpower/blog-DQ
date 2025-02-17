## Linux(Ubuntu24.04)源码编译安装VTK9.2记录

本次安装`VTK9.2`是在`WSL2的Ubuntu24.04`环境下进行。

### 1、安装依赖
首先，确保系统安装了必要的依赖项：
```bash
sudo apt install libxt-dev 
sudo apt install libgl1-mesa-dev libglu1-mesa-dev
sudo apt install libopenmpi-dev openmpi-bin
```

 - **libxt-dev:** 这是X Toolkit库的开发包，提供了X11窗口系统的工具包支持。VTK的图形界面和渲染功能依赖于X11窗口系统，因此需要libxt-dev来确保相关功能正常编译和运行。
 - **libgl1-mesa-dev:** 这是Mesa OpenGL库的开发包，提供了OpenGL的实现。VTK使用OpenGL进行3D图形渲染，因此需要libgl1-mesa-dev来支持OpenGL相关的功能。
 - **libglu1-mesa-dev:** 这是Mesa GLU库的开发包，提供了OpenGL实用库（GLU）。GLU库包含了一些高级OpenGL功能，如曲面细分和几何处理，VTK在渲染复杂图形时会用到这些功能。
 - `libopenmpi-dev:` 这是OpenMPI的开发包，提供了MPI（Message Passing Interface）的实现。MPI是一种用于并行计算的通信协议，常用于高性能计算（HPC）和分布式系统中。VTK支持并行处理（例如并行渲染和数据处理），如果启用了VTK的并行功能（如VTK_USE_MPI），则需要libopenmpi-dev来编译和链接MPI相关的代码。
 - `openmpi-bin:` 这是OpenMPI的运行时工具包，包含了MPI程序的运行环境。在运行使用MPI的VTK程序时，openmpi-bin提供了必要的运行时支持（如mpirun命令）。


 ### 2、获取v9.2版本的VTK源码
clone源码后，checkout切换tag为v9.2.4，如需其它版本可以自行切换。
```bash
git clone https://github.com/Kitware/VTK
cd VTK
git checkout v9.2.4
```


 ### 3、编译安装
创建build文件夹并进入执行cmake： 
```bash
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local \
         -DVTK_GROUP_ENABLE_StandAlone=YES \
         -DVTK_GROUP_ENABLE_MPI=YES \
         -DVTK_USE_MPI=ON \
         -DVTK_BUILD_TESTING=ON \  #可选OFF
         -DVTK_BUILD_EXAMPLES=ON \ #可选OFF
         -DVTK_WRAP_PYTHON=OFF     #此处关闭Python相关功能
```

如需Python相关模块功能，需要将`-DVTK_WRAP_PYTHON=OFF`改为：
```bash
      -DVTK_WRAP_PYTHON=ON \
      -DVTK_PYTHON_VERSION=3 \
      -DPython3_EXECUTABLE=/usr/bin/python3 \
```

cmake执行结束无问题后，使用make编译安装：
```bash
make -j$(nproc)
sudo make install
```

### 4、验证VTK
#### 4.1 验证vtk版本
```bash
#cmake阶段设置了-DVTK_BUILD_TESTING=ON -DVTK_BUILD_EXAMPLES=ON
vtkVersion 
#或者通过头文件查看
cat /usr/local/include/vtk-9.2/vtkVersionMacros.h | grep VTK_VERSION
#define VTK_VERSION "9.2.4"
#define VTK_SOURCE_VERSION "vtk version " VTK_VERSION
#define VTK_VERSION_CHECK(major, minor, build)                                                     \
#define VTK_VERSION_NUMBER                                                                         \
  VTK_VERSION_CHECK(VTK_MAJOR_VERSION, VTK_MINOR_VERSION, VTK_BUILD_VERSION)
```

#### 4.2 验证 Python 绑定(如果开启了Python模块功能）
安装完成后，可以通过以下命令验证 Python 绑定是否成功：
```bash
python3 -c "import vtk; print(vtk.vtkVersion.GetVTKVersion())"
```

### 5、总结
**cmake参数说明：**

-DCMAKE_INSTALL_PREFIX=/usr/local：指定安装路径。

-DVTK_GROUP_ENABLE_StandAlone=YES：启用独立模块。

-DVTK_GROUP_ENABLE_MPI=YES：启用 MPI 支持。

-DVTK_USE_MPI=ON：启用 MPI。

-DVTK_BUILD_TESTING=OFF：禁用测试（加快构建速度）。

-DVTK_WRAP_PYTHON=ON：启用 Python 绑定。

-DVTK_PYTHON_VERSION=3：指定 Python 3 绑定。

如果后续需要安装PCL，一定需要启用MPI支持； 如果不使用 Python，完全可以不开启 DVTK_WRAP_PYTHON 和DVTK_PYTHON_VERSION。这样可以简化构建过程，减少构建时间和安装体积。只需在 CMake 配置中确保 DVTK_WRAP_PYTHON=OFF 即可。
