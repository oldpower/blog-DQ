## PCL源码编译报错[cannot find -lvtkIOMPIImage: No such file or directory ...]

### 1、问题描述
源码编译PCL1.14，遇到如下问题：
```bash
cannot find -lvtkIOMPIImage: No such file or directory
/usr/bin/ld: cannot find -lvtkIOMPIParallel: No such file or directory
/usr/bin/ld: cannot find -lvtkFiltersParallelDIY2: No such file or directory
collect2: error: ld returned 1 exit status
```

### 2、问题原因
在编译 PCL 1.14 时，如果遇到 `cannot find -lvtkIOMPIImage: No such file or directory` 错误，通常是因为系统缺少 VTK 库或相关依赖。 

首先确保已经安装了 VTK 库，并且确保 VTK 库的路径在系统的库搜索路径中，如果确认无问题，那大概率是`VTK版本和PCL版本不匹配`或者是`在安装VTK时未选择安装MPI相关功能`。

我是用的`VTK1.7`版本，因此应该是因为版本不匹配导致的，需更换VTK版本或者降低PCL安装版本来解决问题。

### 3、VTK支持MPI的版本 
VTK从 VTK 6.0 版本开始引入了对 MPI（Message Passing Interface）的支持。MPI 支持主要用于并行计算和大规模数据处理，特别是在科学计算和可视化领域。
 - VTK 6.0：

    - 首次引入了 MPI 支持。

    - 提供了基本的并行数据处理和可视化功能。

 - VTK 7.x：

    - 改进了 MPI 模块的稳定性和性能。

    - 增加了更多的并行算法和工具。

 - VTK 8.x：

    - 进一步优化了 MPI 支持。

    - 提供了更丰富的并行可视化功能。

 - VTK 9.x：

    - 对 MPI 的支持更加成熟。

    - 提供了更高效的并行数据处理和渲染能力。

### 4、VTK 中 MPI 相关的模块
VTK 中与 MPI 相关的模块主要包括：

 - vtkParallelMPI：提供 MPI 并行计算的核心支持。

 - vtkIOParallel：支持并行 I/O 操作。

 - vtkFiltersParallel：提供并行过滤器。

 - vtkRenderingParallel：支持并行渲染。

### 5、如何启用 VTK 的 MPI 支持
在编译 VTK 时，需要显式启用 MPI 支持。以下是具体步骤：
#### 5.1 安装 MPI 库
确保系统中安装了 MPI 库（如 OpenMPI 或 MPICH）：
```bash
sudo apt install openmpi-bin libopenmpi-dev
```
 5.2 配置 VTK 编译选项：
在 CMake 配置中启用 MPI 支持： 
```bash
cmake -DVTK_GROUP_ENABLE_MPI=YES -DVTK_MODULE_ENABLE_VTK_ParallelMPI=YES ..
```
需要注意的是，我的`VTK`版本是`V1.7`，`CMakeLists.txt`中并没有`TK_GROUP_ENABLE_MPI`和`VTK_MODULE_ENABLE_VTK_ParallelMPI`配置选项。我只在`cmake过程的日志`中查找到：
>-- Group MPI modules: vtkFiltersParallelDIY2; vtkFiltersParallelGeometry; vtkFiltersParallelMPI; vtkIOMPIImage; vtkIOMPIParallel; vtkIOParallelNetCDF; vtkParallelMPI; vtkParallelMPI4Py; vtkdiy2

#### 5.3 编译安装
```bash
make -j$(nproc)
sudo make install
```

#### 5.4 验证 MPI 支持
编译完成后，可以检查是否成功启用了 MPI 支持： 
```bash
vtkVersionMacro | grep MPI
```

### 6、注意事项
 - MPI 库版本：

    - 确保使用的 MPI 库（如 OpenMPI 或 MPICH）与 VTK 兼容。

    - 推荐使用较新的 MPI 版本（如 OpenMPI 4.x）。

 - VTK 版本选择：

    - 如果需要 MPI 支持，建议使用 VTK 8.x 或 VTK 9.x，因为这些版本对 MPI 的支持更加完善。

    - PCL和VTK版本对应关系。

 - 并行性能：

    - 在使用 VTK 的 MPI 功能时，注意数据划分和通信开销，以优化并行性能。
        