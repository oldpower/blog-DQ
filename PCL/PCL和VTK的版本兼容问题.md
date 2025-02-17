## PCL和VTK的版本兼容问题

### 1、PCL 和 VTK 版本兼容表

`VTK`（Visualization Toolkit）和 PCL（Point Cloud Library）是两个常用的开源库，它们在处理点云数据和可视化时经常一起使用。由于 PCL 依赖于 VTK 进行可视化功能，因此它们的版本兼容性非常重要。以下是 VTK 和 PCL 的版本兼容表，供参考：

| PCL 版本 | 兼容的 VTK 版本         | 备注                                                                 |
|----------|--------------------------|----------------------------------------------------------------------|
| PCL 1.14 | VTK 9.x, VTK 8.2         | 推荐使用 VTK 9.x，VTK 8.2 也支持。                                   |
| PCL 1.13 | VTK 9.x, VTK 8.2         | VTK 9.x 是首选，VTK 8.2 也兼容。                                     |
| PCL 1.12 | VTK 8.2, VTK 9.x         | VTK 8.2 是主要支持版本，VTK 9.x 可能需要额外配置。                   |
| PCL 1.11 | VTK 8.2                  | 推荐使用 VTK 8.2，VTK 9.x 可能不完全兼容。                           |
| PCL 1.10 | VTK 8.2                  | 仅支持 VTK 8.2，VTK 9.x 不兼容。                                     |
| PCL 1.9  | VTK 8.2                  | 仅支持 VTK 8.2，VTK 9.x 不兼容。                                     |
| PCL 1.8  | VTK 7.x, VTK 8.0         | 推荐使用 VTK 7.x，VTK 8.0 可能部分兼容。                             |
| PCL 1.7  | VTK 7.x                  | 仅支持 VTK 7.x，VTK 8.x 不兼容。                                     |
| PCL 1.6  | VTK 6.x                  | 仅支持 VTK 6.x，VTK 7.x 不兼容。                                     |

### 2、注意事项
 - VTK 9.x 是较新的版本，支持更多的功能和优化，推荐在 PCL 1.12 及以上版本中使用。

 - VTK 8.2 是 PCL 1.10 到 PCL 1.13 的主要支持版本，兼容性较好。

 - VTK 7.x 是 PCL 1.7 到 PCL 1.9 的主要支持版本，不建议在新项目中使用。

 - VTK 6.x 仅适用于 PCL 1.6 及以下版本，已过时。


### 3、如何检查 VTK 和 PCL 版本
#### 3.1 检查 VTK 版本
使用vtkVersionMacro
```bash
vtkVersionMacro
```

或者在 CMake 中查找 VTK 版本： 
```bash
cmake --find-package -DNAME=VTK -DCOMPILER_ID=GNU -DLANGUAGE=CXX -DMODE=COMPILE
-I/usr/local/include/vtk-7.1 -I/home/xxx/miniconda3/envs/env01/include/python3.10
```

#### 3.2 检查 PCL 版本
使用pcl_version
```bash
pcl_version
```
或者在代码中打印版本：
```cpp
#include <pcl/common/common.h>
std::cout << "PCL version: " << PCL_VERSION << std::endl;
```

4、推荐组合
 - PCL 1.14 + VTK 9.x：最新版本，支持最新功能。

 - PCL 1.12 + VTK 8.2：稳定组合，适合大多数项目。

 - PCL 1.10 + VTK 8.2：适合需要稳定性的旧项目。