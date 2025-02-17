## 【PCL】IO（Input/Output）模块介绍和代码示例

✨ Point Cloud Library (PCL) 是一个开源的库，用于处理2D/3D点云和计算机视觉任务。PCL中的IO模块（Input/Output）主要用于点云数据的读取、写入和可视化。IO模块提供了多种文件格式的支持，如PLY、PCD、OBJ等，并且可以方便地将点云数据加载到内存中或从内存中保存到磁盘。

### 1、PCL IO模块的主要功能

1. **读取点云数据**：从磁盘加载点云数据到内存中。
2. **写入点云数据**：将内存中的点云数据保存到磁盘。
3. **可视化点云**：将点云数据可视化，便于调试和分析。

### 2、PCL IO模块的主要类

- `pcl::io::loadPCDFile`: 用于加载PCD格式的点云文件。
- `pcl::io::savePCDFile`: 用于保存点云数据到PCD文件。
- `pcl::io::loadPLYFile`: 用于加载PLY格式的点云文件。
- `pcl::io::savePLYFile`: 用于保存点云数据到PLY文件。
- `pcl::visualization::PCLVisualizer`: 用于点云的可视化。

### 3、应用代码示例

#### 3.1 读取PCD文件

```cpp
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

int main(int argc, char** argv)
{
    // 创建一个点云对象
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    // 从PCD文件加载点云数据
    if (pcl::io::loadPCDFile<pcl::PointXYZ>("input_cloud.pcd", *cloud) == -1)
    {
        PCL_ERROR("Couldn't read file input_cloud.pcd \n");
        return (-1);
    }

    // 打印点云信息
    std::cout << "Loaded "
              << cloud->width * cloud->height
              << " data points from input_cloud.pcd with the following fields: "
              << std::endl;

    for (const auto& point : *cloud)
        std::cout << "    " << point.x
                  << " " << point.y
                  << " " << point.z << std::endl;

    return (0);
}
```

#### 3.2 保存点云数据到PCD文件

```cpp
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

int main(int argc, char** argv)
{
    // 创建一个点云对象
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    // 填充点云数据
    cloud->width = 5;
    cloud->height = 1;
    cloud->points.resize(cloud->width * cloud->height);

    for (auto& point : *cloud)
    {
        point.x = 1024 * rand() / (RAND_MAX + 1.0f);
        point.y = 1024 * rand() / (RAND_MAX + 1.0f);
        point.z = 1024 * rand() / (RAND_MAX + 1.0f);
    }

    // 保存点云数据到PCD文件
    pcl::io::savePCDFileASCII("output_cloud.pcd", *cloud);
    std::cout << "Saved " << cloud->size() << " data points to output_cloud.pcd." << std::endl;

    return (0);
}
```

#### 3.3 读取PLY文件

```cpp
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>

int main(int argc, char** argv)
{
    // 创建一个点云对象
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    // 从PLY文件加载点云数据
    if (pcl::io::loadPLYFile<pcl::PointXYZ>("input_cloud.ply", *cloud) == -1)
    {
        PCL_ERROR("Couldn't read file input_cloud.ply \n");
        return (-1);
    }

    // 打印点云信息
    std::cout << "Loaded "
              << cloud->width * cloud->height
              << " data points from input_cloud.ply with the following fields: "
              << std::endl;

    for (const auto& point : *cloud)
        std::cout << "    " << point.x
                  << " " << point.y
                  << " " << point.z << std::endl;

    return (0);
}
```

#### 3.4 保存点云数据到PLY文件

```cpp
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>

int main(int argc, char** argv)
{
    // 创建一个点云对象
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    // 填充点云数据
    cloud->width = 5;
    cloud->height = 1;
    cloud->points.resize(cloud->width * cloud->height);

    for (auto& point : *cloud)
    {
        point.x = 1024 * rand() / (RAND_MAX + 1.0f);
        point.y = 1024 * rand() / (RAND_MAX + 1.0f);
        point.z = 1024 * rand() / (RAND_MAX + 1.0f);
    }

    // 保存点云数据到PLY文件
    pcl::io::savePLYFileASCII("output_cloud.ply", *cloud);
    std::cout << "Saved " << cloud->size() << " data points to output_cloud.ply." << std::endl;

    return (0);
}
```

#### 3.5 可视化点云

```cpp
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

int main(int argc, char** argv)
{
    // 创建一个点云对象
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    // 从PCD文件加载点云数据
    if (pcl::io::loadPCDFile<pcl::PointXYZ>("input_cloud.pcd", *cloud) == -1)
    {
        PCL_ERROR("Couldn't read file input_cloud.pcd \n");
        return (-1);
    }

    // 创建可视化对象
    pcl::visualization::PCLVisualizer viewer("Point Cloud Viewer");

    // 添加点云到可视化窗口
    viewer.addPointCloud<pcl::PointXYZ>(cloud, "sample cloud");

    // 设置点云大小
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");

    // 主循环
    while (!viewer.wasStopped())
    {
        viewer.spinOnce(100);
    }

    return (0);
}
```

### 4、总结

PCL的IO模块提供了丰富的功能，能够方便地处理点云数据的读取、写入和可视化。通过上述代码示例，你可以轻松地将点云数据加载到内存中、保存到磁盘，并进行可视化操作。这些功能在点云处理任务中非常有用，尤其是在点云数据的预处理和调试阶段。