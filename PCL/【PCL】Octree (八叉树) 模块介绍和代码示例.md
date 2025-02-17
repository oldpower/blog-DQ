## 【PCL】Octree (八叉树) 模块介绍和代码示例

## 1、Octree模块说明
### 1.1 概述
Octree（八叉树）是一种树状数据结构，用于在三维空间中对点云数据进行高效的组织和管理。它将空间递归地划分为八个子空间（即八叉树节点），直到每个节点中的点云数据满足特定的条件（如点的数量或空间分辨率）。Octree广泛应用于点云的压缩、搜索、分割、聚类等任务。

### 1.2 Octree模块的主要功能
PCL中的Octree模块提供了以下主要功能：
- **点云的压缩与解压缩**：通过Octree结构对点云进行压缩存储，减少存储空间。
- **近邻搜索**：快速查找给定点附近的点。
- **体素搜索**：查找位于特定体素（空间立方体）内的点。
- **空间变化检测**：检测两个点云之间的空间变化。
- **点云的分割与聚类**：利用Octree结构对点云进行分割和聚类。

### 1.3 Octree模块的核心类
- **pcl::octree::OctreePointCloud<PointT, LeafContainerT, BranchContainerT>**：这是Octree的核心类，用于管理点云的Octree结构。
  - `PointT`：点云中点的类型，如`pcl::PointXYZ`。
  - `LeafContainerT`：叶子节点的容器类型。
  - `BranchContainerT`：分支节点的容器类型。

## 2、Octree模块主要功能代码示例

---

### 2.1 **点云的压缩与解压缩**
Octree可以用于压缩点云数据，通过减少点的数量或降低分辨率来节省存储空间。

```cpp
#include <pcl/point_cloud.h>
#include <pcl/octree/octree_pointcloud_compression.h>
#include <pcl/io/pcd_io.h>
#include <iostream>

int main()
{
    // 创建一个点云对象
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    // 生成一些随机点云数据
    cloud->width = 1000;
    cloud->height = 1;
    cloud->points.resize(cloud->width * cloud->height);
    for (size_t i = 0; i < cloud->points.size(); ++i)
    {
        cloud->points[i].x = 1024.0f * rand() / (RAND_MAX + 1.0f);
        cloud->points[i].y = 1024.0f * rand() / (RAND_MAX + 1.0f);
        cloud->points[i].z = 1024.0f * rand() / (RAND_MAX + 1.0f);
    }

    // 创建Octree压缩对象
    pcl::io::OctreePointCloudCompression<pcl::PointXYZ> octreeCompression;

    // 压缩点云
    std::stringstream compressedData;
    octreeCompression.encodePointCloud(cloud, compressedData);

    // 解压缩点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr decompressedCloud(new pcl::PointCloud<pcl::PointXYZ>);
    octreeCompression.decodePointCloud(compressedData, decompressedCloud);

    // 输出压缩前后的点云大小
    std::cout << "Original cloud size: " << cloud->points.size() << " points" << std::endl;
    std::cout << "Decompressed cloud size: " << decompressedCloud->points.size() << " points" << std::endl;

    return 0;
}
```

---

### 2.2 **近邻搜索**
Octree可以快速查找给定点附近的点。

```cpp
#include <pcl/point_cloud.h>
#include <pcl/octree/octree_search.h>
#include <pcl/io/pcd_io.h>
#include <iostream>
#include <vector>

int main()
{
    // 创建一个点云对象
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    // 从PCD文件加载点云数据
    if (pcl::io::loadPCDFile<pcl::PointXYZ>("sample.pcd", *cloud) == -1)
    {
        PCL_ERROR("Couldn't read file sample.pcd \n");
        return -1;
    }

    // 创建Octree对象
    float resolution = 0.01f;
    pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> octree(resolution);
    octree.setInputCloud(cloud);
    octree.addPointsFromInputCloud();

    // 定义搜索点
    pcl::PointXYZ searchPoint;
    searchPoint.x = 0.1f;
    searchPoint.y = 0.1f;
    searchPoint.z = 0.1f;

    // 近邻搜索
    int K = 10; // 搜索最近的10个点
    std::vector<int> pointIdxNKNSearch(K);
    std::vector<float> pointNKNSquaredDistance(K);

    if (octree.nearestKSearch(searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
    {
        std::cout << "Nearest neighbor search results:" << std::endl;
        for (size_t i = 0; i < pointIdxNKNSearch.size(); ++i)
        {
            std::cout << "    " << cloud->points[pointIdxNKNSearch[i]].x
                      << " " << cloud->points[pointIdxNKNSearch[i]].y
                      << " " << cloud->points[pointIdxNKNSearch[i]].z
                      << " (squared distance: " << pointNKNSquaredDistance[i] << ")" << std::endl;
        }
    }

    return 0;
}
```

---

### 2.3 **体素搜索**
Octree可以查找位于特定体素（空间立方体）内的点。

```cpp
#include <pcl/point_cloud.h>
#include <pcl/octree/octree_search.h>
#include <pcl/io/pcd_io.h>
#include <iostream>
#include <vector>

int main()
{
    // 创建一个点云对象
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    // 从PCD文件加载点云数据
    if (pcl::io::loadPCDFile<pcl::PointXYZ>("sample.pcd", *cloud) == -1)
    {
        PCL_ERROR("Couldn't read file sample.pcd \n");
        return -1;
    }

    // 创建Octree对象
    float resolution = 0.01f;
    pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> octree(resolution);
    octree.setInputCloud(cloud);
    octree.addPointsFromInputCloud();

    // 定义搜索点
    pcl::PointXYZ searchPoint;
    searchPoint.x = 0.1f;
    searchPoint.y = 0.1f;
    searchPoint.z = 0.1f;

    // 体素搜索
    std::vector<int> pointIdxVec;
    if (octree.voxelSearch(searchPoint, pointIdxVec))
    {
        std::cout << "Voxel search results:" << std::endl;
        for (size_t i = 0; i < pointIdxVec.size(); ++i)
        {
            std::cout << "    " << cloud->points[pointIdxVec[i]].x
                      << " " << cloud->points[pointIdxVec[i]].y
                      << " " << cloud->points[pointIdxVec[i]].z << std::endl;
        }
    }

    return 0;
}
```

---

### 2.4 **空间变化检测**
Octree可以检测两个点云之间的空间变化。

```cpp
#include <pcl/point_cloud.h>
#include <pcl/octree/octree.h>
#include <pcl/io/pcd_io.h>
#include <iostream>

int main()
{
    // 创建两个点云对象
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudA(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudB(new pcl::PointCloud<pcl::PointXYZ>);

    // 加载点云数据
    pcl::io::loadPCDFile<pcl::PointXYZ>("cloudA.pcd", *cloudA);
    pcl::io::loadPCDFile<pcl::PointXYZ>("cloudB.pcd", *cloudB);

    // 创建Octree对象
    float resolution = 0.01f;
    pcl::octree::OctreePointCloudChangeDetector<pcl::PointXYZ> octree(resolution);

    // 添加第一个点云
    octree.setInputCloud(cloudA);
    octree.addPointsFromInputCloud();

    // 切换到变化检测模式
    octree.switchBuffers();

    // 添加第二个点云
    octree.setInputCloud(cloudB);
    octree.addPointsFromInputCloud();

    // 获取变化点索引
    std::vector<int> newPointIdxVector;
    octree.getPointIndicesFromNewVoxels(newPointIdxVector);

    // 输出变化点
    std::cout << "Detected changes: " << newPointIdxVector.size() << " points" << std::endl;
    for (size_t i = 0; i < newPointIdxVector.size(); ++i)
    {
        std::cout << "    " << cloudB->points[newPointIdxVector[i]].x
                  << " " << cloudB->points[newPointIdxVector[i]].y
                  << " " << cloudB->points[newPointIdxVector[i]].z << std::endl;
    }

    return 0;
}
```

---

### 2.5 **点云的分割与聚类**
Octree可以用于点云的分割和聚类。

```cpp
#include <pcl/point_cloud.h>
#include <pcl/octree/octree.h>
#include <pcl/io/pcd_io.h>
#include <pcl/segmentation/extract_clusters.h>
#include <iostream>

int main()
{
    // 创建一个点云对象
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    // 从PCD文件加载点云数据
    if (pcl::io::loadPCDFile<pcl::PointXYZ>("sample.pcd", *cloud) == -1)
    {
        PCL_ERROR("Couldn't read file sample.pcd \n");
        return -1;
    }

    // 创建Octree对象
    float resolution = 0.01f;
    pcl::octree::OctreePointCloud<pcl::PointXYZ> octree(resolution);
    octree.setInputCloud(cloud);
    octree.addPointsFromInputCloud();

    // 提取聚类
    std::vector<pcl::PointIndices> clusterIndices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(0.02f); // 设置聚类容差
    ec.setMinClusterSize(100);     // 设置最小聚类点数
    ec.setMaxClusterSize(25000);   // 设置最大聚类点数
    ec.setSearchMethod(octree.getSearchMethod());
    ec.setInputCloud(cloud);
    ec.extract(clusterIndices);

    // 输出聚类结果
    std::cout << "Number of clusters: " << clusterIndices.size() << std::endl;
    for (size_t i = 0; i < clusterIndices.size(); ++i)
    {
        std::cout << "Cluster " << i << " has " << clusterIndices[i].indices.size() << " points" << std::endl;
    }

    return 0;
}
```

---

### 2.6 总结
以上代码示例展示了PCL中Octree模块的五个主要功能：
1. 点云的压缩与解压缩
2. 近邻搜索
3. 体素搜索
4. 空间变化检测
5. 点云的分割与聚类

通过这些功能，Octree可以高效地处理点云数据，适用于多种应用场景。