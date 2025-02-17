## Linux(Ubuntu24.04)安装Eigen3库

本次安装`Eigen3`是在`WSL2的Ubuntu24.04`环境下进行。

`Eigen3`是一个C++模板库，用于线性代数、矩阵运算和数值计算。它提供了一组高性能的矩阵和向量操作，以及常用的线性代数算法，如矩阵分解、特征值求解和最小二乘解等。

### 1、安装Eigen3
有两种安装方式，一种是apt直接install，另一种是源码编译安装。

#### 1.1 方式1
libeigen3-dev是Eigen库的开发包，包含了开发所需的头文件和库文件。
```bash
sudo apt-get update
sudo apt install libeigen3-dev
```
通常情况，安装位置在 /usr/local/include/eigen3或者/usr/include/eigen3中，ls查看eigen3下有两个文件夹Eigen、unsupported和一个signature_of_eigen3_matrix_library文件。
```bash
ls /usr/include/eigen3/
Eigen  signature_of_eigen3_matrix_library  unsupported
```

#### 1.2 方式2
```bash
#在GitHub上克隆eigen3库文件
git clone https://github.com/OPM/eigen3.git
 
#编译安装
cd eigen3
mkdir build
cd build
cmake ..
sudo make install 
```

#### 1.3 Eigen路径
因为eigen3 被默认安装到了usr/local/include里了（或者是usr/include里，这两个都差不多，都是系统默认的路径），在很多程序中include时经常使用`#include <Eigen/Dense>`而不是使用`#include <eigen3/Eigen/Dense>`所以要做下处理，否则一些程序在编译时会因找不到Eigen/Dense而报错。此时只能在CMakeLists.txt用include_libraries(绝对路径)。解决方式如下：

(1)创建软连接（推荐）
```bash
sudo ln -s /usr/include/eigen3/Eigen /usr/local/include/Eigen
```
(2)或者复制Eigen到上级目录
```bash
sudo cp -r /usr/local/include/eigen3/Eigen /usr/local/include 
```

1.4 查看Eigen3版本
```bash
#使用pkg-config工具查看
pkg-config --modversion eigen3
3.4.0
```

### 2、测试Eigen3
#### 2.1 创建test.cpp
```bash
#建立 test 测试文件
touch test.cpp
#用 gedit 打开此测试文件
vim test.cp
```

2.2 代码
```bash
#include <iostream>
#include <Eigen/Dense>
//using Eigen::MatrixXd;
using namespace Eigen;
using namespace Eigen::internal;
using namespace Eigen::Architecture;
 
using namespace std;
 
int main()
{
        cout<<"*******************1D-object****************"<<endl;
        Vector4d v1;
        v1<< 1,2,3,4;
        cout<<"v1=\n"<<v1<<endl;
 
        VectorXd v2(3);
        v2<<1,2,3;
        cout<<"v2=\n"<<v2<<endl;
 
        Array4i v3;
        v3<<1,2,3,4;
        cout<<"v3=\n"<<v3<<endl;
 
        ArrayXf v4(3);
        v4<<1,2,3;
        cout<<"v4=\n"<<v4<<endl;
}
```

#### 2.3 编译运行 
```bash
g++ test.cpp -o test
./test
*******************1D-object****************
v1=
1
2
3
4
v2=
1
2
3
v3=
1
2
3
4
v4=
1
2
3
```