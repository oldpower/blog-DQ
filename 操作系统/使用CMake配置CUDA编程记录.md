## 使用CMake配置CUDA编程记录

---

[CMake官方文档 - CUDA支持](https://cmake.org/cmake/help/latest/module/FindCUDA.html)

---

### 🐼 位置无关和可分离编译
#### 1. **`set(CMAKE_POSITION_INDEPENDENT_CODE ON)`**
- **作用**：
  - 启用位置无关代码（Position Independent Code, PIC）。
  - 对于 C/C++ 代码，这会生成与位置无关的代码（通常用于共享库 `.so` 或 `.dll` 文件）。
  - 对于 CUDA 代码，这通常没有直接影响，因为 CUDA 设备代码本身是位置无关的。

- **适用场景**：
  - 当你需要编译共享库（动态链接库）时，PIC 是必需的。
  - 如果你在编译可执行文件，通常不需要显式设置 PIC。

- **示例**：
  ```cmake
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # 全局启用 PIC
  ```


#### 2. **`set_target_properties(matrixMul PROPERTIES CUDA_SEPARABLE_COMPILATION ON)`**
- **作用**：
  - 启用 CUDA 可分离编译（Separable Compilation）。
  - 可分离编译允许将 CUDA 代码分成多个编译单元（`.cu` 文件），并在链接时将它们合并。
  - 这对于大型 CUDA 项目非常有用，因为它可以提高编译速度和代码的可维护性。

- **适用场景**：
  - 当你的 CUDA 项目包含多个 `.cu` 文件，并且这些文件需要相互调用设备函数时。
  - 当你需要将 CUDA 代码与主机代码（C++）分开编译时。

- **示例**：
  ```cmake
  set_target_properties(matrixMul PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  ```


#### 3. **两者的结合**
- **`CMAKE_POSITION_INDEPENDENT_CODE ON`**：
  - 主要用于主机代码（C/C++），确保生成的位置无关代码可以用于共享库。
- **`CUDA_SEPARABLE_COMPILATION ON`**：
  - 主要用于 CUDA 设备代码，允许将多个 CUDA 编译单元分开编译并在链接时合并。


#### 4. **实际应用场景**
假设你有一个 CUDA 项目，其中包含多个 `.cu` 文件，并且你希望将这些文件编译成一个可执行文件或共享库。你可以这样配置：

```cmake
cmake_minimum_required(VERSION 3.14)
project(MyCUDAProject LANGUAGES CXX CUDA)

# 启用位置无关代码（适用于共享库）
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 查找 CUDA 工具包
find_package(CUDA REQUIRED)

# 添加可执行文件
add_executable(matrixMul matrixMul.cu helper.cu)

# 启用 CUDA 可分离编译
set_target_properties(matrixMul PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# 添加 CUDA 包含目录
target_include_directories(matrixMul PRIVATE ${CUDA_INCLUDE_DIRS})

# 链接 CUDA 库
target_link_libraries(matrixMul ${CUDA_LIBRARIES})
```

---

### 🐻
