## 【CUDA-BEVFusion】可视化源文件src/common/visualize.cu （二）

CUDA-BEVFusion中，src/common/visualize.cu 源文件的当前部分代码`class  BEVArtistImplement`主要作用是**将3D点云数据和3D边界框（Bounding Box）投影到2D图像平面，并在图像上进行可视化。

## 一、src/common/visualize.cu 部分源码

```cpp
// 定义half5结构体，用于存储5个half类型的值
typedef struct {
  half val[5];
} half5;

// 限制函数：将value限制在[amin, amax]范围内
template <typename _T>
static __host__ __device__ _T limit(_T value, _T amin, _T amax) {
  return value < amin ? amin : (value > amax ? amax : value);
}

// CUDA核函数：将点云数据投影到图像上并绘制
static __global__ void draw_point_to(unsigned int num, const half5* points, float4* view_port, unsigned char* image,
                                     int image_width, int stride, int image_height) {
  unsigned int idx = cuda_linear_index; // 获取当前线程的全局索引
  if (idx >= num) return; // 如果索引超出点云数量，直接返回

  half5 point = points[idx]; // 获取当前点
  float px = point.val[0]; // 点的x坐标
  float py = point.val[1]; // 点的y坐标
  float pz = point.val[2]; // 点的z坐标
  float reflection = point.val[3]; // 反射率（未使用）
  float indensity = point.val[4]; // 强度（未使用）

  // 获取视口变换矩阵的行
  float4 r0 = view_port[0];
  float4 r1 = view_port[1];
  float4 r2 = view_port[2];

  // 将3D点投影到图像平面
  float x = px * r0.x + py * r0.y + pz * r0.z + r0.w;
  float y = px * r1.x + py * r1.y + pz * r1.z + r1.w;
  float w = px * r2.x + py * r2.y + pz * r2.z + r2.w;

  if (w <= 0) return; // 如果投影点在相机后方，直接返回

  x = x / w; // 归一化x坐标
  y = y / w; // 归一化y坐标

  // 检查投影点是否在图像范围内
  if (x < 0 || x >= image_width || y < 0 || y >= image_height) {
    return;
  }

  int ix = static_cast<int>(x); // 计算图像像素的x坐标
  int iy = static_cast<int>(y); // 计算图像像素的y坐标

  // 计算点的深度透明度
  float alpha = limit((pz + 5.0f) / 8.0f, 0.35f, 1.0f);
  unsigned char gray = limit(alpha * 255, 0.0f, 255.0f); // 将透明度转换为灰度值

  // 将灰度值写入图像
  *(uchar3*)&image[iy * stride + ix * 3] = make_uchar3(gray, gray, gray);
}

// Rodrigues旋转公式：根据旋转角度和旋转轴生成旋转矩阵
static std::vector<nvtype::Float4> rodrigues_rotation(float radian, const std::vector<float>& axis) {
  std::vector<nvtype::Float4> output(4); // 输出4x4旋转矩阵
  memset(&output[0], 0, output.size() * sizeof(nvtype::Float4)); // 初始化矩阵为0

  float nx = axis[0]; // 旋转轴x分量
  float ny = axis[1]; // 旋转轴y分量
  float nz = axis[2]; // 旋转轴z分量
  float cos_val = cos(radian); // 旋转角度的余弦值
  float sin_val = sin(radian); // 旋转角度的正弦值
  output[3].w = 1; // 设置矩阵的右下角为1

  float a = 1 - cos_val; // Rodrigues公式中的系数
  float identity[3][3] = { // 单位矩阵
      {1, 0, 0},
      {0, 1, 0},
      {0, 0, 1},
  };

  float M[3][3] = { // 旋转轴的反对称矩阵
      {0, -nz, ny},
      {nz, 0, -nx},
      {-ny, nx, 0}
  };

  // 计算旋转矩阵
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      ((float*)&output[i])[j] = cos_val * identity[i][j] + a * axis[i] * axis[j] + sin_val * M[i][j];
    }
  }
  return output;
}

// 矩阵乘法：计算两个4x4矩阵的乘积
std::vector<nvtype::Float4> matmul(const std::vector<nvtype::Float4>& a, const std::vector<nvtype::Float4>& b) {
  std::vector<nvtype::Float4> c(a.size()); // 输出矩阵
  memset(&c[0], 0, c.size() * sizeof(nvtype::Float4)); // 初始化矩阵为0

  // 计算矩阵乘积
  for (size_t m = 0; m < a.size(); ++m) {
    auto& ra = a[m]; // 矩阵a的第m行
    auto& rc = c[m]; // 矩阵c的第m行
    for (size_t n = 0; n < b.size(); ++n) {
      for (size_t k = 0; k < 4; ++k) {
        auto& rb = b[k]; // 矩阵b的第k行
        ((float*)&rc)[n] += ((float*)&ra)[k] * ((float*)&rb)[n]; // 累加乘积
      }
    }
  }
  return c;
}

// BEVArtistDrawPointCommand结构体：存储点云绘制命令
struct BEVArtistDrawPointCommand {
  const nvtype::half* points_device; // 点云数据（设备端）
  unsigned int number_of_points; // 点云数量
};

// BEVArtistImplement类：实现BEV（鸟瞰图）绘制功能
class BEVArtistImplement : public BEVArtist {
 public:
  virtual ~BEVArtistImplement() {
    if (transform_matrix_device_) checkRuntime(cudaFree(transform_matrix_device_)); // 释放设备端矩阵内存
    if (cuosd_) cuosd_context_destroy(cuosd_); // 销毁cuOSD上下文
  }

  // 初始化函数
  bool init(const BEVArtistParameter& param) {
    param_ = param; // 保存参数
    if (param_.classes.empty()) {
      // 如果没有提供类别配置，则使用默认配置
      param_.classes = {
          {"car", 255, 158, 0},        {"truck", 255, 99, 71},   {"construction_vehicle", 233, 150, 70},
          {"bus", 255, 69, 0},         {"trailer", 255, 140, 0}, {"barrier", 112, 128, 144},
          {"motorcycle", 255, 61, 99}, {"bicycle", 220, 20, 60}, {"pedestrian", 0, 0, 230},
          {"traffic_cone", 47, 79, 79}};
    }

    // 定义LiDAR到图像的变换矩阵
    std::vector<nvtype::Float4> lidar2image = {
        {param_.norm_size / MaxDistance, 0, 0, param_.cx},
        {0, -param_.norm_size / MaxDistance, 0, param_.cy},
        {0, 0, 0, 1},
        {0, 0, 0, 1}};

    transform_matrix_.resize(4); // 初始化变换矩阵
    memset(&transform_matrix_[0], 0, sizeof(nvtype::Float4) * transform_matrix_.size());

    // 计算旋转矩阵
    auto rotation_x = rodrigues_rotation(param.rotate_x / 180.0f * 3.141592653f, {1, 0, 0}); // 绕x轴旋转
    auto rotation_z = rodrigues_rotation(10.0f / 180.0f * 3.141592653f, {0, 0, 1}); // 绕z轴旋转
    transform_matrix_ = matmul(lidar2image, matmul(rotation_x, rotation_z)); // 计算最终变换矩阵

    // 将变换矩阵拷贝到设备端
    checkRuntime(cudaMalloc(&transform_matrix_device_, sizeof(nvtype::Float4) * transform_matrix_.size()));
    checkRuntime(cudaMemcpy(transform_matrix_device_, transform_matrix_.data(), sizeof(nvtype::Float4) * transform_matrix_.size(),
                            cudaMemcpyHostToDevice));

    cuosd_ = cuosd_context_create(); // 创建cuOSD上下文
    return cuosd_ != nullptr; // 返回初始化是否成功
  }

  // 绘制LiDAR点云
  virtual void draw_lidar_points(const nvtype::half* points_device, unsigned int number_of_points) override {
    draw_point_cmds_.emplace_back(BEVArtistDrawPointCommand{points_device, number_of_points}); // 添加点云绘制命令
  }

  // 绘制自车（ego）的边界框
  virtual void draw_ego() override {
    Prediction ego; // 定义自车的边界框
    ego.position.x = 0;
    ego.position.y = 0;
    ego.position.z = 0;
    ego.size.w = 1.5f;
    ego.size.l = 3.0f;
    ego.size.h = 2.0f;
    ego.z_rotation = 0;

    // 将自车的边界框投影到图像平面
    auto points = transformation_predictions(transform_matrix_.data(), {ego});
    size_t num = points.size();
    for (size_t i = 0; i < num; ++i) {
      auto& item = points[i];
      auto& corners = std::get<0>(item); // 边界框的投影角点
      const int idx_of_line[][2] = {
          {0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6}, {6, 7}, {7, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7},
      };

      // 绘制边界框的12条边线
      for (size_t ioff = 0; ioff < sizeof(idx_of_line) / sizeof(idx_of_line[0]); ++ioff) {
        auto& p0 = corners[idx_of_line[ioff][0]]; // 边线的起点
        auto& p1 = corners[idx_of_line[ioff][1]]; // 边线的终点
        cuosd_draw_line(cuosd_, p0.x, p0.y, p1.x, p1.y, 5, {0, 255, 0, 255}); // 绘制绿色边线
      }
    }
  }

  // 绘制预测结果（3D边界框）
  virtual void draw_prediction(const std::vector<Prediction>& predictions, bool take_title) override {
    // 将3D边界框投影到图像平面
    auto points = transformation_predictions(transform_matrix_.data(), predictions);
    size_t num = points.size();
    for (size_t i = 0; i < num; ++i) {
      auto& item = points[i];
      auto& corners = std::get<0>(item); // 边界框的投影角点
      auto label = std::get<1>(item); // 类别ID
      auto score = std::get<2>(item); // 置信度分数

      const int idx_of_line[][2] = {
          {0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6}, {6, 7}, {7, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7},
      };

      // 获取类别名称和颜色
      NameAndColor* name_color = &default_name_color_; // 默认颜色
      if (label >= 0 && label < static_cast<int>(param_.classes.size())) {
        name_color = &param_.classes[label]; // 根据类别ID获取颜色
      }

      // 绘制边界框的12条边线
      for (size_t ioff = 0; ioff < sizeof(idx_of_line) / sizeof(idx_of_line[0]); ++ioff) {
        auto& p0 = corners[idx_of_line[ioff][0]]; // 边线的起点
        auto& p1 = corners[idx_of_line[ioff][1]]; // 边线的终点
        cuosd_draw_line(cuosd_, p0.x, p0.y, p1.x, p1.y, 5, {name_color->r, name_color->g, name_color->b, 255}); // 绘制边线
      }

      // 绘制类别名称和置信度分数
      if (take_title) {
        float size = std::max(std::sqrt(std::pow(corners[6].x - corners[0].x, 2) + std::pow(corners[6].y - corners[0].y, 2)) * 0.02f, 5.0f);
        auto title = nv::format("%s %.2f", name_color->name.c_str(), score); // 生成文本内容
        cuosd_draw_text(cuosd_, title.c_str(), size, UseFont, corners[0].x, corners[0].y, {name_color->r, name_color->g, name_color->b, 255},
                        {255, 255, 255, 200}); // 绘制文本
      }
    }
  }

  // 将绘制结果应用到图像上
  virtual void apply(unsigned char* image_rgb_device, void* stream) override {
    // 绘制点云
    for (size_t i = 0; i < draw_point_cmds_.size(); ++i) {
      auto& item = draw_point_cmds_[i];
      cuda_linear_launch(draw_point_to, static_cast<cudaStream_t>(stream), item.number_of_points,
                         reinterpret_cast<const half5*>(item.points_device), transform_matrix_device_, image_rgb_device,
                         param_.image_width, param_.image_stride, param_.image_height);
    }
    draw_point_cmds_.clear(); // 清空点云绘制命令

    // 应用cuOSD绘制结果
    cuosd_apply(cuosd_, image_rgb_device, nullptr, param_.image_width, param_.image_stride, param_.image_height,
                cuOSDImageFormat::RGB, stream);
  }

 private:
  std::vector<BEVArtistDrawPointCommand> draw_point_cmds_; // 点云绘制命令列表
  std::vector<nvtype::Float4> transform_matrix_; // 变换矩阵（主机端）
  float4* transform_matrix_device_ = nullptr; // 变换矩阵（设备端）
  cuOSDContext_t cuosd_ = nullptr; // cuOSD上下文
  BEVArtistParameter param_; // BEV绘制参数
  NameAndColor default_name_color_{"Unknow", 0, 0, 0}; // 默认类别名称和颜色
};

// 创建BEVArtist对象
std::shared_ptr<BEVArtist> create_bev_artist(const BEVArtistParameter& param) {
  std::shared_ptr<BEVArtistImplement> instance(new BEVArtistImplement()); // 创建实例
  if (!instance->init(param)) { // 初始化
    printf("Failed to create BEVArtist\n"); // 如果初始化失败，打印错误信息
    instance.reset(); // 释放实例
  }
  return instance; // 返回实例
}
```

---
- **输入**：
  - LiDAR点云数据。
  - 3D边界框（目标检测结果）。
  - 自车的位置和尺寸。
- **输出**：
  - 一张包含点云、3D边界框和自车的可视化图像。

---

假设输入是一帧LiDAR点云和检测到的车辆、行人等目标，代码会生成如下可视化结果：
- **点云**：以灰度点的形式显示在图像上，深度越近的点越亮。
- **3D边界框**：用不同颜色的框表示不同类别的目标（如车辆、行人）。
- **自车**：用绿色框表示自车的位置。
- ---

## 二、代码解释


### **重要部分说明**

1. **`half5` 结构体**：
   - 用于存储点云数据，包含5个`half`类型的值：
     - `val[0]`：点的x坐标。
     - `val[1]`：点的y坐标。
     - `val[2]`：点的z坐标。
     - `val[3]`：反射率（未使用）。
     - `val[4]`：强度（未使用）。

2. **`limit` 函数**：
   - 用于将值限制在指定范围内，确保值在`[amin, amax]`之间。

3. **`draw_point_to` CUDA核函数**：
   - 将点云数据投影到图像平面，并根据深度值绘制灰度点。
   - 核心步骤：
     - 获取当前点的坐标。
     - 使用视口变换矩阵将3D点投影到2D图像平面。
     - 检查投影点是否在图像范围内。
     - 根据深度值计算透明度，并绘制灰度点。

4. **`rodrigues_rotation` 函数**：
   - 根据旋转角度和旋转轴生成旋转矩阵（使用Rodrigues旋转公式）。
   - 核心步骤：
     - 计算旋转矩阵的余弦和正弦值。
     - 使用Rodrigues公式生成旋转矩阵。

5. **`matmul` 函数**：
   - 计算两个4x4矩阵的乘积。
   - 核心步骤：
     - 遍历矩阵的行和列，计算乘积并累加结果。

6. **`BEVArtistImplement` 类**：
   - 实现BEV（鸟瞰图）的绘制功能，包括点云、自车和3D边界框的绘制。
   - 核心功能：
     - **`init`**：初始化BEV绘制参数和变换矩阵。
     - **`draw_lidar_points`**：添加点云绘制命令。
     - **`draw_ego`**：绘制自车的边界框。
     - **`draw_prediction`**：绘制3D边界框和类别名称。
     - **`apply`**：将绘制结果应用到图像上。

7. **`create_bev_artist` 函数**：
   - 创建并初始化`BEVArtistImplement`对象。

---

### **步骤顺序**

1. **初始化**：
   - 调用`create_bev_artist`函数创建`BEVArtistImplement`对象。
   - 在`init`函数中：
     - 设置默认类别配置。
     - 计算LiDAR到图像的变换矩阵。
     - 将变换矩阵拷贝到设备端。
     - 创建`cuOSD`上下文。

2. **绘制点云**：
   - 调用`draw_lidar_points`函数，将点云数据添加到绘制命令列表中。

3. **绘制自车**：
   - 调用`draw_ego`函数，绘制自车的边界框。

4. **绘制3D边界框**：
   - 调用`draw_prediction`函数，绘制3D边界框和类别名称。

5. **应用绘制结果**：
   - 调用`apply`函数：
     - 使用CUDA核函数`draw_point_to`绘制点云。
     - 使用`cuosd_apply`将绘制结果应用到图像上。

---

### **核心流程总结**

1. **初始化**：
   - 创建BEV绘制对象，初始化参数和变换矩阵。

2. **数据准备**：
   - 添加点云数据和3D边界框数据。

3. **绘制**：
   - 绘制点云、自车和3D边界框。

4. **结果应用**：
   - 将绘制结果叠加到图像上，并保存或显示。

---

### **关键点**
- **变换矩阵**：用于将3D点云和边界框投影到2D图像平面。
- **CUDA加速**：点云绘制使用CUDA核函数实现高效计算。
- **cuOSD库**：用于在图像上绘制边界框和文本。

---

如果有其他问题或需要进一步解释，请随时告诉我！😊