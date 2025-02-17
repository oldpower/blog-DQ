## 【CUDA-BEVFusion】可视化源文件src/common/visualize.cu （一）


CUDA-BEVFusion中，src/common/visualize.cu 源文件的当前部分代码`class ImageArtistImplement`主要实现了3D边界框（Bounding Box）的投影和可视化功能，使用了CUDA加速的图像绘制库（`cuOSD`）来在图像上绘制3D边界框和相关的文本信息。
                        
## 一、src/common/visualize.cu 部分源码

```cpp
// 定义字体文件路径
#define UseFont "tool/simhei.ttf"
// 定义最大距离（未在代码中使用）
#define MaxDistance 50
// 定义图像缩放插值系数的位数
#define INTER_RESIZE_COEF_BITS 11
// 定义图像缩放插值系数的缩放因子
#define INTER_RESIZE_COEF_SCALE (1 << INTER_RESIZE_COEF_BITS)
// 定义类型转换的位数
#define CAST_BITS (INTER_RESIZE_COEF_BITS << 1)

// 定义Box3DInfo类型，用于存储3D边界框的投影信息
// 包括：投影角点坐标、类别ID、置信度分数、深度值
typedef std::tuple<std::vector<nvtype::Float2>, int, float, float> Box3DInfo;

// 将3D边界框投影到图像平面
std::vector<Box3DInfo> transformation_predictions(const nvtype::Float4* viewport_4x4,
                                                  const std::vector<Prediction>& predictions) {
  if (predictions.empty()) return {}; // 如果预测结果为空，直接返回空列表

  const int number_of_corner = 8; // 3D边界框的角点数量
  std::vector<Box3DInfo> output; // 存储投影结果的列表
  output.reserve(predictions.size()); // 预分配内存

  // 3D边界框的8个角点在局部坐标系中的偏移量
  const nvtype::Float3 offset_of_corners[number_of_corner] = {
      {-1, -1, -1}, {+1, -1, -1}, {+1, +1, -1}, {-1, +1, -1},
      {-1, -1, +1}, {+1, -1, +1}, {+1, +1, +1}, {-1, +1, +1}};

  // 遍历每个预测结果
  for (size_t idx_predict = 0; idx_predict < predictions.size(); ++idx_predict) {
    auto& item = predictions[idx_predict]; // 当前预测结果
    float cos_rotation = cos(item.z_rotation); // 计算旋转角度的余弦值
    float sin_rotation = sin(item.z_rotation); // 计算旋转角度的正弦值

    std::vector<nvtype::Float2> box3d; // 存储当前边界框的投影角点
    box3d.reserve(number_of_corner); // 预分配内存

    // 获取视口变换矩阵的行
    nvtype::Float4 row0 = viewport_4x4[0];
    nvtype::Float4 row1 = viewport_4x4[1];
    nvtype::Float4 row2 = viewport_4x4[2];

    // 计算边界框中心点的深度值
    float zdepth = item.position.x * row2.x + item.position.y * row2.y + item.position.z * row2.z + row2.w;

    // 遍历8个角点，计算其在图像平面上的投影
    for (int idx_corner = 0; idx_corner < number_of_corner; ++idx_corner) {
      auto& offset = offset_of_corners[idx_corner]; // 当前角点的偏移量
      nvtype::Float3 corner; // 存储角点在3D空间中的坐标
      nvtype::Float3 std_corner; // 存储角点在局部坐标系中的坐标

      // 计算角点在局部坐标系中的坐标
      std_corner.x = item.size.w * offset.x * 0.5f;
      std_corner.y = item.size.l * offset.y * 0.5f;
      std_corner.z = item.size.h * offset.z * 0.5f;

      // 计算角点在3D空间中的坐标（考虑旋转）
      corner.x = item.position.x + std_corner.x * cos_rotation + std_corner.y * sin_rotation;
      corner.y = item.position.y + std_corner.x * -sin_rotation + std_corner.y * cos_rotation;
      corner.z = item.position.z + std_corner.z;

      // 将3D坐标投影到图像平面
      float image_x = corner.x * row0.x + corner.y * row0.y + corner.z * row0.z + row0.w;
      float image_y = corner.x * row1.x + corner.y * row1.y + corner.z * row1.z + row1.w;
      float weight = corner.x * row2.x + corner.y * row2.y + corner.z * row2.z + row2.w;

      // 检查投影点是否在图像范围内
      if (image_x <= 0 || image_y <= 0 || weight <= 0) {
        break; // 如果超出范围，跳过当前边界框
      }

      weight = std::max(1e-5f, std::min(1e5f, weight)); // 限制权重范围
      box3d.emplace_back(image_x / weight, image_y / weight); // 存储投影点
    }

    if (box3d.size() != number_of_corner) continue; // 如果角点数量不足，跳过当前边界框

    // 将当前边界框的投影信息存储到输出列表中
    output.emplace_back(box3d, item.id, item.score, zdepth);
  }

  // 按深度值对边界框进行排序（从远到近）
  std::sort(output.begin(), output.end(), [](const Box3DInfo& a, const Box3DInfo& b) {
    return std::get<3>(a) > std::get<3>(b);
  });

  return output; // 返回投影结果
}

// ImageArtistImplement类：实现图像绘制功能
class ImageArtistImplement : public ImageArtist {
 public:
  virtual ~ImageArtistImplement() {
    if (cuosd_) cuosd_context_destroy(cuosd_); // 销毁cuOSD上下文
  }

  // 初始化函数
  bool init(const ImageArtistParameter& param) {
    param_ = param; // 保存参数
    if (param_.classes.empty()) {
      // 如果没有提供类别配置，则使用默认配置
      param_.classes = {
          {"car", 255, 158, 0},        {"truck", 255, 99, 71},   {"construction_vehicle", 233, 150, 70},
          {"bus", 255, 69, 0},         {"trailer", 255, 140, 0}, {"barrier", 112, 128, 144},
          {"motorcycle", 255, 61, 99}, {"bicycle", 220, 20, 60}, {"pedestrian", 0, 0, 230},
          {"traffic_cone", 47, 79, 79}};
    }
    cuosd_ = cuosd_context_create(); // 创建cuOSD上下文
    return cuosd_ != nullptr; // 返回初始化是否成功
  }

  // 绘制预测结果
  virtual void draw_prediction(int camera_index, const std::vector<Prediction>& predictions, bool flipx) override {
    // 将3D边界框投影到图像平面
    auto points = transformation_predictions(this->param_.viewport_nx4x4.data() + camera_index * 4, predictions);
    size_t num = points.size(); // 获取边界框数量

    // 遍历每个边界框
    for (size_t i = 0; i < num; ++i) {
      auto& item = points[i]; // 当前边界框的投影信息
      auto& corners = std::get<0>(item); // 投影角点坐标
      auto label = std::get<1>(item); // 类别ID
      auto score = std::get<2>(item); // 置信度分数

      // 定义边界框的12条边线
      const int idx_of_line[][2] = {
          {0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6}, {6, 7}, {7, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7},
      };

      // 获取类别名称和颜色
      NameAndColor* name_color = &default_name_color_; // 默认颜色
      if (label >= 0 && label < static_cast<int>(param_.classes.size())) {
        name_color = &param_.classes[label]; // 根据类别ID获取颜色
      }

      // 计算边界框的大小
      float size = std::sqrt(std::pow(corners[6].x - corners[0].x, 2) + std::pow(corners[6].y - corners[0].y, 2));
      float minx = param_.image_width; // 边界框的最小x坐标
      float miny = param_.image_height; // 边界框的最小y坐标

      // 绘制边界框的12条边线
      for (size_t ioff = 0; ioff < sizeof(idx_of_line) / sizeof(idx_of_line[0]); ++ioff) {
        auto p0 = corners[idx_of_line[ioff][0]]; // 边线的起点
        auto p1 = corners[idx_of_line[ioff][1]]; // 边线的终点
        if (flipx) {
          // 如果需要水平翻转，则对坐标进行翻转
          p0.x = param_.image_width - p0.x - 1;
          p1.x = param_.image_width - p1.x - 1;
        }
        minx = std::min(minx, std::min(p0.x, p1.x)); // 更新最小x坐标
        miny = std::min(miny, std::min(p0.y, p1.y)); // 更新最小y坐标
        // 绘制边线
        cuosd_draw_line(cuosd_, p0.x, p0.y, p1.x, p1.y, 5, {name_color->r, name_color->g, name_color->b, 255});
      }

      // 计算文本大小
      size = std::max(size * 0.06f, 8.0f);
      // 生成文本内容（类别名称 + 置信度分数）
      auto title = nv::format("%s %.2f", name_color->name.c_str(), score);
      // 绘制文本
      cuosd_draw_text(cuosd_, title.c_str(), size, UseFont, minx, miny, {name_color->r, name_color->g, name_color->b, 255},
                      {255, 255, 255, 200});
    }
  }

  // 将绘制结果应用到图像上
  virtual void apply(unsigned char* image_rgb_device, void* stream) override {
    cuosd_apply(cuosd_, image_rgb_device, nullptr, param_.image_width, param_.image_stride, param_.image_height,
                cuOSDImageFormat::RGB, stream);
  }

 private:
  cuOSDContext_t cuosd_ = nullptr; // cuOSD上下文句柄
  ImageArtistParameter param_; // 图像绘制参数
  NameAndColor default_name_color_{"Unknow", 0, 0, 0}; // 默认类别名称和颜色
};

// 创建ImageArtist对象
std::shared_ptr<ImageArtist> create_image_artist(const ImageArtistParameter& param) {
  std::shared_ptr<ImageArtistImplement> instance(new ImageArtistImplement()); // 创建实例
  if (!instance->init(param)) { // 初始化
    printf("Failed to create ImageArtist\n"); // 如果初始化失败，打印错误信息
    instance.reset(); // 释放实例
  }
  return instance; // 返回实例
}
```

## 二、代码解释

### 1. **宏定义和类型定义**
- **宏定义**：
  - `UseFont`：指定字体文件路径（`simhei.ttf`）。
  - `MaxDistance`：最大距离限制（未在代码中使用）。
  - `INTER_RESIZE_COEF_BITS` 和 `INTER_RESIZE_COEF_SCALE`：用于图像缩放时的插值系数计算。
  - `CAST_BITS`：用于类型转换的位操作。

- **类型定义**：
  - `Box3DInfo`：一个元组类型，用于存储3D边界框的投影信息，包括：
    - `std::vector<nvtype::Float2>`：3D边界框的8个角点在图像平面上的投影坐标。
    - `int`：边界框的类别ID。
    - `float`：边界框的置信度分数。
    - `float`：边界框的深度值（用于排序）。

---

### 2. **`transformation_predictions` 函数**
该函数的作用是将3D边界框的角点投影到图像平面上，并返回投影后的信息。

#### 2.1 **输入参数**
- `viewport_4x4`：4x4的视口变换矩阵，用于将3D点投影到图像平面。
- `predictions`：3D边界框的预测结果列表。

#### 2.2 **实现逻辑**
1. **初始化**：
   - 定义3D边界框的8个角点的偏移量（`offset_of_corners`）。
   - 遍历每个预测结果（`predictions`），计算其8个角点在图像平面上的投影。

2. **投影计算**：
   - 对于每个角点，根据边界框的位置、尺寸和旋转角度，计算其在3D空间中的坐标。
   - 使用视口变换矩阵（`viewport_4x4`）将3D坐标投影到图像平面。
   - 检查投影后的点是否在图像范围内，如果超出范围则跳过该边界框。

3. **存储结果**：
   - 将投影后的角点坐标、类别ID、置信度分数和深度值存储到`Box3DInfo`中。
   - 对所有边界框按深度值进行排序（从远到近），以确保绘制时近处的边界框覆盖远处的边界框。

#### 2.3 **返回值**
- 返回一个`std::vector<Box3DInfo>`，包含所有有效边界框的投影信息。

---

### 3. **`ImageArtistImplement` 类**
该类实现了`ImageArtist`接口，用于在图像上绘制3D边界框和文本信息。

#### 3.1 **成员变量**
- `cuosd_`：`cuOSD`上下文句柄，用于管理绘制操作。
- `param_`：图像绘制参数，包括图像尺寸、类别颜色配置等。
- `default_name_color_`：默认的类别名称和颜色（用于未知类别）。

#### 3.2 **`init` 函数**
- 初始化`ImageArtistImplement`对象。
- 如果没有提供类别配置，则使用默认的类别配置（如`car`、`truck`等）。
- 创建`cuOSD`上下文。

#### 3.3 **`draw_prediction` 函数**
- 在图像上绘制3D边界框和文本信息。

##### 实现逻辑：
1. **投影变换**：
   - 调用`transformation_predictions`函数，将3D边界框投影到图像平面。
2. **绘制边界框**：
   - 遍历每个边界框的投影角点，绘制12条边线（连接8个角点）。
   - 如果`flipx`为`true`，则对投影坐标进行水平翻转。
3. **绘制文本**：
   - 在边界框的左上角绘制类别名称和置信度分数。
   - 使用`cuosd_draw_text`函数绘制文本。
4. **颜色配置**：
   - 根据类别ID从`param_.classes`中获取颜色配置，如果类别ID无效则使用默认颜色。

#### 3.4 **`apply` 函数**
- 将绘制结果应用到图像上。
- 调用`cuosd_apply`函数，将绘制的内容叠加到输入图像上。

---

### 4. **`create_image_artist` 函数**
- 创建并初始化一个`ImageArtistImplement`对象。
- 如果初始化失败，则返回空指针。

---

### 5. **代码总结**
- **功能**：
  - 将3D边界框投影到图像平面，并在图像上绘制边界框和文本信息。
  - 支持多摄像头的图像绘制。
  - 使用CUDA加速的图像绘制库（`cuOSD`）实现高效的绘制操作。

- **核心逻辑**：
  - 通过视口变换矩阵将3D边界框投影到图像平面。
  - 按深度值对边界框进行排序，确保近处的边界框覆盖远处的边界框。
  - 使用`cuOSD`库在图像上绘制边界框和文本。

- **应用场景**：
  - 自动驾驶、机器人感知等领域的3D目标检测结果可视化。



