## 【CUDA-BEVFusion】src/main.cpp—— visualize函数说明

CUDA-BEVFusion中，src/main.cpp—— visualize函数的主要功能是将激光雷达（LiDAR）点云数据、3D边界框（Bounding Box）以及摄像头图像进行可视化，并将结果保存为一张图片。代码使用了CUDA进行加速，并且涉及到多个模块的协同工作，包括LiDAR点云的可视化、3D边界框的绘制、摄像头图像的投影等。

### 1. **函数参数**
   - `bboxes`: 3D边界框的列表，类型为`std::vector<bevfusion::head::transbbox::BoundingBox>`。
   - `lidar_points`: LiDAR点云数据，类型为`nv::Tensor`。
   - `images`: 摄像头图像数据，类型为`std::vector<unsigned char*>`。
   - `lidar2image`: LiDAR到图像的变换矩阵，类型为`nv::Tensor`。
   - `save_path`: 可视化结果的保存路径，类型为`std::string`。
   - `stream`: CUDA流，用于异步操作。
   - --
   - `padding`、`lidar_size`、`content_width`、`content_height`等变量用于定义场景的尺寸和布局。
   - `scene_artist_param`用于设置场景画布的参数，包括宽度、高度、步长等。
   - `scene_device_image`是一个3通道的图像张量，用于存储最终的场景图像。
   - --
   - 使用`scene_artist_param`创建一个场景画布`scene`，并将`scene_device_image`作为画布的背景图像。
   - `bev_artist_param`用于设置BEV可视化的参数，包括图像尺寸、旋转角度、归一化尺寸等。
   - `bev_visualizer`用于绘制LiDAR点云、3D边界框以及自车（ego）的位置。

### 2. **visualize代码**
```cpp
static void visualize(const std::vector<bevfusion::head::transbbox::BoundingBox>& bboxes, const nv::Tensor& lidar_points,
                      const std::vector<unsigned char*> images, const nv::Tensor& lidar2image, const std::string& save_path,
                      cudaStream_t stream) {
  // 将检测框数据拷贝到 predictions 向量中
  std::vector<nv::Prediction> predictions(bboxes.size());
  memcpy(predictions.data(), bboxes.data(), bboxes.size() * sizeof(nv::Prediction));

  // 定义可视化图像的尺寸和布局参数
  int padding = 300;  // 图像边缘的填充大小
  int lidar_size = 1024;  // 点云图像的宽度
  int content_width = lidar_size + padding * 3;  // 最终图像的宽度
  int content_height = 1080;  // 最终图像的高度

  // 初始化 SceneArtist 的参数
  nv::SceneArtistParameter scene_artist_param;
  scene_artist_param.width = content_width;  // 图像宽度
  scene_artist_param.height = content_height;  // 图像高度
  scene_artist_param.stride = scene_artist_param.width * 3;  // 图像步长（每行的字节数）

  // 创建设备上的图像张量，并初始化为全零（黑色）
  nv::Tensor scene_device_image(std::vector<int>{scene_artist_param.height, scene_artist_param.width, 3}, nv::DataType::UInt8);
  scene_device_image.memset(0x00, stream);

  // 将设备图像指针配置到 SceneArtistParameter 中
  scene_artist_param.image_device = scene_device_image.ptr<unsigned char>();
  auto scene = nv::create_scene_artist(scene_artist_param);  // 创建 SceneArtist 实例

  // 初始化 BEVArtist 的参数
  nv::BEVArtistParameter bev_artist_param;
  bev_artist_param.image_width = content_width;  // 图像宽度
  bev_artist_param.image_height = content_height;  // 图像高度
  bev_artist_param.rotate_x = 70.0f;  // 点云绕 x 轴的旋转角度
  bev_artist_param.norm_size = lidar_size * 0.5f;  // 点云的归一化大小
  bev_artist_param.cx = content_width * 0.5f;  // 图像中心点的 x 坐标
  bev_artist_param.cy = content_height * 0.5f;  // 图像中心点的 y 坐标
  bev_artist_param.image_stride = scene_artist_param.stride;  // 图像步长

  // 将点云数据转移到设备上
  auto points = lidar_points.to_device();
  auto bev_visualizer = nv::create_bev_artist(bev_artist_param);  // 创建 BEVArtist 实例

  // 渲染点云、预测框和自车位置
  bev_visualizer->draw_lidar_points(points.ptr<nvtype::half>(), points.size(0));  // 绘制点云
  bev_visualizer->draw_prediction(predictions, false);  // 绘制预测框
  bev_visualizer->draw_ego();  // 绘制自车位置
  bev_visualizer->apply(scene_device_image.ptr<unsigned char>(), stream);  // 将渲染结果应用到图像上

  // 初始化 ImageArtist 的参数
  nv::ImageArtistParameter image_artist_param;
  image_artist_param.num_camera = images.size();  // 相机数量
  image_artist_param.image_width = 1600;  // 相机图像的宽度
  image_artist_param.image_height = 900;  // 相机图像的高度
  image_artist_param.image_stride = image_artist_param.image_width * 3;  // 相机图像的步长
  image_artist_param.viewport_nx4x4.resize(images.size() * 4 * 4);  // 视口矩阵的大小
  memcpy(image_artist_param.viewport_nx4x4.data(), lidar2image.ptr<float>(),
         sizeof(float) * image_artist_param.viewport_nx4x4.size());  // 拷贝视口矩阵数据

  // 定义相机图像的布局参数
  int gap = 0;  // 相机图像之间的间隔
  int camera_width = 500;  // 相机图像的宽度
  int camera_height = static_cast<float>(camera_width / (float)image_artist_param.image_width * image_artist_param.image_height);  // 相机图像的高度
  int offset_cameras[][3] = {
      {-camera_width / 2, -content_height / 2 + gap, 0},  // 相机 1 的位置和翻转标志
      {content_width / 2 - camera_width - gap, -content_height / 2 + camera_height / 2, 0},  // 相机 2 的位置和翻转标志
      {-content_width / 2 + gap, -content_height / 2 + camera_height / 2, 0},  // 相机 3 的位置和翻转标志
      {-camera_width / 2, +content_height / 2 - camera_height - gap, 1},  // 相机 4 的位置和翻转标志
      {-content_width / 2 + gap, +content_height / 2 - camera_height - camera_height / 2, 0},  // 相机 5 的位置和翻转标志
      {content_width / 2 - camera_width - gap, +content_height / 2 - camera_height - camera_height / 2, 1}};  // 相机 6 的位置和翻转标志

  // 创建 ImageArtist 实例
  auto visualizer = nv::create_image_artist(image_artist_param);
  for (size_t icamera = 0; icamera < images.size(); ++icamera) {
    // 计算相机图像在最终图像中的位置
    int ox = offset_cameras[icamera][0] + content_width / 2;
    int oy = offset_cameras[icamera][1] + content_height / 2;
    bool xflip = static_cast<bool>(offset_cameras[icamera][2]);  // 是否水平翻转

    // 在相机图像上绘制预测框
    visualizer->draw_prediction(icamera, predictions, xflip);

    // 将相机图像数据拷贝到设备上
    nv::Tensor device_image(std::vector<int>{900, 1600, 3}, nv::DataType::UInt8);
    device_image.copy_from_host(images[icamera], stream);

    // 如果需要水平翻转，则对图像进行翻转
    if (xflip) {
      auto clone = device_image.clone(stream);
      scene->flipx(clone.ptr<unsigned char>(), clone.size(1), clone.size(1) * 3, clone.size(0), device_image.ptr<unsigned char>(),
                   device_image.size(1) * 3, stream);
      checkRuntime(cudaStreamSynchronize(stream));  // 同步 CUDA 流
    }

    // 将绘制结果应用到相机图像上
    visualizer->apply(device_image.ptr<unsigned char>(), stream);

    // 将相机图像调整大小并放置到最终图像中
    scene->resize_to(device_image.ptr<unsigned char>(), ox, oy, ox + camera_width, oy + camera_height, device_image.size(1),
                     device_image.size(1) * 3, device_image.size(0), 0.8f, stream);
    checkRuntime(cudaStreamSynchronize(stream));  // 同步 CUDA 流
  }

  // 保存最终图像到指定路径
  printf("Save to %s\n", save_path.c_str());
  stbi_write_jpg(save_path.c_str(), scene_device_image.size(1), scene_device_image.size(0), 3,
                 scene_device_image.to_host(stream).ptr(), 100);  // 保存为 JPG 文件
}
```

### 3. 代码功能总结
`visualize` 函数的主要功能是将点云、检测框和相机图像拼接成一张完整的可视化图像，并保存为 JPG 文件。具体步骤包括：
1. **处理检测框数据**：将检测框数据拷贝到 `predictions` 向量中。
2. **初始化场景图像**：创建并初始化设备上的场景图像。
3. **渲染点云和检测框**：使用 `BEVArtist` 渲染点云、检测框和自车位置。
4. **处理相机图像**：将相机图像数据拷贝到设备上，绘制检测框，并根据需要翻转图像。
5. **拼接图像**：将相机图像调整大小并放置到场景图像中。
6. **保存结果**：将最终的可视化图像保存为 JPG 文件。