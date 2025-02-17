## 【CUDA-BEVFusion】可视化源文件src/common/visualize.cu （三）

CUDA-BEVFusion中，src/common/visualize.cu 源文件的当前部分代码`class SceneArtistImplement`的作用是图像处理工具，主要用于图像的缩放、翻转和合成操作。

## 一、src/common/visualize.cu 部分源码

```cpp
static __device__ uchar3 load_pixel(const unsigned char* image, int x, int y, float sx, float sy, int width, int stride, int height) {
  // 计算源图像中的坐标
  float src_x = (x + 0.5f) * sx - 0.5f;
  float src_y = (y + 0.5f) * sy - 0.5f;

  // 获取周围4个像素的坐标
  int y_low = floorf(src_y);
  int x_low = floorf(src_x);
  int y_high = limit(y_low + 1, 0, height - 1);
  int x_high = limit(x_low + 1, 0, width - 1);
  y_low = limit(y_low, 0, height - 1);
  x_low = limit(x_low, 0, width - 1);

  // 计算插值权重
  int ly = rint((src_y - y_low) * INTER_RESIZE_COEF_SCALE);
  int lx = rint((src_x - x_low) * INTER_RESIZE_COEF_SCALE);
  int hy = INTER_RESIZE_COEF_SCALE - ly;
  int hx = INTER_RESIZE_COEF_SCALE - lx;

  // 加载周围4个像素的值
  uchar3 rgb[4];
  rgb[0] = *(uchar3*)&image[y_low * stride + x_low * 3];
  rgb[1] = *(uchar3*)&image[y_low * stride + x_high * 3];
  rgb[2] = *(uchar3*)&image[y_high * stride + x_low * 3];
  rgb[3] = *(uchar3*)&image[y_high * stride + x_high * 3];

  // 双线性插值计算目标像素值
  uchar3 output;
  output.x = (((hy * ((hx * rgb[0].x + lx * rgb[1].x) >> 4)) >> 16) + ((ly * ((hx * rgb[2].x + lx * rgb[3].x) >> 4)) >> 16) + 2) >> 2;
  output.y = (((hy * ((hx * rgb[0].y + lx * rgb[1].y) >> 4)) >> 16) + ((ly * ((hx * rgb[2].y + lx * rgb[3].y) >> 4)) >> 16) + 2) >> 2;
  output.z = (((hy * ((hx * rgb[0].z + lx * rgb[1].z) >> 4)) >> 16) + ((ly * ((hx * rgb[2].z + lx * rgb[3].z) >> 4)) >> 16) + 2) >> 2;
  return output;
}

static __global__ void resize_to_kernel(int nx, int ny, int nz, int x0, int y0, float sx, float sy, const unsigned char* img,
                                        int image_width, int image_stride, int image_height, float alpha, unsigned char* output,
                                        int output_stride) {
  int ox = cuda_2d_x; // 当前线程的x坐标
  int oy = cuda_2d_y; // 当前线程的y坐标
  if (ox >= nx || oy >= ny) return; // 如果坐标超出范围，直接返回

  // 加载插值后的像素值
  uchar3 pixel = load_pixel(img, ox, oy, sx, sy, image_width, image_stride, image_height);

  // 获取目标图像中的像素值
  auto& old = *(uchar3*)(output + output_stride * (oy + y0) + (ox + x0) * 3);

  // 混合插值后的像素值和目标图像中的像素值
  old = make_uchar3(limit(pixel.x * alpha + old.x * (1.0f - alpha), 0.0f, 255.0f),
                    limit(pixel.y * alpha + old.y * (1.0f - alpha), 0.0f, 255.0f),
                    limit(pixel.z * alpha + old.z * (1.0f - alpha), 0.0f, 255.0f));
}

static __global__ void flipx_kernel(int nx, int ny, int nz, const unsigned char* img, int image_stride, unsigned char* output,
                                    int output_stride) {
  int ox = cuda_2d_x; // 当前线程的x坐标
  int oy = cuda_2d_y; // 当前线程的y坐标
  if (ox >= nx || oy >= ny) return; // 如果坐标超出范围，直接返回

  // 将输入图像的像素值水平翻转后写入输出图像
  *(uchar3*)&output[oy * output_stride + ox * 3] = *(uchar3*)&img[oy * image_stride + (nx - ox - 1) * 3];
}

class SceneArtistImplement : public SceneArtist {
 public:
  virtual ~SceneArtistImplement() {
    if (cuosd_) cuosd_context_destroy(cuosd_);
  }

  bool init(const SceneArtistParameter& param) {
    this->param_ = param;
    cuosd_ = cuosd_context_create();
    return cuosd_ != nullptr;
  }

  virtual void flipx(const unsigned char* image_device, int image_width, int image_stride, int image_height,
                     unsigned char* output_device, int output_stride, void* stream) override {
    cudaStream_t _stream = static_cast<cudaStream_t>(stream);
    cuda_2d_launch(flipx_kernel, _stream, image_width, image_height, 1, image_device, image_stride, output_device, output_stride);
  }

  virtual void resize_to(const unsigned char* image, int x0, int y0, int x1, int y1, int image_width, int image_stride,
                         int image_height, float alpha, void* stream) override {
    x0 = limit(x0, 0, param_.width - 1);
    y0 = limit(y0, 0, param_.height - 1);
    x1 = limit(x1, 1, param_.width);
    y1 = limit(y1, 1, param_.height);
    int ow = x1 - x0;
    int oh = y1 - y0;
    if (ow <= 0 || oh <= 0) return;

    float sx = image_width / (float)ow;
    float sy = image_height / (float)oh;
    cudaStream_t _stream = static_cast<cudaStream_t>(stream);
    cuda_2d_launch(resize_to_kernel, _stream, ow, oh, 1, x0, y0, sx, sy, image, image_width, image_stride, image_height, alpha,
                   param_.image_device, param_.stride);
  }

 private:
  SceneArtistParameter param_;
  cuOSDContext_t cuosd_ = nullptr;
};

std::shared_ptr<SceneArtist> create_scene_artist(const SceneArtistParameter& param) {
  std::shared_ptr<SceneArtistImplement> instance(new SceneArtistImplement());
  if (!instance->init(param)) {
    printf("Failed to create SceneArtist\n");
    instance.reset();
  }
  return instance;
}
```

## 二、代码解读


---

### **1. 核心功能**
- **图像缩放**：将输入图像缩放到指定大小，并将结果叠加到目标图像上。
- **图像翻转**：将输入图像水平翻转。
- **图像合成**：将处理后的图像叠加到目标图像上，支持透明度控制。

---

### **2. 关键函数解读**

#### **2.1 `load_pixel` 函数**
- **作用**：从输入图像中加载像素值，支持双线性插值。
- **输入**：
  - `image`：输入图像的像素数据。
  - `x, y`：目标图像的像素坐标。
  - `sx, sy`：缩放比例。
  - `width, stride, height`：输入图像的宽度、步长和高度。
- **处理**：
  - 计算源图像中对应像素的坐标（考虑缩放比例）。
  - 使用双线性插值计算目标像素的值。
- **输出**：插值后的像素值（`uchar3`类型，包含RGB三个通道）。

```cpp
static __device__ uchar3 load_pixel(const unsigned char* image, int x, int y, float sx, float sy, int width, int stride, int height) {
  // 计算源图像中的坐标
  float src_x = (x + 0.5f) * sx - 0.5f;
  float src_y = (y + 0.5f) * sy - 0.5f;

  // 获取周围4个像素的坐标
  int y_low = floorf(src_y);
  int x_low = floorf(src_x);
  int y_high = limit(y_low + 1, 0, height - 1);
  int x_high = limit(x_low + 1, 0, width - 1);
  y_low = limit(y_low, 0, height - 1);
  x_low = limit(x_low, 0, width - 1);

  // 计算插值权重
  int ly = rint((src_y - y_low) * INTER_RESIZE_COEF_SCALE);
  int lx = rint((src_x - x_low) * INTER_RESIZE_COEF_SCALE);
  int hy = INTER_RESIZE_COEF_SCALE - ly;
  int hx = INTER_RESIZE_COEF_SCALE - lx;

  // 加载周围4个像素的值
  uchar3 rgb[4];
  rgb[0] = *(uchar3*)&image[y_low * stride + x_low * 3];
  rgb[1] = *(uchar3*)&image[y_low * stride + x_high * 3];
  rgb[2] = *(uchar3*)&image[y_high * stride + x_low * 3];
  rgb[3] = *(uchar3*)&image[y_high * stride + x_high * 3];

  // 双线性插值计算目标像素值
  uchar3 output;
  output.x = (((hy * ((hx * rgb[0].x + lx * rgb[1].x) >> 4)) >> 16) + ((ly * ((hx * rgb[2].x + lx * rgb[3].x) >> 4)) >> 16) + 2) >> 2;
  output.y = (((hy * ((hx * rgb[0].y + lx * rgb[1].y) >> 4)) >> 16) + ((ly * ((hx * rgb[2].y + lx * rgb[3].y) >> 4)) >> 16) + 2) >> 2;
  output.z = (((hy * ((hx * rgb[0].z + lx * rgb[1].z) >> 4)) >> 16) + ((ly * ((hx * rgb[2].z + lx * rgb[3].z) >> 4)) >> 16) + 2) >> 2;
  return output;
}
```

---

#### **2.2 `resize_to_kernel` CUDA核函数**
- **作用**：将输入图像缩放到指定大小，并将结果叠加到目标图像上。
- **输入**：
  - `nx, ny`：目标图像的宽度和高度。
  - `x0, y0`：目标图像在输出图像中的起始坐标。
  - `sx, sy`：缩放比例。
  - `img`：输入图像的像素数据。
  - `image_width, image_stride, image_height`：输入图像的宽度、步长和高度。
  - `alpha`：透明度。
  - `output`：输出图像的像素数据。
  - `output_stride`：输出图像的步长。
- **处理**：
  - 调用`load_pixel`函数加载插值后的像素值。
  - 将插值后的像素值与目标图像中的像素值进行混合（根据透明度`alpha`）。
- **输出**：处理后的图像。

```cpp
static __global__ void resize_to_kernel(int nx, int ny, int nz, int x0, int y0, float sx, float sy, const unsigned char* img,
                                        int image_width, int image_stride, int image_height, float alpha, unsigned char* output,
                                        int output_stride) {
  int ox = cuda_2d_x; // 当前线程的x坐标
  int oy = cuda_2d_y; // 当前线程的y坐标
  if (ox >= nx || oy >= ny) return; // 如果坐标超出范围，直接返回

  // 加载插值后的像素值
  uchar3 pixel = load_pixel(img, ox, oy, sx, sy, image_width, image_stride, image_height);

  // 获取目标图像中的像素值
  auto& old = *(uchar3*)(output + output_stride * (oy + y0) + (ox + x0) * 3);

  // 混合插值后的像素值和目标图像中的像素值
  old = make_uchar3(limit(pixel.x * alpha + old.x * (1.0f - alpha), 0.0f, 255.0f),
                    limit(pixel.y * alpha + old.y * (1.0f - alpha), 0.0f, 255.0f),
                    limit(pixel.z * alpha + old.z * (1.0f - alpha), 0.0f, 255.0f));
}
```

---

#### **2.3 `flipx_kernel` CUDA核函数**
- **作用**：将输入图像水平翻转。
- **输入**：
  - `nx, ny`：图像的宽度和高度。
  - `img`：输入图像的像素数据。
  - `image_stride`：输入图像的步长。
  - `output`：输出图像的像素数据。
  - `output_stride`：输出图像的步长。
- **处理**：
  - 将输入图像的像素值水平翻转后写入输出图像。
- **输出**：水平翻转后的图像。

```cpp
static __global__ void flipx_kernel(int nx, int ny, int nz, const unsigned char* img, int image_stride, unsigned char* output,
                                    int output_stride) {
  int ox = cuda_2d_x; // 当前线程的x坐标
  int oy = cuda_2d_y; // 当前线程的y坐标
  if (ox >= nx || oy >= ny) return; // 如果坐标超出范围，直接返回

  // 将输入图像的像素值水平翻转后写入输出图像
  *(uchar3*)&output[oy * output_stride + ox * 3] = *(uchar3*)&img[oy * image_stride + (nx - ox - 1) * 3];
}
```

---

#### **2.4 `SceneArtistImplement` 类**
- **作用**：实现图像处理功能（缩放、翻转、合成）。
- **核心方法**：
  - **`flipx`**：调用`flipx_kernel`核函数实现图像水平翻转。
  - **`resize_to`**：调用`resize_to_kernel`核函数实现图像缩放和合成。
- **成员变量**：
  - `param_`：图像处理参数（如宽度、高度、步长等）。
  - `cuosd_`：`cuOSD`上下文句柄。

---

#### **2.5 `create_scene_artist` 函数**
- **作用**：创建并初始化`SceneArtistImplement`对象。
- **返回值**：`SceneArtistImplement`对象的共享指针。

---

### **3. 代码的核心作用总结**
1. **图像缩放**：
   - 使用双线性插值将输入图像缩放到指定大小。
2. **图像翻转**：
   - 将输入图像水平翻转。
3. **图像合成**：
   - 将处理后的图像叠加到目标图像上，支持透明度控制。

---

### **4. 应用场景**
- **自动驾驶**：将LiDAR点云或摄像头图像叠加到BEV（鸟瞰图）上。
- **图像处理**：实现图像的缩放、翻转和合成操作。

---
