# TensorRT INT8 量化YOLO模型—— python

TensorRT 是 NVIDIA 提供的高性能深度学习推理库，支持 INT8 量化以加速模型推理。以下是使用 TensorRT 对 YOLO 模型（如 YOLOv5、YOLOv8 或 YOLOv11）进行 INT8 量化的具体步骤：

---

## **TensorRT INT8 量化的具体步骤**

### **1. 准备工作**
- **环境要求**：
  - 安装 CUDA 和 cuDNN。
  - 安装 TensorRT（建议使用与 CUDA 版本匹配的 TensorRT 版本）。
  - 安装 PyTorch 和 ONNX（用于模型转换）。
- **依赖安装**：
  ```bash
  pip install torch torchvision onnx onnxruntime tensorrt
  ```

- **校准数据集**：
  - 准备一个小型校准数据集（通常 100-1000 张图片即可），用于 TensorRT 的 INT8 校准。

---

### **2. 将 YOLO 模型导出为 ONNX 格式**
TensorRT 支持从 ONNX 格式的模型进行量化。首先需要将 YOLO 模型导出为 ONNX 格式。

#### 示例代码（以 YOLOv5 为例）：
```python
import torch
from models.experimental import attempt_load

# 加载 YOLO 模型
model = attempt_load('yolov5s.pt', map_location=torch.device('cpu'))

# 设置模型为推理模式
model.eval()

# 定义输入张量（batch_size, channels, height, width）
dummy_input = torch.randn(1, 3, 640, 640)

# 导出为 ONNX 格式
torch.onnx.export(
    model,                      # 模型
    dummy_input,                # 输入张量
    "yolov5s.onnx",             # 输出文件名
    opset_version=11,           # ONNX 版本
    input_names=["images"],     # 输入节点名称
    output_names=["output"],    # 输出节点名称
    dynamic_axes={"images": {0: "batch_size"}, "output": {0: "batch_size"}}  # 支持动态 batch size
)
```

---

### **3. 使用 TensorRT 进行 INT8 量化**
TensorRT 提供了 `trtexec` 工具和 Python API 来执行 INT8 量化。以下是使用 Python API 的步骤：

#### (1) 安装 TensorRT Python 包
确保已安装 TensorRT 的 Python 包：
```bash
pip install nvidia-tensorrt
```

#### (2) 编写 INT8 量化代码
以下是一个完整的 INT8 量化示例：

```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

# 1. 创建 TensorRT 日志记录器
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# 2. 创建 TensorRT 构建器
builder = trt.Builder(TRT_LOGGER)

# 3. 创建网络定义
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

# 4. 解析 ONNX 模型
parser = trt.OnnxParser(network, TRT_LOGGER)
with open("yolov5s.onnx", "rb") as model:
    if not parser.parse(model.read()):
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        raise ValueError("Failed to parse ONNX file")

# 5. 配置 INT8 量化
config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.INT8)

# 6. 设置校准数据集
class Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calibration_data, batch_size=1):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.calibration_data = calibration_data
        self.batch_size = batch_size
        self.current_index = 0
        self.device_input = cuda.mem_alloc(self.calibration_data[0].nbytes)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index < len(self.calibration_data):
            batch = self.calibration_data[self.current_index]
            cuda.memcpy_htod(self.device_input, batch)
            self.current_index += 1
            return [int(self.device_input)]
        else:
            return None

    def read_calibration_cache(self):
        return None

    def write_calibration_cache(self, cache):
        pass

# 假设 calibration_data 是一个包含校准数据的列表（numpy 数组）
calibration_data = [np.random.randn(1, 3, 640, 640).astype(np.float32) for _ in range(100)]
calibrator = Calibrator(calibration_data)
config.int8_calibrator = calibrator

# 7. 构建引擎
engine = builder.build_engine(network, config)

# 8. 保存引擎
with open("yolov5s_int8.engine", "wb") as f:
    f.write(engine.serialize())
```

---

### **4. 使用量化后的模型进行推理**
量化后的模型可以加载到 TensorRT 中进行推理。

#### 示例代码：
```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

# 加载引擎
with open("yolov5s_int8.engine", "rb") as f:
    engine_data = f.read()
runtime = trt.Runtime(TRT_LOGGER)
engine = runtime.deserialize_cuda_engine(engine_data)

# 创建执行上下文
context = engine.create_execution_context()

# 分配输入输出内存
inputs, outputs, bindings, stream = [], [], [], cuda.Stream()
for binding in engine:
    size = trt.volume(engine.get_binding_shape(binding)) * engine.get_binding_dtype(binding).itemsize
    device_mem = cuda.mem_alloc(size)
    bindings.append(int(device_mem))
    if engine.binding_is_input(binding):
        inputs.append(device_mem)
    else:
        outputs.append(device_mem)

# 准备输入数据
input_data = np.random.randn(1, 3, 640, 640).astype(np.float32)
cuda.memcpy_htod_async(inputs[0], input_data, stream)

# 执行推理
context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

# 获取输出数据
output_data = np.empty(engine.get_binding_shape(1), dtype=np.float32)
cuda.memcpy_dtoh_async(output_data, outputs[0], stream)
stream.synchronize()

print("推理结果：", output_data)
```

---

### **5. 注意事项**
- **校准数据集**：校准数据集应尽量覆盖实际应用场景中的数据分布。
- **精度验证**：量化后需验证模型的精度是否满足要求。
- **硬件支持**：确保 GPU 支持 INT8 计算（如 NVIDIA Turing 或 Ampere 架构）。

