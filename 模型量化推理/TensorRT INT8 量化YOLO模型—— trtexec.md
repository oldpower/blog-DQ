# TensorRT INT8 量化YOLO模型—— trtexec

TensorRT 提供了 `trtexec` 工具，可以方便地将模型转换为 TensorRT 引擎，并支持 INT8 量化。`trtexec` 是一个命令行工具，适用于快速测试和部署模型，尤其适合对 ONNX 或 UFF 格式的模型进行量化和优化。

以下是使用 `trtexec` 进行 INT8 量化的具体步骤：

---

## **1. 准备工作**
- **安装 TensorRT**：
  - 确保已安装 TensorRT，并且 `trtexec` 工具可用。`trtexec` 通常位于 TensorRT 安装目录的 `bin` 文件夹中。
  - 将 `trtexec` 添加到系统环境变量中，或者直接使用其完整路径。

- **准备校准数据集**：
  - 准备一个小型校准数据集（通常 100-1000 张图片），用于 INT8 量化校准。
  - 校准数据集需要以 TensorRT 支持的格式存储（如 `.npy` 文件或图像文件）。

- **导出模型为 ONNX 格式**：
  - 如果模型是 PyTorch 或 TensorFlow 格式，需要先将其导出为 ONNX 格式。
  - 以 YOLOv5 为例，导出 ONNX 模型的命令如下：
    ```bash
    python export.py --weights yolov5s.pt --include onnx --img 640 --batch 1
    ```

---

## **2. 使用 trtexec 进行 INT8 量化**

### **基本命令**
```bash
trtexec --onnx=yolov5s.onnx --int8 --calib=<校准数据集路径> --saveEngine=yolov5s_int8.engine
```

### **参数说明**
- `--onnx`：指定输入的 ONNX 模型文件。
- `--int8`：启用 INT8 量化模式。
- `--calib`：指定校准数据集的路径（可以是 `.npy` 文件或图像文件夹）。
- `--saveEngine`：指定输出的 TensorRT 引擎文件。
- `--workspace`：设置 GPU 工作空间大小（默认值为 16 MB，可以根据需要调整）。
- `--batch`：设置批量大小（默认为 1）。
- `--verbose`：启用详细日志输出。

---

### **示例：使用图像文件夹作为校准数据集**
假设校准数据集是一个包含图像的文件夹（如 `calib_images/`），可以使用以下命令进行量化：
```bash
trtexec --onnx=yolov5s.onnx --int8 --calib=calib_images/ --saveEngine=yolov5s_int8.engine
```

### **示例：使用 .npy 文件作为校准数据集**
如果校准数据集是 `.npy` 文件（如 `calib_data.npy`），可以使用以下命令：
```bash
trtexec --onnx=yolov5s.onnx --int8 --calib=calib_data.npy --saveEngine=yolov5s_int8.engine
```

---

## **3. 校准数据集的准备**
`trtexec` 支持两种校准数据集格式：
1. **图像文件夹**：
   - 将校准图像存储在一个文件夹中（如 `calib_images/`）。
   - 图像会被自动加载并预处理为模型输入格式。

2. **.npy 文件**：
   - 将校准数据保存为 `.npy` 文件。
   - 文件内容应为 NumPy 数组，形状为 `(N, C, H, W)`，其中：
     - `N` 是样本数量。
     - `C` 是通道数。
     - `H` 是高度。
     - `W` 是宽度。

---

## **4. 验证量化后的模型**
量化完成后，可以使用 `trtexec` 验证量化后的模型性能：
```bash
trtexec --loadEngine=yolov5s_int8.engine
```
- `--loadEngine`：加载量化后的 TensorRT 引擎文件。
- 运行后会输出模型的推理时间、吞吐量等性能指标。

---

## **5. 注意事项**
- **校准数据集**：
  - 校准数据集应尽量覆盖实际应用场景中的数据分布。
  - 数据集大小通常为 100-1000 个样本。

- **精度验证**：
  - 量化后需验证模型的精度是否满足要求。
  - 可以使用原始模型和量化模型在验证数据集上进行对比测试。

- **硬件支持**：
  - 确保 GPU 支持 INT8 计算（如 NVIDIA Turing 或 Ampere 架构）。

---

## **6. 总结**
`trtexec` 是 TensorRT 提供的一个强大工具，可以快速完成模型的 INT8 量化和优化。通过简单的命令行操作，您可以将 YOLO 等模型转换为高效的 TensorRT 引擎，并部署到 NVIDIA GPU 上。