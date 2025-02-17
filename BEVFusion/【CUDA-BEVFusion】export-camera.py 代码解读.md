## 【CUDA-BEVFusion】qat/export-camera.py 代码解读

【CUDA-BEVFusion】qat/export-camera.py代码的主要功能是将一个名为'bevfusion_ptq.pth'的模型导出为 ONNX 格式，支持 INT8 和 FP16 两种精度。
 - Export INT8 model
```bash
python qat/export-camera.py --ckpt=model/resnet50int8/bevfusion_ptq.pth
```

### `qat/export-camera.py
```python
import sys
import warnings
warnings.filterwarnings("ignore")  # 忽略所有警告信息

import argparse  # 用于解析命令行参数
import os  # 用于处理文件和目录路径

import onnx  # ONNX格式支持库
import torch  # PyTorch深度学习框架
from onnxsim import simplify  # 用于简化ONNX模型
from torchpack.utils.config import configs  # 用于加载配置文件
from mmcv import Config  # MMDetection库中的配置工具
from mmdet3d.models import build_model  # 用于构建3D检测模型
from mmdet3d.utils import recursive_eval  # 用于递归评估配置

from torch import nn  # PyTorch中的神经网络模块
from pytorch_quantization.nn.modules.tensor_quantizer import TensorQuantizer  # 量化工具
import lean.quantize as quantize  # 自定义量化模块

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Export bevfusion model")
    parser.add_argument('--ckpt', type=str, default='qat/ckpt/bevfusion_ptq.pth', help="模型检查点路径")
    parser.add_argument('--fp16', action='store_true', help="是否使用FP16精度")
    args = parser.parse_args()
    return args

class SubclassCameraModule(nn.Module):
    """自定义相机模块，用于处理图像和深度信息"""
    def __init__(self, model):
        super(SubclassCameraModule, self).__init__()
        self.model = model  # 传入的模型

    def forward(self, img, depth):
        """前向传播函数"""
        B, N, C, H, W = img.size()  # 获取输入图像的维度
        img = img.view(B * N, C, H, W)  # 将图像展平

        # 通过模型的相机编码器提取特征
        feat = self.model.encoders.camera.backbone(img)
        feat = self.model.encoders.camera.neck(feat)
        if not isinstance(feat, torch.Tensor):
            feat = feat[0]  # 如果特征不是张量，取第一个元素

        BN, C, H, W = map(int, feat.size())
        feat = feat.view(B, int(BN / B), C, H, W)  # 重新调整特征维度

        def get_cam_feats(self, x, d):
            """获取相机特征和深度信息"""
            B, N, C, fH, fW = map(int, x.shape)
            d = d.view(B * N, *d.shape[2:])
            x = x.view(B * N, C, fH, fW)

            d = self.dtransform(d)  # 深度变换
            x = torch.cat([d, x], dim=1)  # 将深度信息和图像特征拼接
            x = self.depthnet(x)  # 通过深度网络处理

            depth = x[:, : self.D].softmax(dim=1)  # 计算深度权重
            feat = x[:, self.D: (self.D + self.C)].permute(0, 2, 3, 1)  # 调整特征维度
            return feat, depth
        
        return get_cam_feats(self.model.encoders.camera.vtransform, feat, depth)

def main():
    """主函数，用于导出模型为ONNX格式"""
    args = parse_args()  # 解析命令行参数

    model = torch.load(args.ckpt).module  # 加载模型检查点
    suffix = "int8"  # 默认使用INT8量化
    if args.fp16:
        suffix = "fp16"  # 如果使用FP16精度，更改后缀
        quantize.disable_quantization(model).apply()  # 禁用量化
        
    data = torch.load("example-data/example-data.pth")  # 加载示例数据
    img = data["img"].data[0].cuda()  # 将图像数据放到GPU上
    points = [i.cuda() for i in data["points"].data[0]]  # 将点云数据放到GPU上

    camera_model = SubclassCameraModule(model)  # 创建自定义相机模块
    camera_model.cuda().eval()  # 将模型放到GPU上并设置为评估模式
    depth = torch.zeros(len(points), img.shape[1], 1, img.shape[-2], img.shape[-1]).cuda()  # 创建深度张量

    downsample_model = model.encoders.camera.vtransform.downsample  # 获取下采样模型
    downsample_model.cuda().eval()  # 将下采样模型放到GPU上并设置为评估模式
    downsample_in = torch.zeros(1, 80, 360, 360).cuda()  # 创建下采样输入张量

    save_root = f"qat/onnx_{suffix}"  # 设置保存路径
    os.makedirs(save_root, exist_ok=True)  # 创建保存目录

    with torch.no_grad():  # 禁用梯度计算
        camera_backbone_onnx = f"{save_root}/camera.backbone.onnx"  # 相机骨干网络的ONNX保存路径
        camera_vtransform_onnx = f"{save_root}/camera.vtransform.onnx"  # 相机变换网络的ONNX保存路径
        TensorQuantizer.use_fb_fake_quant = True  # 启用伪量化
        torch.onnx.export(
            camera_model,
            (img, depth),
            camera_backbone_onnx,
            input_names=["img", "depth"],
            output_names=["camera_feature", "camera_depth_weights"],
            opset_version=13,
            do_constant_folding=True,
        )  # 导出相机骨干网络为ONNX格式

        onnx_orig = onnx.load(camera_backbone_onnx)  # 加载导出的ONNX模型
        onnx_simp, check = simplify(onnx_orig)  # 简化ONNX模型
        assert check, "Simplified ONNX model could not be validated"  # 检查简化后的模型是否有效
        onnx.save(onnx_simp, camera_backbone_onnx)  # 保存简化后的ONNX模型
        print(f"🚀 The export is completed. ONNX save as {camera_backbone_onnx} 🤗, Have a nice day~")

        torch.onnx.export(
            downsample_model,
            downsample_in,
            camera_vtransform_onnx,
            input_names=["feat_in"],
            output_names=["feat_out"],
            opset_version=13,
            do_constant_folding=True,
        )  # 导出相机变换网络为ONNX格式
        print(f"🚀 The export is completed. ONNX save as {camera_vtransform_onnx} 🤗, Have a nice day~")

if __name__ == "__main__":
    main()  # 运行主函数
```

### 代码功能概述：
- 该代码的主要功能是将一个名为 `bevfusion` 的模型导出为 ONNX 格式，支持 INT8 和 FP16 两种精度。
- 代码首先解析命令行参数，加载模型检查点，并根据参数决定是否使用 FP16 精度。
- 然后，代码加载示例数据，并将模型和数据放到 GPU 上。
- 接着，代码定义了一个自定义的相机模块 `SubclassCameraModule`，用于处理图像和深度信息。
- 最后，代码使用 `torch.onnx.export` 将模型导出为 ONNX 格式，并对导出的模型进行简化和保存。

### 主要模块：
1. **`parse_args`**: 解析命令行参数，包括模型检查点路径和是否使用 FP16 精度。
2. **`SubclassCameraModule`**: 自定义的相机模块，用于处理图像和深度信息。
3. **`main`**: 主函数，负责加载模型、处理数据、导出模型为 ONNX 格式，并进行简化和保存。

### 关键点：
- **量化支持**: 代码支持 INT8 量化，并且可以通过 `--fp16` 参数切换到 FP16 精度。
- **ONNX 导出**: 使用 `torch.onnx.export` 将模型导出为 ONNX 格式，并使用 `onnxsim` 对模型进行简化。
- **GPU 加速**: 模型和数据都被放到 GPU 上进行处理，以提高计算效率。