# 【CUDA-BEVFusion】模型量化qat/ptq.py代码说明

`qat/ptq.py`代码实现了一个PTQ（Post Training Quantization）的过程，主要用于对深度学习模型进行量化操作。

---

### 1. **代码结构**
代码的主要功能包括：
- 加载配置文件、模型和数据集。
- 对模型进行量化操作。
- 对量化后的模型进行校准（Calibration）。
- 保存量化后的模型。
- ---
### 2、qat/ptq.py

```python
# 导入必要的库
import sys
import argparse
import copy
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn

# 导入量化工具和自定义函数
import lean.quantize as quantize
import lean.funcs as funcs
from lean.train import qat_train

# 导入配置管理和模型构建工具
from mmcv import Config
from torchpack.environ import auto_set_run_dir, set_run_dir
from torchpack.utils.config import configs

# 导入数据集和模型构建工具
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model
from mmdet3d.utils import get_root_logger, convert_sync_batchnorm, recursive_eval

# 导入模型加载和保存工具
from mmcv.runner import load_checkpoint, save_checkpoint
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.cnn import resnet
from mmcv.cnn.utils.fuse_conv_bn import _fuse_conv_bn

# 导入量化卷积模块
from pytorch_quantization.nn.modules.quant_conv import QuantConv2d, QuantConvTranspose2d

# 定义函数：融合卷积层和批归一化层
def fuse_conv_bn(module):
    """
    将卷积层（Conv2d 或 QuantConv2d）和其后的批归一化层（BatchNorm）融合。
    融合后的卷积层会替代原来的卷积层，批归一化层会被替换为 Identity（即不做任何操作）。
    这种融合可以减少模型的计算量，提升推理速度。
    """
    last_conv = None
    last_conv_name = None

    for name, child in module.named_children():
        if isinstance(child, (nn.modules.batchnorm._BatchNorm, nn.SyncBatchNorm)):
            if last_conv is None:  # 只融合在卷积层之后的批归一化层
                continue
            fused_conv = _fuse_conv_bn(last_conv, child)  # 融合卷积层和批归一化层
            module._modules[last_conv_name] = fused_conv  # 替换原来的卷积层
            module._modules[name] = nn.Identity()  # 将批归一化层替换为 Identity
            last_conv = None
        elif isinstance(child, QuantConv2d) or isinstance(child, nn.Conv2d):
            last_conv = child  # 记录当前的卷积层
            last_conv_name = name
        else:
            fuse_conv_bn(child)  # 递归处理子模块
    return module

# 定义函数：加载模型
def load_model(cfg, checkpoint_path=None):
    """
    根据配置文件构建模型，并加载预训练权重（如果提供了 checkpoint_path）。
    """
    model = build_model(cfg.model)  # 构建模型
    if checkpoint_path is not None:
        checkpoint = load_checkpoint(model, checkpoint_path, map_location="cpu")  # 加载预训练权重
    return model

# 定义函数：量化模型
def quantize_net(model):
    """
    对模型的各个部分进行量化操作：
    - 量化激光雷达分支（lidar.backbone）。
    - 量化相机分支（camera）。
    - 量化融合模块（fuser）。
    - 量化解码器（decoder）。
    - 对激光雷达分支进行层融合（layer_fusion_bn）。
    """
    quantize.quantize_encoders_lidar_branch(model.encoders.lidar.backbone)  # 量化激光雷达分支
    quantize.quantize_encoders_camera_branch(model.encoders.camera)  # 量化相机分支
    quantize.replace_to_quantization_module(model.fuser)  # 量化融合模块
    quantize.quantize_decoder(model.decoder)  # 量化解码器
    model.encoders.lidar.backbone = funcs.layer_fusion_bn(model.encoders.lidar.backbone)  # 层融合
    return model

# 主函数
def main():
    """
    主函数，执行以下流程：
    1. 初始化量化工具。
    2. 加载配置文件和模型。
    3. 对模型进行量化操作。
    4. 对量化后的模型进行校准。
    5. 保存量化后的模型。
    """
    quantize.initialize()  # 初始化量化工具

    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", metavar="FILE", default="bevfusion/configs/nuscenes/det/transfusion/secfpn/camera+lidar/resnet50/convfuser.yaml", help="config file")
    parser.add_argument("--ckpt", default="model/resnet50/bevfusion-det.pth", help="the checkpoint file to resume from")
    parser.add_argument("--calibrate_batch", type=int, default=300, help="calibrate batch")
    args = parser.parse_args()

    args.ptq_only = True  # 设置为仅执行 PTQ（Post Training Quantization）

    # 加载配置文件
    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)

    # 设置保存路径
    save_path = 'qat/ckpt/bevfusion_ptq.pth'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 设置随机种子
    if cfg.seed is not None:
        print(f"Set random seed to {cfg.seed}, deterministic mode: {cfg.deterministic}")
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if cfg.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # 构建数据集和数据加载器
    dataset_train = build_dataset(cfg.data.train)
    dataset_test = build_dataset(cfg.data.test)
    print('train nums:{} val nums:{}'.format(len(dataset_train), len(dataset_test)))
    distributed = False
    data_loader_train = build_dataloader(
        dataset_train,
        samples_per_gpu=1,  # 每个 GPU 的样本数
        workers_per_gpu=1,  # 每个 GPU 的工作线程数
        dist=distributed,
        seed=cfg.seed,
    )
    print('DataLoad Info:', data_loader_train.batch_size, data_loader_train.num_workers)

    # 创建模型并加载预训练权重
    model = load_model(cfg, checkpoint_path=args.ckpt)
    model = quantize_net(model)  # 量化模型
    model = fuse_conv_bn(model)  # 融合卷积层和批归一化层
    model = MMDataParallel(model, device_ids=[0])  # 包装为多 GPU 模型
    model.eval()  # 设置为评估模式

    # 校准模型
    print("🔥 start calibrate 🔥")
    quantize.set_quantizer_fast(model)  # 设置量化器为快速模式
    quantize.calibrate_model(model, data_loader_train, 0, None, args.calibrate_batch)  # 使用训练数据进行校准

    # 禁用部分层的量化
    quantize.disable_quantization(model.module.encoders.lidar.backbone.conv_input).apply()
    quantize.disable_quantization(model.module.decoder.neck.deblocks[0][0]).apply()
    quantize.print_quantizer_status(model)  # 打印量化器状态

    # 保存量化后的模型
    print(f"Done due to ptq only! Save checkpoint to {save_path} 🤗")
    model.module.encoders.lidar.backbone = funcs.fuse_relu_only(model.module.encoders.lidar.backbone)
    torch.save(model, save_path)
    return

# 程序入口
if __name__ == "__main__":
    main()
```
---


### 3. **代码运行流程**
1. 加载配置文件和模型。
2. 对模型进行量化操作。
3. 对量化后的模型进行校准。
4. 保存量化后的模型。

---

### 4. **关键点**
- **量化**：将浮点数模型转换为低精度（如INT8）模型，以减少计算和存储开销。
- **校准**：通过少量数据调整量化参数，以减少量化带来的精度损失。
- **融合**：将卷积层和批归一化层融合，提升推理速度。

---

### 5. **适用场景**
- 该代码适用于需要对深度学习模型进行量化压缩的场景，尤其是资源受限的部署环境（如嵌入式设备、移动端等）。
- 通过PTQ，可以在不重新训练模型的情况下，快速获得量化模型。