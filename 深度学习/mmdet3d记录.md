

CUDA 版本是 11.4，需要确保安装与之兼容的 PyTorch 版本，并且确认所有依赖项都正确配置。以下是详细的步骤在 Conda 环境中安装支持 CUDA 11.4 的 PyTorch 和 MMDetection3D。

### 步骤 1: 创建并激活虚拟环境

如果你还没有创建一个专门用于 MMDetection3D 的 Conda 环境，可以按以下命令创建并激活它：

```shell
conda create --name mmdet3d python=3.8 -y
conda activate mmdet3d
```

这里我们选择了 Python 3.8 版本作为示例，但请根据 MMDetection3D 的官方文档确认支持的 Python 版本。

### 步骤 2: 安装支持 CUDA 11.4 的 PyTorch

访问 [PyTorch 官方网站](https://pytorch.org/get-started/locally/) 并选择适合你环境的选项（CUDA 11.4）。这将生成一个适合你需求的安装命令。对于 Conda 用户，你可以使用如下命令：

```shell
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

注意：虽然你需要 CUDA 11.4 支持，但在 PyTorch 官网给出的选项中可能没有直接匹配 CUDA 11.4 的版本。通常情况下，选择接近的 CUDA 版本（如 11.3）是可以兼容的。如果确实需要严格匹配 CUDA 11.4，可以尝试通过 pip 安装特定版本：

```shell
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu114
```

### 步骤 3: 安装 MMCV

MMCV 是 MMDetection3D 所依赖的一个基础库。由于不同的 PyTorch 和 CUDA 版本组合对 MMCV 有特定的要求，因此需要从 MMCV 的 [官方文档](https://mmdetection3d.readthedocs.io/zh-cn/latest/get_started.html) 查找适合你环境的安装指令。
```bash
# 克隆 MMCV 仓库
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv

# 切换到适合你需求的分支或标签
git checkout v1.4.4  # 示例版本，请根据需要调整

# 编译并安装
MMCV_WITH_OPS=1 pip install -e .
```


---

### 🚀 Demo

**MVX-Net**
```bash
python demo/multi_modality_demo.py demo/data/kitti/kitti_000008.bin demo/data/kitti/kitti_000008.png demo/data/kitti/kitti_000008_infos.pkl configs/mvxnet/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class.py checkpoints/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class_20210831_060805-83442923.pth --out-dir output/demo_test --show
```

**VoteNet**
```bash
python demo/pcd_demo.py ./demo/data/scannet/scene0000_00.npy  ./configs/votenet/votenet_8xb8_scannet-3d.py  /mnt/d/wsl_workspace/votenet_8x8_scannet-3d-18class_20210823_234503-cf8134fa.pth
```