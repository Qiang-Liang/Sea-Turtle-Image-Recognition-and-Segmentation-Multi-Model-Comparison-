## 海龟图像识别与分割（多模型对比）

本仓库包含基于多种深度学习与传统方法的海龟图像识别与语义分割实验，涵盖 UNet、DeepLabV3、YOLOv8（分割/检测）以及传统机器学习基线。所有实验以 Jupyter Notebook 形式组织，便于快速复现与对比。

### 目录结构
- `deeplabv3.ipynb`：使用 DeepLabV3 进行语义分割（可选骨干：ResNet 等）
- `traditional_model.ipynb`：传统方法（如颜色/纹理特征 + 经典分类/分割）
- `unet.ipynb`：UNet 语义分割
- `yolov8.ipynb`：YOLOv8 检测/分割（Ultralytics）

### 项目目标
- 对比多种模型在海龟图像上的识别与分割效果
- 形成从数据准备、训练、评估到可视化的完整实验流程
- 为后续研究与工程落地提供可复现的基线与脚手架

---

## 环境与依赖

建议使用 Conda 创建独立环境（Windows 10/11 已测试）：

```bash
conda create -n turtle-seg python=3.9 -y
conda activate turtle-seg

# 基础科学计算
pip install numpy pandas scikit-learn matplotlib seaborn opencv-python

# 深度学习（根据你的显卡选择 CUDA/CPU 版本）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# 若无 NVIDIA CUDA，请改用 CPU 版本（去掉 --index-url）

# 语义分割模型与工具
pip install segmentation-models-pytorch

# YOLOv8（Ultralytics）
pip install ultralytics

# 笔记本环境
pip install jupyter ipywidgets
```

可选依赖（根据需要）：
- `albumentations`：数据增强
- `tqdm`：训练进度条
- `onnxruntime`：推理部署验证

> 注：不同 GPU/CUDA 版本的 PyTorch 安装方式略有差异，请参考官方说明。

---

## 数据准备

请准备海龟数据集，建议包含：
- 原始图像：`images/`
- 分割标注（掩码，单通道或索引色）：`masks/`
- 训练/验证/测试划分文件或按照目录划分

将数据路径配置为相对路径并在各 Notebook 顶部进行修改。例如：

```python
DATA_DIR = "./data/turtle"
IMAGES_DIR = f"{DATA_DIR}/images"
MASKS_DIR = f"{DATA_DIR}/masks"
```

如果使用 YOLOv8，请准备一个 `dataset.yaml`（示例）：

```yaml
path: ./data/turtle
train: images/train
val: images/val
test: images/test
names:
  0: turtle
```

---

## 快速开始

1) 启动 Jupyter：
```bash
conda activate turtle-seg
jupyter notebook
```

2) 打开相应的 Notebook 并按顺序运行所有单元格：
- `unet.ipynb`：构建数据集 → 定义 UNet → 训练 → 评估 → 可视化
- `deeplabv3.ipynb`：加载 DeepLabV3 → 迁移学习 → 评估 → 可视化
- `yolov8.ipynb`：基于 `ultralytics` 训练/验证（检测或分割任务）
- `traditional_model.ipynb`：特征工程 + 传统模型基线对比

3) 结果产出：
- 训练日志、混淆矩阵、PR/ROC 曲线、mIoU、Dice、mAP 等指标
- 预测可视化（叠加掩码/轮廓）与示例图保存

---

## 复现实验（示例指令）

UNet/DeepLabV3 训练与评估通常在 Notebook 内直接运行；YOLOv8 也支持命令行：

```bash
# 语义分割（YOLOv8-seg 示例）
yolo task=segment mode=train model=yolov8n-seg.pt data=./data/turtle/dataset.yaml epochs=100 imgsz=640

# 验证
yolo task=segment mode=val model=runs/segment/train/weights/best.pt data=./data/turtle/dataset.yaml imgsz=640

# 推理
yolo task=segment mode=predict model=runs/segment/train/weights/best.pt source=./data/turtle/images/test
```

---

## 评估指标

分割：
- 像素精度（PA）、交并比（IoU/mIoU）、Dice/F1、边界 F-score（可选）

检测：
- mAP@0.5、mAP@0.5:0.95、Precision/Recall

传统方法：
- 可比对 Accuracy、Recall、Precision、F1 等

各 Notebook 中均包含指标计算与可视化示例。

---

## 可视化与结果展示

建议在运行结束后，将关键对比图（如不同模型的定性可视化、指标对比表）保存到：
- `results/figures/`
- `results/metrics/`

README 可附上若干代表性结果图（如掩码叠加效果、误检/漏检案例）。

---

## 常见问题（FAQ）

- 显存不足：
  - 减小 `batch_size`/`imgsz`，或使用更小模型（如 `yolov8n-seg`）
  - 使用混合精度（AMP）或裁剪输入图像
- 模型收敛慢：
  - 调整学习率、优化器；增加数据增强；使用预训练权重
- 标注格式不一致：
  - 统一掩码为单通道（类索引），并在数据加载时映射到一致的标签 ID

---

## 贡献

欢迎提交 Issue 或 Pull Request：
- 新的数据加载器/增强策略
- 额外模型（如 UNet++、DeepLabV3+、HRNet、SAM 微调等）
- 训练技巧、部署脚本（ONNX/TensorRT）

---

## 许可证

本项目采用 MIT License，详见 `LICENSE`。

---

## 引用

如本项目或基线对你有帮助，请在论文或报告中引用本仓库。

```
@misc{9517py,
  title        = {9517py: Sea Turtle Image Recognition and Segmentation with Multiple Models},
  author       = {Damocles Sisyphus},
  year         = {2025},
  howpublished = {GitHub repository},
  note         = {https://github.com/<your-account>/9517py}
}
```

---

## 致谢

感谢开源社区与相关数据集/标注工具的贡献者，也感谢 PyTorch、Ultralytics 等优秀开源项目。


