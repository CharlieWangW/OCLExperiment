# OCL + CLIP Object Inference Experiment

本项目实现了一个基于 OCL 数据集和 OpenAI CLIP 的物体概念推理实验。当前版本主要用于本地小规模验证：先确认 `Dataset`、`DataLoader`、CLIP GPU 推理、category top-k accuracy、attribute mAP 和 affordance mAP 的完整流程可以跑通，再切换到服务器完整 OCL 数据集做正式实验。

## 实验目标

对应任务要求，本项目完成了以下内容：

- 基于 `torch.utils.data.Dataset` 实现 OCL object-level 数据加载。
- 读取 OCL annotation pickle，并将 image-level 标注展开为 object-level samples。
- 根据 object category 的样本数排序，保留样本数最多的 10 个类别。
- 根据 bounding box 裁剪每个物体区域，并接入 CLIP 的图像预处理。
- 构建 `torch.utils.data.DataLoader`，测试 batch size 为 4。
- 使用 OpenAI CLIP 模型进行 zero-shot 推理。
- 推理时将模型切到 `eval()`，并冻结所有参数的梯度计算。
- 使用图像特征和文本类别特征的余弦相似度进行 category 排序，计算 top-k accuracy。
- 对 attribute 和 affordance 预测分数计算 mAP。

## 代码结构

```text
.
├── ocl_clip/
│   ├── __init__.py
│   └── data.py                 # OCLDataset 与 batch collate
├── check_environment.py        # 检查依赖是否可导入
├── check_dataloader.py         # 检查 Dataset/DataLoader
├── run_ocl_clip.py             # CLIP 推理与指标计算
├── make_ocl_coco_subset.py     # 构建小型 OCL/COCO 子集的辅助脚本
├── ocl_coco_subset/            # 仓库内置小数据集
└── outputs/
    └── ocl_clip_metrics_gpu.json
```

## 数据集

正式 OCL 数据集位于服务器路径：

```text
/data/DATA/OCL_DATA/OCL_data
```

其中 `data/resource` 是标注文件目录，其余目录为源图片数据。

本地实验没有下载完整 COCO/OCL 数据集，而是使用仓库内置的小数据集：

```text
ocl_coco_subset/
```

本地小数据集信息：

| Item | Value |
| --- | --- |
| COCO images | 20 |
| Object-level samples | 40 |
| Split | valtest |
| Top categories | 10 |
| Batch size | 4 |

本地选出的 10 个类别为：

```text
banana, broccoli, carrot, car, truck, sheep, motorcycle, cattle, horse, boat
```

## 环境

本地 GPU 实验环境如下：

| Item | Value |
| --- | --- |
| Python | 3.11.15 |
| PyTorch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| GPU | NVIDIA GeForce RTX 4060 |
| CUDA available | True |

GPU 检查输出：

```text
2.11.0+cu128
True
NVIDIA GeForce RTX 4060
```

如果 `torch.cuda.is_available()` 不是 `True`，不要直接切到 CPU 跑实验，应先检查 CUDA 版 PyTorch、NVIDIA driver 和当前 Python 环境。

## 安装依赖

建议先创建干净环境，例如使用 venv、conda、miniforge 或 uv。下面以 pip 为例。

先根据机器 CUDA/driver 情况安装 CUDA 版 PyTorch。当前本地机器使用的是 CUDA 12.8 wheel：

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

然后安装其它依赖和 OpenAI CLIP：

```bash
python -m pip install numpy pillow tqdm scikit-learn git+https://github.com/openai/CLIP.git
```

如果当前环境没有 `git` 命令，也可以直接从 GitHub 源码压缩包安装 CLIP：

```bash
python -m pip install numpy pillow tqdm scikit-learn https://github.com/openai/CLIP/archive/refs/heads/main.zip
```

检查依赖：

```bash
python check_environment.py
```

本地输出：

```text
python: C:\Users\pc\Documents\OCL\OCLExperiment\envs\ocl_clip\python.exe
version: 3.11.15
torch        OK
torchvision  OK
PIL          OK
numpy        OK
sklearn      OK
tqdm         OK
clip         OK
```

## DataLoader 检查

运行：

```bash
python check_dataloader.py --data-root ocl_coco_subset --split valtest --batch-size 4
```

本地输出摘要：

```text
samples: 40
selected categories: ['banana', 'broccoli', 'carrot', 'car', 'truck', 'sheep', 'motorcycle', 'cattle', 'horse', 'boat']
batch image tensor: (4, 3, 224, 224)
batch categories: ['banana', 'broccoli', 'carrot', 'broccoli']
```

这说明 OCL 标注解析、object-level sample 展开、bbox crop、CLIP preprocess 和 batch collate 都能正常工作。

## 运行 CLIP 推理

本地 GPU 小实验命令：

```bash
python run_ocl_clip.py \
  --data-root ocl_coco_subset \
  --split valtest \
  --batch-size 4 \
  --num-workers 0 \
  --device cuda \
  --max-samples 40 \
  --output outputs/ocl_clip_metrics_gpu.json
```

使用的 CLIP 模型：

```text
ViT-B/32
```

推理过程：

- `clip.load("ViT-B/32", device=device)` 加载 CLIP。
- `model.eval()` 切换到推理模式。
- 遍历 `model.parameters()`，设置 `requires_grad_(False)`。
- 对 category、attribute、affordance 分别构造 prompt 并编码文本特征。
- 对每个 batch 的 object crop 编码图像特征。
- 将图像特征和文本特征归一化后做点积，等价于余弦相似度。
- category 使用相似度排序计算 top-k accuracy。
- attribute 和 affordance 使用所有 label 的相似度分数计算 mAP。

## 本地实验结果

结果文件：

```text
outputs/ocl_clip_metrics_gpu.json
```

完整结果：

```json
{
  "data_root": "C:\\Users\\pc\\Documents\\OCL\\OCLExperiment\\ocl_coco_subset",
  "split": "valtest",
  "clip_model": "ViT-B/32",
  "num_object_samples": 40,
  "selected_categories": [
    "banana",
    "broccoli",
    "carrot",
    "car",
    "truck",
    "sheep",
    "motorcycle",
    "cattle",
    "horse",
    "boat"
  ],
  "category_topk_accuracy": {
    "top1": 0.9,
    "top3": 0.975,
    "top5": 1.0
  },
  "attribute_mAP": 0.3110507660278995,
  "affordance_mAP": 0.4154354548479642,
  "num_attributes": 45,
  "num_affordances": 118
}
```

指标汇总：

| Metric | Value |
| --- | ---: |
| Category top-1 accuracy | 0.900 |
| Category top-3 accuracy | 0.975 |
| Category top-5 accuracy | 1.000 |
| Attribute mAP | 0.3111 |
| Affordance mAP | 0.4154 |
| Object samples | 40 |
| Attributes | 45 |
| Affordances | 118 |

## 结论

本地小规模实验已经验证：

- OCL 数据结构可以被正确解析为 object-level samples。
- DataLoader 可以按 `batch_size=4` 正常生成 CLIP 输入。
- CUDA 版 PyTorch 可用，实验实际运行在 NVIDIA GPU 上。
- CLIP ViT-B/32 可以完成 category、attribute 和 affordance 三类 zero-shot 推理。
- category top-k accuracy、attribute mAP 和 affordance mAP 指标均已产出。

需要注意的是，本结果只是基于 `ocl_coco_subset/` 的本地 smoke test，不是完整 OCL benchmark 结果。正式汇报完整性能时，应在服务器完整数据集 `/data/DATA/OCL_DATA/OCL_data` 上运行同一套 pipeline，并根据需要移除或调整 `--max-samples`。
