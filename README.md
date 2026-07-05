# OCL + CLIP Object Inference Experiment

## 1. 实验内容

1. 实现了基于 `torch.utils.data.Dataset` 的 `OCLDataset`。
2. 从 OCL 标注文件中解析 image-level annotations，并展开为 object-level samples。
3. 按 object category 的样本数排序，保留样本数最多的 10 个类别。
4. 对每个 object 根据 bbox 进行裁剪，并使用 CLIP preprocess 做图像预处理。
5. 构建 PyTorch `DataLoader`，设置 `batch_size = 4`。
6. 使用 OpenAI CLIP `ViT-B/32` 模型进行 zero-shot 推理。
7. 推理时使用 GPU，模型设置为 `eval()`，并冻结全部参数。
8. 根据 image feature 与 category text feature 的余弦相似度计算 top-k accuracy。
9. 对 attribute 和 affordance 计算 mAP 指标。

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

## 实验结果


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

