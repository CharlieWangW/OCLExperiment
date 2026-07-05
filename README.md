# OCL CLIP Inference Experiment

This repository contains a small zero-shot CLIP inference experiment for Object Concept Learning (OCL). It includes:

- a custom `torch.utils.data.Dataset`
- top-10 category filtering by object sample count
- `DataLoader(batch_size=4)` support
- OpenAI CLIP inference
- category top-k accuracy
- attribute and affordance mAP
- a helper to build a tiny COCO/OCL subset locally

## Setup

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python check_environment.py
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python check_environment.py
```

## Build A Small Real Subset

The repository does not include COCO images or OCL annotation files. To build the small subset locally, first clone the official OCL repository and download COCO `val2014.zip`:

- OCL official repo: https://github.com/silicx/ObjectConceptLearning
- COCO images: http://images.cocodataset.org/zips/val2014.zip

Then run:

```bash
python make_ocl_coco_subset.py \
  --official-root ObjectConceptLearning \
  --split test \
  --coco-zip /path/to/val2014.zip \
  --output ocl_coco_subset \
  --top-categories 10 \
  --per-category 4
```

This creates `ocl_coco_subset/`, a small local dataset with real COCO images and OCL annotations. It is ignored by git.

## Run The Small Subset

Check the Dataset/DataLoader first:

```bash
python check_dataloader.py --data-root ocl_coco_subset --batch-size 4
```

Run CLIP inference on CPU:

```bash
python run_ocl_clip.py \
  --data-root ocl_coco_subset \
  --split valtest \
  --batch-size 4 \
  --num-workers 0 \
  --device cpu \
  --max-samples 40
```

If CUDA PyTorch is installed, use:

```bash
python run_ocl_clip.py \
  --data-root ocl_coco_subset \
  --split valtest \
  --batch-size 4 \
  --num-workers 0 \
  --device cuda \
  --max-samples 40
```

The metrics JSON is written to `outputs/ocl_clip_metrics.json`.
