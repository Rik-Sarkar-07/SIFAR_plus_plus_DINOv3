# SIFAR++: Temporal Prompting for Efficient Video Understanding with Image Vision Transformers

Official implementation of **SIFAR++**, a parameter-efficient framework for video understanding that extends the Super Image (SIFAR) representation with **learnable temporal prompt tokens** for Vision Transformers.

The framework enables image-pretrained ViTs (e.g., DinoV3) to capture temporal information without introducing heavy temporal backbones.

---

## Overview

```
Video Frames
      │
      ▼
 Super Image (SIFAR)
      │
      ▼
Vision Transformer
+ Learnable Temporal Prompt Tokens
      │
      ▼
Video Classification
```

Temporal prompts are prepended to the patch embeddings before passing them through the Vision Transformer.

```python
prompt = nn.Parameter(torch.randn(1, num_prompts, dim))
x = torch.cat([prompt.expand(B, -1, -1), patch_tokens], dim=1)
```

---

# Repository Features

- ✅ SIFAR++ Temporal Prompting
- ✅ DinoV3 Small & Base finetuning
- ✅ Support for Kinetics-400
- ✅ Support for Something-Something V2
- ✅ Distributed multi-GPU training
- ✅ PyAV video loading
- ✅ MSN-pretrained checkpoint finetuning
- ✅ Standard DinoV3 finetuning

---

# Dataset Preparation

Create two annotation files

```
train.txt
val.txt
```

Each line should follow the format

```
video_path label start_frame end_frame
```

Example

```
/raid/abircs/Datasets/Kinetics400/train_256/playing_drums/GJJUUAxgIYo_000007_000017.mp4 1 300 230
```

where

| Field | Description |
|-------|-------------|
| video_path | Path to the video |
| label | Integer class label |
| start_frame | Starting frame index |
| end_frame | Ending frame index |

Place both annotation files inside a single directory and provide that directory using

```bash
--data_dir /path/to/dataset
```

Also update

```
video_dataset_config.py
```

with your dataset configuration.

---

# Environment

Install dependencies using either

```bash
pip install -r requirements.txt
```

or

```bash
conda env create -f env.yaml
```

---

# Training

## Common Arguments

| Argument | Kinetics-400 | Something-Something V2 | Description |
|-----------|--------------|------------------------|-------------|
| `--class_numbers` | 400 | 174 | Number of classes |
| `--model` | `dino_small` / `dino_base` | Same | Backbone |
| `--duration` | 16 | 16 | Number of sampled frames |

---

# 1. Standard DinoV3 Finetuning (SIFAR++)

Use

```
--dino_model_path
```

Do **not** enable

```
--msn_pretraining
```

## DinoV3 Small

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=28529 main.py \
  --data_dir /home/dasabir/orcd/scratch/datasets/Kinetics400_sifar \
  --use_pyav \
  --dataset kinetics400 \
  --opt adamw --lr 5e-4 --epochs 30 --sched cosine \
  --duration 16 --batch-size 4 --super_img_rows 4 \
  --num_workers 16 --disable_scaleup \
  --mixup 0.8 --cutmix 1.0 --drop-path 0.05 \
  --pretrained --warmup-epochs 5 --no-amp \
  --model dino_small \
  --output_dir output/test_1 \
  --weight-decay 0.01 \
  --clip-grad 1.0 \
  --class_numbers 400 \
  --dino_model_path path/to/dinov3_small.pth
```

---

## DinoV3 Base

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=28529 main.py \
  --data_dir /home/dasabir/orcd/scratch/datasets/Kinetics400_sifar \
  --use_pyav \
  --dataset kinetics400 \
  --opt adamw --lr 5e-4 --epochs 30 --sched cosine \
  --duration 16 --batch-size 4 --super_img_rows 4 \
  --num_workers 16 --disable_scaleup \
  --mixup 0.8 --cutmix 1.0 --drop-path 0.05 \
  --pretrained --warmup-epochs 5 --no-amp \
  --model dino_base \
  --output_dir output/test_1 \
  --weight-decay 0.01 \
  --clip-grad 1.0 \
  --class_numbers 400 \
  --dino_model_path path/to/dinov3_base.pth
```

---

# 2. MSN-Pretrained DinoV3 Finetuning

Enable

```bash
--msn_pretraining True
```

and provide

```
--msn_model_path
```

Example

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=28529 main.py \
  --data_dir /home/dasabir/orcd/scratch/datasets/Kinetics400_sifar \
  --use_pyav \
  --dataset kinetics400 \
  --opt adamw --lr 5e-4 --epochs 30 --sched cosine \
  --duration 16 --batch-size 4 --super_img_rows 4 \
  --num_workers 16 --disable_scaleup \
  --mixup 0.8 --cutmix 1.0 --drop-path 0.05 \
  --pretrained --warmup-epochs 5 --no-amp \
  --model dino_small \
  --output_dir output/test_1 \
  --weight-decay 0.01 \
  --clip-grad 1.0 \
  --class_numbers 400 \
  --msn_pretraining True \
  --msn_model_path path/to/checkpoint.pth
```

---

# Pretrained Checkpoints

| Model | Link |
|-------|------|
| DinoV3 Base (LVD Pretraining) | https://drive.google.com/drive/folders/1e1PLmTIKrMzgnhplKvWIleIFo5iknFI6 |
| MSN-pretrained DinoV3 | https://drive.google.com/drive/folders/14y_jOugOtlDFJOaRJaehSfcXvxRG6lvb |

---

---

# Citation

The accompanying paper is currently under review. Citation information will be added once the manuscript becomes publicly available.

---


---

# Contact

**Sudipta Sarkar**

📧 sudiptasarkar3600@gmail.com

---

## Acknowledgements

This repository builds upon the excellent work of:

- DinoV3
- Vision Transformer (ViT)
- SIFAR
- MSN
