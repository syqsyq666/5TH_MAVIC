## PBVS 2026 (Track C) — SAR Classification (3rd Place Solution)

This repository contains the training and inference code for the PBVS 2026 Multi-modal Aerial View Image Challenge (MAVIC) .

## Quickstart (inference only)

- Put test SAR images into `datasets/test/`
- Make sure you have trained weights under `checkpoints/` (or update paths in `test.py`)
- Run:

```bash
mkdir -p out
python test.py
```

The submission file will be written to `out/results.csv` with columns: `image_id`, `class_id`, `score`.

## Repository layout

- **`run_train.py`**: unified training entrypoint (recommended)
- **`norm_resnet50_SAR.py`**: ResNet101-based cross-domain training (EO + SAR)
- **`efficient_SAR.py`**: EfficientNet-B0-based cross-domain training (EO + SAR)
- **`test.py`**: inference script (reads `datasets/test/`, writes `out/results.csv`)
- **`datasets/`**: dataset root (you provide)
- **`checkpoints/`**: training outputs (created automatically)
- **`resnet50-0676ba61.pth`**, **`efficientnet-b0-355c32eb.pth`**: backbone initialization weights (already in repo)

## Environment setup

You have two common options.

### Option A (Conda, reproducible)

This repo includes an explicit Conda spec file (despite the name `environment.yml`).

```bash
conda create -n pbvs2026 --file environment.yml
conda activate pbvs2026
```

### Option B (pip)

1) Install PyTorch + torchvision first (choose the right CUDA build for your machine), then:

```bash
pip install -r requirements.txt
```

Notes:
- `requirements.txt` does **not** pin `torch/torchvision`, so you must install them yourself.

## Data preparation

Training scripts use `torchvision.datasets.ImageFolder`, so your folders must follow ImageFolder format:

```text
datasets/
  train/
    EO_Train/
      <class_0>/
        *.png
      <class_1>/
        *.png
      ...
    SAR_Train/
      <class_0>/
        *.png
      <class_1>/
        *.png
      ...
  test/
    *.png
```

- **Training requires both** `datasets/train/EO_Train/` and `datasets/train/SAR_Train/`.
- **Inference uses only** `datasets/test/`.
- For inference, filenames must contain digits, because `test.py` extracts `image_id` with a regex (e.g. `Gotcha16664030.png` → `16664030`).

## Training

`test.py` is **inference only**. For training, use `run_train.py` (recommended) or run training scripts directly.

### Recommended (unified entrypoint)

```bash
# Train ResNet101 only
python run_train.py --model resnet --gpus 0

# Train EfficientNet-B0 only
python run_train.py --model efficient --gpus 0

# Train both sequentially (default)
python run_train.py --model both --gpus 0,1
```

- `--gpus` sets `CUDA_VISIBLE_DEVICES`. The training scripts will automatically choose 2 GPUs if available, otherwise fall back to 1 GPU / CPU.

### Run a single training script

```bash
# ResNet101 (EO + SAR)
python norm_resnet50_SAR.py

# EfficientNet-B0 (EO + SAR)
python efficient_SAR.py
```

### Checkpoints (outputs)

By default, the scripts save to:

- ResNet: `checkpoints/resnet101/`
- EfficientNet: `checkpoints/efficientnet_b0/`

Each training script saves:
- Epoch checkpoints like `SAR_cross_domain_*_epoch_<N>.pth`
- Final models:
  - `SAR_cross_domain_resnet50.pth` / `EO_cross_domain_resnet50.pth` (ResNet script)
  - `SAR_cross_domain_efficientB0_final.pth` / `EO_cross_domain_efficientB0_final.pth` (EfficientNet script)

Important:
- Some copies of this repo may already contain ResNet checkpoints under `checkpoints/resnet50/` (legacy folder name). If your weights are there, either **edit the paths** in `test.py`, or move/symlink the directory to match what `test.py` expects.

## Inference (generate submission)

1) Put test images into `datasets/test/`
2) Ensure the `out/` directory exists:

```bash
mkdir -p out
```

3) Edit model paths in `test.py` if needed:

- `model_path_resnet` (ResNet checkpoint)
- `model_path_efficient` (EfficientNet checkpoint)

4) Run:

```bash
python test.py
```

Output:
- `out/results.csv`

## Common issues / troubleshooting

- **`FileNotFoundError: 训练数据目录不存在: .../datasets/train/EO_Train`**
  - You only have SAR training data. Training requires EO + SAR. Add `EO_Train` in ImageFolder format.

- **`FileNotFoundError` for backbone weights (`resnet50-0676ba61.pth` / `efficientnet-b0-355c32eb.pth`)**
  - Training scripts currently contain a `pretrained_path` hardcoded to an absolute path. Update `pretrained_path` in:
    - `norm_resnet50_SAR.py`
    - `efficient_SAR.py`
  - Point it to the corresponding `.pth` file in the repo root.

- **`invalid device ordinal` / GPU errors**
  - Use `CUDA_VISIBLE_DEVICES` (via `run_train.py --gpus ...`) and set `device_ids` in `test.py` to match your actual visible GPU indices.



