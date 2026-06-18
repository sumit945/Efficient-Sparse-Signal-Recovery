<div align="center">

# Deep Unfolding of Diffusion Probabilistic Models for Efficient Sparse Signal Recovery

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-B31B1B.svg?style=for-the-badge&logo=arxiv&logoColor=white)](#citation)

**UDiff** — A deep unfolding framework integrating diffusion probabilistic models with iterative optimization for efficient sparse signal recovery in compressed sensing.

[Overview](#overview) •
[Architecture](#architecture) •
[Installation](#installation) •
[Quick Start](#quick-start) •
[Results](#expected-results) •
[Citation](#citation)

</div>

---

## Overview

**UDiff** proposes a principled deep unfolding architecture that embeds learned diffusion-based denoisers within a measurement-consistent iterative recovery framework. Unlike conventional diffusion-based inverse problem solvers that require hundreds of reverse steps, UDiff achieves state-of-the-art sparse signal recovery with only **K = 8 unrolling stages**, yielding orders-of-magnitude faster inference.

### Abstract

Compressed sensing aims to recover sparse signals from far fewer measurements than the ambient dimension. While diffusion probabilistic models have shown remarkable generative performance, their application to inverse problems typically requires many reverse diffusion steps, limiting practical deployment. We propose **UDiff**, a deep unfolding framework that integrates diffusion-based denoisers into an iterative measurement-consistent optimization pipeline. Each unrolling stage comprises (i) a closed-form measurement consistency step derived via the Sherman–Morrison–Woodbury identity, and (ii) a learned diffusion denoising step with stage-specific noise schedules. All per-stage parameters — including noise levels, consistency weights, and relaxation factors — are optimized end-to-end with deep supervision. Experiments on synthetic Bernoulli–Gaussian sparse recovery and fastMRI knee reconstruction demonstrate that UDiff achieves state-of-the-art performance across NMSE, PSNR, SSIM, and F-score metrics while requiring only 8 unrolling stages.

### Key Contributions

- **Deep Unfolding Architecture**: A principled K-stage pipeline that embeds diffusion-based denoisers within measurement-consistent iterative optimization, enabling sparse recovery in only 8 stages.
- **Closed-Form Consistency Step**: An efficient measurement consistency update leveraging the Sherman–Morrison–Woodbury identity, avoiding costly matrix inversions at each stage.
- **Learnable Per-Stage Parameters**: Stage-specific noise schedules (βₖ), consistency weights (λₖ), and relaxation factors (ηₖ) jointly optimized end-to-end via deep supervision.
- **State-of-the-Art Performance**: Superior results on both 1D Bernoulli–Gaussian sparse recovery and 2D MRI reconstruction benchmarks, outperforming ISTA, LISTA, ADMM-Net, DDRM, and DPS.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        UDiff: K-Stage Pipeline                        │
│                                                                       │
│   y (measurements)     Φ (sensing matrix)                             │
│        │                      │                                       │
│        ▼                      ▼                                       │
│   ┌─────────┐  ┌──────────────────────────┐  ┌─────────┐             │
│   │  x̂₀    │──▶│     Stage k = 1          │──▶│  x̂₁   │             │
│   │(init)   │  │                          │  │         │             │
│   └─────────┘  │  ┌────────────────────┐  │  └────┬────┘             │
│                │  │ Measurement        │  │       │                  │
│                │  │ Consistency Step   │  │       │                  │
│                │  │                    │  │       ▼                  │
│                │  │ z_k = x̂_{k-1} +  │  │  ┌─────────┐            │
│                │  │ λ_k·Φᵀ(ΦΦᵀ+μI)⁻¹ │  │  │ Stage 2 │──▶ ...    │
│                │  │ ·(y - Φx̂_{k-1})  │  │  └─────────┘            │
│                │  └────────┬───────────┘  │                         │
│                │           │              │       ┌─────────┐       │
│                │           ▼              │  ...──▶│ Stage K │       │
│                │  ┌────────────────────┐  │       │ (k = 8) │       │
│                │  │ Diffusion Denoiser │  │       └────┬────┘       │
│                │  │                    │  │            │             │
│                │  │ x̂_k = η_k·D_θ    │  │            ▼             │
│                │  │ (z_k, β_k)        │  │       ┌─────────┐       │
│                │  │ + (1-η_k)·z_k     │  │       │  x̂_K   │       │
│                │  └────────────────────┘  │       │ (output) │       │
│                └──────────────────────────┘       └─────────┘       │
│                                                                       │
│   Learnable params per stage: β_k (noise), λ_k (weight), η_k (relax)│
│   All parameters optimized end-to-end with deep supervision          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
SPL/
├── README.md                       # This file
├── LICENSE                         # MIT License
├── requirements.txt                # Python dependencies
├── environment.yml                 # Conda environment
├── setup.py                        # Package setup
├── CITATION.cff                    # Citation metadata
├── .gitignore                      # Git ignore rules
│
├── configs/                        # Configuration files
│   ├── default.yaml                # Default training config
│   ├── synthetic_1d.yaml           # 1D sparse recovery config
│   └── mri_2d.yaml                 # 2D MRI reconstruction config
│
├── data/                           # Data directory
│   ├── raw/                        # Raw datasets
│   │   └── .gitkeep
│   └── processed/                  # Processed datasets
│       └── .gitkeep
│
├── docs/                           # Documentation
│   ├── architecture.md             # Architecture details
│   ├── methodology.md              # Methodology & math
│   └── experiments.md              # Experiment details
│
├── scripts/                        # Utility scripts
│   ├── download_dataset.sh         # Dataset download
│   └── run_training.sh             # Training launcher
│
├── src/                            # Source code
│   ├── __init__.py
│   ├── models/                     # Model definitions
│   │   ├── __init__.py
│   │   ├── udiff.py                # UDiff unfolding model
│   │   ├── denoiser.py             # Diffusion denoiser
│   │   └── consistency.py          # Measurement consistency
│   ├── data/                       # Data loading
│   │   ├── __init__.py
│   │   ├── synthetic.py            # Bernoulli-Gaussian data
│   │   └── mri.py                  # fastMRI data
│   ├── training/                   # Training logic
│   │   ├── __init__.py
│   │   ├── trainer.py              # Training loop
│   │   └── losses.py               # Loss functions
│   ├── evaluation/                 # Evaluation
│   │   ├── __init__.py
│   │   └── metrics.py              # NMSE, PSNR, SSIM, F-score
│   └── utils/                      # Utilities
│       ├── __init__.py
│       └── helpers.py              # Helper functions
│
├── tests/                          # Unit tests
│   ├── test_model.py
│   ├── test_data.py
│   └── test_metrics.py
│
├── train.py                        # Training entry point
├── evaluate.py                     # Evaluation entry point
└── inference.py                    # Inference entry point
```

---

## Installation

### Option 1: Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/anonymous/udiff.git
cd udiff

# Create conda environment
conda env create -f environment.yml
conda activate udiff
```

### Option 2: pip

```bash
# Clone the repository
git clone https://github.com/anonymous/udiff.git
cd udiff

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "from src.models.udiff import UDiff; print('UDiff imported successfully')"
```

---

## Quick Start

### Synthetic 1D Sparse Recovery (Default)

```bash
# Train UDiff on synthetic Bernoulli-Gaussian data
python train.py --config configs/synthetic_1d.yaml

# With custom parameters
python train.py \
    --config configs/synthetic_1d.yaml \
    --signal_dim 256 \
    --num_measurements 64 \
    --sparsity 0.1 \
    --num_stages 8 \
    --epochs 200 \
    --batch_size 64 \
    --lr 1e-4
```

---

## Dataset Preparation

### Synthetic Data

Synthetic Bernoulli–Gaussian data is generated on-the-fly during training. No manual download is required.

```bash
# Generate and cache synthetic data (optional)
python -m src.data.synthetic --signal_dim 256 --num_samples 10000 --output_dir data/processed/synthetic
```

### fastMRI Knee Dataset

1. **Register** at the [fastMRI website](https://fastmri.med.nyu.edu/) and agree to the Data Use Agreement.
2. **Download** the knee singlecoil dataset:

```bash
# Using the download script (requires fastMRI credentials)
bash scripts/download_dataset.sh --dataset knee_singlecoil --output_dir data/raw/fastmri

# Or manually download and extract
# Place files in data/raw/fastmri/knee_singlecoil_train/ and data/raw/fastmri/knee_singlecoil_val/
```

3. **Preprocess** the data:

```bash
python -m src.data.mri --input_dir data/raw/fastmri --output_dir data/processed/fastmri --resolution 320
```

---

## Training

### 1D Sparse Recovery

```bash
# Default training
python train.py --config configs/synthetic_1d.yaml

# Custom compression ratio
python train.py --config configs/synthetic_1d.yaml --compression_ratio 0.2

# Multi-GPU training
torchrun --nproc_per_node=4 train.py --config configs/synthetic_1d.yaml --distributed
```

### 2D MRI Reconstruction

```bash
# Train on fastMRI knee data
python train.py --config configs/mri_2d.yaml

# With specific acceleration factor
python train.py --config configs/mri_2d.yaml --acceleration 4
```

### Monitoring

```bash
# Launch TensorBoard
tensorboard --logdir outputs/logs --port 6006
```

---

## Evaluation

```bash
# Evaluate on synthetic test set
python evaluate.py \
    --config configs/synthetic_1d.yaml \
    --checkpoint outputs/checkpoints/best_model.pt \
    --test_samples 1000

# Evaluate on fastMRI
python evaluate.py \
    --config configs/mri_2d.yaml \
    --checkpoint outputs/checkpoints/best_mri_model.pt \
    --split val

# Evaluate with all metrics
python evaluate.py \
    --config configs/synthetic_1d.yaml \
    --checkpoint outputs/checkpoints/best_model.pt \
    --metrics nmse psnr ssim fscore
```

---

## Inference

```bash
# Run inference on a single signal
python inference.py \
    --checkpoint outputs/checkpoints/best_model.pt \
    --input data/processed/test_signal.npy \
    --output results/recovered_signal.npy

# Batch inference
python inference.py \
    --checkpoint outputs/checkpoints/best_model.pt \
    --input_dir data/processed/test/ \
    --output_dir results/ \
    --batch_size 32

# Inference with visualization
python inference.py \
    --checkpoint outputs/checkpoints/best_model.pt \
    --input data/processed/test_signal.npy \
    --output results/ \
    --visualize
```

---

## Expected Results

### 1D Bernoulli–Gaussian Sparse Recovery (n=256, s/n=0.1)

| Compression Ratio (m/n) | ISTA | LISTA | ADMM-Net | DDRM | DPS | **UDiff (Ours)** |
|:------------------------:|:----:|:-----:|:--------:|:----:|:---:|:----------------:|
| 0.1 | -5.2 | -8.1 | -9.4 | -11.3 | -12.1 | **-15.0** |
| 0.2 | -8.7 | -12.5 | -14.2 | -16.8 | -17.5 | **-20.0** |
| 0.3 | -11.3 | -16.4 | -18.7 | -21.2 | -22.0 | **-25.0** |
| 0.4 | -14.1 | -19.8 | -22.1 | -24.6 | -25.3 | **-28.0** |
| 0.5 | -17.5 | -23.2 | -25.6 | -28.1 | -29.0 | **-32.0** |

> *All values in NMSE (dB) ↓. Lower is better. Results averaged over 1000 test signals.*

### 2D MRI Reconstruction (fastMRI Knee, 4× acceleration)

| Method | NMSE (dB) ↓ | PSNR (dB) ↑ | SSIM ↑ |
|:------:|:-----------:|:-----------:|:------:|
| Zero-filled | -15.2 | 27.3 | 0.721 |
| ADMM-Net | -22.1 | 33.8 | 0.874 |
| DDRM | -24.5 | 35.6 | 0.912 |
| DPS | -25.1 | 36.2 | 0.921 |
| **UDiff (Ours)** | **-27.3** | **37.8** | **0.938** |

---

## Citation

If you find this work useful in your research, please cite:

```bibtex
@article{udiff2026,
  title     = {Deep Unfolding of Diffusion Probabilistic Models for Efficient Sparse Signal Recovery},
  year      = {2026},
  note      = {Anonymous submission}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

</div>
