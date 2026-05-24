# GAN-based Retinal Image Super-Resolution

<p align="center">
  <strong>Enhancing retinal fundus image resolution using Generative Adversarial Networks</strong><br>
  <em>Accessible medical imaging through computational super-resolution</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11-blue?logo=python" alt="Python 3.11">
  <img src="https://img.shields.io/badge/framework-PyTorch-red?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/CUDA-11.8-green?logo=nvidia" alt="CUDA 11.8">
  <img src="https://img.shields.io/badge/status-research-yellow" alt="Research">
  <img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="MIT">
</p>

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Super-Resolution (SRGAN)](#super-resolution-srgan)
  - [Classification (VGG16)](#classification-vgg16)
  - [Experimental Design](#experimental-design)
- [Key Results](#key-results)
  - [SRGAN Performance](#srgan-performance)
  - [Classification Performance](#classification-performance)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Project Roadmap](#project-roadmap)
- [Documentation](#documentation)
- [Author](#author)

---

## Overview

### The Problem

Low-cost medical imaging devices often produce low-resolution images due to hardware constraints. In retinal diagnostics, this limits the ability to detect early signs of diabetic retinopathy — a leading cause of preventable blindness worldwide.

### The Solution

This project leverages **Generative Adversarial Networks (GANs)** to computationally enhance retinal fundus images, enabling high-quality diagnostics from affordable imaging hardware. By applying super-resolution techniques originally developed for natural images, we demonstrate significant quality improvements on medical fundus photography.

### Key Contributions

- **SRGAN-based super-resolution** for retinal fundus images across 9 degradation scenarios (3 degradation levels × 3 scale factors)
- **Best SR performance:** PSNR **37.43 dB**, SSIM **0.944** at ×2 magnification
- **Transfer learning classifier** achieving **64.43% accuracy** on 5-class diabetic retinopathy grading
- **Comprehensive degradation pipeline** simulating real-world camera optics (blur → downsample → noise)
- **Full documentation** of methodology, results, and future work

---

## Dataset

### Source

**Kaggle:** [Binary Classification Data — APTOS and Messidor](https://www.kaggle.com/datasets/anikbhowmickae20b102/binary-classification-data-aptos-and-messidor)

Combines two major retinal image datasets:
- **APTOS** (Asia Pacific Tele-Ophthalmology Society)
- **Messidor** (French research program)

### Class Distribution

| DR Severity | Label | Samples | Percentage |
|:-----------:|:-----:|:-------:|:----------:|
| No DR | 0 | 1,804 | 50.4% |
| Mild | 1 | 350 | 9.8% |
| Moderate | 2 | 954 | 26.7% |
| Severe | 3 | 182 | 5.1% |
| Proliferative | 4 | 288 | 8.0% |

### Preprocessing Pipeline

```mermaid
flowchart LR
    A["Raw APTOS + Messidor<br/>~40,000 images"] --> B["Deduplication<br/>(name + MD5)"]
    B --> C["Label alignment<br/>(CSV cross-reference)"]
    C --> D["Stratified split<br/>70/15/15"]
    D --> E["On-the-fly degradation<br/>(blur → downsample → noise)"]
    E --> F["HR/LR pairs<br/>(dynamic generation)"]
```

**Key decision:** HR images are stored; all degradations are applied dynamically during training to ensure reproducibility and prevent data leakage.

---

## Methodology

### Super-Resolution (SRGAN)

Two architectures were developed:

#### Full SRGAN (Baseline Design)

| Component | Architecture | Parameters |
|-----------|-------------|:----------:|
| **Generator** | SRResNet: 16× ResBlocks + PixelShuffle upsampling | ~1.55M (×4) |
| **Discriminator** | 8× Conv-BN-LeakyReLU blocks + classifier head | ~2.8M |
| **Loss** | Content (VGG-19 relu3_3) + Adversarial + L1 | — |

```mermaid
flowchart LR
    subgraph Generator
        A["LR Image"] --> B["16× ResBlocks"]
        B --> C["PixelShuffle<br/>Upsampling"]
        C --> D["SR Image"]
    end
    subgraph Discriminator
        E["HR or SR"] --> F["8× Conv Blocks"]
        F --> G["Real / Fake"]
    end
    D --> E
```

#### Simplified SRGAN (srgan2 — Trained)

Due to GPU constraints (GTX 1650, 4 GB VRAM), a lighter architecture was used for full training:

```python
nn.Sequential(
    Conv2d(3, 64, 3) → ReLU,
    Conv2d(64, 64, 3) → ReLU,
    Conv2d(64, 3 × scale², 3),
    PixelShuffle(scale),
)
```

**Loss:** L1 only | **Optimizer:** Adam (lr=1×10⁻⁴) | **Epochs:** 3 | **Batch size:** 4

### Classification (VGG16)

A VGG16 pre-trained on ImageNet was adapted for DR severity grading:

```
VGG16 (frozen convolutional features)
    ↓
Classifier: Linear(25088, 4096) → ReLU → Dropout(0.5)
          → Linear(4096, 4096) → ReLU → Dropout(0.5)
          → Linear(4096, 5)
```

- **Loss:** Weighted CrossEntropyLoss (inverse frequency weights)
- **Optimizer:** Adam (lr=1×10⁻⁴, classifier only)
- **Training time:** 104 minutes
- **Key finding:** Frozen features outperformed fine-tuning (64.43% vs 62.76%) due to limited training data

### Experimental Design

**9 models** exploring the interaction between degradation level and scale factor:

| | ×2 | ×4 | ×8 |
|---|:---:|:---:|:---:|
| **A — Standard** (Bicubic, σ=1.0, σₙ=5) | M1 | M2 | M3 |
| **B — Moderate** (Bilinear, σ=1.5, σₙ=10) | M4 | M5 | M6 |
| **C — Severe** (Lanczos, σ=2.0, Poisson) | M7 | M8 | M9 |

[Full experimental design details →](docs/technical/experimental-design.md)

---

## Key Results

### SRGAN Performance

| Model | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | Best For |
|:-----:|:----------:|:------:|:-------:|----------|
| **M4** | **37.43** | **0.944** | **0.043** | Best overall (×2, Bilinear) |
| M1 | 35.19 | 0.922 | 0.060 | Standard scenario (×2) |
| M5 | 32.01 | 0.860 | 0.138 | Moderate scenario (×4) |
| M8 | 31.20 | 0.855 | 0.171 | Severe scenario (×4) |
| M2 | 30.84 | 0.817 | 0.206 | Standard scenario (×4) |
| M6 | 27.49 | 0.729 | 0.414 | Moderate scenario (×8) |
| M9 | 27.52 | 0.765 | 0.410 | Severe scenario (×8) |

[Full results →](docs/technical/results.md)

**Key findings:**
- ×2 models achieve **excellent** reconstruction (avg PSNR > 35 dB, SSIM > 0.92)
- ×4 models achieve **good** reconstruction (avg PSNR > 30 dB, SSIM > 0.80)
- Scenario B (Bilinear) **consistently outperforms** Bicubic and Lanczos — smoother LR inputs are easier to reconstruct

### Classification Performance

| Model | Accuracy | Best For |
|-------|:--------:|----------|
| **Frozen VGG16** | **64.43%** | Limited data scenarios |
| Fine-tuned VGG16 | 62.76% | Larger datasets |
| Ensemble | 63.31% | Stability |

**Primary limitation:** Class imbalance (50:1 ratio between healthy and minority DR stages)

---

## Repository Structure

```
ProyectoAvanzado2/
│
├── README.md                        ← This file — project overview
├── AGENTS.md                        ← Agent configuration (opencode/Claude)
├── CONTEXT.md                       ← Domain glossary for AI agents
│
├── notebooks/                       ← Jupyter notebooks (all experiments)
│   ├── data_exploration.ipynb       → Dataset exploration
│   ├── model_desing.ipynb           → Architecture design decisions
│   ├── model_preparation.ipynb      → Data cleaning and preprocessing
│   ├── SRGAN_model_implementation.ipynb → Full SRGAN implementation
│   ├── srgan2.ipynb                 → Simplified SRGAN (9 trained models)
│   ├── Transfer_learnig_prediction_DL_model.ipynb → VGG16 classification
│   └── models/                      → Trained model weights (.pth)
│       ├── classification/          → VGG16 classifiers
│       └── Super-resolution/        → SRGAN variants (srgan2, srgan31, srgan32)
│
├── scripts/                         ← Utility Python modules
│   ├── useful_functions.py          → CSV/image processing utilities
│   ├── download.py                  → Kaggle dataset downloader
│   └── calculos.py                  → Physics calculations (auxiliary)
│
├── data/                            ← Datasets (gitignored due to size)
│   ├── raw/                         → Original Kaggle download
│   └── processed/                   → Cleaned M3 dataset
│
└── docs/                            ← Detailed documentation
    ├── stages/                      → One document per development stage
    │   ├── 01-literature-review.md
    │   ├── 02-dataset-preparation.md
    │   ├── 03-baseline-model.md
    │   ├── 04-training-optimization.md
    │   ├── 05-evaluation.md
    │   ├── 06-system-integration.md
    │   └── 07-final-analysis.md
    ├── technical/                   → Architecture and results reference
    │   ├── experimental-design.md   → 9-model matrix explained
    │   └── results.md               → Full metrics tables
    ├── agents/                      → Agent workflow configuration
    ├── images/                      → Figures and diagrams
    └── adr/                         → Architecture Decision Records
```

---

## Installation

### Prerequisites

- Python 3.11+
- PyTorch 2.1+ (CUDA recommended)
- NVIDIA GPU with 4+ GB VRAM (optional, for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/JuanAngel4/ProyectoAvanzado2.git
cd ProyectoAvanzado2

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate    # Windows

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install jupyter numpy opencv-python scikit-image lpips matplotlib kagglehub
```

### Dataset Download

```bash
# Option 1: Automated download
python scripts/download.py

# Option 2: Manual download from Kaggle
# 1. Go to: https://www.kaggle.com/datasets/anikbhowmickae20b102/binary-classification-data-aptos-and-messidor
# 2. Download and extract to data/raw/
```

---

## Usage

### Running Notebooks

```bash
jupyter notebook notebooks/
```

Key notebooks:
- **`srgan2.ipynb`** — Train/evaluate SRGAN models (9 configurations)
- **`SRGAN_model_implementation.ipynb`** — Full SRGAN architecture
- **`Transfer_learnig_prediction_DL_model.ipynb`** — VGG16 classification
- **`model_preparation.ipynb`** — Dataset preparation pipeline

### Inference (from notebooks)

```python
# Load trained model
model = torch.load('notebooks/models/Super-resolution/srgan2/model_4.pth')

# Super-resolve an image
sr_image = model(lr_image.unsqueeze(0))
```

---

## Project Roadmap

| Stage | Description | Status |
|:-----:|-------------|:------:|
| 1 | Literature Review & Problem Definition | ✅ Complete |
| 2 | Dataset Preparation & Preprocessing | ✅ Complete |
| 3 | Baseline Model Implementation (SRGAN) | ✅ Complete |
| 4 | Model Training & Optimization | ✅ Complete |
| 5 | Quantitative & Qualitative Evaluation | ✅ Complete |
| 6 | System Integration & Scalability | ⏳ Pending |
| 7 | Final Analysis & Future Work | ⏳ Pending |

---

## Documentation

Each project stage is documented in detail:

| Stage | Document |
|:-----:|----------|
| 1 | [Literature Review](docs/stages/01-literature-review.md) |
| 2 | [Dataset Preparation](docs/stages/02-dataset-preparation.md) |
| 3 | [Baseline Model](docs/stages/03-baseline-model.md) |
| 4 | [Training & Optimization](docs/stages/04-training-optimization.md) |
| 5 | [Evaluation](docs/stages/05-evaluation.md) |
| 6 | [System Integration](docs/stages/06-system-integration.md) |
| 7 | [Final Analysis](docs/stages/07-final-analysis.md) |

**Technical references:**
- [Experimental Design](docs/technical/experimental-design.md) — 9-model matrix
- [Results](docs/technical/results.md) — Full metrics tables

---

## Author

**Juan Angel** — Physics Engineer | Data Science Specialist

[GitHub](https://github.com/JuanAngel4) — [LinkedIn](https://linkedin.com/in/juanangel) — gamezdiazjuanangel@gmail.com

---

<p align="center">
  <sub>Built with ❤️ for accessible medical imaging</sub>
</p>
