# Context: GAN-based Retinal Image Super-Resolution

## Domain

Medical imaging, retinal fundus photography, diabetic retinopathy (DR) diagnosis, deep learning, super-resolution, generative adversarial networks (GANs).

## Problem

Low-cost retinal imaging devices produce low-resolution images that limit diagnostic accuracy. Computational super-resolution can enhance image quality to compensate for hardware constraints, making medical imaging more accessible in resource-constrained environments.

## Core Terms

| Term | Definition |
|------|------------|
| **SRGAN** | Super-Resolution Generative Adversarial Network — a GAN architecture for upscaling low-resolution images while generating realistic textures |
| **SRResNet** | The generator sub-network of SRGAN; a residual network with 16 ResBlocks and PixelShuffle upsampling |
| **Degradation pipeline** | Sequential process: Gaussian blur → downsampling → noise addition, simulating real-world image quality loss |
| **Perceptual loss** | Content loss computed from VGG-19 feature activations (ReLU3_3), measuring semantic similarity rather than pixel-level error |
| **PixelShuffle** | Efficient sub-pixel convolution layer for upscaling feature maps to higher resolution |
| **PSNR** | Peak Signal-to-Noise Ratio — standard pixel-level fidelity metric (higher is better) |
| **SSIM** | Structural Similarity Index — measures perceived visual quality (0-1, higher is better) |
| **LPIPS** | Learned Perceptual Image Patch Similarity — learned metric that correlates with human perception (lower is better) |
| **DR grading** | Diabetic Retinopathy severity classification: 0 (none), 1 (mild), 2 (moderate), 3 (severe), 4 (proliferative) |
| **ResBlock** | Residual block: Conv2d → BatchNorm → PReLU → Conv2d → BatchNorm + skip connection |
| **APTOS** | Asia Pacific Tele-Ophthalmology Society dataset — source of retinal fundus images |
| **Messidor** | French research program dataset for diabetic retinopathy — combined with APTOS |

## Architecture

This project explores two parallel tracks:
1. **Super-resolution** — SRGAN-based enhancement of retinal fundus images across 9 degradation scenarios (3 degradation levels × 3 scale factors)
2. **Classification** — VGG16 transfer learning for diabetic retinopathy severity grading on the enhanced images

## Key Technical Decisions

- **On-the-fly degradation**: HR images stored; LR images generated dynamically during training via parameterized degradation, ensuring reproducibility
- **Camera-motivated degradation order**: Blur (lens) → downsample (discretization) → noise (sensor electronics)
- **SRGAN over ESRGAN**: Simpler architecture sufficient for medical domain; lower computational requirements; easier training stability
- **Frozen VGG16 > fine-tuned**: Small dataset (2,504 training samples) makes fine-tuning prone to overfitting; frozen features with weighted classifier yield better generalization
- **LPIPS for perceptual quality**: In addition to traditional PSNR/SSIM, because medical image quality correlates better with learned perceptual metrics

## Repository Structure

```
/                   → Root
├── README.md       → Project overview and quick-start
├── AGENTS.md       → Agent configuration and installed skills
├── CONTEXT.md      → Domain glossary (this file)
├── notebooks/      → Jupyter notebooks for all experiments
├── scripts/        → Utility Python modules
├── data/           → Raw and processed datasets
│   ├── raw/        → Original Kaggle download
│   └── processed/  → Cleaned, unified, split datasets
├── docs/           → Detailed documentation
│   ├── stages/     → One document per development stage
│   ├── technical/  → Architecture and results reference
│   ├── agents/     → Agent workflow configuration
│   ├── images/     → Figures and diagrams
│   └── adr/        → Architecture Decision Records
└── models/         → Trained model weights (.pth)
```
