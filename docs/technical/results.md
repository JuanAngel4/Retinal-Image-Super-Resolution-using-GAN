# Results

## SRGAN Super-Resolution

### Full Metrics Table

| Model | Scenario | Scale | PSNR (dB) ↑ | SSIM ↑ | MSE ↓ | LPIPS ↓ |
|:-----:|:--------:|:----:|:----------:|:-----:|:-----:|:-------:|
| M1 | A — Standard | ×2 | **35.19** | **0.922** | 3.18e-4 | 0.0601 |
| M2 | A — Standard | ×4 | 30.84 | 0.817 | 9.12e-4 | 0.2064 |
| M3 | A — Standard | ×8 | 26.93 | 0.666 | 2.58e-3 | 0.4887 |
| M4 | B — Moderate | ×2 | **37.43** | **0.944** | 1.91e-4 | 0.0429 |
| M5 | B — Moderate | ×4 | 32.01 | 0.860 | 6.96e-4 | 0.1379 |
| M6 | B — Moderate | ×8 | 27.49 | 0.729 | 2.19e-3 | 0.4135 |
| M7 | C — Severe | ×2 | 34.79 | 0.918 | 3.48e-4 | 0.0799 |
| M8 | C — Severe | ×4 | 31.20 | 0.855 | 8.16e-4 | 0.1709 |
| M9 | C — Severe | ×8 | 27.52 | 0.765 | 2.09e-3 | 0.4100 |

### Best Model: M4 (Scenario B, ×2)

```
PSNR:  37.43 dB  — Excellent pixel-level fidelity
SSIM:  0.944     — Near-perfect structural preservation
LPIPS: 0.0429    — Very low perceptual distance
MSE:   1.91e-4   — Minimal pixel error
```

---

### Aggregated by Scale

| Scale | Avg PSNR | Avg SSIM | Avg LPIPS |
|-------|:--------:|:--------:|:---------:|
| ×2 | 35.80 dB | 0.928 | 0.061 |
| ×4 | 31.35 dB | 0.844 | 0.172 |
| ×8 | 27.31 dB | 0.720 | 0.437 |

**Trend:** Clear inverse relationship between scale factor and quality metrics. ×2 models achieve excellent reconstruction; ×8 models are limited but functional.

---

### Aggregated by Scenario

| Scenario | Avg PSNR | Avg SSIM | Avg LPIPS | Interpretation |
|----------|:--------:|:--------:|:---------:|----------------|
| A — Bicubic | 30.99 dB | 0.802 | 0.252 | Solid baseline |
| B — Bilinear | **32.31 dB** | **0.844** | **0.198** | **Best — smoother inputs** |
| C — Lanczos | 31.17 dB | 0.846 | 0.220 | Competitive with A |

**Why Scenario B wins:** Bilinear downsampling produces the least aliased LR images. The model learns a simpler mapping function compared to edge-preserving Bicubic/Lanczos, resulting in higher reconstruction fidelity.

---

### Performance Visualization

```
PSNR by Model
 38 +                     M4 (37.43)
    |
 36 +          M1 (35.19)
    |               M7 (34.79)
 34 +
    |
 32 +                         M5 (32.01)    M8 (31.20)
    |                    M2 (30.84)
 30 +
    |
 28 +                                                  M6 (27.49)  M9 (27.52)
    |                                             M3 (26.93)
 26 +
    +-------+-------+-------+-------+-------+-------+-------+-------+-------+
          M1      M2      M3      M4      M5      M6      M7      M8      M9
```

---

## VGG16 Classification

### Model Comparison

| Model | Test Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|-------|:------------:|:-----------------:|:--------------:|:----------:|
| Frozen VGG16 | **64.43%** | 0.58 | 0.54 | 0.54 |
| Fine-tuned VGG16 | 62.76% | 0.56 | 0.53 | 0.52 |
| Ensemble | 63.31% | 0.57 | 0.54 | 0.53 |

### Loss Curves

| Metric | Frozen | Fine-tuned |
|--------|:------:|:----------:|
| Final Train Loss | 0.8445 | — |
| Final Val Loss | 1.0513 | 0.9723 |
| Epochs | 13 | 13 |
| Training time | 104 min | ~150 min |

### Confusion Pattern

The classifier performs best on:
- **Class 0 (No DR):** High sensitivity (majority class, 1,262 samples)
- **Class 2 (Moderate):** Moderate performance (668 samples)

Performs poorly on:
- **Class 1 (Mild):** Often confused with class 0 or 2 (245 samples)
- **Class 3 (Severe):** Low sensitivity (127 samples)
- **Class 4 (Proliferative):** Low sensitivity (202 samples)

**Root cause:** Class imbalance — the model rarely sees minority classes.

---

## Saved Models

### Location: `notebooks/models/`

```
models/
├── classification/
│   ├── best_model.pth            → Frozen VGG16 (537 MB) — ✓ Best
│   ├── best_model_improved.pth   → Fine-tuned VGG16 (514 MB)
│   └── vgg16_retinal_model.pth   → Earlier version (537 MB)
│
└── Super-resolution/
    ├── srgan2/                   → 9 models, ~2-6 MB each
    │   ├── model_1.pth  ...  model_9.pth
    ├── srgan31/                  → 9 models (alternate config)
    │   └── model_1.pth  ...  model_9.pth
    └── srgan32/                  → 9 models (alternate config)
        └── model_1.pth  ...  model_9.pth
```
