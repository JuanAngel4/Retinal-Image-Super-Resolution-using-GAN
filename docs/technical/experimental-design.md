# Experimental Design

## Overview

The experimental design explores the interaction between **3 degradation scenarios** and **3 scale factors**, producing 9 distinct models. Each combination tests how well the SRGAN can recover high-resolution details under different levels of image quality degradation.

---

## The 9-Model Matrix

| | ×2 (256×256 → 512×512) | ×4 (128×128 → 512×512) | ×8 (64×64 → 512×512) |
|---|:---:|:---:|:---:|
| **A — Standard** | M1 | M2 | M3 |
| **B — Moderate** | M4 | M5 | M6 |
| **C — Severe** | M7 | M8 | M9 |

---

## Scenario Definitions

### Scenario A: Standard (Literature Baseline)

Represents typical degradation found in academic benchmarks.

| Parameter | Value | Motivation |
|-----------|-------|------------|
| Downsampling | Bicubic | Standard interpolation, good edge preservation |
| Blur | Gaussian, σ=1.0 | Mild optical defocus |
| Noise | Gaussian, σₙ=5 | Low sensor noise |
| **Clinical analogue** | Good quality fundus camera | |

### Scenario B: Moderate (Common Real-World)

Represents the expected quality from mid-range medical cameras.

| Parameter | Value | Motivation |
|-----------|-------|------------|
| Downsampling | Bilinear | Smoother, more realistic for low-cost sensors |
| Blur | Gaussian, σ=1.5 | Noticeable defocus |
| Noise | Gaussian, σₙ=10 | Elevated sensor noise |
| **Clinical analogue** | Mid-range portable fundus camera | |

### Scenario C: Severe (Low-Cost Devices)

Represents the worst-case quality from ultra-low-cost or smartphone-based imaging.

| Parameter | Value | Motivation |
|-----------|-------|------------|
| Downsampling | Lanczos | Sharp edges but introduces ringing |
| Blur | Gaussian, σ=2.0 | Significant defocus |
| Noise | Poisson | Photon-counting noise (low-light conditions) |
| **Clinical analogue** | Smartphone-based retinal imaging | |

---

## Scale Factor Analysis

| Scale | LR Resolution | Information Loss | Clinical Relevance |
|-------|:-------------:|:----------------:|:------------------:|
| ×2 | 256×256 | Minimal — 75% of pixels preserved | Suitable for mild enhancement |
| ×4 | 128×128 | Significant — 94% of pixels lost | Standard SR target, balances quality and challenge |
| ×8 | 64×64 | Extreme — 98.4% of pixels lost | Tests model limits, useful for extreme low-res scenarios |

---

## Degradation Pipeline Detail

```python
def degrade_image(hr, scale, scenario):
    # Step 1: Blur (simulates lens optics)
    img = gaussian_blur(hr, sigma=scenario.blur_sigma)
    
    # Step 2: Downsample (simulates sensor discretization)
    img = downsample(img, scale, method=scenario.downsample_method)
    
    # Step 3: Noise (simulates sensor electronics)
    img = add_noise(img, type=scenario.noise_type, sigma=scenario.noise_sigma)
    
    return img
```

This order **(blur → downsample → noise)** follows the physical imaging pipeline: optics cause blur before the sensor discretizes the signal, and electronics add noise after capture.

---

## Training Configuration (srgan2)

All 9 models share the same training hyperparameters:

| Parameter | Value |
|-----------|-------|
| Architecture | 3× Conv → PixelShuffle |
| Loss function | L1 |
| Optimizer | Adam (lr=1×10⁻⁴) |
| Batch size | 4 |
| Epochs | 3 |
| Training samples | 3,578 |
| Validation split | 10% |
| Hardware | GTX 1650 4 GB |
| Time per model | ~30 min |
| **Total training time** | **~4.5 hours** |

---

## Why This Design?

1. **Isolate variable effects**: Each scenario has fixed degradation params, each scale varies only the upsampling factor
2. **Clinical grounding**: Degradation levels map to real-world device quality
3. **Comprehensive coverage**: 9 points in the degradation × scale space
4. **Practical feasibility**: Each model trains in ~30 min on available hardware
