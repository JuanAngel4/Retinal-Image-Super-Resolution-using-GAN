# Stage 1: Literature Review and Problem Definition

**Status:** ✅ Complete

## Objective

Review the state-of-the-art in Super-Resolution techniques for medical imaging, with emphasis on GAN-based approaches. Establish the theoretical foundation and select the baseline architecture.

---

## Reviewed Architectures

### SRGAN (Ledig et al., 2017)

**Paper:** [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802)

**Key Contributions:**
- First GAN-based super-resolution architecture achieving photo-realistic outputs at 4× upscaling
- Introduced **perceptual loss** using VGG-19 feature maps instead of pixel-wise MSE
- Generator: SRResNet with 16 residual blocks and PixelShuffle upsampling
- Discriminator: 8-layer convolutional network classifying real vs. generated HR images
- **Finding:** SRGAN outputs are preferred by human raters over MSE-optimized networks despite lower PSNR

**Relevance:** Baseline architecture for this project. Suitable for medical images due to its balance of quality and computational efficiency.

### ESRGAN (Wang et al., 2018)

**Paper:** [ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks](https://arxiv.org/abs/1809.00219)

**Key Improvements Over SRGAN:**
- Residual-in-Residual Dense Block (RRDB) replacing basic ResBlocks
- Relativistic discriminator (RaGAN) instead of standard GAN discriminator
- Improved perceptual loss using features before activation

**Assessment:** Higher quality but significantly more parameters and training time. Not selected due to GPU constraints (GTX 1650, 4 GB VRAM).

### Medical Super-Resolution Literature

**Key Findings from Survey:**
- Retinal fundus SR primarily uses SRGAN and adaptations
- Most works achieve 2-4× magnification
- PSNR ranges from 28-38 dB depending on dataset and scale
- Evaluation combines PSNR/SSIM with clinical expert assessment

---

## Baseline Selection

| Criterion | SRGAN | ESRGAN |
|-----------|-------|--------|
| Parameters | ~1.55M (×4) | ~16M |
| Training time (GTX 1650) | ~2-4 hours | ~12+ hours |
| Perceptual quality | High | Very High |
| Implementation complexity | Moderate | High |
| Prior medical applications | Extensive | Limited |
| **Selected** | **✓** | ✗ |

**Rationale:** SRGAN provides the best balance of quality, training feasibility on available hardware, and proven applicability to medical imaging.

---

## Evaluation Metrics Defined

| Metric | Range | Target | Purpose |
|--------|-------|--------|---------|
| **PSNR** | 0-∞ dB | >30 dB | Pixel-level fidelity |
| **SSIM** | 0-1 | >0.85 | Structural similarity |
| **LPIPS** | 0-∞ | <0.15 | Perceptual quality |
| **Visual comparison** | Qualitative | — | Expert assessment |

---

## Deliverables

- Literature review summary — this document
- Architecture selection rationale — SRGAN
- Defined evaluation criteria — PSNR, SSIM, LPIPS
