# EdgeAuth 🔍

> **On-Device AI-Generated Image Detection via a Calibrated Hybrid CNN-ViT Cascade**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-CIFAKE-orange.svg)](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
[![Accuracy](https://img.shields.io/badge/Val%20Acc-97.42%25-brightgreen.svg)]()
[![AUC](https://img.shields.io/badge/AUC-0.978-brightgreen.svg)]()

Diffusion models now generate synthetic images that fool human observers nearly as often as real ones. **EdgeAuth** is a four-stage cascade architecture for authenticating AI-generated images **entirely on-device** and requires no cloud, no privacy exposure, no latency.

---

## ✨ Highlights

| Metric | Value |
|---|---|
| Validation Accuracy (Hybrid CNN-ViT) | **97.42%** |
| AUC-ROC | **0.978** |
| Improvement over ResNet-50 baseline | **+8.90 pp** |
| Trainable parameters | **0.72M / 95.4M (0.76%)** |
| FastPath throughput | **47.1% of traffic** |
| Expected Calibration Error (ECE) | **0.0183** |
| Full cascade decisive accuracy | **96.21%** |

---

## 🏗️ Architecture

EdgeAuth is a **four-stage cascade** that routes images through progressively more expensive components only when needed:

```
Input Image
    │
    ▼
┌─────────────────────────────┐
│  Stage 1: FastPath (128px)  │  ← MobileNet-style CNN, 3.21M params
│  score < 0.20 → Exit: REAL  │    47.1% of traffic exits here
└────────────┬────────────────┘
             │ (remaining 52.9%)
             ▼
┌────────────────────────────────────────┐
│  Stage 2: Hybrid CNN-ViT (192px)       │  ← ResNet-50 stem (frozen, stages 1-3)
│  score < 0.35 → Exit: REAL            │      + 12-layer ViT with LoRA (r=16)
└────────────┬───────────────────────────┘    30.6% of traffic exits here
             │ (remaining 22.3%)
             ▼
┌───────────────────────────────┐
│  Stage 3: Guard MLP           │  ← 3-layer MLP, 71.4K params
│  G(x) → FAKE or UNCERTAIN    │    DPO-aligned for asymmetric cost modeling
└───────────────────────────────┘
```

### Key Components

**Hybrid CNN-ViT Backbone**
- ResNet-50 convolutional stem (stages 1–3, frozen) extracts a `1024×12×12` feature map
- Features are linearly projected to 768-d tokens and processed by 12 pre-LayerNorm transformer blocks
- LoRA adapters (rank=16, alpha=32) applied to Q and V projections in all attention layers
- Only **0.76%** of parameters are updated during training

**Guard MLP**
- Predicts backbone correctness from 4 scalar features: `score`, `conf`, `var_mean`, `max_norm`
- Trained on a 70/30 split of the validation set, then aligned with **Direct Preference Optimization (DPO)**
- Emits a three-valued output: `REAL` / `FAKE` / `UNCERTAIN`

**FastPath Pre-filter**
- MobileNet-style depthwise-separable CNN (11 blocks, 32→1024 channels)
- Processes 128×128 images at ~93.1% accuracy
- Routes obvious REAL cases before engaging the expensive backbone

---

## 📊 Results

### Component Performance on CIFAKE

| Component | Val Acc | AUC | Params | Trainable |
|---|---|---|---|---|
| Baseline: ResNet-50 + LogReg | 88.52% | 0.892 | 25.5M | ~2K |
| FastPath CNN (128px) | 93.10% | 0.951 | 3.21M | 3.21M |
| **Hybrid CNN-ViT (LoRA r=16)** | **97.42%** | **0.978** | 95.4M | **0.72M** |
| Guard MLP (post-DPO) | 92.10%* | — | 71.4K | 71.4K |
| Full Cascade (decisive) | 96.21% | — | ~99M | — |

*Guard accuracy measures correctness-prediction (recall = 91.3%, FPR = 8.7%)

### LoRA Rank Ablation

| Configuration | Trainable Params | Val Acc |
|---|---|---|
| Full fine-tune | 95.4M | 97.21% |
| LoRA r=8 | 0.38M | 96.58% |
| **LoRA r=16 (EdgeAuth)** | **0.72M** | **97.42%** |
| LoRA r=32 | 1.42M | 97.39% |

LoRA r=16 beats full fine-tuning — the low-rank constraint acts as implicit regularization.

---

## 🔬 Why CNN + ViT?

Frequency-domain analysis of CIFAKE reveals that AI-generated images imprint **structured spectral artifacts** in the 2-D Fourier spectrum arising from periodic upsampling in the diffusion UNet decoder. These artifacts are:

- **Local** → captured by convolutional weight-sharing
- **Global** → their co-occurrence across the image requires transformer self-attention

A hand-crafted high-frequency energy ratio achieves AUC = 0.623 with zero parameters, confirming the signal is real but not trivially separable. The Hybrid CNN-ViT is therefore architecturally motivated by the signal structure, not arbitrary paradigm stacking.

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch torchvision timm
pip install numpy pandas matplotlib scikit-learn
```

### Dataset

Download [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) (120,000 images, 50/50 real/fake split):

```
data/
├── train/
│   ├── REAL/   # 50,000 images
│   └── FAKE/   # 50,000 images
└── test/
    ├── REAL/   # 10,000 images
    └── FAKE/   # 10,000 images
```

### Training

Open and run the companion notebook:

```bash
jupyter notebook EdgeAuth_.ipynb
```

The notebook covers the full pipeline in order:
1. Exploratory analysis (pixel statistics + frequency-domain signature)
2. Hybrid CNN-ViT training with LoRA
3. FastPath training
4. Guard MLP training + DPO alignment
5. Cascade evaluation and ablations

---

## 📁 Repository Structure

```
EdgeAuth/
├── EdgeAuth_.ipynb     # Full training + evaluation notebook
├── EdgeAuth_.pdf       # Research paper
├── EdgeAuth.pptx       # Presentation slides
└── README.md
```

---

## 🧠 Technical Details

### Preprocessing & Augmentation

CIFAKE images are natively 32×32 and are upsampled to avoid exploiting JPEG metadata artifacts:

| Transform | Value |
|---|---|
| Resize (ViT backbone) | 192×192 (bilinear) |
| Resize (FastPath) | 128×128 (bilinear) |
| Random horizontal flip | p = 0.50 |
| Gaussian blur (σ ∈ [0.1, 1.5]) | p = 0.30 |
| Color jitter (brightness/contrast/saturation ±0.10, hue ±0.02) | p = 0.30 |
| Normalization | ImageNet mean/std |

### Training Hyperparameters

| Hyperparameter | CNN-ViT | Guard MLP | FastPath |
|---|---|---|---|
| Optimizer | AdamW | AdamW | AdamW |
| Learning rate | 1e-3 | 1e-3 | 3e-4 |
| Weight decay | 0.05 | 1e-2 | 1e-4 |
| LR schedule | Cosine | Cosine | Cosine |
| Batch size | 64 | 256 | 64 |
| Epochs | 5 | 20 | 5 |
| Loss | Focal (γ=2) + conf BCE | Cross-entropy | Cross-entropy |
| Mixed precision | BF16/FP16 | FP32 | FP32 |

### Cascade Decision Logic

| Stage | Condition | Action | Traffic |
|---|---|---|---|
| Stage 1 – FastPath | score < 0.20 | Exit: REAL | 47.1% |
| Stage 2 – Hybrid ViT | score < 0.35 | Exit: REAL | 30.6% |
| Stage 3 – Guard MLP | G(x) = REPORT | Emit: FAKE / UNCERTAIN | 22.3% |

---

## ⚠️ Limitations

- Evaluated exclusively on CIFAKE (Stable Diffusion 1.4 + CIFAR-10 at 32×32)
- Generalization to SDXL, DALL-E 3, consistency models, and flow matching is untested
- The 32×192px upsampling may suppress artifacts that manifest only at higher native resolutions
- Future work: evaluate on [GenImage](https://github.com/GenImage-Dataset/GenImage) or DGM4; test adversarial post-processing robustness

---

## 📄 Citation

If you use EdgeAuth in your research, please cite:

```bibtex
@misc{edgeauth2024,
  title={EdgeAuth: On-Device AI-Generated Image Detection via a Calibrated Hybrid CNN-ViT Cascade},
  author={Chatterjee, Prithwiraj},
  year={2024},
  url={https://github.com/Prithwi13/EdgeAuth}
}
```

---

## 📚 References

Key works this builds on:

- Wang et al. (2020) — CNN-generated images are surprisingly easy to spot
- Frank et al. (2020) — Frequency analysis for deepfake recognition
- Corvi et al. (2023) — Intriguing properties of diffusion models
- Hu et al. (2022) — LoRA: Low-rank adaptation of large language models
- Rafailov et al. (2023) — Direct Preference Optimization
- Bird & Lotfi (2024) — CIFAKE dataset

---

<div align="center">
  <sub>Built by <a href="https://github.com/Prithwi13">Prithwiraj Chatterjee</a></sub>
</div>
