<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0C1A35,100:0EB39E&height=200&section=header&text=LSB%20Steganalysis&fontSize=52&fontColor=ffffff&fontAlignY=38&desc=Deep%20Learning%20%E2%80%94%20Detect%20the%20Invisible&descAlignY=58&descSize=18" width="100%"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Kaggle](https://img.shields.io/badge/Kaggle-BOSSbase-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://kaggle.com)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research-F59E0B?style=for-the-badge)]()

<br/>

> **Can a neural network detect a single flipped bit hidden inside 65,000 pixels?**
> This project answers that question — using a dual-backbone CNN with gated attention and a trainable SRM adapter.

<br/>

</div>

---

## 📖 Table of Contents

- [🔍 What Is Steganography?](#-what-is-steganography)
- [🎯 Project Goal](#-project-goal)
- [📦 Dataset](#-dataset)
- [🏗️ Architecture](#️-architecture)
- [⚗️ The Critical Data Fix](#️-the-critical-data-fix)
- [🔬 Model Components](#-model-components)
- [🎓 Training Strategy](#-training-strategy)
- [🛡️ Regularisation](#️-regularisation)
- [📊 Evaluation](#-evaluation)
- [🚀 Getting Started](#-getting-started)
- [📁 Repository Structure](#-repository-structure)
- [📈 Results](#-results)
- [🧠 Deep Learning Q&A](#-deep-learning-qa)
- [⚠️ Lessons Learned](#️-lessons-learned)
- [📜 License](#-license)

---

## 🔍 What Is Steganography?

**Steganography** is the art of hiding information inside ordinary-looking data — in this case, inside images. The method used here is **LSB (Least Significant Bit)** embedding:

```
Original pixel:  10110110  (182)
After embedding: 10110111  (183)  ← 1-bit change, invisible to the human eye
```

At **0.4 bits per pixel**, roughly **40% of all pixels** are silently modified — each by at most ±1 gray level. The result is visually identical to the original.

**Steganalysis** is the reverse challenge: training a model to detect whether an image carries hidden data, despite having no visual cue to go on.

---

## 🎯 Project Goal

| | |
|---|---|
| **Task** | Binary image classification: **Cover** vs **Stego** |
| **Signal** | ±1 gray level change in ≤40% of pixels |
| **Difficulty** | Signal-to-noise ratio is near zero for standard CNNs |
| **Primary metric** | **AUC-ROC** (threshold-free, robust on balanced data) |
| **Dataset** | BOSSbase 1.01 — 10,000 grayscale images, 256×256 px |

---

## 📦 Dataset

**BOSSbase 1.01** is the standard benchmark for steganography research, hosted on Kaggle as [`lijiyu/bossbase`](https://kaggle.com/datasets/lijiyu/bossbase).

```
boss_256_0.4/
├── cover/     ← 10,000 clean grayscale images (256×256)
└── stego/     ← 10,000 LSB-embedded counterparts (0.4 bpp)

boss_256_0.4_test/
├── cover/
└── stego/
```

> ⚠️ **Critical caveat:** The Kaggle dataset stores images as **JPEG** — a lossy format. JPEG re-compression introduces quantization noise of ±1 to ±10 gray levels, which is orders of magnitude larger than the ±1 LSB signal. The stego images in the raw dataset are **effectively destroyed**.
>
> This project fixes this with a lossless PNG regeneration step — see [The Critical Data Fix](#️-the-critical-data-fix).

---

## 🏗️ Architecture

The full model pipeline — **StegDetectorV3**:

```
┌─────────────────────────────────────────────────────────────────┐
│                       Input Image (256×256)                     │
└────────────────────────────┬────────────────────────────────────┘
                             │  Extract N random patches (48×48)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              ① Trainable SRM Adapter                            │
│   5×5 depthwise conv (SRM-init) → abs() → BatchNorm → 1×1 conv │
│   Amplifies ±1 LSB residuals; learns the optimal high-pass filter│
└────────────────────────────┬────────────────────────────────────┘
                             │  Noise residual map [0, 1]
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              ② Dual-Backbone Feature Extractor                  │
│                                                                 │
│   ConvNeXt-Tiny ──► proj (384-dim) ─┐                          │
│                                      ├─► concat → 768-dim       │
│   DenseNet-121  ──► proj (384-dim) ─┘                          │
│                                                                 │
│   Phase 1: fully frozen  │  Phase 2: late blocks  │  Phase 3: all│
└────────────────────────────┬────────────────────────────────────┘
                             │  Feature bag (B × N × 768)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              ③ Gated Attention MIL                              │
│                                                                 │
│   aₖ = softmax( wᵀ · (tanh(Vhₖ) × sigmoid(Uhₖ)) )            │
│   bag = Σ aₖ · hₖ                                              │
│                                                                 │
│   Sigmoid gate suppresses clean patches; concentrates on stego  │
└────────────────────────────┬────────────────────────────────────┘
                             │  Single 768-dim bag embedding
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              ④ Classifier MLP                                   │
│   768 → 256 (LayerNorm + GELU + Dropout 0.4)                   │
│   256 → 64  (GELU)                                             │
│    64 → 2   (logits)  ──►  Cover / Stego                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## ⚗️ The Critical Data Fix

This was the single most important discovery in the entire project.

### ❌ The Problem

```python
# What the broken Kaggle dataset did:
cover_jpeg = load("cover.jpg")          # JPEG decompressed — OK
stego_pixels = lsb_embed(cover_jpeg)    # Embed ±1 in pixels — OK
save(stego_pixels, "stego.jpg")         # ← JPEG re-compression DESTROYS embedding!

# Diagnostic:
diff(cover, stego) → Unique diffs: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#                                   ^^^^^^^^ JPEG noise, NOT steganography
```

The model trained for 100+ combined epochs across earlier versions learning **random JPEG quantization noise** — completely unrelated to steganography. AUC stayed at **0.50** (random chance).

### ✅ The Fix

```python
# What this project does:
cover_jpeg  = load("cover.jpg")             # Load JPEG
cover_clean = decompress_to_pixels(cover)   # Raw pixels — no JPEG
stego_clean = lsb_embed(cover_clean)        # Embed ±1

save(cover_clean, "cover.png", lossless=True)  # ← PNG, lossless
save(stego_clean, "stego.png", lossless=True)  # ← PNG, lossless

# Verification:
diff(cover, stego) → Unique diffs: [0, 1]
#                    Max diff: 1   Mean diff: 0.199 ✓
```

**Rule:** LSB steganography can only be trained on **lossless image formats**. JPEG is fundamentally incompatible.

---

## 🔬 Model Components

### ① Trainable SRM Adapter

The **Spatial Rich Model (SRM)** is a classical steganalysis tool — a set of hand-crafted high-pass filters that amplify pixel residuals. This project makes them **learnable**:

```python
class TrainableSRMAdapter(nn.Module):
    def __init__(self):
        self.dw = nn.Conv2d(3, 3, kernel_size=5, padding=2, groups=3)  # SRM-initialized
        self.bn = nn.BatchNorm2d(3)
        self.pw = nn.Conv2d(3, 3, kernel_size=1)                        # learned mix

    def forward(self, x):
        r = torch.abs(self.bn(self.dw(x * 255.0)))   # abs() because residuals are ±1
        return torch.clamp(self.pw(r), 0, 3) / 3.0
```

| Design choice | Reason |
|---|---|
| `abs()` instead of `ReLU` | LSB residuals are symmetric ±1; ReLU would discard half the signal |
| SRM kernel initialization | Starts at the known-good solution; free to adapt from there |
| `x * 255.0` scaling | Amplifies the ±1 residual before filtering |
| Learnable (not fixed) | Optimal filter for this specific dataset may differ from classical SRM |

---

### ② Dual-Backbone Feature Extractor

Two pretrained backbones are fused to capture **complementary aspects** of the LSB residual:

| Backbone | Strength | Why it helps |
|---|---|---|
| **ConvNeXt-Tiny** | Large 7×7 depthwise kernels | Global texture correlations across the patch |
| **DenseNet-121** | Dense skip connections | Multi-scale feature reuse from all depths |

Each backbone's output is projected to 384-dim via `Linear → LayerNorm → GELU → Dropout(0.2)`, then concatenated to a **768-dim fused vector** per patch.

---

### ③ Gated Attention MIL

**Multiple Instance Learning (MIL)** treats each image as a *bag* of patches. Most patches are clean and uninformative — only a few carry the LSB signal. The attention mechanism learns which patches to focus on.

**Standard attention** (Ilse et al. 2018):
```
aₖ = softmax( wᵀ · tanh(Vhₖ) )
```

**Gated attention** (this project):
```
aₖ = softmax( wᵀ · (tanh(Vhₖ) × sigmoid(Uhₖ)) )
```

The sigmoid branch acts as a **gate** — it can zero out uninformative patches entirely, whereas standard attention only re-weights them. This is critical when the signal is sparse.

---

## 🎓 Training Strategy

Training uses a **3-phase progressive fine-tuning schedule** to avoid catastrophic forgetting of pretrained ImageNet knowledge:

```
Epochs 1–5   │ Phase 1 │ ALL backbone layers FROZEN
             │         │ Only MIL head + SRM adapter train (lr = 1e-3)
             │         │ Warms up randomly-initialized components safely
─────────────┼─────────┼──────────────────────────────────────────────
Epochs 6–20  │ Phase 2 │ Late blocks UNFROZEN
             │         │ ConvNeXt stages[2-3], DenseBlock 3-4 (lr = 1e-4)
             │         │ Re-specialises high-level features for noise residuals
─────────────┼─────────┼──────────────────────────────────────────────
Epochs 21–40 │ Phase 3 │ ALL layers UNFROZEN with layer-wise LR decay
             │         │ Early blocks:  lr × 0.10  (near-frozen)
             │         │ Mid blocks:    lr × 0.50
             │         │ Late blocks:   lr × 1.00  (lr = 2e-5)
             │         │ Head + SRM:    lr = 1e-3
```

**Early stopping** with patience = 5 per phase:
- Phase 2 plateau → jumps to Phase 3 early (no wasted epochs)
- Phase 3 plateau → full stop, loads best checkpoint

**Optimizer:** `AdamW` with decoupled weight decay (`1e-4`). Cosine LR annealing within each phase.

---

## 🛡️ Regularisation

Every regularisation technique was chosen for a specific reason — nothing is boilerplate:

| Technique | Where | Purpose |
|---|---|---|
| **Batch Normalisation** | SRM adapter | Stabilises tiny noise residual distributions across batch |
| **Dropout 0.4** | Classifier head | Prevents co-adaptation of neurons on spurious features |
| **Dropout 0.25** | Gated attention | Avoids over-reliance on a single patch |
| **Label Smoothing 0.1** | CrossEntropyLoss | Prevents overconfident outputs; breaks the 0.693 loss plateau |
| **Mixup α=0.2** | Training loop | Interpolates bag pairs; further disrupts the 0.693 collapse |
| **Gradient Clipping 1.0** | All phases | Prevents exploding gradients during backbone fine-tuning |
| **Weight Decay 1e-4** | AdamW | L2 regularisation — penalises large weights globally |
| **Layer-wise LR decay** | Phase 3 | Early layers get LR × 0.1 — prevents catastrophic forgetting |

> 💡 **The 0.693 plateau:** `CrossEntropyLoss(0.5, 0.5) = ln(2) ≈ 0.693`. When a model predicts 50/50 for every sample, loss collapses here. Label smoothing + Mixup break this symmetry.

---

## 📊 Evaluation

### Metrics

| Metric | Role |
|---|---|
| **AUC-ROC** | Primary metric — threshold-free, measures ranking quality across all operating points |
| **Accuracy** | Overall correctness on balanced test set |
| **Precision / Recall / F1** | Per-class breakdown; F1 penalises both false positives and false negatives |
| **Confusion Matrix** | Shows cover-classified-as-stego vs stego-missed (false negatives) |

### Visualisations

The notebook generates four diagnostic plots:

```
📈 training_curves.png   — loss, accuracy, AUC per epoch with phase shading
📉 evaluation.png        — ROC curve + confusion matrix + score distribution
🎨 learned_srm.png       — original SRM kernels vs what the adapter learned
👁️  attention.png         — top-K attended patches per image (border = attention weight)
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch torchvision timm Pillow numpy scikit-learn matplotlib tqdm
```

### Running on Kaggle (recommended)

1. Add the dataset: `lijiyu/bossbase`
2. Enable GPU accelerator (T4 or P100)
3. Run all cells in order — **Cell 4 must run first** (lossless data regeneration)

### Running locally

```bash
git clone https://github.com/your-username/lsb-steganalysis
cd lsb-steganalysis

# Update paths in Cell 3 (CFG dict) to point to your local dataset
# Then run the notebook cell by cell
jupyter notebook steganography.ipynb
```

### Key configuration (Cell 3)

```python
CFG = {
    'patch_size'      : 48,      # px — patch extracted per bag
    'num_patches'     : 32,      # patches per image at train time
    'num_patches_test': 50,      # larger bag at test time for stability
    'backbone_a'      : 'convnext_tiny',
    'backbone_b'      : 'densenet121',
    'fused_dim'       : 768,     # concatenated feature dimension
    'phase1_end'      : 5,       # epoch at which Phase 1 ends
    'phase2_end'      : 20,
    'phase3_end'      : 40,
    'lr_head'         : 1e-3,
    'lr_finetune'     : 1e-4,
    'lr_fullfinetune' : 2e-5,
    'mixup_alpha'     : 0.2,
    'label_smoothing' : 0.1,
    'batch_size'      : 8,
}
```

---

## 📁 Repository Structure

```
📦 lsb-steganalysis/
│
├── 📓 steganography.ipynb        ← Final notebook (StegDetectorV3)
│
├── 📂 drafts/
│   ├── 📓 steganography_v1.ipynb ← EfficientNet-B0 attempt (AUC 0.50)
│   └── 📓 steganography_v2.ipynb ← SRNet attempt (dataset still broken)
│
├── 📂 scripts/
│   └── 🐍 Steganography.py       ← Standalone LSB embed/extract utility
│
├── 📊 results/
│   ├── 🖼️  training_curves.png
│   ├── 🖼️  evaluation.png
│   ├── 🖼️  learned_srm.png
│   ├── 🖼️  attention.png
│   └── 📄 metrics.json
│
├── 📊 Steganalysis_Soutenance.pptx  ← Presentation slides
└── 📄 README.md
```

---

## 📈 Results

### Version Comparison

| | **v1** | **v2** | **Final** ✓ |
|---|:---:|:---:|:---:|
| Data format | JPEG ✗ | JPEG ✗ | Lossless PNG ✓ |
| Backbone | EfficientNet-B0 | Shallow SRNet | ConvNeXt-T + DenseNet-121 |
| SRM strategy | Post-CNN (broken) | Fixed, pre-CNN | Trainable adapter |
| Attention | Standard MIL | Standard MIL | Gated MIL |
| Training phases | 2 | 2 | 3 + Early Stopping |
| Mixup | ✗ | ✗ | ✓ (α=0.2) |
| **AUC-ROC** | **0.50** | **≈0.50** | **Meaningful ↑** |

### AUC-ROC Interpretation Guide

```
≥ 0.90  ██████████  Excellent
≥ 0.85  █████████░  Strong
≥ 0.80  ████████░░  Good
≥ 0.70  ███████░░░  Acceptable
≥ 0.60  █████░░░░░  Weak
  0.50  ████░░░░░░  Random chance (coin flip)
```

> The first two versions both scored **0.50** — the model was learning random JPEG noise, not steganography. The lossless PNG fix was the turning point.

---

## 🧠 Deep Learning Q&A

Answers to the core deep learning questions relevant to this project:

<details>
<summary><strong>Q: Why CNN over MLP or RNN?</strong></summary>

Images have **spatial locality** — nearby pixels share structure. CNNs exploit this via weight sharing and local receptive fields, making them orders of magnitude more parameter-efficient than MLPs for image tasks. RNNs and LSTMs are designed for sequences (time series, text). Transformers suit long-range dependencies but require far more data. For small-patch steganalysis where the signal is local and high-frequency, **CNNs (ConvNeXt, DenseNet) are the natural choice**.

</details>

<details>
<summary><strong>Q: How did you handle gradient vanishing / exploding?</strong></summary>

**Exploding:** `gradient clipping (max norm = 1.0)` applied in all three training phases. Prevents destructive parameter updates when fine-tuning deep pretrained models.

**Vanishing:** Multiple complementary strategies:
- **GELU activations** — non-zero gradient for negative inputs (unlike ReLU's dead zone)
- **LayerNorm** in the classifier — stabilises activations without the batch dependency of BatchNorm
- **BatchNorm** in the SRM adapter — normalises residual signal distributions
- **DenseNet skip connections** — native gradient highway through all depths
- **Layer-wise LR decay** in Phase 3 — early layers receive LR × 0.1, avoiding destructive updates while allowing gradual adaptation

</details>

<details>
<summary><strong>Q: What is the role of BatchNorm, Dropout, and Early Stopping?</strong></summary>

**BatchNorm (SRM adapter):** LSB residuals are extremely small — normalising them across the batch prevents gradient instability and helps the model learn consistent residual patterns regardless of image brightness or contrast.

**Dropout 0.4 (classifier):** Prevents the classifier head from memorising specific patch combinations rather than learning generalised detection features.

**Dropout 0.25 (attention):** Prevents the MIL aggregator from always attending to the same patch positions regardless of content.

**Early Stopping:** Monitors AUC on the test set with patience = 5 per phase. In Phase 2, plateau triggers a jump to Phase 3 (instead of stopping entirely — the additional unfreezing often breaks the plateau). In Phase 3, plateau triggers full termination and restoration of the best checkpoint.

</details>

<details>
<summary><strong>Q: Why CrossEntropyLoss + AdamW? Justify both choices.</strong></summary>

**CrossEntropyLoss with label_smoothing=0.1:** Standard choice for classification. Label smoothing softens hard targets from {0, 1} to {0.05, 0.95}, which:
1. Prevents the model from becoming overconfident (a known calibration failure)
2. Breaks the **0.693 plateau** — when a model predicts 50/50, `CE(0.5, 0.5) = ln(2) ≈ 0.693`; smoothed labels make this plateau shallower and easier to escape

**AdamW (not Adam):** In standard Adam, weight decay is coupled with the adaptive learning rate, which diminishes its regularisation effect. AdamW **decouples** weight decay — L2 penalty is applied directly to the weights, independently of the gradient scaling. This is the theoretically correct formulation and consistently outperforms Adam on fine-tuning tasks.

</details>

---

## ⚠️ Lessons Learned

The three-version journey taught more than just architecture choices:

```
🔴 Version 1 — Wrong architecture
   ├── EfficientNet-B0 suppresses high-freq noise (that's ImageNet's job)
   ├── SRM placed after CNN — too late, signal already destroyed
   └── AUC: 0.50

🟡 Version 2 — Right architecture, wrong data
   ├── Replaced backbone with shallow SRNet from scratch ✓
   ├── Moved SRM before CNN ✓
   ├── But dataset still JPEG — embedding completely corrupted
   └── AUC: ≈0.50

🟢 Final — Everything fixed
   ├── Lossless PNG regeneration (THE critical fix) ✓
   ├── Trainable SRM adapter ✓
   ├── Dual backbone (ConvNeXt-T + DenseNet-121) ✓
   ├── Gated Attention MIL ✓
   └── AUC: meaningful signal
```

**The #1 lesson:** Verify your data signal before writing a single line of model code.
A simple `np.unique(np.abs(cover - stego))` would have revealed the JPEG corruption immediately. If that array shows anything other than `[0, 1]`, your LSB dataset is broken.

---

## 🐍 Standalone Steganography Utility

The repository also includes a standalone Python script for embedding and extracting LSB messages:

```bash
# Embed a message
python Steganography.py embed input.jpg output.png "secret message" 42

# Extract a message  
python Steganography.py extract output.png 42
```

> **Note:** Output must always be `.png`. The seed (42 above) acts as the key — without the correct seed, extraction is impossible.

---

## 📜 License

This project is released under the [MIT License](LICENSE). Free to use, modify, and distribute with attribution.

---

<div align="center">

<br/>

**Built for the Deep Learning module — M. Abdallah Khemais**

<br/>

[![Made with PyTorch](https://img.shields.io/badge/Made%20with-PyTorch-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org)
[![BOSSbase Dataset](https://img.shields.io/badge/Dataset-BOSSbase%201.01-20BEFF?style=flat-square&logo=kaggle)](https://kaggle.com)
[![Model](https://img.shields.io/badge/Model-StegDetectorV3-0EB39E?style=flat-square)]()

<br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0EB39E,100:0C1A35&height=100&section=footer" width="100%"/>

</div>
