# 🎙️ Authentication using Voice Biometrics

A deep-learning project for **text-independent speaker verification** using Siamese Networks. Three distinct approaches are implemented with increasing architectural sophistication — from a custom CNN+GRU network, to VGGish transfer learning, to a full production-grade **ECAPA-TDNN** with Online Triplet Loss.

---

## 📑 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Approaches](#approaches)
  - [Approach 1 — Custom CNN + GRU Siamese Network](#approach-1--custom-cnn--gru-siamese-network)
  - [Approach 2 — VGGish-Based Siamese Network (Transfer Learning)](#approach-2--vggish-based-siamese-network-transfer-learning)
  - [Approach 3 — ECAPA-TDNN with Online Triplet Loss ⭐](#approach-3--ecapa-tdnn-with-online-triplet-loss-)
- [Dataset Structure](#dataset-structure)
- [Installation](#installation)
- [How to Run Approach 3](#how-to-run-approach-3)
- [Data Path Reference](#data-path-reference)
- [Evaluation](#evaluation)
- [Results](#results)
- [Future Work](#future-work)

---

## Overview

This project tackles the **speaker verification** problem: given two audio clips, determine whether they were uttered by the **same person**. This is a core task in voice-based biometric authentication systems.

All approaches use a **Siamese Network** architecture that maps voice recordings into a compact embedding space where:
- Embeddings of the **same speaker** are pulled **close together**
- Embeddings of **different speakers** are pushed **far apart**

---

## Project Structure

### Approach 1 — Custom CNN + GRU

| File | Contents |
|---|---|
| `preprocess.py` | Audio trimming, normalisation, resampling |
| `feature_extraction.py` | 80-band log-mel spectrogram extraction |
| `speaker_pairup.py` | On-the-fly speaker pair generation |
| `model.py` | SiameseNetwork: CNNEncoder + GRU + ProjectionHead |
| `loss.py` | Contrastive Loss (cosine distance) |
| `train.py` | Training loop with best-model checkpointing |
| `evaluate.py` | EER computation and optimal threshold |

### Approach 2 — VGGish Transfer Learning

| File | Contents |
|---|---|
| `dataset_pairup.py` | On-the-fly speaker pair generation |
| `model.py` | SiameseNet: frozen VGGish features + projection head |
| `loss.py` | Contrastive Loss |
| `train.py` | Training loop (AdamW, lr=1e-4) |
| `training_with_early_stopping.py` | Training variant with early stopping |
| `evaluate.py` | EER + similarity score distribution plot |

### Approach 3 — ECAPA-TDNN ⭐

| File | Contents |
|---|---|
| `config.yaml` | Centralised hyperparameters |
| `preprocess.py` | Waveform loading, VAD trim, torchaudio normalisation |
| `feature_extraction.py` | `LogMelExtractor` — 80-band, instance-normalised |
| `dataset_pairup.py` | `SpeakerPairDataset` — speaker-disjoint splits, SpecAugment (80%), 50/50 balance |
| `model.py` | ECAPA-TDNN backbone, SpeechBrain 1.0+ pretrained weights, dropout=0.5, expanded projection head |
| `loss.py` | `OnlineTripletLoss` (semi-hard mining), `ContrastiveLoss`, `TripletLoss` |
| `train.py` | GPU training loop — AMP, AdamW (wd=0.01), OneCycleLR, early stopping (patience=15), per-epoch JSON log |
| `evaluate.py` | EER, minDCF, DET curve, similarity distribution, t-SNE embeddings |
| `diagnose_data.py` | Data quality audit — speaker overlap, balance, length distribution |
| `diagnose_model.py` | Model audit — embedding norms, gradient flow, pretrained weight loading |
| `diagnose_training.py` | Training dynamics — loss curves, EER over epochs, overfitting detection |
| `quick_test.py` | 5-epoch ablation test to verify pipeline end-to-end |

#### Execution Order for Approach 3

| Step | Script | Command |
|---|---|---|
| 1 | `diagnose_data.py` | `python Approach3/diagnose_data.py Data --out_dir Approach3/diagnostics` |
| 2 | `diagnose_model.py` | `python Approach3/diagnose_model.py --out_dir Approach3/diagnostics` |
| 3 | `quick_test.py` *(optional)* | `python Approach3/quick_test.py Data --fix all --epochs 5` |
| 4 | `train.py` | `python Approach3/train.py Data --save_dir Approach3/checkpoints --epochs 100` |
| 5 | `diagnose_training.py` *(optional)* | `python Approach3/diagnose_training.py --log Approach3/checkpoints/train_log.csv --out_dir Approach3/diagnostics` |
| 6 | `evaluate.py` | `python Approach3/evaluate.py Approach3/checkpoints/trials.tsv Approach3/checkpoints/best_model.pth --results_dir Approach3/results` |

### Root

| File / Folder | Contents |
|---|---|
| `Data/` | Raw audio files (~1,440 utterances, 24 speakers) |
| `Approach3/checkpoints/` | `best_model.pth`, `latest_model.pth`, `train_log.csv`, `epoch_log.json` |
| `Approach3/results/` | `eer_report.txt`, `results.json`, `det_curve.png`, `similarity_distribution.png`, `tsne_embeddings.png` |
| `siamese_vggish*.pth` | Saved VGGish Siamese model weights |
| `utils.ipynb` | Utility notebook for exploration |
| `pyproject.toml` | Project dependencies |

---

## Approaches

### Approach 1 — Custom CNN + GRU Siamese Network

A fully custom architecture trained from scratch on raw mel spectrogram features.

#### Architecture

```
Input Spectrogram (1 × 80 × T)
    │
    ▼
CNNEncoder
  ├── Conv2d(1→32, 3×3) + ReLU + MaxPool2d  →  (32 × 40 × T/2)
  └── Conv2d(32→64, 3×3) + ReLU + MaxPool2d →  (64 × 20 × T/4)
    │
    ▼  reshape → (B, T/4, 64×20)
TemporalEncoder
  └── GRU(input=64×20, hidden=128) → last hidden state → (B, 128)
    │
    ▼
ProjectionHead
  └── Linear(128→128) + L2 Normalisation → 128-D unit-sphere embedding
```

#### Training Details

| Hyperparameter | Value |
|---|---|
| Epochs | 10 |
| Batch size | 16 |
| Optimiser | Adam (lr=1e-3) |
| Loss | Contrastive Loss (cosine distance, margin=1.0) |
| Features | 80-band log-mel, N_FFT=400, hop=160 |
| Max time frames | 400 (pad/crop) |

---

### Approach 2 — VGGish-Based Siamese Network (Transfer Learning)

Leverages the **pretrained VGGish** model (trained on AudioSet) as a frozen feature extractor.

#### Architecture

```
Input Spectrogram (1 × 64 × 400)
    │
    ▼
VGGishEncoder
  ├── Frozen VGGish convolutional features   →  (B, 512, H, W)
  ├── AdaptiveAvgPool2d(1, 1)               →  (B, 512, 1, 1)
  ├── Flatten                               →  (B, 512)
  └── Linear(512→128) + L2 Normalise       →  (B, 128)
```

#### Training Details

| Hyperparameter | Value |
|---|---|
| Epochs | 30 (early stopping variant available) |
| Batch size | 16 |
| Optimiser | Adam (lr=1e-4, trainable params only) |
| Loss | Contrastive Loss |
| Features | 64 mel bands, max 400 time frames (pad/crop) |
| Backbone | VGGish from `harritaylor/torchvggish` (frozen) |

---

### Approach 3 — ECAPA-TDNN with Online Triplet Loss ⭐

The most advanced approach, implementing the full **ECAPA-TDNN** architecture with **Online Triplet Loss** (semi-hard negative mining) — robust for small speaker datasets without needing a large class-weight matrix.

#### Architecture

```
Input (B, 80, T)  ← variable length, no padding required at input
    │
    ▼
ECAPA-TDNN Backbone
  ├── Conv1d(80→512, k=5)  +  BN  +  ReLU
  ├── SE-Res2Block (dilation=2)  → e1
  ├── SE-Res2Block (dilation=3)  → e2
  ├── SE-Res2Block (dilation=4)  → e3
  └── MFA: concat(e1,e2,e3) → Conv1d(1536→1536)    →  (B, 1536, T)
    │
    ▼
Attentive Statistical Pooling
  ├── Self-attention weights (softmax over T)
  ├── Weighted mean                               →  (B, 1536)
  └── Weighted std                                →  (B, 1536)
  concat → (B, 3072)
    │
    ▼
Projection Head
  ├── Linear(3072→256) + BatchNorm1d + ReLU + Dropout(0.3)
  └── Linear(256→128) + BatchNorm1d + L2 Normalise  →  (B, 128)
```

Additionally, **frame-level dropout (p=0.5)** is applied after the backbone before pooling.

**SE-Res2Block** = Squeeze-and-Excitation + Res2Net multi-scale convolution + residual connection.

#### Loss Function

```
Total Loss = OnlineTripletLoss(all_emb, speaker_labels) + 0.1 × ContrastiveLoss(emb_a, emb_b, pair_labels)
```

**OnlineTripletLoss** mines semi-hard negatives within each mini-batch:

```
L = max(d(a,p) − d(a,n) + margin, 0)    margin=0.5
```

This works well with small speaker counts (24 speakers) since no class-weight matrix is needed.

#### Training Features

| Feature | Detail |
|---|---|
| Backbone | SpeechBrain 1.0+ pretrained ECAPA-TDNN (VoxCeleb) — fine-tuned |
| Optimiser | AdamW (lr=**1e-4**, weight_decay=**0.01**) |
| LR Schedule | **OneCycleLR** — 30% linear warmup → cosine decay (per-batch) |
| Precision | Mixed precision (AMP) — **GPU required** |
| Grad clipping | max-norm=1.0 |
| Loss | OnlineTripletLoss (margin=0.5) + 0.1 × ContrastiveLoss |
| Frame dropout | p=0.5 (before pooling) |
| Augmentation | **SpecAugment 80%** — FreqMask(15) + TimeMask(20) on train split |
| Early stopping | Monitors validation EER, patience=**15** |
| Speaker split | 80/20 — **strictly speaker-disjoint** (checked at 3 levels) |
| Input | Dynamic length; padded per-batch by `collate_fn` |
| Epochs | Up to **100** |
| Batch size | **64** |
| Pairs/epoch | 10,000 (50% positive / 50% negative, enforced) |
| Pre-flight checks | Asserts speaker disjointness, embedding norms |
| Checkpoints | `best_model.pth` + `latest_model.pth` |
| Logging | `train_log.csv` + `epoch_log.json` (per-epoch metrics) |

---

## Dataset Structure

All approaches expect raw audio organised as **speaker-ID folders**:

```
Data/
└── raw/
    ├── speaker_0001/
    │   ├── utterance_01.wav
    │   ├── utterance_02.wav
    │   └── ...
    ├── speaker_0002/
    │   └── ...
    └── ...
```

> **Each speaker folder must have at least 2 utterances** for pair sampling to work.

For Approach 1 & 2, preprocessed features are saved as `.npy` files:
```
data/
├── processed/   ← resampled WAV files (from Approach1/preprocess.py)
└── features/    ← .npy log-mel spectrograms
```

For **Approach 3**, feature extraction is done on-the-fly — no pre-saved `.npy` files are needed.

---

## Installation

### With uv (recommended)

```bash
uv sync
```

### With pip

```bash
pip install torch torchvision torchaudio librosa scikit-learn numpy matplotlib \
            soundfile noisereduce pedalboard speechbrain tqdm seaborn scipy
```


---

### Dataset layout expected

Approach 3 reads raw `.wav` files directly — no preprocessing step needed.

```
Data/
├── Actor_01/
│   ├── 03-01-01-01-01-01-01.wav
│   └── ...
├── Actor_02/
│   └── ...
└── ...  (24 actors / speakers)
```

Each speaker folder must contain **≥ 2 utterances** for positive pair sampling.

---

### Step 1 — Diagnose data pipeline

Checks for speaker overlap, utterance overlap, pair balance, and audio length distribution.
Saves report + plot to `Approach3/diagnostics/`.

```powershell
python Approach3/diagnose_data.py Data --out_dir Approach3/diagnostics
```

Expected output summary:
```
Total files   : 1440  |  Total speakers: 24
Speaker overlap  : ✅ PASS (0 shared speakers)
Utterance overlap: ✅ PASS
Pair balance     : ✅ PASS (~50% positive)
```

---

### Step 2 — Diagnose model architecture

Verifies embedding norms, gradient flow, and SpeechBrain pretrained weight loading.

```powershell
# First run (no checkpoint yet)
python Approach3/diagnose_model.py --out_dir Approach3/diagnostics

# After training (with checkpoint)
python Approach3/diagnose_model.py --checkpoint Approach3/checkpoints/best_model.pth --out_dir Approach3/diagnostics
```

---

### Step 3 — Quick 5-epoch ablation test _(optional but recommended)_

Runs a short 5-epoch training cycle to verify the pipeline end-to-end before committing to full training.

```powershell
python Approach3/quick_test.py Data --fix all --epochs 5
```

Expected result (RAVDESS, 24 speakers):
```
Epoch 5/5 — loss=0.0235  val_eer=13.25%
RESULT: final_eer = 13.25%  (target < 20%)
✅ PASS — proceed to full training
```

`--fix` options: `data` | `arcface` | `pretrained` | `all`

---

### Step 4 — Full training

```powershell
python Approach3/train.py Data --save_dir Approach3/checkpoints --epochs 100 --batch_size 64 --lr 1e-4
```

**All arguments:**

| Argument | Default | Description |
|---|---|---|
| `data_dir` | *(required)* | Root dir — speaker subdirs with `.wav` files |
| `--save_dir` | `checkpoints` | Saves `best_model.pth`, `latest_model.pth`, `train_log.csv`, `epoch_log.json` |
| `--epochs` | `100` | Maximum training epochs |
| `--batch_size` | `64` | Batch size (minimum 32 recommended) |
| `--lr` | `1e-4` | Initial learning rate (for fine-tuning pretrained backbone) |
| `--triplet_margin` | `0.5` | Triplet loss margin |
| `--resume` | `None` | Path to a checkpoint to resume from |
| `--seed` | `42` | Global random seed |

Training prints per-epoch metrics and saves a CSV + JSON log:
```
[Epoch 001/100] loss=0.4821  val_eer=22.33%  val_loss=0.3012  lr=3.33e-05
[Epoch 013/100] loss=0.1953  val_eer=18.15%  val_loss=0.2764  lr=8.67e-05
  ✓ New best EER: 18.15%
```

**Resume from checkpoint:**
```powershell
python Approach3/train.py Data --save_dir Approach3/checkpoints --epochs 100 --resume Approach3/checkpoints/latest_model.pth
```

---

### Step 5 — Diagnose training dynamics _(after training)_

Plots loss curves, EER over epochs, LR schedule, and detects overfitting.

```powershell
python Approach3/diagnose_training.py \
  --log Approach3/checkpoints/train_log.csv \
  --checkpoint Approach3/checkpoints/best_model.pth \
  --out_dir Approach3/diagnostics
```

---

### Step 6 — Evaluate

First generate a trials file (tab-separated: `path_a \t path_b \t label`, 1=same speaker, 0=different):

```powershell
# Auto-generate trials.tsv from your Data/ directory
python -c "
import random, pathlib, itertools
data = pathlib.Path('Data'); rng = random.Random(42)
spk_files = {s.name: [str(f) for ext in ('*.wav','*.flac') for f in s.glob(ext)] for s in data.iterdir() if s.is_dir()}
pairs = [(a,b,1) for spk,fs in spk_files.items() for a,b in itertools.combinations(fs[:10],2)]
negs = []
while len(negs)<len(pairs):
    s1,s2=rng.sample(list(spk_files),2); negs.append((rng.choice(spk_files[s1]),rng.choice(spk_files[s2]),0))
all_p=pairs+negs; rng.shuffle(all_p)
open('Approach3/checkpoints/trials.tsv','w').writelines(f'{a}	{b}	{l}
' for a,b,l in all_p)
print(f'{len(all_p)} pairs written')
"

python Approach3/evaluate.py Approach3/checkpoints/trials.tsv Approach3/checkpoints/best_model.pth --results_dir Approach3/results
```

**All arguments:**

| Argument | Default | Description |
|---|---|---|
| `trials_file` | *(required)* | Path to the TSV trial pairs file — **no header row** |
| `model_path` | *(required)* | Path to `best_model.pth` checkpoint |
| `--results_dir` | `results` | Directory for output artefacts |
| `--n_mels` | `80` | Must match training config |
| `--channels` | `512` | Must match training config |
| `--embedding_dim` | `128` | Must match training config |
| `--p_target` | `0.01` | Prior probability for minDCF |

**Output artefacts** (saved to `Approach3/results/`):

| File | Description |
|---|---|
| `eer_report.txt` | EER %, minDCF, optimal threshold, trial count |
| `results.json` | Machine-readable metrics (always up-to-date) |
| `det_curve.png` | Detection Error Tradeoff (DET) curve |
| `similarity_distribution.png` | Histogram: same vs different speaker scores |
| `tsne_embeddings.png` | t-SNE of speaker embeddings |

---

## Data Path Reference

| Approach | Script | Where to set the data path |
|---|---|---|
| **Approach 1** | `preprocess.py` | Hardcoded `RAW_DATA_DIR` — edit line 6 |
| **Approach 1** | `feature_extraction.py` | Hardcoded `PROCESSED_DATA_DIR` — edit line 5-6 |
| **Approach 1** | `train.py` | Hardcoded `FEATURES_DATA_DIR` — edit line 72 |
| **Approach 1** | `evaluate.py` | Hardcoded `FEATURES_DATA_DIR` — edit line 77 |
| **Approach 2** | `train.py` | Hardcoded in `train()` function — edit line 47 |
| **Approach 2** | `evaluate.py` | Hardcoded in `run_evaluation()` — edit line 18 |
| **Approach 3** | `train.py` | ✅ CLI argument `data_dir` — no edits needed |
| **Approach 3** | `evaluate.py` | ✅ CLI arguments `trials_file` + `model_path` — no edits needed |

> ⚠️ **Approach 1 and 2 have Linux-style hardcoded paths** (`/home/rohithkaki/...`). Before running these, update the path constants at the top of each script to point to your local `Data/` directory.

**For Approach 1 & 2 on Windows, replace paths like:**
```python
# Old (Linux path)
FEATURES_DATA_DIR = "/home/rohithkaki/Voice_Biometrics/data/features"

# New (Windows path — use raw string or forward slashes)
FEATURES_DATA_DIR = r"D:\Authentication_using_Voice_Biometrics-main\Data\features"
```

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **EER** (Equal Error Rate) | Point where FAR = FRR. **Lower is better.** |
| **minDCF** | Minimum Detection Cost Function. **Lower is better.** |
| **Optimal Threshold** | Cosine similarity threshold at the EER operating point |

---

## Results

| | Approach 1 (CNN+GRU) | Approach 2 (VGGish) | Approach 3 (ECAPA-TDNN) |
|---|---|---|---|
| Backbone | Trained from scratch | Pretrained VGGish (frozen) | **Pretrained SpeechBrain ECAPA-TDNN** |
| Input features | 80-band log-mel | 64-band mel | 80-band log-mel (instance norm) |
| Input handling | Fixed 400 frames | Fixed 400 frames | **Dynamic length** |
| Loss | Contrastive | Contrastive | **OnlineTripletLoss (margin=0.5) + Contrastive** |
| Augmentation | None | None | **SpecAugment 80%** (FreqMask+TimeMask) |
| GPU required | No | No | **Yes (CUDA)** |
| Epochs | 10 | 30 | Up to 100 (early stopping, patience=15) |
| Embedding dim | 128 | 128 | 128 |
| Val EER (15 epochs) | — | — | **18.15%** |
| **Eval EER** (2160 trials) | — | — | **✅ 5.83%** |
| **minDCF** (p=0.01) | — | — | **0.2731** |
| **Threshold** | — | — | **0.699** |
| Metrics | EER | EER | **EER + minDCF + DET curve** |
| Plots | — | Score distribution | DET, score dist., t-SNE |

Training loss curve: [`loss_plot.png`](loss_plot.png)  
Result images: [`Results_images/`](Results_images/)

---

## Future Work

| Feature | Status |
|---|---|
| Pretrained backbone (SpeechBrain 1.0+) | ✅ Done |
| Speaker-disjoint splits | ✅ Enforced at 3 levels |
| Online Triplet Loss (semi-hard mining) | ✅ Done |
| SpecAugment data augmentation (80%) | ✅ Done |
| OneCycleLR scheduler | ✅ Done |
| Per-epoch JSON logging | ✅ `epoch_log.json` |
| Dropout regularisation (p=0.5) | ✅ Frame + projection head |
| Hard negative mining (embedding pool) | 🔲 Planned |
| Dataset scale | 🔲 VoxCeleb1/2 (1000+ speakers) |
| SubCenterArcFace | 🔲 Planned |
| VoxCeleb1-H benchmark evaluation | 🔲 Planned |

---

## References

- [VGGish](https://github.com/harritaylor/torchvggish) — PyTorch port of Google's VGGish audio feature extractor  
- [ECAPA-TDNN paper](https://arxiv.org/abs/2005.07143) — Emphasized Channel Attention, Propagation and Aggregation in TDNN  
- [ArcFace paper](https://arxiv.org/abs/1801.07698) — Additive Angular Margin Loss for Deep Face Recognition  
- [Speaker Verification – Short Technical Notes](Speaker%20Verification%20%E2%80%93%20Short%20Technical%20Notes.pdf) — included in this repository  
- Hadsell et al. (2006) — Contrastive Loss ([paper](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf))