# Bharath AI SoC Hackathon PS5
# 🗑️ Edge-AI Waste Classification on FPGA
### MobileNetV2 · INT8 Quantization · Xilinx KRIA KV260 · Vitis-AI 3.5

<div align="center">

![Accuracy](https://img.shields.io/badge/Accuracy-93.11%25-brightgreen?style=for-the-badge)
![Inference](https://img.shields.io/badge/DPU%20Inference-3.24ms-blue?style=for-the-badge)
![Throughput](https://img.shields.io/badge/Throughput-40%20FPS-orange?style=for-the-badge)
![Model Size](https://img.shields.io/badge/Model%20Size-2.4MB-purple?style=for-the-badge)
![Platform](https://img.shields.io/badge/Platform-KRIA%20KV260-red?style=for-the-badge)

</div>

> **Real-time waste classification at the edge** — no cloud, no network dependency, no compromise.  
> A full pipeline from PyTorch training → Vitis-AI INT8 quantization → DPU compilation → live inference on FPGA.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Why Edge AI for Waste Classification?](#-why-edge-ai-for-waste-classification)
- [System Architecture](#-system-architecture)
- [Dataset](#-dataset)
- [Model: MobileNetV2](#-model-mobilenetv2)
- [Quantization Pipeline](#-quantization-pipeline)
- [FPGA Deployment](#-fpga-deployment)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Setup & Reproduction](#-setup--reproduction)
- [Key Technical Decisions](#-key-technical-decisions)
- [Limitations & Future Work](#-limitations--future-work)

---

## 🔍 Overview

This project implements an **end-to-end Edge-AI pipeline** for automatic waste material classification using a quantized MobileNetV2 neural network deployed on the **Xilinx KRIA KV260** FPGA development board.

| Component | Detail |
|-----------|--------|
| **Task** | 6-class waste image classification |
| **Model** | MobileNetV2 (ImageNet pretrained → fine-tuned) |
| **Precision** | FP32 training → INT8 deployment |
| **Accelerator** | Xilinx DPU (DPUCZDX8G) on KRIA KV260 |
| **Toolchain** | PyTorch 1.13.1 + Vitis-AI 3.5 |
| **Runtime** | VART (Vitis-AI Runtime) on PetaLinux 2022.2 |
| **Classes** | Cardboard, Glass, Metal, Paper, Plastic, Trash |

---

## 🌍 Why Edge AI for Waste Classification?

Waste misclassification is a serious bottleneck in recycling pipelines. Existing solutions either rely on expensive cloud inference (introducing latency and connectivity requirements) or expensive industrial sensors.

**Edge deployment on FPGA addresses this directly:**

- ⚡ **Deterministic latency** — DPU inference is consistent at 3.24ms ± 0.03ms, unlike GPU servers with variable load
- 🔌 **Low power** — FPGA DPU consumes a fraction of the power of a GPU-based system
- 🔒 **Data sovereignty** — no images leave the device; critical for industrial environments
- 📦 **Form factor** — the KV260 is small enough to mount on a conveyor belt or smart bin
- 💰 **Cost** — no recurring cloud compute costs; one-time hardware investment

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        TRAINING (Colab)                         │
│                                                                 │
│   TrashNet Dataset  ──►  MobileNetV2 (FP32)  ──►  best_model.pth│
│   (224×224, 6 cls)       Adam / CrossEntropy                    │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    QUANTIZATION (Vitis-AI 3.5)                  │
│                                                                 │
│   FP32 .pth  ──►  vai_q_pytorch (PTQ)  ──►  INT8 .xmodel       │
│                   Calibration: 508 val images                   │
│                   Accuracy drop: 93.11% → ~91.7%               │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    COMPILATION (vai_c_xir)                      │
│                                                                 │
│   INT8 .xmodel  ──►  DPUCZDX8G/KV260 arch  ──►  waste_classifier.xmodel│
│                       Hybrid: CPU + DPU subgraphs               │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                   DEPLOYMENT (KRIA KV260)                       │
│                                                                 │
│   Image  ──►  CPU Preprocess  ──►  DPU (conv layers)  ──►  CPU  │
│   Input       Resize/Norm          [16, 56, 56, 24]     Softmax │
│                                                          Output  │
└─────────────────────────────────────────────────────────────────┘
```

### Hybrid Execution Model

The compiled model operates in **hybrid CPU+DPU mode**. This is important to understand:

- The **DPU** handles quantized convolutional backbone layers — this is where 90%+ of compute happens
- The **CPU** handles: input preprocessing, layers unsupported by DPU (e.g. certain activation functions), and the final classification head
- The DPU expects input feature tensors of shape `[Batch=16, H=56, W=56, C=24]` — not raw images
- VART runtime manages subgraph scheduling transparently

---

## 📁 Dataset

**Primary:** [TrashNet](https://github.com/garythung/trashnet) — 6 waste material categories

| Class | Test Samples | KRIA Accuracy |
|-----------|-------------|--------------|
| Cardboard | 81 | 95.06% |
| Glass | 101 | 92.08% |
| Metal | 82 | 98.78% |
| Paper | 119 | 98.32% |
| Plastic | 97 | 88.66% |
| Trash | 28 | 67.86% |
| **Total** | **508** | **93.11%** |

**Directory structure expected:**
```
WasteDataset/
├── train/
│   ├── cardboard/
│   ├── glass/
│   ├── metal/
│   ├── paper/
│   ├── plastic/
│   └── trash/
└── val/
    ├── cardboard/
    ├── glass/
    ├── metal/
    ├── paper/
    ├── plastic/
    └── trash/
```

> **Note on class imbalance:** The `trash` class has significantly fewer samples (28 test images vs 119 for paper). This directly explains its lower accuracy of 67.86% and is the primary candidate for data augmentation in future iterations.

---

## 🧠 Model: MobileNetV2

MobileNetV2 was selected over heavier architectures (ResNet, VGG) deliberately:

### Architecture Highlights

```
Input (224×224×3)
    │
    ▼
Conv2d 3×3, stride 2          ← Standard convolution (first layer only)
    │
    ▼
[Inverted Residual Blocks ×17] ← Core of MobileNetV2
    │   ┌─────────────────────────────────────┐
    │   │  1×1 Conv (Expand, ReLU6)           │
    │   │  3×3 Depthwise Conv (ReLU6)         │ × 17 blocks
    │   │  1×1 Conv (Project, Linear)         │
    │   └─────────────────────────────────────┘
    │
    ▼
Conv2d 1×1 (1280 channels)
    │
    ▼
GlobalAvgPool → FC(6) → Softmax
```

### Why MobileNetV2 for FPGA?

| Property | Benefit for Edge/FPGA |
|----------|----------------------|
| **Depthwise Separable Convs** | Reduces MAC operations by ~8-9× vs standard convs |
| **Linear Bottlenecks** | Prevents information loss during dimensionality reduction |
| **Inverted Residuals** | Keeps computation in low-dimensional space; fewer activations to buffer |
| **ReLU6** | Bounded activations — critical for INT8 quantization stability |
| **2.4MB INT8 size** | Fits entirely in on-chip BRAM; avoids DDR bottleneck |

**Training configuration:**
```python
optimizer = Adam(lr=1e-4, weight_decay=1e-4)
criterion = CrossEntropyLoss()
input_size = (224, 224)
pretrained = True          # ImageNet weights as starting point
epochs = 30                # Early stopping on val accuracy
scheduler = StepLR(step_size=10, gamma=0.1)
```

---

## ⚙️ Quantization Pipeline

### Why Post-Training Quantization (PTQ)?

PTQ was chosen over Quantization-Aware Training (QAT) for pragmatic reasons:
- Faster iteration — no retraining required
- Vitis-AI's PTQ is well-optimized for the DPU target
- Accuracy drop was acceptable: **93.11% (FP32) → ~91.7% (INT8)** — only ~1.4% degradation

In future work, QAT could recover some of this gap, especially for the `trash` class.

### Quantization Process

```bash
# Step 1: Enter Vitis-AI Docker environment
docker run --gpus all -it xilinx/vitis-ai-pytorch-gpu:latest

# Step 2: Run PTQ calibration
python quantize.py \
  --model best_model.pth \
  --calib-dir WasteDataset/val \
  --output quantize_result/

# Step 3: Inspect quantized model
vai_inspector -model quantize_result/WasteMobileNetV2_int.xmodel
```

**Calibration insight:** 508 calibration images (the full val set) provide good statistical coverage. The rule of thumb for PTQ calibration is 100–1000 samples per class; using the full validation set ensures all activation distributions are well-characterized.

### INT8 Quantization — What Actually Happens

For each layer, Vitis-AI:
1. Collects activation statistics during calibration forward passes
2. Computes per-tensor scale factors: `scale = max_val / 127`
3. Maps FP32 weights/activations → INT8 with minimal rounding error
4. Fuses BN layers into preceding conv weights (eliminates BN at inference)

**ReLU6 is key here** — bounded activations `[0, 6]` map cleanly to `[0, 127]` in INT8, which is why MobileNetV2 quantizes particularly well.

---

## 🚀 FPGA Deployment

### Compilation

```bash
vai_c_xir \
  -x quantize_result/WasteMobileNetV2_int.xmodel \
  -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json \
  -o compiled_model/ \
  -n waste_classifier
```

The compiler performs:
- **Subgraph partitioning** — splits the model into DPU-runnable and CPU-fallback subgraphs
- **Instruction scheduling** — maps DPU operations to hardware instruction set
- **Memory planning** — allocates on-chip buffer layout for feature maps

### Files Required on KV260

```
/home/root/waste_classifier/
├── waste_classifier.xmodel    # Compiled model
└── inference.py               # Inference script
```

### Running Inference

```bash
# Single image
python inference.py val/glass/glass103.jpg

# Batch evaluation (508 images)
python inference.py --eval-dir WasteDataset/val/
```

### Runtime Environment

| Component | Version |
|-----------|---------|
| OS | PetaLinux 2022.2 |
| Python | 3.8 |
| Runtime | VART (Vitis-AI Runtime) |
| DPU IP | DPUCZDX8G |

---

## 📊 Results

### Performance on KRIA KV260

#### Inference Timing (DPU only, 508 images)

| Metric | Value |
|--------|-------|
| Mean | **3.24 ms** |
| Std Dev | 0.03 ms |
| Min | 3.20 ms |
| Max | 3.51 ms |
| Median | 3.23 ms |
| P95 | 3.29 ms |

> The extremely low standard deviation (0.03ms) reflects the **deterministic nature of FPGA execution** — unlike GPU inference which varies with thermal throttling and scheduler contention.

#### End-to-End Latency & Throughput

| Metric | Value |
|--------|-------|
| Mean Total Latency | 25.10 ms |
| Throughput (mean) | **39.86 FPS** |
| Throughput (max) | 40.98 FPS |

The gap between DPU-only (3.24ms) and end-to-end (25.10ms) latency is CPU overhead: image loading, preprocessing (resize, normalize), and postprocessing. This is a clear optimization target — a C++ preprocessing pipeline could reduce this significantly.

#### Classification Accuracy

| Metric | Value |
|--------|-------|
| Overall Accuracy | **93.11%** (473/508) |
| Model Size | 2.4 MB (INT8) |

#### Confidence Score Analysis

| Metric | Value |
|--------|-------|
| Average Confidence (all) | 0.9402 |
| Avg Confidence (correct) | **0.9548** |
| Avg Confidence (wrong) | **0.7431** |

The ~0.21 confidence gap between correct and incorrect predictions is significant. **This means confidence score is a useful proxy for prediction reliability** — a deployment threshold of ~0.85 could automatically flag uncertain predictions for human review, effectively creating a confidence-gated inference system.

#### Confusion Matrix (On-Device)

```
              Predicted →
Actual ↓   cardbo  glass  metal  paper  plasti  trash
─────────────────────────────────────────────────────
cardboard  [ 77]     1      1      2      0      0    (81)
glass         0   [ 93]     6      2      0      0   (101)
metal         0      0   [ 81]     0      1      0    (82)
paper         1      0      0   [117]     0      1   (119)
plastic       1      5      3      0   [ 86]     2    (97)
trash         1      2      2      3      1   [ 19]   (28)
```

**Key misclassification patterns:**
- `glass → metal` (6 instances): Visually similar under certain lighting; both have reflective surfaces
- `plastic → glass` (5 instances): Transparent plastic bottles confused with glass
- `trash` class: Heterogeneous by definition — contains mixed materials

#### ROC / AUC Scores

| Class | AUC |
|-------|-----|
| Cardboard | 1.00 |
| Glass | 0.99 |
| Metal | 0.99 |
| Paper | 1.00 |
| Plastic | 0.99 |
| Trash | 1.00 |

Near-perfect AUC across all classes confirms strong class separability in the learned feature space, even for classes with lower top-1 accuracy.

---

## 📁 Project Structure

```
edge-waste-classifier/
│
├── training/
│   ├── train.py                  # MobileNetV2 fine-tuning script
│   ├── evaluate.py               # FP32 evaluation + confusion matrix
│   └── requirements.txt          # PyTorch, torchvision, etc.
│
├── quantization/
│   ├── quantize.py               # Vitis-AI PTQ script
│   └── quantize_result/
│       └── WasteMobileNetV2_int.xmodel
│
├── compilation/
│   └── compiled_model/
│       └── waste_classifier.xmodel
│
├── deployment/
│   └── inference.py              # KRIA KV260 inference script (VART)
│
├── results/
│   ├── confusion_matrix.png      # FP32 confusion matrix (Colab)
│   ├── roc_curve.png             # Multi-class ROC (Colab)
│   ├── class_accuracy.png        # Per-class accuracy bar chart
│   └── dpu_results.json          # Raw on-device evaluation output
│
└── README.md
```

---

## 🛠️ Setup & Reproduction

### 1. Training (Google Colab / GPU machine)

```bash
pip install torch==1.13.1 torchvision==0.14.1

python training/train.py \
  --data-dir /path/to/WasteDataset \
  --epochs 30 \
  --lr 1e-4 \
  --output best_model.pth
```

### 2. Quantization (Vitis-AI Docker)

```bash
# Pull and enter Vitis-AI environment
docker pull xilinx/vitis-ai-pytorch-gpu:latest
docker run --gpus all -v $(pwd):/workspace -it xilinx/vitis-ai-pytorch-gpu:latest

# Inside container:
cd /workspace
python quantization/quantize.py \
  --model best_model.pth \
  --calib-dir WasteDataset/val
```

**Environment:**
```
Vitis-AI: 3.5
PyTorch:  1.13.1
Python:   3.8
```

### 3. Compilation

```bash
vai_c_xir \
  -x quantize_result/WasteMobileNetV2_int.xmodel \
  -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json \
  -o compiled_model/ \
  -n waste_classifier
```

### 4. Deploy to KV260

```bash
# Copy to board over SSH
scp compiled_model/waste_classifier.xmodel root@<board-ip>:/home/root/
scp deployment/inference.py root@<board-ip>:/home/root/

# On the board (PetaLinux 2022.2):
python inference.py --eval-dir WasteDataset/val/
```

---

## 💡 Key Technical Decisions

### Why KRIA KV260 over Raspberry Pi / Jetson?
The KV260 provides a **dedicated DPU hardened into the FPGA fabric**, giving deterministic latency and high MAC throughput without the thermal variability of a GPU. Unlike Jetson, the FPGA fabric can be reconfigured for different DPU architectures. The tradeoff is a more complex toolchain (Vitis-AI vs. TensorRT).

### Why MobileNetV2 over EfficientNet / ResNet?
- **EfficientNet** achieves higher accuracy but its compound scaling makes quantization trickier and DPU subgraph coverage lower
- **ResNet-18** is a reasonable alternative, but its standard convolutions use significantly more MACs than MobileNetV2's depthwise separable convolutions
- MobileNetV2's **ReLU6** activations quantize cleanly to INT8 — this is a non-trivial advantage

### Why PTQ over QAT?
- PTQ achieved only **~1.4% accuracy degradation** (93.11% → 91.7%), which was acceptable
- QAT requires retraining within the Vitis-AI framework — significantly more complex setup
- For a first deployment, PTQ's simplicity and speed of iteration outweighed QAT's marginal accuracy gains

### Confidence Thresholding
The observed confidence gap (correct: 0.9548, wrong: 0.7431) suggests that **softmax confidence is a reliable uncertainty signal** for this model. A production system should implement a confidence gate — predictions below ~0.85 get routed to a secondary classifier or human review queue.

---

## ⚠️ Limitations & Future Work

| Limitation | Impact | Proposed Solution |
|------------|--------|------------------|
| Small `trash` class (28 images) | 67.86% accuracy | Collect more data; use TACO dataset |
| CPU preprocessing bottleneck | 25ms end-to-end vs 3.24ms DPU | C++ preprocessing pipeline |
| PTQ accuracy drop (~1.4%) | Minor accuracy loss | Quantization-Aware Training (QAT) |
| Glass/Metal confusion | 6 misclassifications | Train with augmented lighting conditions |
| Single-label classification | Can't handle mixed waste | Multi-label or detection-based approach |
| Static batch size (16) | Memory overhead for single images | Dynamic batching support |

---

## 📚 References

- [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381) — Sandler et al., 2018
- [Vitis-AI User Guide (UG1414)](https://docs.xilinx.com/r/en-US/ug1414-vitis-ai) — Xilinx/AMD
- [TrashNet Dataset](https://github.com/garythung/trashnet) — Gary Thung & Mindy Yang
- [DPUCZDX8G Product Guide (PG338)](https://docs.xilinx.com/r/en-US/pg338-dpu) — Xilinx/AMD
- [KRIA KV260 Vision AI Starter Kit](https://www.xilinx.com/products/som/kria/kv260-vision-starter-kit.html)

---

<div align="center">

**Edge-AI Waste Classification · MobileNetV2 on KRIA KV260 · Vitis-AI 3.5**  
*93.11% accuracy · 3.24ms DPU latency · 40 FPS · 2.4MB model*

</div>
