# D-SENet-MambaOut: Robust Vibration Signal Recognition for Phase-Sensitive Optical Time-Domain Reflectometer (Φ-OTDR)

This is our paper
[Dual-Channel SENet Based on MambaOut for OTDR Vibration Event Recognition](https://www.sciencedirect.com/science/article/abs/pii/S0030399226002860)
implemented in PyTorch using [MambaOut](https://github.com/yuweihao/MambaOut?tab=readme-ov-file#mambaout-do-we-really-need-mamba-for-vision-cvpr-2025) as the D-SENet-MambaOut algorithm.


---

Figure 3: Recurrence plots (RP) of typical signal sequences. Left: Time-domain signal sequence diagram; Right: Corresponding RP. Each row shows a representative signal and its corresponding recurrence plot. From top to bottom: \textbf{np1}: Constant function ($y=1$); \textbf{np2}: Linear function ($y = x$); \textbf{np3}: Sine function ($y = \sin(x)$); \textbf{np4}: Periodic window function (alternating between 0 and 1); \textbf{np5}: Modulated sine function ($y = \sin(100x) \cdot \sin(20x)$). }

Figure 4: Visualization of time series data and the corresponding RP images under weak
and strong augmentation: (a) Wind, (b) Man-made, (c) Excavation. The red dashed
vertical lines validate the correlation between the images and the characteristics of the time
series signals, demonstrating how the augmentation techniques affect the representation
of the data. Weak augmentation preserves the overall structure and global context, while
strong augmentation emphasizes local features and high-frequency details.

Figure 7: Diagram of the D-SE-MambaOut architecture. The STEM module converts
raw pixels into primary semantic features, implementing 4× spatial downsampling with
two 3 × 3 convolutions (stride=2), expanding the number of channels from 3 to 64. The
design incorporates LayerNorm and GELU activations for channel attention, and permute
operations to decouple spatial and channel dimensions. The model progresses through
four stages of hierarchical feature extraction and downsampling, building a multi-scale
feature pyramid. The final multichannel tensor undergoes squeeze and excitation process-
ing by D-SENet (section 3.1.2) to achieve dynamic channel weighting and feature fusion
optimization.

---

# Requirements
Like [MambaOut](https://github.com/yuweihao/MambaOut?tab=readme-ov-file#mambaout-do-we-really-need-mamba-for-vision-cvpr-2025), it requires PyTorch and timm.
```cmd
pip install timm==0.6.11
```

Due to the confidential nature of the collected data, the dataset cannot be made publicly available.
```cmd
.
├── 0wind
│   ├── test
│   ├── train
│   └── val
├── 1manual
│   ├── test
│   ├── train
│   └── val
└── 2digger
    ├── test
    ├── train
    └── val
```

---

# Repository Structure

---

```cmd
.
├── txt2RP.py
├── D-SE-Mambaout/
├── SE-Mambaout/
├── Mambaout/
├── end2end/
├── model_utils/
├── other_utils/
└── README.md
```
Below is a detailed explanation of each module.

---

## txt2RP.py

This script converts raw .txt vibration signal files into Recurrence Plot (RP) images.

---

## Model Variants (Paper Comparison Models)

The repository includes three core experimental model variants used in the paper for ablation and comparison.

### A. Mambaout (Baseline Model)
This folder implements the pure MambaOut-based model without SE attention.

Structure:

* mambaout_withoutAttention.py → model definition
* my_mamba_withoutSEAttention.py → running interface (entry script)
* model_utils/ → data loading & training utilities
* best_model.pth → trained checkpoint

### B. SE-Mambaout
This variant integrates Squeeze-and-Excitation attention into the MambaOut backbone.

Structure:

* mambaout.py → SE-integrated model
* loss_separation.py → customized loss separation mechanism
* my_mamba.py → running interface
* models_*.pth → trained weights

### C. D-SE-Mambaout (Proposed Model)

This is the full proposed model in the paper.

Structure:

* mambaout_weak_strong.py → dual-channel model definition
* my_mamba_weak_strong.py → main running interface
* model_utils/data_weak_strong.py → dual-input data loader
* best_model.pth → best trained model

---

## End-to-End Benchmark

The `end2end` module provides a comprehensive benchmark pipeline:

> **txt → Recurrence Plot (RP) → tensor conversion → model inference**

This script is designed to evaluate whether the reported throughput includes RP generation, addressing concerns about real-world deployment latency.

---

### Purpose

In the original pipeline, RP images were generated and saved to disk.
However, disk I/O significantly dominates latency.

This benchmark script explicitly separates:

1. **Model-only inference speed** (forward pass only)
2. **End-to-end latency (compute-only)**
   txt → RP → tensor → inference (no disk saving)
3. **Full end-to-end latency (including disk saving)**
   txt → RP → save_png → inference

This allows transparent and reproducible latency reporting.

---

### Output Files

All benchmark results are saved to `--out_dir`:

| File                        | Description                                                       |
| --------------------------- | ----------------------------------------------------------------- |
| `e2e_timing_per_sample.csv` | Per-sample timing breakdown                                       |
| `e2e_timing_summary.csv`    | Per-class + overall statistics (mean / median / p95 / throughput) |
| `e2e_meta.json`             | Hardware, software, and benchmark configuration metadata          |

---

### Typical Usage

#### Model-Only Inference (Network Forward Only)

```bash
python end2end_rp_infer_benchmark.py \
    --bench model_only \
    --model mambaout \
    --batch_size 64 \
    --iters 2000
```

Measures pure forward-pass speed.

---

#### End-to-End (Compute Only, No Disk Saving)

```bash
python end2end_rp_infer_benchmark.py \
    --bench e2e \
    --model mambaout \
    --N 1024 --eps 0.05 --steps 3 --block 512 \
    --dir_wind data/0wind \
    --dir_manual data/1manual \
    --dir_digger data/2digger \
    --batch_size 8 \
    --max_per_class 200 \
    --transform minimal
```

Measures full computational pipeline:

```
txt → RP generation → tensor → model inference
```

No disk writing included.

---

#### End-to-End Including PNG Saving (I/O Overhead)

```bash
python end2end_rp_infer_benchmark.py \
    --bench e2e \
    --save_png \
    --save_dir rp_debug_pngs
```

This includes:

```
txt → RP → save_png → inference
```

Used to evaluate logging/debug overhead.

---

### Dual-Input Model (D-SE-Mambaout)

For the dual-channel model (`dse_mambaout`), two input modes are available:

| Option                    | Description                                  | Speed   |
| ------------------------- | -------------------------------------------- | ------- |
| `--transform minimal`     | Same view fed to both inputs                 | Fastest |
| `--transform weak_strong` | Weak/strong augmented views (training-style) | Slower  |

The `weak_strong` mode reproduces the full training-time data augmentation pipeline.

---

### Benchmark Philosophy

This script ensures:

* Transparent separation of compute vs I/O cost
* Reproducible timing statistics
* Hardware-aware benchmarking
* Per-class latency reporting

It validates the real-time capability of the proposed model under practical deployment conditions.



---

# Citation
If you find our work helpful for your research, please consider citing the following BibTeX entry.
```text
@article{NIANCHAO2026114935,
title = {Mambaout-driven dual-channel SENet for Φ-OTDR vibration event recognition},
journal = {Optics & Laser Technology},
volume = {199},
pages = {114935},
year = {2026}ma ,
issn = {0030-3992},
doi = {https://doi.org/10.1016/j.optlastec.2026.114935},
url = {https://www.sciencedirect.com/science/article/pii/S0030399226002860},
author = {Liu Nianchao and Zixia Hu and Yiming Zhao and Hongwei Liu and Huan Wang and Ran Yang and Sheng Liang}
}
```

---
# Acknowledgment
在此感谢西安交通大学穆廷魁教授为我提供的宝贵指导。