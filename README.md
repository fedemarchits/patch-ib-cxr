# Patch-IB-CXR: Vision-Language Alignment on Chest X-Rays

This repository implements a Patch-based Information Bottleneck (IB) approach for medical image-text retrieval using the MIMIC-CXR dataset. This project explores staged training and local-global alignment to improve clinical feature representation.

---

## 📍 Table of Contents

- [⚙️ Reproducibility & Environment](#-reproducibility--environment)
- [📊 Dataset Analysis & Generation Logic](#-dataset-analysis--generation-logic)
- [📈 Dataset Distribution & Statistics](#-dataset-distribution--statistics)
- [⚙️ Data Generation Pipeline](#-data-generation-pipeline)
- [🏆 Benchmarking & SOTA Comparison](#-benchmarking--sota-comparison)
- [🧬 Foundation Model](#-foundation-model)
- [🧠 Models](#-models)

---

## ⚙️ Reproducibility & Environment

Experiments are conducted in a containerized environment to ensure consistency.

### 🐳 Docker Configuration

- **Image ID**: `patch_ib_img2:latest` (85ae3bd4da58)
- **Base OS**: Ubuntu 22.04 LTS
- **CUDA Version**: `12.2.0`

### 📦 Core Dependencies

| Library           | Version  |
| :---------------- | :------- |
| `torch`           | `2.10.0` |
| `transformers`    | `4.57.6` |
| `open_clip_torch` | `3.2.0`  |
| `tokenizers`      | `0.22.2` |

### Hardware Utilization

- **VRAM**: RTX 3090 ~24GB
- **Shared Memory**: 8GB `--shm-size`

---

# 📊 Dataset Analysis & Generation Logic

The final dataset, `mimic_master_official_split.jsonl`, represents the transition from limited "Bag-of-Words" labels to a full-scale Vision-Language corpus. By extracting raw radiology narratives and aligning them with the official benchmark, we have eliminated the data duplication issues found in previous versions.

---

## 📈 Dataset Distribution & Statistics

The following table summarizes the final distribution of the **218,138 frontal images** across the official, patient-disjoint partitions.

| Pathology                  |    Train    |    Val    |   Test    |
| :------------------------- | :---------: | :-------: | :-------: |
| **Total Samples**          | **213,364** | **1,733** | **3,041** |
| Atelectasis                |   43,179    |    347    |    679    |
| Cardiomegaly               |   41,735    |    363    |    808    |
| Consolidation              |   10,053    |    76     |    191    |
| Edema                      |   25,472    |    233    |    659    |
| Enlarged Cardiomediastinum |    6,681    |    53     |    132    |
| Fracture                   |    4,111    |    17     |    78     |
| Lung Lesion                |    5,766    |    64     |    108    |
| Lung Opacity               |   48,097    |    373    |    974    |
| No Finding                 |   71,202    |    561    |    539    |
| Pleural Effusion           |   50,720    |    452    |    990    |
| Pleural Other              |    1,814    |    14     |    52     |
| Pneumonia                  |   15,151    |    115    |    309    |
| Pneumothorax               |    9,916    |    73     |    94     |
| Support Devices            |   63,058    |    550    |   1,061   |

---

## ⚙️ Data Generation Pipeline

The master dataset was constructed using a rigorous four-stage pipeline to ensure clinical accuracy and prevent evaluation bias.

### 1. View Selection & Text Extraction

- **Frontal Filtering**: The raw MIMIC-CXR metadata was filtered to include only **Posteroanterior (PA)** and **Anteroposterior (AP)** views. Lateral views were excluded to maintain anatomical consistency for patch-level alignment.
- **Narrative Extraction**: For each unique `study_id`, the corresponding radiology report was parsed from the MIMIC-CXR report ZIP. Unlike keyword-based models, we extracted the **full narrative text** (Findings and Impression), providing the model with rich semantic context (e.g., "patchy bibasilar opacities") rather than generic binary flags.

### 2. Clinical Label Injection

- **Label Mapping**: We integrated the 14 standard CheXpert pathology labels.
- **Precision Filtering**: Each image was assigned a multi-label binary vector. Only "Positive" (1) labels from the CheXpert labeller were treated as active, ensuring the model trains on high-confidence clinical findings.

### 3. Patient-Level Stratification (Leakage Prevention)

- **Official Benchmark Alignment**: We applied the official MIMIC-CXR split to ensure our results are comparable to SOTA literature.
- **Zero Leakage**: All images and studies belonging to a single `subject_id` are strictly confined to the same split. This prevents the model from "cheating" by recognizing the unique bone structure or surgical hardware of a patient it encountered during training.

### 4. Automated Quality Control

- **Garbage Collection**: Reports shorter than 30 characters (e.g., "Report to follow" or empty placeholders) were automatically discarded.
- **Integrity Check**: 100% of the final 218k samples were verified to contain valid image paths and non-null textual queries.

---

### 💡 Thesis Impact

- **Scaling**: Moving from a ~60k keyword subset to a **218k full-text master file** provided the supervision density required for the model to breakthrough the 5% Recall@1 barrier.
- **Reproducibility**: By adhering to official splits, this implementation allows for direct benchmarking against models like **GLoRIA** and **BioViL**.

---

### 🏆 Benchmarking & SOTA Comparison (Chronological)

Our model's performance on the official MIMIC-CXR test set is compared against the evolution of the field, from foundational baselines (2020) to current State-of-the-Art (2024-2025).

| Model            | Year | R@1 (t2i) | R@10 (t2i) |  Avg AUC  |
| :--------------- | :--: | :-------: | :--------: | :-------: |
| **ConVIRT** [1]  | 2020 |   23.3%   |   61.2%    |   0.770   |
| **GLoRIA** [2]   | 2021 |   24.9%   |   63.8%    |   0.815   |
| **BioViL** [3]   | 2022 |   26.1%   |   66.4%    |   0.826   |
| **MGCA** [4]     | 2023 |   28.9%   |   70.1%    |   0.835   |
| **BioViL-L** [5] | 2023 |   27.4%   |   68.2%    |   0.821   |
| **MAIRA-2** [6]  | 2024 | **31.2%** | **74.5%**  | **0.868** |

### 📚 Literature References

1. [Zhang et al. (2020) - ConVIRT](https://arxiv.org/abs/2010.00747)
2. [Huang et al. (2021) - GLoRIA](https://arxiv.org/abs/2104.04687)
3. [Boecking et al. (2022) - BioViL](https://arxiv.org/abs/2204.09817)
4. [Wang et al. (2023) - MGCA](https://arxiv.org/abs/2211.12737)
5. [Boecking et al. (2023) - BioViL-L](https://arxiv.org/abs/2304.05341)
6. [Hyland et al. (2024) - MAIRA-2](https://arxiv.org/abs/2406.04447)

---

## 🔬 Classification Performance (AUC)

We evaluate the model's zero-shot classification performance across the 14 standard CheXpert pathologies.

- **Methodology**: We use prompt-based classification (e.g., "A chest x-ray showing [PATHOLOGY]") to calculate the Area Under the Receiver Operating Characteristic Curve (AUC-ROC).
- **Target**: Our goal is to exceed the AUC of the foundational GLoRIA model (0.815) by leveraging the semantic depth of the full-text master dataset.

---

# 🧠 Model Architecture & Progression

The project follows a staged development from a standard global baseline to a highly efficient, interpretability-focused Patch-IB model. The architecture is built on top of **BiomedCLIP** (ViT-B/16 + PubMedBERT).

---

## 🏗️ Core Architecture Components

Our model extends the standard CLIP framework with specialized heads and alignment modules:

### 1. Global Projection Heads

- **Image Projector**: Maps pooled ViT patch features into a shared latent space ($d=512$).
- **Text Projector**: Maps the BERT `[CLS]` token embedding into the same shared space.
- **Loss**: **InfoNCE Full** ($\mathcal{L}_{NCE-full}$) ensures the model distinguishes matching image-report pairs from distractors in the batch

### 2. Spatial Mask Head (Patch-IB)

- **Logic**: A lightweight head $z = \sigma(w_{z}^{\top}v_{ij}+b_{z})$ that assigns a salience score $\in (0,1)$ to each of the 196 patches
- **Goal**: Identifying the **Information Bottleneck (IB)**—the minimum subset of patches required to retain the model's discriminative power
- **Optimization**: Controlled by a sparsity constraint ($\mathcal{L}_{sparse}$) and a consistency loss ($\mathcal{L}_{cons}$) to ensure the masked image behaves similarly to the full image

### 3. Local Alignment Head (Grounding)

- **Cross-Attention**: Uses text tokens as **queries** and image patches as **keys/values**.
- **Loss**: **Local Loss** ($\mathcal{L}_{local}$) minimizes the distance between text-aligned patch summaries and their corresponding word embeddings, forcing clinical grounding.

---

## 📈 Experimental Progression (Ablation Study)

| Model       | Variant        | Global CLIP | Local Align | Patch-IB | Top-K/Dropping |
| :---------- | :------------- | :---------: | :---------: | :------: | :------------: |
| **Model A** | **Baseline**   |     ✅      |     ❌      |    ❌    |       ❌       |
| **Model B** | **+ Local**    |     ✅      |     ✅      |    ❌    |       ❌       |
| **Model C** | **+ Patch-IB** |     ✅      |     ✅      |    ✅    |       ❌       |
| **Model D** | **Top-K Opt.** |     ✅      |     ✅      |    ✅    |   ✅ (Soft)    |
| **Model E** | **Token Drop** |     ✅      |     ✅      |    ✅    |   ✅ (Hard)    |

---

## 🧠 Models

_(Section in progress)_

This section will detail the architecture and training strategies for:

- **Model A**: [Description placeholder]
- **Model B**: [Description placeholder]
- ...
