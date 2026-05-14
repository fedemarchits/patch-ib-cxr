<h1 align="center">🫁 Patch-IB-CXR</h1>
<h3 align="center">Patch-level Information Bottleneck for Vision-Language Alignment on Chest X-Rays</h3>

<p align="center">
  <em>From global CLIP contrastive learning to text-conditioned patch dropping — building interpretable, sparse, and grounded radiology representations on MIMIC-CXR.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.10-EE4C2C?logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/CUDA-12.2-76B900?logo=nvidia&logoColor=white" />
  <img src="https://img.shields.io/badge/Backbone-BiomedCLIP-1f77b4" />
  <img src="https://img.shields.io/badge/Dataset-MIMIC--CXR-7f3fbf" />
  <img src="https://img.shields.io/badge/Grounding-MS--CXR-2ca02c" />
  <img src="https://img.shields.io/badge/License-MIT-yellow" />
</p>

<p align="center">
  <img src="thesisPdf/images/FinalArchitecture.png" alt="Patch-IB-CXR full architecture" width="92%"/>
  <br>
  <em>Figure — Patch-IB-CXR end-to-end: BiomedCLIP backbone, FILIP probes for fine-grained text-conditioned scoring, an intra-ViT patch-dropping gate, and the global + local + sparsity objectives that shape the latent space.</em>
</p>

---

## ✨ TL;DR

- **Backbone**: BiomedCLIP (ViT-B/16 + PubMedBERT), full-text MIMIC-CXR (213k frontal images, official patient-disjoint split).
- **Idea**: replace dense patch attention with a **text-conditioned bottleneck** — keep only the patches a radiology report actually needs.
- **Mechanisms studied**: global InfoNCE, local FILIP/cosine alignment, mid-fusion cross-attention, STE / Top-K / Gumbel masks, intra-ViT patch dropping, soft sigmoid gating.
- **Best trade-offs**: **Model F** (FILIP-drop @ layer 6) — best Purity + strong NMI on ~50% patches; **Model F-adaptive** — best faithfulness (Del 0.032 ↓ / Ins 0.785 ↑).
- **Side-products**: phrase grounding on MS-CXR, faithfulness (deletion/insertion) curves, full ablation against ConVIRT / GLoRIA / BioViL / MGCA / MAIRA-2.

---

## 📍 Table of Contents

- [⚙️ Reproducibility & Environment](#-reproducibility--environment)
- [📊 Dataset Analysis & Generation Logic](#-dataset-analysis--generation-logic)
- [📈 Dataset Distribution & Statistics](#-dataset-distribution--statistics)
- [⚙️ Data Generation Pipeline](#-data-generation-pipeline)
- [🏆 Benchmarking & SOTA Comparison](#-benchmarking--sota-comparison)
- [🧬 Foundation Model](#-foundation-model)
- [🧠 Models](#-models)
- [🖼️ Visual Mechanisms Gallery](#-visual-mechanisms-gallery)
- [🩻 Phrase Grounding on MS-CXR](#-phrase-grounding-on-ms-cxr)
- [📉 Faithfulness — Deletion & Insertion](#-faithfulness--deletion--insertion)
- [🌌 Embedding Space — UMAP Gallery](#-embedding-space--umap-gallery)
- [🏁 Final Leaderboard](#-final-leaderboard)

---

## 🧭 What the patches look like

<p align="center">
  <img src="thesisPdf/images/cxr_patches_zoom.png" alt="ViT-B/16 patchification on a chest X-ray" width="85%"/>
  <br>
  <em>Figure — A 224×224 frontal chest X-ray is tokenized into 14×14 = <strong>196 patches</strong>. Patch-IB asks: <em>which subset of these 196 carries the report's signal?</em></em>
</p>

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

## 📊 Dataset Analysis & Generation Logic

The final dataset, `mimic_master_official_split.jsonl` considers only frontal AP and PA images for a total of 213364 samples.

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

#### 💡 Thesis Impact

- **Scaling**: Moving from a ~60k keyword subset to a **218k full-text master file** provided the supervision density required for the model to breakthrough the 5% Recall@1 barrier.
- **Reproducibility**: By adhering to official splits, this implementation allows for direct benchmarking against models like **GLoRIA** and **BioViL**.

---

## 🏆 Benchmarking & SOTA Comparison (Chronological)

Our model's performance on the official MIMIC-CXR test set is compared against the evolution of the field, from foundational baselines (2020) to current State-of-the-Art (2024-2025).

| Model            | Year | R@1 (t2i) | R@10 (t2i) |  Avg AUC  |
| :--------------- | :--: | :-------: | :--------: | :-------: |
| **ConVIRT** [1]  | 2020 |   23.3%   |   61.2%    |   0.770   |
| **GLoRIA** [2]   | 2021 |   24.9%   |   63.8%    |   0.815   |
| **BioViL** [3]   | 2022 |   26.1%   |   66.4%    |   0.826   |
| **MGCA** [4]     | 2023 |   28.9%   |   70.1%    |   0.835   |
| **BioViL-L** [5] | 2023 |   27.4%   |   68.2%    |   0.821   |
| **MAIRA-2** [6]  | 2024 | **31.2%** | **74.5%**  | **0.868** |

#### 📚 Literature References

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

## 🧠 Model Architecture & Progression

The project follows a staged development from a standard global baseline to a highly efficient, interpretability-focused Patch-IB model. The architecture is built on top of **BiomedCLIP** (ViT-B/16 + PubMedBERT).

---

### 🏗️ Core Architecture Components

Our model extends the standard CLIP framework with specialized heads and alignment modules:

#### 1. Global Projection Heads

- **Image Projector**: Maps pooled ViT patch features into a shared latent space ($d=512$).
- **Text Projector**: Maps the BERT `[CLS]` token embedding into the same shared space.
- **Loss**: **InfoNCE Full** ($\mathcal{L}_{NCE-full}$) ensures the model distinguishes matching image-report pairs from distractors in the batch

#### 2. Spatial Mask Head (Patch-IB)

- **Logic**: A lightweight head $z = \sigma(w_{z}^{\top}v_{ij}+b_{z})$ that assigns a salience score $\in (0,1)$ to each of the 196 patches
- **Goal**: Identifying the **Information Bottleneck (IB)**—the minimum subset of patches required to retain the model's discriminative power
- **Optimization**: Controlled by a sparsity constraint ($\mathcal{L}_{sparse}$) and a consistency loss ($\mathcal{L}_{cons}$) to ensure the masked image behaves similarly to the full image

<p align="center">
  <img src="thesisPdf/images/IB.png" alt="Information Bottleneck principle" width="55%"/>
  <br>
  <em>Figure — The Information Bottleneck principle behind Patch-IB: compress the input <code>X</code> through a minimal sufficient code <code>Z</code> that still predicts <code>Y</code>.</em>
</p>

#### 3. Local Alignment Head (Grounding)

- **Cross-Attention**: Uses text tokens as **queries** and image patches as **keys/values**.
- **Loss**: **Local Loss** ($\mathcal{L}_{local}$) minimizes the distance between text-aligned patch summaries and their corresponding word embeddings, forcing clinical grounding.

<table align="center">
  <tr>
    <td align="center"><img src="thesisPdf/images/FILIP_fig1.png" width="100%"/><br><em>FILIP: fine-grained token-patch similarity</em></td>
    <td align="center"><img src="thesisPdf/images/crossattn_vs_filip.png" width="100%"/><br><em>Cross-attention vs FILIP scoring</em></td>
  </tr>
</table>

---

### 📈 Experimental Progression (Ablation Study)

| Model       | Variant        | Global CLIP | Local Align | Patch-IB | Top-K/Dropping |
| :---------- | :------------- | :---------: | :---------: | :------: | :------------: |
| **Model A** | **Baseline**   |     ✅      |     ❌      |    ❌    |       ❌       |
| **Model B** | **+ Local**    |     ✅      |     ✅      |    ❌    |       ❌       |
| **Model C** | **+ Patch-IB** |     ✅      |     ✅      |    ✅    |       ❌       |
| **Model D** | **Top-K Opt.** |     ✅      |     ✅      |    ✅    |   ✅ (Soft)    |
| **Model E** | **Token Drop** |     ✅      |     ✅      |    ✅    |   ✅ (Hard)    |

---

### ⚙️ Common Training and Evaluation Settings

Unless otherwise specified for a particular model, the following configurations and practices apply across all experiments:

#### Foundation Model

All experiments in this repository utilize **BiomedCLIP** (`microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`) as the base foundation model. This model was specifically designed for biomedical image-text processing and pretrained on **PMC-15M**, a large-scale dataset of 15 million figure-caption pairs from biomedical research articles.

#### 🏗️ Model Architecture

BiomedCLIP follows a dual-stream Transformer architecture, consisting of a vision encoder and a text encoder projected into a shared latent space.

##### 1. Vision Encoder: ViT-B/16

- **Architecture**: Vision Transformer (ViT) Base.
- **Input Resolution**: $224 \times 224$ pixels.
- **Patch Size**: $16 \times 16$ pixels, resulting in 196 image patches.
- **Function**: The image is treated as a sequence of patches, where each patch is embedded and processed through 12 Transformer layers to capture spatial and semantic hierarchies.

##### 2. Text Encoder: PubMedBERT

- **Architecture**: BERT-base.
- **Pretraining**: Unlike standard CLIP (which uses a generic text encoder), BiomedCLIP uses **PubMedBERT**, which was pretrained from scratch on the full text of PubMed abstracts and articles.
- **Max Length**: 256 tokens.
- **Function**: This allows the model to deeply understand complex medical terminology (e.g., "cardiac silhouette", "interstitial opacities") that generic encoders often fail to represent accurately.

#### 🛰️ Pretraining Objective

BiomedCLIP was pretrained using a standard **Contrastive Language-Image Pretraining (CLIP)** objective. The model learns by maximizing the cosine similarity between matched image-caption pairs and minimizing it for unmatched pairs within a batch (InfoNCE loss).

##### Key Statistics:

| Feature           | Specification               |
| :---------------- | :-------------------------- |
| **Dataset**       | PMC-15M (15 Million Pairs)  |
| **Domain**        | Biomedical / Clinical       |
| **Embedding Dim** | 768 (projected to 512)      |
| **Tokenization**  | WordPiece (domain-specific) |

#### 🔗 Model Source

- **HuggingFace Hub**: [microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)

<p align="center">
  <img src="thesisPdf/images/CLIP_structure.png" alt="CLIP dual-stream backbone" width="70%"/>
  <br>
  <em>Figure — The dual-stream contrastive backbone we inherit from BiomedCLIP and extend with patch-level heads.</em>
</p>

#### Data Configuration

- **Dataset**: MIMIC-CXR (`mimic_master_official_split.jsonl`)
- **Image Root**: `/datasets/MIMIC-CXR/files`
- **Image Size**: 224x224 pixels
- **Data Loaders**: 4 workers per DataLoader, ensuring efficient data loading.

#### Training Strategy

##### Optimization Strategy and Learning Rate Dynamics

For the training and fine-tuning of the model, I selected the **AdamW** optimizer. This choice is driven by its ability to handle the decoupled weight decay required for Transformer stability and its prevalence in state-of-the-art Vision-Language Model (VLM) literature.

###### Learning Rate Selection

The base learning rate was set to **$5.0 \times 10^{-6}$**. In the context of deep learning, this is considered a **very low learning rate**. This conservative choice is intentional for several reasons:

- **Foundation Model Preservation**: Since the backbones (ViT-B/16 and PubMedBERT) are already pretrained on massive datasets (PMC-15M), a high learning rate would risk "catastrophic forgetting," where the model loses its broad medical knowledge in favor of overfitting to the specific nuances of the local dataset.
- **Fine-Grained Alignment**: The **LocalAlignModule** and **Patch-IB** masking heads require precise, incremental updates to find the delicate mathematical balance between global retrieval and local anatomical grounding.
- **Stability with Small Batches**: Given the high memory requirements of processing high-resolution medical images, low learning rates ensure stable gradient updates even when using gradient accumulation.

During the initial hyperparameter search, higher learning rates (e.g., $1.0 \times 10^{-4}$) were evaluated but ultimately discarded due to several critical failure modes:

- **Validation Instability**: Large weight updates caused the model to overshoot optimal regions in the loss landscape, leading to erratic oscillations in validation metrics.
- **Optimization Convergence**: The complexity of the multi-task objective—balancing Global, Local, and Sparsity losses—required a narrower "search corridor." High learning rates prevented the model from settling into a stable joint minimum.
- **Representation Collapse**: Most significantly, higher rates frequently triggered a collapse in the latent space, where the encoders (ViT and BERT) mapped diverse inputs to nearly identical embeddings. This "shortcut" learning resulted in low training loss but a total failure in retrieval (Recall) and discriminative tasks (AUC).

By transitioning to a more conservative base learning rate of **$5.0 \times 10^{-6}$** combined with **Layer-wise Learning Rate Decay (LLRD)**, the training process achieved the necessary granularity to refine the pretrained BiomedCLIP backbones while successfully grounding the new **LocalAlignModule** and **Patch-IB** heads.

> 🔍 I've noticed that with a higher **lr** and without **LLRD** the model would quickly forget some pre-learned knowledge in the first epochs, leading to lower performances at the end in terms of Retrievial (AUC as well, but less).

###### 🧬 Layer-wise Learning Rate Decay (LLRD)

To further refine the training process, I implemented **LLRD** with a factor of **0.85**. This technique acknowledges that different layers of a Transformer capture different levels of abstraction. By applying a decay factor, we ensure that the foundational "low-level" layers (like early edge detectors in the ViT) remain stable, while the "high-level" semantic layers and the custom heads remain more plastic for task-specific adaptation.

When the `llrd_factor` is set to `0.85`, the learning rates are distributed as follows:

| Parameter Group | Parameters  | Learning Rate |
| :-------------- | :---------- | :------------ |
| `embeddings`    | 741,888     | 7.11e-07      |
| `layer_0`       | 7,077,888   | 8.37e-07      |
| `layer_1`       | 7,077,888   | 9.84e-07      |
| `layer_2`       | 7,077,888   | 1.16e-06      |
| `layer_3`       | 7,077,888   | 1.36e-06      |
| `layer_4`       | 7,077,888   | 1.60e-06      |
| `layer_5`       | 7,077,888   | 1.89e-06      |
| `layer_6`       | 7,077,888   | 2.22e-06      |
| `layer_7`       | 7,077,888   | 2.61e-06      |
| `layer_8`       | 7,077,888   | 3.07e-06      |
| `layer_9`       | 7,077,888   | 3.61e-06      |
| `layer_10`      | 7,077,888   | 4.25e-06      |
| `layer_11`      | 7,077,888   | 5.00e-06      |
| `head`          | 3,309,568   | 5.00e-06      |
| `other`         | 108,789,504 | 5.00e-06      |
| `no_decay`      | 227,841     | 5.00e-06      |

###### ⚖️ Decoupled and Selective Weight Decay

The **AdamW** optimizer applies a weight decay of **0.01**. Following the implementation standards of models like **BiomedCLIP** and **BioViL**, I utilized **Selective Weight Decay**.

- **Decoupling**: Weight decay is decoupled from the gradient update, allowing for stronger regularization without dampening the adaptive learning rate.
- **Exclusion**: Weight decay is explicitly omitted for **biases** and **LayerNorm** parameters (`no_decay` group). This prevents the model from being penalized for maintaining architectural stability and baseline normalization.

##### Mixed Precision AMP

Automatic Mixed Precision (AMP) is enabled (`use_amp: true`) for performance efficiency.

##### Gradient Accumulation

To keepp the models as much as possible comparable with each other, and to not get OOM on the RTX 3090 I had to decrease **batch size** and increase **gradient accumulation** with growing complexity of models.

In generale gradient is computed on a fixed size of **196 samples**.

#### Early Stopping

- **Metric**: Combined metric, calculated as a weighted average of Mean Recall@K and Mean AUC (`0.7 * Recall + 0.3 * AUC` in most cases, or `0.6 * Recall + 0.4 * AUC` for Model A).
- **Patience**: can be different across models since show different behaviours.
- **AUC Evaluation**: Mean AUC is computed every epoch (`eval_auc_every: 1`) on the validation set for accurate early stopping, it:
  - **extracts embeddings**,
  - a **lightweight classifier** is trained for few iterations (around 200) to predict probabilities,
  - computes **Mean AUC**.

#### Evaluation

- **Metrics**: Standard evaluation includes:
  - **Retrieval**: Recall@K (R@1, R@5, R@10) for both Image-to-Text (I2T) and Text-to-Image (T2I).
  - **Classification**: Mean Area Under the Receiver Operating Characteristic Curve (AUC) and Mean Average Precision (AP) from a linear probe.
- **Hardware**: Evaluations are typically performed on an RTX 3090 GPU.

---

## 🧠 Models

This section will detail the architecture and training strategies for:

### Model A: Global CLIP Baseline (Contrastive Only)

**Architecture**: Model A serves as the foundational baseline. It utilizes a **BiomedCLIP** (ViT-B/16 for vision and PubMedBERT for text) as its backbone. It exclusively relies on global image and text embeddings, which are mapped into a shared latent space ($d=512$) via projection heads. Masking and local alignment features are **disabled**.

The total loss for Model A is the InfoNCE Full loss, defined as:

<!-- prettier-ignore -->
$$ L_{total} = L_{NCE-full} = -\frac{1}{N} \sum*{i=1}^{N} \left[ \log \frac{\exp(\mathbf{v}\_i \cdot \mathbf{t}\_i / \tau)}{\sum*{j=1}^{N} \exp(\mathbf{v}_i \cdot \mathbf{t}\_j / \tau)} + \log \frac{\exp(\mathbf{t}\_i \cdot \mathbf{v}\_i / \tau)}{\sum_{j=1}^{N} \exp(\mathbf{t}\_i \cdot \mathbf{v}\_j / \tau)} \right] $$

Where:

- $N$ is the **batch size**,
- $\mathbf{v}_i$ and $\mathbf{t}_i$ are the image and text **embeddings** for the $i$-th sample,
- $\tau$ is the **temperature parameter**.

**Configuration**:

| Parameter                     | Value      | Description                                              |
| :---------------------------- | :--------- | :------------------------------------------------------- |
| `contrastive_weight_i2t`      | `0.5`      | Weight for image-to-text loss                            |
| `contrastive_weight_t2i`      | `0.5`      | Weight for text-to-image loss                            |
| `epochs`                      | `40`       | Maximum training epochs                                  |
| `lr` (fine-tuning)            | `1.0e-5`   | Base learning rate for fine-tuning phase                 |
| `warmup_epochs`               | `3`        | Epochs with frozen backbone (Phase 1)                    |
| `warmup_lr` (Phase 1)         | `1.0e-4`   | Learning rate for Phase 1                                |
| `weight_decay`                | `0.05`     | AdamW weight decay                                       |
| `warmup_steps`                | `1000`     | LR warmup steps (linear)                                 |
| `early_stopping_metric`       | `combined` | Metric to monitor: Weighted Recall + AUC                 |
| `early_stopping_patience`     | `4`        | Epochs without improvement before stopping               |
| `batch_size`                  | `96`       | Batch size per GPU                                       |
| `gradient_accumulation_steps` | `2`        | Accumulate gradients over N steps (effective batch: 192) |
| `use_amp`                     | `true`     | Enable mixed precision (FP16)                            |
| `llrd_factor`                 | `0.85`     | Layer-wise LR decay factor                               |

Observations:

- temperature initially set to 0.1 was increased to 0.2, since it controls the "sharpness" of the probability distribution over the pairs in the batch. The reason why I increased it is that it stabilizes the loss that

##### Learning Rate Distribution (LLRD)

When `llrd_factor` is set to `0.85`, the learning rates are distributed across the model's layers as follows:

| Parameter Group | Parameters  | Learning Rate |
| :-------------- | :---------- | :------------ |
| `embeddings`    | 741,888     | 7.11e-07      |
| `layer_0`       | 7,077,888   | 8.37e-07      |
| `layer_1`       | 7,077,888   | 9.84e-07      |
| `layer_2`       | 7,077,888   | 1.16e-06      |
| `layer_3`       | 7,077,888   | 1.36e-06      |
| `layer_4`       | 7,077,888   | 1.60e-06      |
| `layer_5`       | 7,077,888   | 1.89e-06      |
| `layer_6`       | 7,077,888   | 2.22e-06      |
| `layer_7`       | 7,077,888   | 2.61e-06      |
| `layer_8`       | 7,077,888   | 3.07e-06      |
| `layer_9`       | 7,077,888   | 3.61e-06      |
| `layer_10`      | 7,077,888   | 4.25e-06      |
| `layer_11`      | 7,077,888   | 5.00e-06      |
| `head`          | 3,309,568   | 5.00e-06      |
| `other`         | 108,789,504 | 5.00e-06      |
| `no_decay`      | 227,841     | 5.00e-06      |

#### Evaluation Results (Test Set)

##### Performance Metrics

| Metric | $I2T$ (Image-to-Text) | $T2I$ (Text-to-Image) |
| :----- | :-------------------: | :-------------------: |
| $R@1$  |        23.69%         |        22.38%         |
| $R@5$  |        50.37%         |        49.95%         |
| $R@10$ |        64.76%         |        62.87%         |

| Metric                |   Value   |
| :-------------------- | :-------: |
| Mean $AUC$ (CheXpert) | **0.768** |
| Mean $AP$             |   0.343   |

##### Efficiency Metrics (Test Set)

| Metric     | Value   | Unit    |
| :--------- | :------ | :------ |
| Throughput | 71.84   | img/sec |
| Peak VRAM  | 3576.84 | MB      |

**Training Progress Visualizations**:

<table>
    <tr>
        <td style="text-align: center;">
            <img src="imgs/model-a-staged-training_train_val_loss.png" alt="Model A Training & Validation Loss" style="width: 100%;"/>
            <br>Figure: Model A Training and Validation Loss over Epochs_
        </td>
        <td style="text-align: center;">
            <img src="imgs/model-a-staged-training_combined_metric.png" alt="Model A Combined Metric" style="width: 100%;"/>
            <br>Figure: Model A Combined Metric (Recall + AUC) over Epochs_
        </td>
    </tr>
    <tr>
        <td style="text-align: center;">
            <img src="imgs/model-a-staged-training_mean_recall.png" alt="Model A Mean Retrieval Recall" style="width: 100%;"/>
            <br>Figure: Model A Mean Retrieval Recall over Epochs_
        </td>
        <td style="text-align: center;">
            <img src="imgs/model-a-staged-training_mean_auc.png" alt="Model A Mean Classification AUC" style="width: 100%;"/>
            <br>Figure: Model A Mean Classification AUC over Epochs_
        </td>
    </tr>
    <tr>
        <td style="text-align: center;">
            <img src="imgs/model-a-staged-training_temperature.png" alt="Model A Learning Rate Schedule" style="width: 100%;"/>
            <br>Figure: Model A Learning Rate Schedule over Training Steps_
        </td>
        <td style="text-align: center;">
            <img src="imgs/model-a-staged-training_learning_rate.png" alt="Model A Learning Rate Schedule" style="width: 100%;"/>
            <br>Figure: Model A Learning Rate Schedule over Training Steps_
        </td>
    </tr>
</table>

#### Efficiency

<table align="center">
  <tr>
    <td align="center"><img src="imgs/model-a-staged-training_gpu_memory.png" width="100%"/><br><em>GPU memory across staged training</em></td>
    <td align="center"><img src="imgs/model-a-staged-training_gpu_utilization.png" width="100%"/><br><em>GPU utilization (RTX 3090)</em></td>
    <td align="center"><img src="imgs/model-a-staged-training_temperature.png" width="100%"/><br><em>Learned contrastive temperature</em></td>
  </tr>
</table>

---

### Model B: + Local Alignment

#### Local Alignment

At the beginning the first attempt has been done using **MSE** as follows:

<!-- prettier-ignore -->
$$
L_{\mathrm{MSE}}
= \frac{1}{\sum_{i=1}^{N} \sum_{k=1}^{L} m_{ik}}
\sum_{i=1}^{N} \sum_{k=1}^{L}
m_{ik} \,
\left\| \tilde{\mathbf{v}}_{ik} - \mathbf{t}_{ik} \right\|^{2}
$$

where:

- $N$ = batch size,
- $L$ = max number of tokens,
- $m_{ik}$ is the valid-token mask (1 if token $k$ of sample $i$ is not padding)
- $\tilde{\mathbf{v}}_{ik}$ is the **normalized** image feature aligned to the $k$-th text token of sample $i$,
- $\mathbf{t}_{ik}$ is the **normalized** embedding of the $k$-th text token of sample $i$.

The whole loss will result in:

<!-- prettier-ignore -->
$$
L_{\mathrm{total}} = L_{\mathrm{NCE\text{-}full}} + \lambda_{\mathrm{local}} \, L_{\mathrm{MSE}}
$$

The first attempts to balance these two losses were quite tricky since $L_{\mathrm{MSE}}$ and $L_{\mathrm{NCE\text{-}full}}$ have different nature and required extremely high values for $\lambda_{\mathrm{local}}$, but, since both embeddings were L2-normalized before computing MSE its very similar to **Cosine Similarity**:

$$
\left\| \tilde{\mathbf{v}}_{ik} - \mathbf{t}_{ik} \right\|^{2}
= 2 \left( 1 - \cos\bigl(\tilde{\mathbf{v}}_{ik}, \mathbf{t}_{ik}\bigr) \right)
$$

That's why the loss I've used at the end:

$$
L_{\mathrm{cos}} =
\frac{1}{\sum_{i=1}^{N} \sum_{k=1}^{L} m_{ik}}
\sum_{i=1}^{N} \sum_{k=1}^{L}
m_{ik}\,\bigl(1 - \cos(\tilde{\mathbf{v}}_{ik}, \mathbf{t}_{ik})\bigr)
$$

and final loss for **model B**:

$$
L_{\mathrm{total}} = L_{\mathrm{NCE\text{-}full}} + \lambda_{\mathrm{local}} \, L_{\mathrm{cos}}
$$

In this way $\lambda_{\mathrm{local}}$ can be more contained since both losses now speek same language.

#### Results

#### Loss Balance

In order to keep track of the influence of the two losses during training I've kept an eye on both, the **Loss Contribution** and its **Gradient Contribution**.

| Metrica | $I2T$ (Image-to-Text) | $T2I$ (Text-to-Image) |
| :------ | :-------------------: | :-------------------: |
| $R@1$   |        24.16%         |        22.10%         |
| $R@5$   |        52.12%         |        49.40%         |
| $R@10$  |        65.20%         |        62.15%         |

| Metrica Clinica       |  Valore   |
| :-------------------- | :-------: |
| Mean $AUC$ (CheXpert) | **0.761** |

### Model C

| Metrica | $I2T$ (Image-to-Text) | $T2I$ (Text-to-Image) |
| :------ | :-------------------: | :-------------------: |
| $R@1$   |        26.10%         |        24.45%         |
| $R@5$   |        54.84%         |        52.91%         |
| $R@10$  |        68.53%         |        66.81%         |

| Metrica Clinica       |  Valore   |
| :-------------------- | :-------: |
| Mean $AUC$ (CheXpert) | **0.819** |

### Model D

| Metrica | $I2T$ (Image-to-Text) | $T2I$ (Text-to-Image) |
| :------ | :-------------------: | :-------------------: |
| $R@1$   |        24.98%         |        22.85%         |
| $R@5$   |        52.65%         |        50.79%         |
| $R@10$  |        64.82%         |        65.95%         |

| Metrica Clinica       |  Valore   |
| :-------------------- | :-------: |
| Mean $AUC$ (CheXpert) | **0.795** |

## Efficiency

| Modello             | Throughput (img/sec) | Avg Step Time (ms) | Latency (ms/img) | Peak VRAM (MB) |  GFLOPs   | Patch Usage |
| :------------------ | :------------------: | :----------------: | :--------------: | :------------: | :-------: | :---------: |
| **A (Baseline)**    |        82.40         |       42.60        |       1.33       |    3210.15     |   28.11   |    100%     |
| **B (Local Align)** |      **67.55**       |     **62.52**      |     **1.95**     |  **3640.07**   | **30.24** |    100%     |
| **C (Patch-IB)**    |        58.45         |       71.85        |       2.25       |    3855.20     |   27.17   |    ~80%     |
| **D (Top-K)**       |      **92.15**       |     **40.12**      |     **1.25**     |  **3120.45**   | **21.40** | ~**12.7%**  |

---

### Model E: Text-agnostic PatchScorerMLP (Post-ViT)

A first patch-selection ablation: a tiny MLP scores every patch from its ViT feature alone — no text conditioning. Layer-6 read-out, K = 118 patches kept.

<p align="center">
  <img src="thesisPdf/images/Post-ViT.png" alt="Post-ViT scorer architecture" width="80%"/>
  <br>
  <em>Figure — Post-ViT scoring: patches go through the full 12-block ViT, then a lightweight head selects a subset for downstream losses.</em>
</p>

<p align="center">
  <img src="thesisPdf/images/postvit_variants.png" alt="Post-ViT variants" width="78%"/>
  <br>
  <em>Figure — Post-ViT variants explored (MLP scorer, FILIP-conditioned scorer, hard vs soft selection).</em>
</p>

---

### Model F: FILIP-Drop inside the ViT

Model F injects the patch bottleneck **inside** the ViT: FILIP-scored patches at an intermediate layer (4, 6 or 9) are dropped before the remaining transformer blocks. This is the configuration that delivers the best Purity / Faithfulness trade-off.

<p align="center">
  <img src="thesisPdf/images/IntraVitArch.png" alt="Intra-ViT FILIP drop architecture" width="88%"/>
  <br>
  <em>Figure — Intra-ViT FILIP drop: text-conditioned similarities at layer ℓ produce a hard top-K gate, the surviving patches feed the remaining blocks.</em>
</p>

<p align="center">
  <img src="thesisPdf/images/intravit_variants.png" alt="Intra-ViT variants" width="80%"/>
  <br>
  <em>Figure — Drop-layer ablation: shallow (4) vs mid (6) vs deep (9) injection. Layer 6 hits the best Purity/NMI sweet-spot.</em>
</p>

<p align="center">
  <img src="thesisPdf/images/ste_vs_topk.png" alt="STE vs TopK" width="62%"/>
  <br>
  <em>Figure — Mask discreteness: STE-thresholded mask (left) vs differentiable Top-K with straight-through gradient (right).</em>
</p>

#### Qualitative Patch Selection — Model F (drop @ layer 6)

<table align="center">
  <tr>
    <td align="center"><img src="Results/Model-F/Model-F-6/visualizations/mask_sample_0.png" width="100%"/></td>
    <td align="center"><img src="Results/Model-F/Model-F-6/visualizations/mask_sample_3.png" width="100%"/></td>
    <td align="center"><img src="Results/Model-F/Model-F-6/visualizations/mask_sample_5.png" width="100%"/></td>
  </tr>
  <tr>
    <td align="center"><img src="Results/Model-F/Model-F-6/visualizations/mask_sample_7.png" width="100%"/></td>
    <td align="center"><img src="Results/Model-F/Model-F-6/visualizations/mask_sample_8.png" width="100%"/></td>
    <td align="center"><img src="Results/Model-F/Model-F-6/visualizations/mask_sample_9.png" width="100%"/></td>
  </tr>
</table>

<p align="center"><em>Text-conditioned FILIP-drop mask: kept patches concentrate on lungs / mediastinum / support devices — the regions referenced by the report.</em></p>

---

### Model F-adaptive: STE-threshold mask

Same FILIP probe, but instead of a fixed K we apply a learnable threshold via STE — the model picks **how many** patches to keep per image. Yields the strongest faithfulness numbers (Del ↓ 0.032 / Ins ↑ 0.785).

<table align="center">
  <tr>
    <td align="center"><img src="Results/Model-F/Model-F-adaptive/visualizations/mask_sample_2.png" width="100%"/></td>
    <td align="center"><img src="Results/Model-F/Model-F-adaptive/visualizations/mask_sample_4.png" width="100%"/></td>
    <td align="center"><img src="Results/Model-F/Model-F-adaptive/visualizations/mask_sample_6.png" width="100%"/></td>
  </tr>
</table>

---

### Model H: Soft Sigmoid Gating

Differentiable alternative to hard dropping: every patch survives but is scaled by a soft sigmoid gate. Best NMI, slightly worse faithfulness than F-adaptive.

<p align="center">
  <img src="thesisPdf/images/softgate_viz.png" alt="Soft sigmoid gating" width="62%"/>
  <br>
  <em>Figure — Soft sigmoid gate: continuous (0,1) weights replace the hard top-K, keeping gradients smooth everywhere.</em>
</p>

---

## 🖼️ Visual Mechanisms Gallery

### Patch attention — Model C (STE mask + cosine local)

<table align="center">
  <tr>
    <td align="center"><img src="Results/Model-C/Model-C1/visualizations/attention_sample_0.png" width="100%"/></td>
    <td align="center"><img src="Results/Model-C/Model-C1/visualizations/attention_sample_3.png" width="100%"/></td>
    <td align="center"><img src="Results/Model-C/Model-C1/visualizations/attention_sample_5.png" width="100%"/></td>
  </tr>
  <tr>
    <td align="center"><img src="Results/Model-C/Model-C1/visualizations/attention_sample_7.png" width="100%"/></td>
    <td align="center"><img src="Results/Model-C/Model-C1/visualizations/attention_sample_8.png" width="100%"/></td>
    <td align="center"><img src="Results/Model-C/Model-C1/visualizations/attention_sample_9.png" width="100%"/></td>
  </tr>
</table>

### Mid-fusion FILIP — Model B

Cross-attention probes at layers 4 + 8 + 12. The per-token text-to-visual maps reveal which patches each report word attends to.

<table align="center">
  <tr>
    <td align="center"><img src="Results/Model-B/Model-B-filip-mid/visualizations/filip_alignment_sample_0.png" width="100%"/></td>
    <td align="center"><img src="Results/Model-B/Model-B-filip-mid/visualizations/filip_alignment_sample_2.png" width="100%"/></td>
    <td align="center"><img src="Results/Model-B/Model-B-filip-mid/visualizations/filip_alignment_sample_4.png" width="100%"/></td>
  </tr>
  <tr>
    <td align="center"><img src="Results/Model-B/Model-B-filip-mid/visualizations/midfusion_t2v_sample_1.png" width="100%"/></td>
    <td align="center"><img src="Results/Model-B/Model-B-filip-mid/visualizations/midfusion_t2v_sample_3.png" width="100%"/></td>
    <td align="center"><img src="Results/Model-B/Model-B-filip-mid/visualizations/midfusion_t2v_sample_6.png" width="100%"/></td>
  </tr>
</table>

### Top-K dropping — Model D

Aggressive sparsity (k ≈ 0.25–0.40) — the model survives with ~12% of patches.

<table align="center">
  <tr>
    <td align="center"><img src="Results/Model-D/Model-D-topk/visualizations/midfusion_t2v_sample_0.png" width="100%"/></td>
    <td align="center"><img src="Results/Model-D/Model-D-topk/visualizations/midfusion_t2v_sample_3.png" width="100%"/></td>
    <td align="center"><img src="Results/Model-D/Model-D-topk/visualizations/midfusion_t2v_sample_5.png" width="100%"/></td>
  </tr>
</table>

---

## 🩻 Phrase Grounding on MS-CXR

MS-CXR provides expert bounding boxes for clinical phrases. We use the same FILIP probes — no extra supervision — to localize the phrase referenced by the report.

<table align="center">
  <tr>
    <td align="center"><img src="thesisPdf/images/PneunomiaFILIPDrop.png" width="100%"/><br><em>Model F — FILIP-drop @ layer 6</em></td>
    <td align="center"><img src="thesisPdf/images/PneunomiaAdaptive.png" width="100%"/><br><em>Model F-adaptive — STE threshold</em></td>
  </tr>
</table>

<p align="center"><em>Phrase: "right lower lobe pneumonia". Both variants concentrate mass on the right lung base; the adaptive variant produces a tighter footprint.</em></p>

---

## 📉 Faithfulness — Deletion & Insertion

For each model we sort patches by importance, then progressively delete (most-important first → metric should drop fast) or insert them (least-important first → metric should rise fast). Lower **Del AUC** and higher **Ins AUC** mean the salience map is faithful to the model.

<p align="center">
  <img src="thesisPdf/images/del_ins_curve.png" alt="Deletion / Insertion curves" width="65%"/>
  <br>
  <em>Figure — Deletion (↓ better) and Insertion (↑ better) curves comparing baseline attribution, B-FILIP-mid, and Model F-adaptive.</em>
</p>

| Model                 |  Del AUC ↓   |  Ins AUC ↑   |  Gap (Ins − Del) |
| :-------------------- | :----------: | :----------: | :--------------: |
| B (FILIP mid-fusion)  |    0.368     |    0.463     |      +0.095      |
| F (layer 6)           |    0.205     |    0.595     |      +0.390      |
| F-layer9              |    0.183     |    0.678     |      +0.495      |
| **F-adaptive**        |  **0.032**   |  **0.785**   |    **+0.753**    |
| H (soft gate)         |    0.299     |    0.681     |      +0.382      |

---

## 🌌 Embedding Space — UMAP Gallery

UMAP projections of the test set CLS embeddings, coloured by the dominant CheXpert label. Tighter, more cleanly separated clusters → better semantic structure (also reflected in NMI / Purity).

<table align="center">
  <tr>
    <td align="center"><img src="Results/Model-A/visualizations/umap_embeddings.png" width="100%"/><br><em>Model A — global contrastive only</em></td>
    <td align="center"><img src="Results/Model-B/Model-B-postViT/visualizations/umap_embeddings.png" width="100%"/><br><em>Model B — post-ViT FILIP</em></td>
  </tr>
  <tr>
    <td align="center"><img src="Results/Model-C/Model-C1/visualizations/umap_embeddings.png" width="100%"/><br><em>Model C — STE mask + cosine local</em></td>
    <td align="center"><img src="Results/Model-F/Model-F-6/visualizations/umap_embeddings.png" width="100%"/><br><em>Model F — FILIP-drop @ layer 6</em></td>
  </tr>
  <tr>
    <td align="center"><img src="Results/Model-F/Model-F-adaptive/visualizations/umap_embeddings.png" width="100%"/><br><em>Model F-adaptive</em></td>
    <td align="center"><img src="Results/Model-F/Model-F-9/visualizations/umap_embeddings.png" width="100%"/><br><em>Model F — drop @ layer 9</em></td>
  </tr>
</table>

### Pairwise cluster agreement (Model F)

<table align="center">
  <tr>
    <td align="center"><img src="Results/Model-F/Model-F-6/visualizations/pairwise_nmi_heatmap.png" width="100%"/><br><em>NMI</em></td>
    <td align="center"><img src="Results/Model-F/Model-F-6/visualizations/pairwise_ari_heatmap.png" width="100%"/><br><em>ARI</em></td>
    <td align="center"><img src="Results/Model-F/Model-F-6/visualizations/pairwise_purity_heatmap.png" width="100%"/><br><em>Purity</em></td>
  </tr>
</table>

---

## 🏁 Final Leaderboard

| Model                     | I2T R@1    | T2I R@1    | R@10 (avg) |    NMI     |   Purity   | Del ↓ | Ins ↑ | Patch Usage |
| :------------------------ | :--------: | :--------: | :--------: | :--------: | :--------: | :---: | :---: | :---------: |
| A — Baseline              |   17.37    |   17.82    |    56.9    |   0.1318   |   0.4228   |   —   |   —   |    100%     |
| B — cosine mid-fusion     | **18.70**  | **19.12**  |    58.2    |   0.1320   |   0.4313   |   —   |   —   |    100%     |
| B — FILIP mid-fusion      |   18.28    | **19.12**  |    58.6    |   0.1495   |   0.4501   | 0.368 | 0.463 |    100%     |
| B — multiscale probes     |   18.33    |   18.43    |    57.9    | **0.1536** |   0.4501   | 0.452 | 0.472 |    100%     |
| C — STE + cosine          |   14.23    |   13.39    |    49.2    |   0.1443   |   0.4486   | 0.482 | 0.416 |     50%     |
| D — Top-K (k = 0.4)       |   18.33    |   18.33    |    58.5    |   0.1184   |   0.4342   | 0.424 | 0.423 |    ~40%     |
| E — PatchScorerMLP        |   15.18    |   14.65    |    50.3    |   0.1437   |   0.4269   | 0.458 | 0.470 |     60%     |
| F-layer4                  |   18.43    |   18.70    |    57.1    |   0.1474   |   0.4414   | 0.316 | 0.496 |     50%     |
| **F (layer 6)**           |   17.96    |   18.17    |    58.0    |   0.1523   | **0.4645** | 0.205 | 0.595 |     50%     |
| F-layer9                  |   17.49    |   17.96    |    57.6    | **0.1542** |   0.4486   | 0.183 | 0.678 |     50%     |
| **F-adaptive**            |   17.86    |   17.28    |    57.2    |   0.1435   |   0.4428   | **0.032** | **0.785** |   adaptive  |
| G — staged 2-step drop    |   17.38    |   17.02    |    57.4    |   0.1460   |   0.4327   | 0.253 | 0.622 |     50%     |
| **H — soft gate**         |   16.96    |   16.96    |    55.6    | **0.1540** |   0.4530   | 0.299 | 0.681 |  soft 100%  |

<sub>Full per-config breakdown and notes in <a href="RESULTS.md"><code>RESULTS.md</code></a>.</sub>

<p align="center">
  <img src="thesisPdf/images/results_table.png" alt="Final results table" width="92%"/>
</p>

---

## 📂 Repository Layout

```
patch-ib-cxr/
├── configs/                 # YAML configs for every Model A→H variant
├── data/                    # MIMIC-CXR + MS-CXR loaders, master JSONL builder
├── engine/                  # trainer, validator, evaluator (retrieval + AUC + faithfulness)
├── models/
│   └── full_model.py        # ModelABaseline, ModelE, ModelF, ModelFAdaptive, ModelH
├── ms-cxr/                  # MS-CXR phrase-grounding subset + scripts
├── thesisPdf/               # LaTeX thesis + all architecture figures (images/)
├── Results/                 # per-model checkpoints, visualizations, eval JSON
├── imgs/                    # training curves (TensorBoard exports)
├── train.py · eval.sh       # entry points
├── evaluate_grounding_only.py
└── RESULTS.md               # detailed per-variant analysis
```

---

## ▶️ Quick Start

```bash
# 1. Build the container (CUDA 12.2 + PyTorch 2.10)
docker build -t patch_ib_img2:latest .

# 2. Run a single model
bash run_docker.sh
python train.py --config configs/model_f_filip_drop.yaml

# 3. Evaluate retrieval + AUC + faithfulness
bash eval.sh logs/model_f_layer6/best_model.pt

# 4. Phrase grounding on MS-CXR
python evaluate_grounding_only.py --checkpoint logs/model_f_layer6/best_model.pt
```

---

## 📜 Citation

If you find this work useful, please cite the underlying thesis (`thesisPdf/main.pdf`) and the BiomedCLIP backbone:

```bibtex
@mastersthesis{marchi2026patchib,
  author  = {Federico Marchi},
  title   = {Patch-level Information Bottleneck for Vision-Language
             Alignment on Chest X-Rays},
  school  = {[Your University]},
  year    = {2026}
}
```

---

<p align="center"><sub>Built on top of <a href="https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224">BiomedCLIP</a> · MIMIC-CXR · MS-CXR.</sub></p>
