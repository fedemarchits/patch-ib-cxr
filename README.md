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

#### 3. Local Alignment Head (Grounding)

- **Cross-Attention**: Uses text tokens as **queries** and image patches as **keys/values**.
- **Loss**: **Local Loss** ($\mathcal{L}_{local}$) minimizes the distance between text-aligned patch summaries and their corresponding word embeddings, forcing clinical grounding.

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

<table>
    <tr>
        <td style="text-align: center;">
            <img src="imgs/model-a-staged-training_gpu_memory.png" alt="Model A Training & Validation Loss" style="width: 100%;"/>
            <br>Figure: Model A Training and Validation Loss over Epochs_
        </td>
        <td style="text-align: center;">
            <img src="imgs/model-a-staged-training_gpu_utilization.png" alt="Model A Combined Metric" style="width: 100%;"/>
            <br>Figure: Model A Combined Metric (Recall + AUC) over Epochs_
        </td>
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
