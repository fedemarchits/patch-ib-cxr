# Configuration Guidelines

This document explains all configuration options available for training and evaluation.

---

## Table of Contents
1. [Experiment](#experiment)
2. [Data Configuration](#data-configuration)
3. [Model Configuration](#model-configuration)
4. [Training Configuration](#training-configuration)
5. [WandB Configuration](#wandb-configuration)
6. [Model Variants Quick Reference](#model-variants-quick-reference)

---

## Experiment

```yaml
experiment_name: "model_a_baseline"
```

| Option | Type | Description |
|--------|------|-------------|
| `experiment_name` | string | Name for logging and identification |

---

## Data Configuration

```yaml
data:
  json_path: "/workspace/mimic_master_official_split.jsonl"
  image_root: "/datasets/MIMIC-CXR/files"
  batch_size: 64
  num_workers: 4
  image_size: 224
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `json_path` | string | required | Path to JSONL dataset file |
| `image_root` | string | required | Root directory for images |
| `batch_size` | int | 32 | Batch size per GPU |
| `num_workers` | int | 4 | DataLoader workers |
| `image_size` | int | 224 | Input image resolution (224 for ViT-B/16) |

### Effective Batch Size
```
effective_batch_size = batch_size × gradient_accumulation_steps
```
Example: `batch_size=64` + `gradient_accumulation_steps=2` = 128 effective

---

## Model Configuration

### Backbone

```yaml
model:
  vision_backbone: "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
  text_backbone: "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
```

| Option | Type | Description |
|--------|------|-------------|
| `vision_backbone` | string | HuggingFace model ID for vision encoder |
| `text_backbone` | string | HuggingFace model ID for text encoder |

### Contrastive Loss

```yaml
model:
  temperature: 0.1
  contrastive_weight_i2t: 0.5
  contrastive_weight_t2i: 0.5
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `temperature` | float | 0.1 | Initial temperature (learned via `logit_scale`) |
| `contrastive_weight_i2t` | float | 0.5 | Weight for image-to-text loss |
| `contrastive_weight_t2i` | float | 0.5 | Weight for text-to-image loss |

**Note**: The actual temperature used is `exp(logit_scale)` which is learned. The `temperature` config is not directly used when `logit_scale` is present.

### Patch-IB (Information Bottleneck)

```yaml
model:
  use_masking: true
  mask_ratio: 0.5
  use_sparsity_loss: true
  sparsity_weight: 10.0
  consistency_weight: 1.0
  consistency_include_negatives: false
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `use_masking` | bool | false | Enable Patch-IB masking |
| `mask_ratio` | float | 0.5 | Target sparsity (0.5 = keep 50% of patches) |
| `use_sparsity_loss` | bool | false | Enable sparsity regularization |
| `sparsity_weight` | float | 10.0 | Weight for sparsity loss (λ_sparse) |
| `consistency_weight` | float | 1.0 | Weight for consistency loss (μ) |
| `consistency_include_negatives` | bool | false | Include negative pairs in consistency loss |

**Patch-IB Loss Formula**:
```
L = L_NCE_full + L_NCE_mask + λ_sparse × L_sparse + μ × L_consistency
```

### Model D: Top-K Patch Selection

```yaml
model:
  use_masking: true
  use_topk_masking: true
  k_ratio: 0.25
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `use_topk_masking` | bool | false | Use Top-K selection instead of threshold |
| `k_ratio` | float | 0.25 | Fraction of patches to keep (0.25 = 49 of 196 patches) |

**Top-K Benefits**:
- Guaranteed sparsity: exactly K patches are always selected
- More efficient local alignment: only K patches used as K/V
- Reduced GPU memory compared to Model C
- Better interpretability: can visualize which patches are selected

**k_ratio Examples**:
- `k_ratio: 0.5` = 98 patches (similar compression to Model C)
- `k_ratio: 0.25` = 49 patches (4x compression)
- `k_ratio: 0.125` = 24 patches (8x compression)

### Local Alignment

```yaml
model:
  use_local_alignment: true
  local_alignment_temperature: 0.7
  local_alignment_weight: 100
  local_alignment_warmup_steps: 500
  local_alignment_n_heads: 4
  local_alignment_dropout: 0.1
  use_uncertainty_weighting: false
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `use_local_alignment` | bool | false | Enable local token-patch alignment |
| `local_alignment_temperature` | float | 1.0 | Temperature for attention softmax |
| `local_alignment_weight` | float | 0.1 | Weight for local alignment loss (λ_local) |
| `local_alignment_warmup_steps` | int | 0 | Steps to linearly warmup local weight |
| `local_alignment_n_heads` | int | 4 | Number of attention heads |
| `local_alignment_dropout` | float | 0.1 | Dropout in attention module |
| `use_uncertainty_weighting` | bool | false | Learn loss weights automatically (Kendall et al.) |

**Local Alignment Architecture**:
- Query: Text tokens (B, L, 512)
- Key/Value: Image patches (B, 196, 512)
- Output: Aligned visual features per token

---

## Training Configuration

### Basic Training

```yaml
training:
  epochs: 40
  lr: 5.0e-6
  weight_decay: 0.01
  warmup_steps: 1000
  device: "cuda"
  use_amp: true
  gradient_accumulation_steps: 2
  print_freq: 50
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `epochs` | int | 40 | Maximum training epochs |
| `lr` | float | 5e-6 | Base learning rate |
| `weight_decay` | float | 0.01 | AdamW weight decay |
| `warmup_steps` | int | 500 | LR warmup steps (linear) |
| `device` | string | "cuda" | Device ("cuda" or "cpu") |
| `use_amp` | bool | true | Enable mixed precision (FP16) |
| `gradient_accumulation_steps` | int | 1 | Accumulate gradients over N steps |
| `print_freq` | int | 50 | Print frequency (batches) |

### Learning Rate Schedule

The scheduler uses **cosine decay with warmup**:
```
LR Schedule:
  [0, warmup_steps]: Linear 0 → lr
  [warmup_steps, total_steps]: Cosine lr → 0
```

### Early Stopping

```yaml
training:
  early_stopping_metric: "combined"
  early_stopping_patience: 7
  combined_weights:
    recall: 0.7
    auc: 0.3
  eval_auc_every: 1
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `early_stopping_metric` | string | "loss" | Metric to monitor: "loss", "recall", "auc", "combined" |
| `early_stopping_patience` | int | 5 | Epochs without improvement before stopping |
| `combined_weights.recall` | float | 0.7 | Weight for mean recall in combined metric |
| `combined_weights.auc` | float | 0.3 | Weight for AUC in combined metric |
| `eval_auc_every` | int | 1 | Compute AUC every N epochs (expensive) |

**Combined Metric Formula**:
```
combined = 0.7 × mean_recall + 0.3 × (AUC × 100)
```

### Fine-tuning Options

```yaml
training:
  freeze_backbone: false
  llrd_factor: 0.85
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `freeze_backbone` | bool | false | Freeze vision+text encoders |
| `llrd_factor` | float | 0.9 | Layer-wise LR decay factor |

**LLRD (Layer-wise Learning Rate Decay)**:
```
Layer i LR = base_lr × (llrd_factor ^ (num_layers - 1 - i))

Example with llrd_factor=0.85, 12 layers:
  Layer 0 (earliest):  lr × 0.85^11 = 0.17 × lr
  Layer 6 (middle):    lr × 0.85^5  = 0.44 × lr
  Layer 11 (latest):   lr × 0.85^0  = 1.0 × lr
```

### Staged Training

```yaml
training:
  staged_training: true
  warmup_epochs: 3
  warmup_lr: 1.0e-4
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `staged_training` | bool | false | Enable two-phase training |
| `warmup_epochs` | int | 3 | Epochs with frozen backbone (Phase 1) |
| `warmup_lr` | float | 1e-4 | Learning rate for Phase 1 |

**Staged Training Phases**:
```
Phase 1 (epochs 0 to warmup_epochs-1):
  - Backbone: FROZEN
  - Training: logit_scale + projection heads only
  - LR: warmup_lr (higher, e.g., 1e-4)

Phase 2 (epochs warmup_epochs to end):
  - Backbone: UNFROZEN
  - Training: All parameters with LLRD
  - LR: lr (lower, e.g., 5e-6)
```

---

## WandB Configuration

```yaml
wandb:
  enable: true
  project: "Thesis-PatchIB"
  entity: null
  run_name: "model-a-baseline"
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enable` | bool | false | Enable Weights & Biases logging |
| `project` | string | "my-thesis-project" | W&B project name |
| `entity` | string | null | W&B team/username (null = default) |
| `run_name` | string | "default-run" | Name for this run |

---

## Model Variants Quick Reference

### Model A: Baseline (Contrastive Only)

```yaml
model:
  use_masking: false
  use_local_alignment: false
```

**Loss**: `L = L_NCE(img, txt)`

### Model B: Baseline + Local Alignment

```yaml
model:
  use_masking: false
  use_local_alignment: true
  local_alignment_weight: 100
  local_alignment_warmup_steps: 500
```

**Loss**: `L = L_NCE + λ_local × L_local`

### Model C: Patch-IB (Full)

```yaml
model:
  use_masking: true
  mask_ratio: 0.5
  sparsity_weight: 10.0
  consistency_weight: 1.0
  use_local_alignment: true
  local_alignment_weight: 100
```

**Loss**: `L = L_NCE_full + L_NCE_mask + λ_sparse × L_sparse + μ × L_cons + λ_local × L_local`

### Model D: Top-K Patch Selection (Efficient)

```yaml
model:
  use_masking: true
  use_topk_masking: true
  k_ratio: 0.25
  sparsity_weight: 5.0
  consistency_weight: 1.0
  use_local_alignment: true
  local_alignment_weight: 100
```

**Loss**: Same as Model C, but with guaranteed sparsity from Top-K selection

**Key Differences from Model C**:
1. **Guaranteed sparsity**: Always selects exactly K patches (vs. learned threshold)
2. **Efficient local alignment**: Only Top-K patches used as K/V (vs. all 196)
3. **Lower sparsity_weight**: Sparsity is guaranteed, so loss just helps learn rankings
4. **Memory savings**: ~4x fewer patches in attention computation (with k_ratio=0.25)

---

## Recommended Configurations

### For RTX 3090 (24GB VRAM)

```yaml
data:
  batch_size: 64
training:
  gradient_accumulation_steps: 2  # effective batch = 128
  use_amp: true
```

### For Limited VRAM (< 16GB)

```yaml
data:
  batch_size: 32
training:
  gradient_accumulation_steps: 4  # effective batch = 128
  use_amp: true
```

### For Quick Testing

```yaml
data:
  batch_size: 16
training:
  epochs: 5
  early_stopping_patience: 2
  eval_auc_every: 5  # Skip AUC computation
```

---

## Metrics Logged to WandB

### Training (per step)
- `train/step_loss` - Total loss
- `train/contrastive_loss_raw` - Raw contrastive loss
- `train/learning_rate` - Current LR
- `efficiency/step_time_ms` - Time per step
- `efficiency/throughput_img_per_sec` - Images processed per second
- `mask/*` - Patch masking statistics (if enabled)
- `loss_balance/*` - Loss component contributions

### Validation (per epoch)
- `val/loss` - Validation loss
- `val/mean_recall` - Mean R@K across all metrics
- `val/i2t_R@1`, `val/i2t_R@5`, `val/i2t_R@10` - Image-to-text retrieval
- `val/t2i_R@1`, `val/t2i_R@5`, `val/t2i_R@10` - Text-to-image retrieval
- `val/auc` - Mean AUC from linear probe
- `val/combined_metric` - Combined early stopping metric
