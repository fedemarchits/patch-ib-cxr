"""
Optimizer utilities for fine-tuning pretrained models.
Includes Layer-wise Learning Rate Decay (LLRD) and backbone freezing.
"""

import torch
from torch.optim import AdamW


def get_parameter_groups(model, base_lr, weight_decay=0.01, llrd_factor=0.9, freeze_backbone=False):
    """
    Create parameter groups with optional Layer-wise Learning Rate Decay (LLRD).

    Args:
        model: The model (ModelABaseline or similar)
        base_lr: Base learning rate for the top layers
        weight_decay: Weight decay for regularization
        llrd_factor: Decay factor per layer (e.g., 0.9 means each lower layer gets 0.9x the LR)
        freeze_backbone: If True, freeze the entire backbone and only train projection/head

    Returns:
        List of parameter groups for optimizer
    """

    if freeze_backbone:
        return _get_frozen_backbone_groups(model, base_lr, weight_decay)
    else:
        return _get_llrd_groups(model, base_lr, weight_decay, llrd_factor)


def _get_frozen_backbone_groups(model, base_lr, weight_decay):
    """
    Freeze backbone (both ViT and BERT), only train logit_scale and other heads.
    """
    # Freeze entire backbone (ViT + BERT)
    for param in model.backbone.parameters():
        param.requires_grad = False

    # Re-enable logit_scale (it's part of backbone but we want to train it)
    if hasattr(model, 'logit_scale'):
        model.logit_scale.requires_grad = True

    # Collect trainable parameters
    trainable_params = []

    # logit_scale
    if hasattr(model, 'logit_scale'):
        trainable_params.append({
            'params': [model.logit_scale],
            'lr': base_lr,
            'weight_decay': 0.0,  # No weight decay on scale
            'name': 'logit_scale'
        })

    # Local alignment projections (if present)
    if hasattr(model, 'patch_proj'):
        trainable_params.append({
            'params': list(model.patch_proj.parameters()),
            'lr': base_lr,
            'weight_decay': weight_decay,
            'name': 'patch_proj'
        })

    if hasattr(model, 'token_proj'):
        trainable_params.append({
            'params': list(model.token_proj.parameters()),
            'lr': base_lr,
            'weight_decay': weight_decay,
            'name': 'token_proj'
        })

    # Uncertainty weighting parameters (if present)
    if hasattr(model, 'log_var_contrastive'):
        trainable_params.append({
            'params': [model.log_var_contrastive],
            'lr': base_lr,
            'weight_decay': 0.0,
            'name': 'log_var_contrastive'
        })

    if hasattr(model, 'log_var_local'):
        trainable_params.append({
            'params': [model.log_var_local],
            'lr': base_lr,
            'weight_decay': 0.0,
            'name': 'log_var_local'
        })

    # Mask head (if present)
    if hasattr(model, 'mask_head'):
        trainable_params.append({
            'params': list(model.mask_head.parameters()),
            'lr': base_lr,
            'weight_decay': weight_decay,
            'name': 'mask_head'
        })

    # Local alignment module (if present)
    if hasattr(model, 'local_align'):
        trainable_params.append({
            'params': list(model.local_align.parameters()),
            'lr': base_lr,
            'weight_decay': weight_decay,
            'name': 'local_align'
        })

    # Mid-fusion cross-attention modules (if present)
    if hasattr(model, 'mid_fusion_modules'):
        trainable_params.append({
            'params': list(model.mid_fusion_modules.parameters()),
            'lr': base_lr,
            'weight_decay': weight_decay,
            'name': 'mid_fusion_modules'
        })

    # Mid-fusion local loss projections (if present)
    if hasattr(model, 'mid_fusion_patch_projs'):
        trainable_params.append({
            'params': list(model.mid_fusion_patch_projs.parameters()) +
                      list(model.mid_fusion_token_projs.parameters()),
            'lr': base_lr,
            'weight_decay': weight_decay,
            'name': 'mid_fusion_projections'
        })

    # Multi-scale probe projection heads (if present)
    if hasattr(model, 'probe_patch_projs'):
        trainable_params.append({
            'params': list(model.probe_patch_projs.parameters()) +
                      list(model.probe_token_projs.parameters()),
            'lr': base_lr,
            'weight_decay': weight_decay,
            'name': 'probe_projections'
        })

    return trainable_params


def _extract_layer_idx(name, key):
    """Extract integer layer index from a parameter name after the given key.
    E.g. _extract_layer_idx("...blocks.5.mlp...", "blocks") -> 5
         _extract_layer_idx("...encoder.layer.3.attention...", "layer") -> 3
    """
    parts = name.split('.')
    for j, part in enumerate(parts):
        if part == key and j + 1 < len(parts):
            try:
                return int(parts[j + 1])
            except ValueError:
                pass
    return None


def _get_llrd_groups(model, base_lr, weight_decay, llrd_factor):
    """
    Layer-wise Learning Rate Decay for fine-tuning.
    Earlier layers get smaller learning rates.
    Applied to both ViT (visual.trunk.blocks) and BERT (text.transformer.encoder.layer).
    Both encoders have 12 layers and share the same LR schedule per depth.
    """
    param_groups = []

    # Get the visual encoder blocks
    visual_model = model.backbone.clip_model.visual

    # Count number of transformer blocks (ViT)
    if hasattr(visual_model, 'blocks'):
        num_layers = len(visual_model.blocks)
    elif hasattr(visual_model, 'trunk') and hasattr(visual_model.trunk, 'blocks'):
        num_layers = len(visual_model.trunk.blocks)
    else:
        print("Warning: Could not detect transformer blocks, using uniform LR")
        return [{'params': model.parameters(), 'lr': base_lr, 'weight_decay': weight_decay}]

    # Build layer index -> LR mapping
    # Layer 0 (earliest) gets the smallest LR
    # Layer N-1 (latest) gets base_lr
    layer_lrs = {}
    for i in range(num_layers):
        layer_lrs[i] = base_lr * (llrd_factor ** (num_layers - 1 - i))

    # Embeddings (both ViT and BERT) get the smallest LR
    embedding_lr = base_lr * (llrd_factor ** num_layers)

    # Group parameters
    embedding_params = []
    vit_layer_params = {i: [] for i in range(num_layers)}
    bert_layer_params = {i: [] for i in range(num_layers)}
    head_params = []
    other_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Skip weight decay for certain parameters
        if 'bias' in name or 'norm' in name or 'logit_scale' in name or 'log_var' in name:
            no_decay_params.append(param)
            continue

        # Classify parameter by layer
        if 'backbone' in name:
            # ---- ViT embeddings ----
            if 'patch_embed' in name or 'cls_token' in name or 'pos_embed' in name:
                embedding_params.append(param)

            # ---- ViT transformer blocks ----
            elif 'blocks' in name:
                layer_idx = _extract_layer_idx(name, 'blocks')
                if layer_idx is not None and layer_idx in vit_layer_params:
                    vit_layer_params[layer_idx].append(param)
                else:
                    other_params.append(param)

            # ---- BERT embeddings ----
            elif 'text' in name and 'embeddings' in name:
                embedding_params.append(param)

            # ---- BERT transformer layers ----
            elif 'encoder.layer' in name:
                layer_idx = _extract_layer_idx(name, 'layer')
                if layer_idx is not None and layer_idx in bert_layer_params:
                    bert_layer_params[layer_idx].append(param)
                else:
                    other_params.append(param)

            # ---- Head / projection layers ----
            elif 'head' in name or 'proj' in name:
                head_params.append(param)
            else:
                other_params.append(param)
        else:
            # Non-backbone params (mid-fusion, local alignment, etc.) get base_lr
            head_params.append(param)

    # Create parameter groups
    if embedding_params:
        param_groups.append({
            'params': embedding_params,
            'lr': embedding_lr,
            'weight_decay': weight_decay,
            'name': 'embeddings'
        })

    for i in range(num_layers):
        # ViT layer i
        if vit_layer_params[i]:
            param_groups.append({
                'params': vit_layer_params[i],
                'lr': layer_lrs[i],
                'weight_decay': weight_decay,
                'name': f'vit_layer_{i}'
            })
        # BERT layer i (same LR as ViT layer i)
        if bert_layer_params[i]:
            param_groups.append({
                'params': bert_layer_params[i],
                'lr': layer_lrs[i],
                'weight_decay': weight_decay,
                'name': f'bert_layer_{i}'
            })

    if head_params:
        param_groups.append({
            'params': head_params,
            'lr': base_lr,
            'weight_decay': weight_decay,
            'name': 'head'
        })

    if other_params:
        param_groups.append({
            'params': other_params,
            'lr': base_lr,
            'weight_decay': weight_decay,
            'name': 'other'
        })

    if no_decay_params:
        param_groups.append({
            'params': no_decay_params,
            'lr': base_lr,
            'weight_decay': 0.0,
            'name': 'no_decay'
        })

    return param_groups


def create_optimizer(model, cfg):
    """
    Create optimizer with proper parameter groups based on config.

    Config options:
        training.lr: Base learning rate
        training.weight_decay: Weight decay
        training.freeze_backbone: If True, freeze backbone
        training.llrd_factor: Layer-wise LR decay factor (default 0.9)
    """
    base_lr = float(cfg['training']['lr'])
    weight_decay = float(cfg['training'].get('weight_decay', 0.01))
    freeze_backbone = cfg['training'].get('freeze_backbone', False)
    llrd_factor = cfg['training'].get('llrd_factor', 0.9)

    param_groups = get_parameter_groups(
        model,
        base_lr=base_lr,
        weight_decay=weight_decay,
        llrd_factor=llrd_factor,
        freeze_backbone=freeze_backbone
    )

    # Print info about parameter groups
    total_params = 0
    trainable_params = 0
    print("\n" + "="*60)
    print("Optimizer Parameter Groups:")
    print("="*60)

    for group in param_groups:
        n_params = sum(p.numel() for p in group['params'])
        total_params += n_params
        trainable_params += n_params
        print(f"  {group.get('name', 'unnamed'):20s}: {n_params:>10,} params, lr={group['lr']:.2e}")

    # Count frozen params
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  {'[frozen]':20s}: {frozen_params:>10,} params")
    print("="*60)
    print(f"  Total trainable: {trainable_params:,}")
    print(f"  Total frozen:    {frozen_params:,}")
    print(f"  Total:           {trainable_params + frozen_params:,}")
    print("="*60 + "\n")

    optimizer = AdamW(param_groups, lr=base_lr)

    return optimizer
