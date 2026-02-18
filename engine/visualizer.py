"""
Visualization module for attention masks and patch importance.

Saves attention heatmaps overlaid on original images during evaluation.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# Lazy imports for matplotlib (not required for training)
plt = None
cm = None


def _ensure_matplotlib():
    """Lazy import matplotlib when actually needed for visualization."""
    global plt, cm
    if plt is None:
        try:
            import matplotlib.pyplot as _plt
            import matplotlib.cm as _cm
            plt = _plt
            cm = _cm
        except ImportError:
            raise ImportError(
                "matplotlib is required for visualization. "
                "Install it with: pip install matplotlib"
            )


def visualize_attention_samples(
    model,
    dataloader,
    device,
    output_dir,
    num_samples=10,
    use_amp=False,
    image_size=224,
    patch_size=16
):
    """
    Visualize attention masks and patch importance for sample images.

    Args:
        model: The trained model
        dataloader: DataLoader with images
        device: Device to use
        output_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
        use_amp: Whether to use automatic mixed precision
        image_size: Original image size (224)
        patch_size: Patch size (16 for ViT-B/16)

    Saves:
        - attention_sample_{i}.png: Attention heatmap overlay
        - mask_sample_{i}.png: Patch importance mask (if masking enabled)
    """
    _ensure_matplotlib()
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    # Calculate grid size
    grid_size = image_size // patch_size  # 14 for 224/16

    samples_saved = 0
    print(f"\n[Visualizer] Generating {num_samples} attention visualizations...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Visualizing")):
            if batch is None:
                continue

            images, text = batch[0].to(device), batch[1].to(device)
            batch_size = images.shape[0]

            # Forward pass
            with torch.amp.autocast(device_type=device, enabled=use_amp):
                outputs = model(images, text)

            # Unpack outputs
            img_emb, txt_emb, importance_logits, local_features, img_emb_full = outputs

            # Process each image in batch
            for i in range(batch_size):
                if samples_saved >= num_samples:
                    break

                # Get original image (denormalize)
                img_tensor = images[i].cpu()
                img_np = denormalize_image(img_tensor)

                # === Visualize Patch Importance Mask (if masking enabled) ===
                if importance_logits is not None:
                    mask_logits = importance_logits[i].cpu()  # (196,)
                    mask_probs = torch.sigmoid(mask_logits)
                    hard_mask = (mask_logits > 0).float()

                    # Reshape to grid
                    mask_grid = mask_probs.view(grid_size, grid_size).numpy()
                    hard_mask_grid = hard_mask.view(grid_size, grid_size).numpy()

                    # Save mask visualization
                    save_mask_overlay(
                        img_np, mask_grid, hard_mask_grid,
                        os.path.join(output_dir, f"mask_sample_{samples_saved}.png"),
                        title=f"Patch Importance Mask (Sample {samples_saved})"
                    )

                # === Visualize Local Alignment Attention (if enabled) ===
                if local_features is not None and len(local_features) >= 5:
                    patch_feat, token_feat, attn_mask, aligned_feat, attn_weights = local_features

                    # attn_weights: (B, L, M) where M=196 patches
                    attn = attn_weights[i].cpu()  # (L, 196)

                    # Average attention across all tokens (or use specific tokens)
                    avg_attn = attn.mean(dim=0)  # (196,)
                    attn_grid = avg_attn.view(grid_size, grid_size).numpy()

                    # Also get max attention per patch (which token attends most)
                    max_attn = attn.max(dim=0)[0]  # (196,)
                    max_attn_grid = max_attn.view(grid_size, grid_size).numpy()

                    # Save attention visualization
                    save_attention_overlay(
                        img_np, attn_grid, max_attn_grid,
                        os.path.join(output_dir, f"attention_sample_{samples_saved}.png"),
                        title=f"Local Alignment Attention (Sample {samples_saved})"
                    )

                samples_saved += 1

            if samples_saved >= num_samples:
                break

    print(f"[Visualizer] Saved {samples_saved} visualizations to {output_dir}")
    return samples_saved


def denormalize_image(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Denormalize image tensor back to [0, 1] range.

    Args:
        img_tensor: (C, H, W) normalized tensor

    Returns:
        img_np: (H, W, C) numpy array in [0, 1]
    """
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)

    img = img_tensor * std + mean
    img = torch.clamp(img, 0, 1)
    img_np = img.permute(1, 2, 0).numpy()

    return img_np


def save_mask_overlay(img_np, soft_mask, hard_mask, save_path, title="Patch Mask"):
    """
    Save visualization of patch importance mask overlaid on image.

    Args:
        img_np: (H, W, C) original image
        soft_mask: (grid_h, grid_w) soft mask probabilities
        hard_mask: (grid_h, grid_w) binary mask
        save_path: Path to save the figure
        title: Figure title
    """
    _ensure_matplotlib()
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # 1. Original image
    axes[0].imshow(img_np)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # 2. Soft mask heatmap
    soft_mask_upsampled = upsample_mask(soft_mask, img_np.shape[:2])
    axes[1].imshow(img_np)
    im1 = axes[1].imshow(soft_mask_upsampled, cmap='jet', alpha=0.5, vmin=0, vmax=1)
    axes[1].set_title(f"Soft Mask (mean={soft_mask.mean():.2f})")
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. Hard mask overlay
    hard_mask_upsampled = upsample_mask(hard_mask, img_np.shape[:2])
    masked_img = img_np * hard_mask_upsampled[:, :, None]
    axes[2].imshow(masked_img)
    axes[2].set_title(f"Hard Mask ({hard_mask.sum():.0f}/{hard_mask.size} patches)")
    axes[2].axis('off')

    # 4. Grid view of mask
    axes[3].imshow(soft_mask, cmap='viridis', vmin=0, vmax=1)
    axes[3].set_title("Mask Grid")
    # Add grid lines
    for x in range(soft_mask.shape[1] + 1):
        axes[3].axvline(x - 0.5, color='white', linewidth=0.5)
    for y in range(soft_mask.shape[0] + 1):
        axes[3].axhline(y - 0.5, color='white', linewidth=0.5)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_attention_overlay(img_np, avg_attn, max_attn, save_path, title="Attention"):
    """
    Save visualization of attention weights overlaid on image.

    Args:
        img_np: (H, W, C) original image
        avg_attn: (grid_h, grid_w) average attention across tokens
        max_attn: (grid_h, grid_w) max attention per patch
        save_path: Path to save the figure
        title: Figure title
    """
    _ensure_matplotlib()
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # 1. Original image
    axes[0].imshow(img_np)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # 2. Average attention heatmap
    avg_attn_upsampled = upsample_mask(avg_attn, img_np.shape[:2])
    axes[1].imshow(img_np)
    im1 = axes[1].imshow(avg_attn_upsampled, cmap='hot', alpha=0.6)
    axes[1].set_title("Avg Attention (all tokens)")
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. Max attention heatmap
    max_attn_upsampled = upsample_mask(max_attn, img_np.shape[:2])
    axes[2].imshow(img_np)
    im2 = axes[2].imshow(max_attn_upsampled, cmap='hot', alpha=0.6)
    axes[2].set_title("Max Attention per patch")
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # 4. Grid view
    axes[3].imshow(avg_attn, cmap='viridis')
    axes[3].set_title("Attention Grid")
    for x in range(avg_attn.shape[1] + 1):
        axes[3].axvline(x - 0.5, color='white', linewidth=0.5)
    for y in range(avg_attn.shape[0] + 1):
        axes[3].axhline(y - 0.5, color='white', linewidth=0.5)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def upsample_mask(mask, target_size):
    """
    Upsample a grid mask to target image size using bilinear interpolation.

    Args:
        mask: (grid_h, grid_w) numpy array
        target_size: (H, W) target size

    Returns:
        upsampled: (H, W) numpy array
    """
    mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
    upsampled = F.interpolate(
        mask_tensor,
        size=target_size,
        mode='bilinear',
        align_corners=False
    )
    return upsampled.squeeze().numpy()


def visualize_token_attention(
    model,
    dataloader,
    device,
    output_dir,
    num_samples=5,
    tokens_to_show=5,
    use_amp=False
):
    """
    Visualize attention for specific text tokens.

    Shows which image patches each text token attends to.

    Args:
        model: The trained model
        dataloader: DataLoader with images
        device: Device to use
        output_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
        tokens_to_show: Number of tokens to show per sample
        use_amp: Whether to use automatic mixed precision
    """
    _ensure_matplotlib()
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    grid_size = 14  # 224 / 16

    samples_saved = 0
    print(f"\n[Visualizer] Generating per-token attention visualizations...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Token Attention"):
            if batch is None:
                continue

            images, text = batch[0].to(device), batch[1].to(device)

            with torch.amp.autocast(device_type=device, enabled=use_amp):
                outputs = model(images, text)

            img_emb, txt_emb, importance_logits, local_features, img_emb_full = outputs

            if local_features is None or len(local_features) < 5:
                print("[Visualizer] Local features not available, skipping token attention")
                return 0

            patch_feat, token_feat, attn_mask, aligned_feat, attn_weights = local_features

            for i in range(min(images.shape[0], num_samples - samples_saved)):
                img_np = denormalize_image(images[i].cpu())
                attn = attn_weights[i].cpu()  # (L, 196)
                mask = attn_mask[i].cpu()  # (L,)

                # Get valid token indices
                valid_indices = torch.where(mask > 0)[0]
                num_valid = min(len(valid_indices), tokens_to_show)

                if num_valid == 0:
                    continue

                # Create figure with subplots for each token
                fig, axes = plt.subplots(1, num_valid + 1, figsize=(4 * (num_valid + 1), 4))

                # Original image
                axes[0].imshow(img_np)
                axes[0].set_title("Original")
                axes[0].axis('off')

                # Attention for each token
                for j, token_idx in enumerate(valid_indices[:num_valid]):
                    token_attn = attn[token_idx].view(grid_size, grid_size).numpy()
                    token_attn_up = upsample_mask(token_attn, img_np.shape[:2])

                    axes[j + 1].imshow(img_np)
                    im = axes[j + 1].imshow(token_attn_up, cmap='hot', alpha=0.6)
                    axes[j + 1].set_title(f"Token {token_idx.item()}")
                    axes[j + 1].axis('off')

                plt.suptitle(f"Per-Token Attention (Sample {samples_saved})")
                plt.tight_layout()
                plt.savefig(
                    os.path.join(output_dir, f"token_attention_sample_{samples_saved}.png"),
                    dpi=150, bbox_inches='tight'
                )
                plt.close()

                samples_saved += 1

            if samples_saved >= num_samples:
                break

    print(f"[Visualizer] Saved {samples_saved} token attention visualizations")
    return samples_saved


def visualize_mid_fusion_attention(
    model,
    dataloader,
    device,
    output_dir,
    num_samples=10,
    use_amp=False,
    image_size=224,
    patch_size=16
):
    """
    Visualize cross-attention maps from mid-fusion modules.

    For each fusion layer, shows the text-to-image (t2v) attention averaged
    across text tokens, indicating which image regions the text attends to.

    Args:
        model: Model with mid-fusion modules
        dataloader: DataLoader with images
        device: Device to use
        output_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
        use_amp: Whether to use automatic mixed precision
        image_size: Original image size (224)
        patch_size: Patch size (16 for ViT-B/16)
    """
    _ensure_matplotlib()
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    grid_size = image_size // patch_size  # 14

    # Enable attention weight storage
    model.store_attention_weights = True

    # Get fusion layer names for titles
    fusion_layers = [idx + 1 for idx in model.mid_fusion_layer_indices]  # 1-indexed
    num_modules = len(fusion_layers)

    samples_saved = 0
    print(f"\n[Visualizer] Generating {num_samples} mid-fusion attention visualizations...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Mid-Fusion Attention"):
            if batch is None:
                continue

            images, text = batch[0].to(device), batch[1].to(device)

            with torch.amp.autocast(device_type=device, enabled=use_amp):
                model(images, text)

            attn_weights = model._mid_fusion_attn_weights
            if attn_weights is None or len(attn_weights) == 0:
                print("[Visualizer] No mid-fusion attention weights found, skipping")
                model.store_attention_weights = False
                return 0

            for i in range(images.shape[0]):
                if samples_saved >= num_samples:
                    break

                img_np = denormalize_image(images[i].cpu())

                # --- Overview figure: one column per fusion layer ---
                fig, axes = plt.subplots(1, num_modules + 1, figsize=(5 * (num_modules + 1), 5))

                axes[0].imshow(img_np)
                axes[0].set_title("Original Image")
                axes[0].axis('off')

                for m, (v2t_w, t2v_w) in enumerate(attn_weights):
                    # t2v_w: (B, L_text, N_vit) where N_vit = 197 (CLS + 196 patches)
                    t2v = t2v_w[i].cpu()  # (L, 197)

                    # Exclude CLS token (position 0 of ViT keys) -> (L, 196)
                    t2v_patches = t2v[:, 1:]

                    # Average across text tokens -> (196,)
                    avg_attn = t2v_patches.mean(dim=0)
                    attn_grid = avg_attn.view(grid_size, grid_size).numpy()
                    attn_up = upsample_mask(attn_grid, img_np.shape[:2])

                    axes[m + 1].imshow(img_np)
                    im = axes[m + 1].imshow(attn_up, cmap='hot', alpha=0.6)
                    axes[m + 1].set_title(f"Layer {fusion_layers[m]} (t2v)")
                    axes[m + 1].axis('off')
                    plt.colorbar(im, ax=axes[m + 1], fraction=0.046, pad=0.04)

                plt.suptitle(f"Mid-Fusion Cross-Attention (Sample {samples_saved})")
                plt.tight_layout()
                plt.savefig(
                    os.path.join(output_dir, f"midfusion_t2v_sample_{samples_saved}.png"),
                    dpi=150, bbox_inches='tight'
                )
                plt.close()

                # --- v2t overview: which text tokens each patch attends to ---
                fig, axes = plt.subplots(1, num_modules + 1, figsize=(5 * (num_modules + 1), 5))

                axes[0].imshow(img_np)
                axes[0].set_title("Original Image")
                axes[0].axis('off')

                for m, (v2t_w, t2v_w) in enumerate(attn_weights):
                    # v2t_w: (B, N_vit, L_text) where N_vit = 197
                    v2t = v2t_w[i].cpu()  # (197, L)

                    # Patches only (exclude CLS query) -> (196, L)
                    v2t_patches = v2t[1:, :]

                    # Entropy of each patch's attention over text tokens
                    # High entropy = patch attends broadly, Low = focused on specific tokens
                    v2t_probs = v2t_patches + 1e-8
                    entropy = -(v2t_probs * torch.log(v2t_probs)).sum(dim=-1)  # (196,)
                    entropy_grid = entropy.view(grid_size, grid_size).numpy()
                    entropy_up = upsample_mask(entropy_grid, img_np.shape[:2])

                    axes[m + 1].imshow(img_np)
                    im = axes[m + 1].imshow(entropy_up, cmap='viridis', alpha=0.6)
                    axes[m + 1].set_title(f"Layer {fusion_layers[m]} (v2t entropy)")
                    axes[m + 1].axis('off')
                    plt.colorbar(im, ax=axes[m + 1], fraction=0.046, pad=0.04)

                plt.suptitle(f"Mid-Fusion v2t Attention Entropy (Sample {samples_saved})")
                plt.tight_layout()
                plt.savefig(
                    os.path.join(output_dir, f"midfusion_v2t_entropy_sample_{samples_saved}.png"),
                    dpi=150, bbox_inches='tight'
                )
                plt.close()

                samples_saved += 1

            if samples_saved >= num_samples:
                break

    # Disable attention weight storage
    model.store_attention_weights = False
    model._mid_fusion_attn_weights = None

    print(f"[Visualizer] Saved {samples_saved} mid-fusion attention visualizations to {output_dir}")
    return samples_saved


def visualize_mid_fusion_token_attention(
    model,
    dataloader,
    device,
    output_dir,
    num_samples=5,
    tokens_to_show=5,
    use_amp=False,
    image_size=224,
    patch_size=16
):
    """
    Visualize per-token cross-attention maps at each mid-fusion layer.

    Shows which image patches each individual text token attends to,
    one figure per sample with rows = fusion layers, columns = tokens.

    Args:
        model: Model with mid-fusion modules
        dataloader: DataLoader with images
        device: Device to use
        output_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
        tokens_to_show: Number of text tokens to display per sample
        use_amp: Whether to use automatic mixed precision
        image_size: Original image size
        patch_size: Patch size
    """
    _ensure_matplotlib()
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    grid_size = image_size // patch_size

    model.store_attention_weights = True

    fusion_layers = [idx + 1 for idx in model.mid_fusion_layer_indices]
    num_modules = len(fusion_layers)

    samples_saved = 0
    print(f"\n[Visualizer] Generating per-token mid-fusion attention visualizations...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Token Mid-Fusion Attn"):
            if batch is None:
                continue

            images, text = batch[0].to(device), batch[1].to(device)

            with torch.amp.autocast(device_type=device, enabled=use_amp):
                model(images, text)

            attn_weights = model._mid_fusion_attn_weights
            if attn_weights is None or len(attn_weights) == 0:
                model.store_attention_weights = False
                return 0

            # Build attention mask for valid tokens
            attention_mask = (text != 0).long()

            for i in range(images.shape[0]):
                if samples_saved >= num_samples:
                    break

                img_np = denormalize_image(images[i].cpu())
                mask = attention_mask[i].cpu()
                valid_indices = torch.where(mask > 0)[0]
                num_valid = min(len(valid_indices), tokens_to_show)

                if num_valid == 0:
                    continue

                # rows = fusion layers, columns = original + tokens
                fig, axes = plt.subplots(
                    num_modules, num_valid + 1,
                    figsize=(4 * (num_valid + 1), 4 * num_modules)
                )
                if num_modules == 1:
                    axes = axes[np.newaxis, :]

                for m, (v2t_w, t2v_w) in enumerate(attn_weights):
                    t2v = t2v_w[i].cpu()  # (L, 197)
                    t2v_patches = t2v[:, 1:]  # (L, 196)

                    axes[m, 0].imshow(img_np)
                    axes[m, 0].set_title(f"Layer {fusion_layers[m]}")
                    axes[m, 0].axis('off')

                    for j, token_idx in enumerate(valid_indices[:num_valid]):
                        token_attn = t2v_patches[token_idx]  # (196,)
                        attn_grid = token_attn.view(grid_size, grid_size).numpy()
                        attn_up = upsample_mask(attn_grid, img_np.shape[:2])

                        axes[m, j + 1].imshow(img_np)
                        axes[m, j + 1].imshow(attn_up, cmap='hot', alpha=0.6)
                        if m == 0:
                            axes[m, j + 1].set_title(f"Token {token_idx.item()}")
                        axes[m, j + 1].axis('off')

                plt.suptitle(f"Per-Token Mid-Fusion Attention (Sample {samples_saved})")
                plt.tight_layout()
                plt.savefig(
                    os.path.join(output_dir, f"midfusion_token_attn_sample_{samples_saved}.png"),
                    dpi=150, bbox_inches='tight'
                )
                plt.close()

                samples_saved += 1

            if samples_saved >= num_samples:
                break

    model.store_attention_weights = False
    model._mid_fusion_attn_weights = None

    print(f"[Visualizer] Saved {samples_saved} per-token mid-fusion visualizations")
    return samples_saved


def visualize_filip_alignment(
    model,
    dataloader,
    device,
    output_dir,
    num_samples=5,
    tokens_to_show=6,
    use_amp=False,
    image_size=224,
    patch_size=16
):
    """
    Visualize FILIP local alignment: which image regions each word aligns to.

    For each mid-fusion layer, computes cosine similarity between projected
    patch and token features, then shows a heatmap for the top-N tokens
    (by max similarity to any patch) with the actual decoded word as title.

    Args:
        model: Model with mid-fusion + FILIP projection layers
        dataloader: DataLoader with (images, text, ...) batches
        device: Device to use
        output_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
        tokens_to_show: Number of top tokens to display per sample
        use_amp: Whether to use automatic mixed precision
        image_size: Original image size
        patch_size: ViT patch size
    """
    _ensure_matplotlib()
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    grid_size = image_size // patch_size  # 14

    # We need the tokenizer to decode token IDs back to words
    import open_clip
    wrapper = open_clip.get_tokenizer(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
    hf_tok = wrapper.tokenizer

    if not hasattr(model, 'use_mid_fusion_local_loss') or not model.use_mid_fusion_local_loss:
        print("[Visualizer] FILIP projection layers not found, skipping")
        return 0

    fusion_layers = [idx + 1 for idx in model.mid_fusion_layer_indices]
    num_modules = len(fusion_layers)

    samples_saved = 0
    print(f"\n[Visualizer] Generating {num_samples} FILIP alignment visualizations...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="FILIP Alignment"):
            if batch is None:
                continue

            images, text = batch[0].to(device), batch[1].to(device)

            with torch.amp.autocast(device_type=device, enabled=use_amp):
                _, _, _, local_features, _ = model(images, text)

            if local_features is None or not isinstance(local_features, list):
                print("[Visualizer] No mid-fusion local features, skipping")
                return 0

            # Build attention mask for valid tokens (exclude padding)
            attention_mask = (text != 0).long()

            for i in range(images.shape[0]):
                if samples_saved >= num_samples:
                    break

                img_np = denormalize_image(images[i].cpu())
                mask = attention_mask[i].cpu()  # (L,)
                text_ids = text[i].cpu()  # (L,)

                # Decode all tokens for this sample
                token_words = []
                for tid in text_ids:
                    if tid.item() == 0:
                        token_words.append("[PAD]")
                    else:
                        token_words.append(hf_tok.decode([tid.item()]).strip())

                # Get valid token indices (skip [CLS], [SEP], [PAD])
                valid_mask = mask.bool().clone()
                # Skip CLS (index 0) and SEP tokens
                valid_mask[0] = False
                for j in range(len(text_ids)):
                    if token_words[j] in ("[CLS]", "[SEP]", "[PAD]"):
                        valid_mask[j] = False
                valid_indices = torch.where(valid_mask)[0]

                if len(valid_indices) == 0:
                    continue

                # --- Figure: rows = layers, cols = original + top tokens ---
                n_tokens_show = min(tokens_to_show, len(valid_indices))
                fig, axes = plt.subplots(
                    num_modules, n_tokens_show + 1,
                    figsize=(3.5 * (n_tokens_show + 1), 3.5 * num_modules)
                )
                if num_modules == 1:
                    axes = axes[np.newaxis, :]

                for m, (pf, tf, am) in enumerate(local_features):
                    # pf: (B, M, D), tf: (B, L, D) — projected features
                    patch_feat = F.normalize(pf[i].cpu().float(), dim=-1)  # (M, D)
                    token_feat = F.normalize(tf[i].cpu().float(), dim=-1)  # (L, D)

                    # Cosine similarity: (L, M)
                    cos_sim = token_feat @ patch_feat.t()  # (L, M)

                    # For each valid token, get its max similarity to any patch
                    valid_cos = cos_sim[valid_indices]  # (N_valid, M)
                    max_per_token = valid_cos.max(dim=1).values  # (N_valid,)

                    # Top tokens by max similarity
                    top_k = min(n_tokens_show, len(max_per_token))
                    top_indices_in_valid = max_per_token.topk(top_k).indices
                    top_token_indices = valid_indices[top_indices_in_valid]

                    # Original image in first column
                    axes[m, 0].imshow(img_np)
                    axes[m, 0].set_title(f"Layer {fusion_layers[m]}", fontsize=10, fontweight='bold')
                    axes[m, 0].axis('off')

                    for j, tok_idx in enumerate(top_token_indices):
                        tok_idx = tok_idx.item()
                        word = token_words[tok_idx]
                        sim_to_patches = cos_sim[tok_idx]  # (M,)
                        sim_grid = sim_to_patches.view(grid_size, grid_size).numpy()
                        sim_up = upsample_mask(sim_grid, img_np.shape[:2])

                        axes[m, j + 1].imshow(img_np)
                        im = axes[m, j + 1].imshow(
                            sim_up, cmap='hot', alpha=0.65,
                            vmin=sim_grid.min(), vmax=sim_grid.max()
                        )
                        score = max_per_token[top_indices_in_valid[j]].item()
                        axes[m, j + 1].set_title(f'"{word}" ({score:.2f})', fontsize=9)
                        axes[m, j + 1].axis('off')

                # Build caption from all valid tokens
                caption = " ".join(token_words[idx.item()] for idx in valid_indices)
                if len(caption) > 100:
                    caption = caption[:100] + "..."

                plt.suptitle(
                    f"FILIP Alignment (Sample {samples_saved})\n{caption}",
                    fontsize=10, y=1.02
                )
                plt.tight_layout()
                plt.savefig(
                    os.path.join(output_dir, f"filip_alignment_sample_{samples_saved}.png"),
                    dpi=150, bbox_inches='tight'
                )
                plt.close()

                samples_saved += 1

            if samples_saved >= num_samples:
                break

    print(f"[Visualizer] Saved {samples_saved} FILIP alignment visualizations to {output_dir}")
    return samples_saved
