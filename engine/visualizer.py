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
