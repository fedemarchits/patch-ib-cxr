"""
MS-CXR Phrase Grounding Evaluation.

Evaluates whether the model's local alignment (FILIP or cross-attention)
produces heatmaps that correctly localize radiological findings.

Metrics:
- CNT (Pointing Game): Is the max activation inside the ground truth bbox?
- mIoU: Mean IoU between thresholded heatmap and GT bbox mask.

Reference: MS-CXR v1.1.0 (Boecking et al., PhysioNet)
"""

import os
import csv
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
from torchvision import transforms


def load_ms_cxr_annotations(csv_path, split="test"):
    """
    Load MS-CXR annotations and group by (dicom_id, label_text).

    Returns:
        list of dicts, each with:
            - dicom_id, category_name, label_text, path
            - bboxes: list of (x, y, w, h) in original image coords
            - image_width, image_height
    """
    raw = defaultdict(lambda: {
        "bboxes": [], "category_name": None, "label_text": None,
        "path": None, "image_width": None, "image_height": None,
    })

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] != split:
                continue

            key = (row["dicom_id"], row["label_text"])
            entry = raw[key]
            entry["dicom_id"] = row["dicom_id"]
            entry["category_name"] = row["category_name"]
            entry["label_text"] = row["label_text"]
            entry["path"] = row["path"]
            entry["image_width"] = int(row["image_width"])
            entry["image_height"] = int(row["image_height"])
            entry["bboxes"].append((
                int(row["x"]), int(row["y"]),
                int(row["w"]), int(row["h"]),
            ))

    samples = list(raw.values())
    print(f"   >> Loaded {len(samples)} phrase-bbox groups from {split} split")

    # Stats
    categories = defaultdict(int)
    for s in samples:
        categories[s["category_name"]] += 1
    for cat, count in sorted(categories.items()):
        print(f"      {cat:30s}: {count}")

    return samples


def _bbox_to_mask(bboxes, image_width, image_height, grid_size=14):
    """
    Convert list of bboxes to a binary mask on the patch grid.

    Args:
        bboxes: list of (x, y, w, h) in original image coordinates
        image_width, image_height: original image dimensions
        grid_size: patch grid size (14 for ViT-B/16 with 224px)

    Returns:
        mask: (grid_size, grid_size) binary numpy array
    """
    mask = np.zeros((grid_size, grid_size), dtype=np.float32)

    for (x, y, w, h) in bboxes:
        # Convert bbox to grid coordinates
        x1 = int(x / image_width * grid_size)
        y1 = int(y / image_height * grid_size)
        x2 = min(grid_size, int((x + w) / image_width * grid_size) + 1)
        y2 = min(grid_size, int((y + h) / image_height * grid_size) + 1)

        mask[y1:y2, x1:x2] = 1.0

    return mask


def _compute_cnt(heatmap, gt_mask):
    """
    Pointing game (CNT): is the argmax of the heatmap inside the GT mask?

    Args:
        heatmap: (H, W) numpy array
        gt_mask: (H, W) binary numpy array

    Returns:
        1.0 if correct, 0.0 otherwise
    """
    max_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    return float(gt_mask[max_idx] > 0)


def _compute_iou(heatmap, gt_mask, threshold_percentile=50):
    """
    IoU between thresholded heatmap and GT bbox mask.

    Args:
        heatmap: (H, W) numpy array
        gt_mask: (H, W) binary numpy array
        threshold_percentile: percentile threshold for binarizing heatmap

    Returns:
        IoU score (float)
    """
    threshold = np.percentile(heatmap, threshold_percentile)
    pred_mask = (heatmap >= threshold).astype(np.float32)

    intersection = (pred_mask * gt_mask).sum()
    union = ((pred_mask + gt_mask) > 0).sum()

    if union == 0:
        return 0.0
    return float(intersection / union)


def evaluate_phrase_grounding(
    model, csv_path, image_root, device,
    use_amp=False, image_size=224, grid_size=14, split="test"
):
    """
    Evaluate phrase grounding on MS-CXR using FILIP local features.

    For each (image, phrase) pair:
    1. Tokenize the phrase and run through model
    2. Compute cosine similarity between phrase tokens and image patches
    3. Compare resulting heatmap against GT bounding boxes

    Args:
        model: Model with mid-fusion + FILIP projection layers
        csv_path: Path to MS_CXR_Local_Alignment CSV
        image_root: Root directory for MIMIC-CXR images (parent of 'files/')
        device: Device to use
        use_amp: Whether to use AMP
        image_size: Model input size
        grid_size: Patch grid size (14 for ViT-B/16)
        split: Which split to evaluate ('test', 'val', 'train')

    Returns:
        dict with per-category and overall CNT and mIoU
    """
    import open_clip

    model.eval()
    print(f"\n[Grounding] MS-CXR Phrase Grounding Evaluation ({split} split)...")

    # Load annotations
    samples = load_ms_cxr_annotations(csv_path, split=split)
    if len(samples) == 0:
        print("   >> No samples found!")
        return {}

    # Setup tokenizer and image transform
    tokenizer = open_clip.get_tokenizer(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Check model has FILIP projections
    has_filip = (
        hasattr(model, 'use_mid_fusion_local_loss') and model.use_mid_fusion_local_loss
    )
    if not has_filip:
        print("   >> No mid-fusion projection layers found, skipping grounding eval")
        return {}

    fusion_layers = [idx + 1 for idx in model.mid_fusion_layer_indices]
    num_modules = len(fusion_layers)

    # Per-layer, per-category results
    results_per_layer = {m: defaultdict(lambda: {"cnt": [], "iou": []}) for m in range(num_modules)}
    skipped = 0

    with torch.no_grad():
        for sample in tqdm(samples, desc="Grounding"):
            # Load image
            img_path = os.path.join(image_root, sample["path"])
            if not os.path.exists(img_path):
                skipped += 1
                continue

            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                skipped += 1
                continue

            img_tensor = transform(img).unsqueeze(0).to(device)  # (1, 3, 224, 224)

            # Tokenize the phrase
            text_tokens = tokenizer(sample["label_text"]).to(device)  # (1, L)

            # Forward pass
            with torch.amp.autocast(device_type=str(device), enabled=use_amp):
                _, _, _, local_features, _ = model(img_tensor, text_tokens)

            if local_features is None or not isinstance(local_features, list):
                skipped += 1
                continue

            # Build attention mask (skip CLS, SEP, PAD)
            attention_mask = (text_tokens[0] != 0).cpu()
            # Also skip CLS (pos 0) and find SEP to skip it
            valid_mask = attention_mask.clone()
            valid_mask[0] = False  # CLS
            # Find SEP: first 0-to-nonzero transition after valid tokens
            for j in range(len(valid_mask)):
                if not attention_mask[j]:
                    break
                if j > 0 and j == attention_mask.sum().item() - 1:
                    valid_mask[j] = False  # SEP is last valid token

            valid_indices = torch.where(valid_mask)[0]
            if len(valid_indices) == 0:
                skipped += 1
                continue

            # GT mask on patch grid
            gt_mask = _bbox_to_mask(
                sample["bboxes"],
                sample["image_width"], sample["image_height"],
                grid_size=grid_size
            )

            if gt_mask.sum() == 0:
                skipped += 1
                continue

            category = sample["category_name"]

            # Evaluate each layer
            for m, (pf, tf, am) in enumerate(local_features):
                # pf: (1, M, D), tf: (1, L, D)
                patch_feat = F.normalize(pf[0].cpu().float(), dim=-1)  # (M, D)
                token_feat = F.normalize(tf[0].cpu().float(), dim=-1)  # (L, D)

                # Cosine similarity for valid phrase tokens: (N_valid, M)
                phrase_sim = token_feat[valid_indices] @ patch_feat.t()

                # Average across tokens → (M,) heatmap
                heatmap = phrase_sim.mean(dim=0).numpy()
                heatmap = heatmap.reshape(grid_size, grid_size)

                cnt = _compute_cnt(heatmap, gt_mask)
                iou = _compute_iou(heatmap, gt_mask)

                results_per_layer[m][category]["cnt"].append(cnt)
                results_per_layer[m][category]["iou"].append(iou)

    if skipped > 0:
        print(f"   >> Skipped {skipped} samples (missing images or invalid)")

    # Aggregate results
    output = {}

    for m in range(num_modules):
        layer_name = f"layer_{fusion_layers[m]}"
        print(f"\n   >> Results for {layer_name}:")
        print(f"      {'Category':30s} {'CNT':>8s} {'mIoU':>8s} {'N':>6s}")
        print(f"      {'-'*56}")

        all_cnt = []
        all_iou = []

        for category in sorted(results_per_layer[m].keys()):
            r = results_per_layer[m][category]
            cnt_mean = np.mean(r["cnt"]) if r["cnt"] else 0
            iou_mean = np.mean(r["iou"]) if r["iou"] else 0
            n = len(r["cnt"])

            all_cnt.extend(r["cnt"])
            all_iou.extend(r["iou"])

            print(f"      {category:30s} {cnt_mean:8.3f} {iou_mean:8.3f} {n:6d}")

            output[f"grounding/{layer_name}/{category}/cnt"] = cnt_mean
            output[f"grounding/{layer_name}/{category}/miou"] = iou_mean

        overall_cnt = np.mean(all_cnt) if all_cnt else 0
        overall_iou = np.mean(all_iou) if all_iou else 0
        print(f"      {'-'*56}")
        print(f"      {'OVERALL':30s} {overall_cnt:8.3f} {overall_iou:8.3f} {len(all_cnt):6d}")

        output[f"grounding/{layer_name}/overall_cnt"] = overall_cnt
        output[f"grounding/{layer_name}/overall_miou"] = overall_iou

    return output
