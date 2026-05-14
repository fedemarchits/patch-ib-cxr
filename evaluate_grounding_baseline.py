"""
Zero-shot BiomedCLIP grounding baseline.

Computes phrase grounding (CNT + mIoU) using the pretrained BiomedCLIP model
directly — no fine-tuning, no learned projection heads.

FILIP scores are computed from:
  - Visual: final ViT block patch features (layer 12), projected to 512-d via BiomedCLIP's visual head
  - Text:   final PubMedBERT token features, projected to 512-d via BiomedCLIP's text projection

This gives the zero-shot grounding performance of the base model, which serves
as a baseline to compare against all fine-tuned models.

Usage:
    python evaluate_grounding_baseline.py \
        --ms_cxr_csv ms-cxr/1.1.0/MS_CXR_Local_Alignment_v1.1.0.csv \
        --ms_cxr_image_root Results/ms_cxr_images \
        --output_json Results/grounding_baseline.json
"""

import torch
import torch.nn.functional as F
import open_clip
import numpy as np
import json
import argparse
import os
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from datetime import datetime

from engine.grounding_evaluator import (
    load_ms_cxr_annotations, _bbox_to_mask, _compute_cnt, _compute_iou
)

MODEL_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


def get_patch_and_token_features(clip_model, images, text_tokens, device):
    """
    Extract final-layer patch features (visual) and token features (text)
    from BiomedCLIP, both projected into the shared 512-d embedding space.

    Returns:
        patch_feats: (B, 196, 512) normalized patch features
        token_feats: (B, L, 512)  normalized token features
        attn_mask:   (B, L)       1=real token, 0=padding
    """
    visual_model = clip_model.visual
    text_model = clip_model.text

    # ── Visual: run ViT trunk, get all patch tokens at final layer ──────────
    vit_trunk = visual_model.trunk
    x = vit_trunk.patch_embed(images)
    x = vit_trunk._pos_embed(x)
    x = vit_trunk.patch_drop(x)
    x = vit_trunk.norm_pre(x)
    for block in vit_trunk.blocks:
        x = block(x)
    x = vit_trunk.norm(x)  # (B, 197, 768)

    patch_feats_768 = x[:, 1:, :]  # drop CLS → (B, 196, 768)

    # Apply visual projection head (768 → 512)
    if hasattr(visual_model, 'proj') and visual_model.proj is not None:
        patch_feats = patch_feats_768 @ visual_model.proj.to(patch_feats_768.dtype)
    elif hasattr(visual_model, 'head') and visual_model.head is not None:
        # head expects (B, D) — apply per-patch
        B, N, D = patch_feats_768.shape
        patch_feats = visual_model.head(patch_feats_768.view(B * N, D)).view(B, N, -1)
    else:
        patch_feats = patch_feats_768  # fallback: use 768-d features

    patch_feats = F.normalize(patch_feats.float(), dim=-1)  # (B, 196, 512)

    # ── Text: run PubMedBERT, get all token hidden states ───────────────────
    attn_mask = (text_tokens != 0).long()  # (B, L)

    out = text_model.transformer(
        input_ids=text_tokens,
        attention_mask=attn_mask
    )
    token_feats_768 = out.last_hidden_state  # (B, L, 768)

    # Apply text projection (768 → 512) if available
    if hasattr(text_model, 'proj') and text_model.proj is not None:
        proj = text_model.proj
        if callable(proj) and not isinstance(proj, torch.Tensor):
            # Sequential or nn.Module
            B, L, D = token_feats_768.shape
            token_feats = proj(token_feats_768.view(B * L, D)).view(B, L, -1)
        else:
            token_feats = token_feats_768 @ proj.to(token_feats_768.dtype)
    else:
        token_feats = token_feats_768

    token_feats = F.normalize(token_feats.float(), dim=-1)  # (B, L, 512)

    return patch_feats, token_feats, attn_mask


def evaluate_biomedclip_grounding(ms_cxr_csv, image_root, device,
                                   image_size=224, grid_size=14, split="test"):
    print(f"Loading BiomedCLIP ({MODEL_NAME})...")
    clip_model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME, device=device)
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    clip_model.eval()
    print("   >> Model loaded.")

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    samples = load_ms_cxr_annotations(ms_cxr_csv, split=split)
    if not samples:
        return {}

    results = defaultdict(lambda: {"cnt": [], "iou": []})
    skipped = 0

    with torch.no_grad():
        for sample in tqdm(samples, desc="Grounding (BiomedCLIP baseline)"):
            img_path = os.path.join(image_root, sample["path"])
            if not os.path.exists(img_path):
                img_path = os.path.join(image_root, os.path.basename(sample["path"]))
            if not os.path.exists(img_path):
                skipped += 1
                continue

            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                skipped += 1
                continue

            img_tensor = transform(img).unsqueeze(0).to(device)
            text_tokens = tokenizer(sample["label_text"]).to(device)

            gt_mask = _bbox_to_mask(
                sample["bboxes"],
                sample["image_width"], sample["image_height"],
                grid_size=grid_size
            )
            if gt_mask.sum() == 0:
                skipped += 1
                continue

            patch_feats, token_feats, attn_mask = get_patch_and_token_features(
                clip_model, img_tensor, text_tokens, device
            )

            # FILIP: per-patch max cosine sim over valid text tokens
            valid_mask = (attn_mask[0] == 1)  # (L,)
            valid_mask[0] = False  # skip CLS
            # skip SEP (last valid token)
            valid_indices = valid_mask.nonzero(as_tuple=True)[0]
            if len(valid_indices) <= 1:
                skipped += 1
                continue
            valid_indices = valid_indices[:-1]  # remove SEP

            # patch_feats: (1, 196, 512), token_feats: (1, L, 512)
            pf = patch_feats[0]  # (196, 512)
            tf = token_feats[0][valid_indices]  # (N_valid, 512)

            sim = tf @ pf.t()  # (N_valid, 196)
            heatmap = sim.mean(dim=0).cpu().numpy().reshape(grid_size, grid_size)

            cnt = _compute_cnt(heatmap, gt_mask)
            iou = _compute_iou(heatmap, gt_mask)

            results[sample["category_name"]]["cnt"].append(cnt)
            results[sample["category_name"]]["iou"].append(iou)

    if skipped > 0:
        print(f"   >> Skipped {skipped} samples")

    output = {}
    all_cnt, all_iou = [], []

    print(f"\n   >> Results for BiomedCLIP zero-shot (final layer):")
    print(f"      {'Category':30s} {'CNT':>8s} {'mIoU':>8s} {'N':>6s}")
    print(f"      {'-'*56}")

    for category in sorted(results.keys()):
        r = results[category]
        cnt_mean = np.mean(r["cnt"]) if r["cnt"] else 0
        iou_mean = np.mean(r["iou"]) if r["iou"] else 0
        n = len(r["cnt"])
        all_cnt.extend(r["cnt"])
        all_iou.extend(r["iou"])
        print(f"      {category:30s} {cnt_mean:8.3f} {iou_mean:8.3f} {n:6d}")
        output[f"grounding/biomedclip_baseline/{category}/cnt"] = cnt_mean
        output[f"grounding/biomedclip_baseline/{category}/miou"] = iou_mean

    overall_cnt = np.mean(all_cnt) if all_cnt else 0
    overall_iou = np.mean(all_iou) if all_iou else 0
    print(f"      {'-'*56}")
    print(f"      {'OVERALL':30s} {overall_cnt:8.3f} {overall_iou:8.3f} {len(all_cnt):6d}")
    output["grounding/biomedclip_baseline/overall_cnt"] = overall_cnt
    output["grounding/biomedclip_baseline/overall_miou"] = overall_iou

    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ms_cxr_csv", type=str, required=True)
    parser.add_argument("--ms_cxr_image_root", type=str, default="Results/ms_cxr_images")
    parser.add_argument("--output_json", type=str, default="Results/grounding_baseline.json")
    args = parser.parse_args()

    device = "cpu"  # safe for Mac/CPU-only

    metrics = evaluate_biomedclip_grounding(
        ms_cxr_csv=args.ms_cxr_csv,
        image_root=args.ms_cxr_image_root,
        device=device,
    )

    metrics["timestamp"] = datetime.now().isoformat()
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n   >> Saved to {args.output_json}")


if __name__ == "__main__":
    main()
