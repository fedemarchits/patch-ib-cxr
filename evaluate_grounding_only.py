"""
Grounding-only evaluation script.

Runs phrase grounding (CNT + mIoU) for ALL model families:
  - B/C/D mid-fusion models: uses evaluate_phrase_grounding() (multi-layer FILIP via local_features)
  - F/G/H intra-ViT drop models: uses get_intermediate_patch_scores() (single-layer FILIP at drop_layer)
  - Mask grounding (Recall/Precision/IoU) for C/D masking models

Results are MERGED into the existing eval_results.json without touching
retrieval, clustering, or deletion/insertion metrics.

Usage:
    python evaluate_grounding_only.py \
        --config configs/model_f_filip_drop.yaml \
        --checkpoint logs/model_f_filip_drop/best_model.pt \
        --output_dir logs/model_f_filip_drop \
        --ms_cxr_csv /path/to/MS_CXR_Local_Alignment_v1.0.0.csv \
        --ms_cxr_image_root /datasets/MIMIC-CXR
"""

import torch
import torch.nn.functional as F
import yaml
import argparse
import os
import json
import numpy as np
from collections import defaultdict
from datetime import datetime
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

from models.full_model import ModelABaseline, ModelE, ModelF, ModelFAdaptive, ModelG, ModelH
from engine.grounding_evaluator import (
    evaluate_phrase_grounding, evaluate_mask_grounding,
    load_ms_cxr_annotations, _bbox_to_mask, _compute_cnt, _compute_iou
)


def load_model(cfg, checkpoint_path, device):
    # Force CPU in config so backbone doesn't try to init on CUDA
    cfg.setdefault('training', {})['device'] = device
    if cfg.get('model', {}).get('use_soft_modulation', False):
        model = ModelH(cfg)
        print("Instantiated ModelH")
    elif cfg.get('model', {}).get('use_gradual_drop', False):
        model = ModelG(cfg)
        print("Instantiated ModelG")
    elif cfg.get('model', {}).get('use_filip_adaptive', False):
        model = ModelFAdaptive(cfg)
        print("Instantiated ModelFAdaptive")
    elif cfg.get('model', {}).get('use_filip_drop', False):
        model = ModelF(cfg)
        print("Instantiated ModelF")
    elif cfg.get('model', {}).get('use_mid_drop', False):
        model = ModelE(cfg)
        print("Instantiated ModelE")
    else:
        model = ModelABaseline(cfg)
        print("Instantiated ModelABaseline")

    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
        print(f"   >> Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"   >> WARNING: checkpoint not found: {checkpoint_path}")

    model.to(device)
    model.eval()
    return model


def evaluate_filip_drop_grounding(model, csv_path, image_root, device,
                                   use_amp=False, image_size=224, grid_size=14, split="test"):
    """
    Phrase grounding for F/G/H models using get_intermediate_patch_scores().

    These models compute per-patch FILIP scores at drop_layer via probe_patch_proj
    and probe_token_proj — the same projection heads used during training.
    The score vector (196,) is reshaped to (14,14) and evaluated with CNT + mIoU.

    This is directly comparable to evaluate_phrase_grounding() for B/C/D models.
    """
    import open_clip

    if not hasattr(model, 'get_intermediate_patch_scores'):
        print("   >> Model does not have get_intermediate_patch_scores(), skipping")
        return {}

    drop_layer = getattr(model, 'drop_layer', '?')
    print(f"   >> F/G/H grounding via FILIP scores at drop_layer={drop_layer}")

    tokenizer = open_clip.get_tokenizer(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    samples = load_ms_cxr_annotations(csv_path, split=split)
    if len(samples) == 0:
        return {}

    results = defaultdict(lambda: {"cnt": [], "iou": []})
    skipped = 0

    with torch.no_grad():
        for sample in tqdm(samples, desc="Grounding (F/G/H)"):
            img_path = os.path.join(image_root, sample["path"])
            if not os.path.exists(img_path):
                # Fallback: flat directory (just basename)
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

            with torch.amp.autocast(device_type="cpu", enabled=False):
                # (1, 196) per-patch FILIP scores at drop_layer
                scores = model.get_intermediate_patch_scores(img_tensor, text_tokens)

            heatmap = scores[0].cpu().float().numpy().reshape(grid_size, grid_size)

            cnt = _compute_cnt(heatmap, gt_mask)
            iou = _compute_iou(heatmap, gt_mask)

            results[sample["category_name"]]["cnt"].append(cnt)
            results[sample["category_name"]]["iou"].append(iou)

    if skipped > 0:
        print(f"   >> Skipped {skipped} samples (missing images or invalid GT)")

    output = {}
    layer_name = f"filip_layer_{drop_layer}"
    all_cnt, all_iou = [], []

    print(f"\n   >> Results for {layer_name}:")
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
        output[f"grounding/{layer_name}/{category}/cnt"] = cnt_mean
        output[f"grounding/{layer_name}/{category}/miou"] = iou_mean

    overall_cnt = np.mean(all_cnt) if all_cnt else 0
    overall_iou = np.mean(all_iou) if all_iou else 0
    print(f"      {'-'*56}")
    print(f"      {'OVERALL':30s} {overall_cnt:8.3f} {overall_iou:8.3f} {len(all_cnt):6d}")
    output[f"grounding/{layer_name}/overall_cnt"] = overall_cnt
    output[f"grounding/{layer_name}/overall_miou"] = overall_iou

    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--ms_cxr_csv", type=str, required=True)
    parser.add_argument("--ms_cxr_image_root", type=str, default="/datasets/MIMIC-CXR")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cpu"
    use_amp = cfg.get('training', {}).get('use_amp', False)

    print(f"\n{'='*60}")
    print(f"GROUNDING-ONLY EVALUATION")
    print(f"Config:     {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device:     {device}")
    print(f"{'='*60}\n")

    model = load_model(cfg, args.checkpoint, device)

    grounding_metrics = {}

    # Path A: F/G/H models — use get_intermediate_patch_scores()
    if hasattr(model, 'get_intermediate_patch_scores'):
        print("\n[1] F/G/H Phrase Grounding (FILIP at drop_layer)...")
        grounding_metrics.update(evaluate_filip_drop_grounding(
            model=model,
            csv_path=args.ms_cxr_csv,
            image_root=args.ms_cxr_image_root,
            device=device,
            use_amp=use_amp,
        ))

    # Path B: B/C/D mid-fusion models — use standard evaluate_phrase_grounding()
    else:
        print("\n[1] Phrase Grounding (CNT + mIoU via mid-fusion FILIP)...")
        grounding_metrics.update(evaluate_phrase_grounding(
            model=model,
            csv_path=args.ms_cxr_csv,
            image_root=args.ms_cxr_image_root,
            device=device,
            use_amp=use_amp,
        ))

    # Path C: Mask grounding for C/D models (always attempted, returns {} if not applicable)
    print("\n[2] Mask Grounding (Recall/Precision/IoU for masking models)...")
    grounding_metrics.update(evaluate_mask_grounding(
        model=model,
        csv_path=args.ms_cxr_csv,
        image_root=args.ms_cxr_image_root,
        device=device,
        use_amp=use_amp,
    ))

    if not grounding_metrics:
        print("\n   >> No grounding metrics produced. Model has no mid-fusion or FILIP drop layers.")
        print("   >> eval_results.json unchanged.")
        return

    # Merge into existing eval_results.json
    json_path = os.path.join(args.output_dir, "eval_results.json")
    existing = {}
    if os.path.exists(json_path):
        with open(json_path) as f:
            existing = json.load(f)
        print(f"\n   >> Loaded existing results from {json_path}")
    else:
        print(f"\n   >> No existing JSON found, creating new one at {json_path}")

    existing.update(grounding_metrics)
    existing["grounding_eval_timestamp"] = datetime.now().isoformat()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(existing, f, indent=2)

    print(f"\n   >> Saved merged results to {json_path}")
    print(f"   >> {len(grounding_metrics)} grounding keys added/updated.")


if __name__ == "__main__":
    main()
