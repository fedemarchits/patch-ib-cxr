"""
Quick script to visualize MS-CXR samples with bounding boxes.
Run on server: python show_ms_cxr_sample.py [--n 5] [--category Cardiomegaly]
Saves PNGs to ms_cxr_examples/ directory.
"""

import csv
import os
import argparse
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont

MS_CXR_CSV = "/datasets/ms-cxr/1.1.0/MS_CXR_Local_Alignment_v1.1.0.csv"
IMAGE_ROOT = "/datasets/MIMIC-CXR"
OUTPUT_DIR = "ms_cxr_examples"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=8, help="Number of samples to show")
    parser.add_argument("--category", type=str, default=None, help="Filter by category")
    parser.add_argument("--split", type=str, default="test", help="Split to use")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load and group annotations
    grouped = defaultdict(lambda: {"bboxes": [], "category": None, "text": None, "path": None, "iw": 0, "ih": 0})

    with open(MS_CXR_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] != args.split:
                continue
            if args.category and row["category_name"] != args.category:
                continue

            key = (row["dicom_id"], row["label_text"])
            entry = grouped[key]
            entry["category"] = row["category_name"]
            entry["text"] = row["label_text"]
            entry["path"] = row["path"]
            entry["iw"] = int(row["image_width"])
            entry["ih"] = int(row["image_height"])
            entry["bboxes"].append((
                int(row["x"]), int(row["y"]),
                int(row["w"]), int(row["h"]),
            ))

    samples = list(grouped.values())
    print(f"Found {len(samples)} samples in {args.split} split")

    # Show category distribution
    cats = defaultdict(int)
    for s in samples:
        cats[s["category"]] += 1
    print("\nCategory distribution:")
    for cat, count in sorted(cats.items()):
        print(f"  {cat:30s}: {count}")

    # Pick diverse samples (one per category if possible)
    if args.category is None:
        selected = []
        for cat in sorted(cats.keys()):
            cat_samples = [s for s in samples if s["category"] == cat]
            selected.append(cat_samples[0])
            if len(selected) >= args.n:
                break
        # Fill remaining with any samples
        if len(selected) < args.n:
            for s in samples:
                if s not in selected:
                    selected.append(s)
                    if len(selected) >= args.n:
                        break
    else:
        selected = samples[:args.n]

    # Draw bounding boxes
    colors = {
        "Atelectasis": "red", "Cardiomegaly": "blue", "Consolidation": "green",
        "Edema": "yellow", "Lung Opacity": "orange", "Pleural Effusion": "cyan",
        "Pneumonia": "magenta", "Pneumothorax": "lime",
    }

    for i, sample in enumerate(selected):
        img_path = os.path.join(IMAGE_ROOT, sample["path"])
        if not os.path.exists(img_path):
            print(f"  [SKIP] {img_path} not found")
            continue

        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        color = colors.get(sample["category"], "red")

        for (x, y, w, h) in sample["bboxes"]:
            # Draw bbox with thick border
            for offset in range(3):
                draw.rectangle(
                    [x - offset, y - offset, x + w + offset, y + h + offset],
                    outline=color
                )

        # Add text label at top
        label = f'{sample["category"]}: "{sample["text"]}"'
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        except (IOError, OSError):
            font = ImageFont.load_default()

        # Background for text
        bbox = draw.textbbox((10, 10), label, font=font)
        draw.rectangle([bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2], fill="black")
        draw.text((10, 10), label, fill=color, font=font)

        save_path = os.path.join(OUTPUT_DIR, f"ms_cxr_{args.split}_{i}_{sample['category']}.png")
        img.save(save_path)
        print(f"  Saved: {save_path}  ({sample['category']}: {sample['text']}, {len(sample['bboxes'])} boxes)")

    print(f"\nDone! Check {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
