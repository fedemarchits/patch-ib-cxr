import torch
import yaml
import argparse
import os
import json
from datetime import datetime

# Absolute imports (run from project root)
from models.full_model import ModelABaseline
from data.dataset import create_dataloaders
from engine.evaluator import Evaluator
from engine.visualizer import (
    visualize_attention_samples, visualize_token_attention,
    visualize_mid_fusion_attention, visualize_mid_fusion_token_attention,
    visualize_filip_alignment
)
from engine.grounding_evaluator import evaluate_phrase_grounding, evaluate_mask_grounding

# Try to import wandb (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def save_results_json(results, filepath):
    """Save results to a JSON file, handling numpy types."""
    def convert_to_serializable(obj):
        if hasattr(obj, 'item'):  # numpy/torch scalar
            return obj.item()
        elif hasattr(obj, 'tolist'):  # numpy array
            return obj.tolist()
        return obj

    serializable = {k: convert_to_serializable(v) for k, v in results.items()}
    with open(filepath, 'w') as f:
        json.dump(serializable, f, indent=2)
    return serializable


def create_efficiency_report(eff_metrics, output_dir):
    """Create a formatted efficiency report text file."""
    report_lines = [
        "=" * 60,
        "EFFICIENCY REPORT",
        "=" * 60,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "--- Performance Metrics ---",
        f"Throughput:           {eff_metrics.get('throughput_img_per_sec', 0):.2f} img/sec",
        f"Avg Step Time:        {eff_metrics.get('avg_step_time_ms', 0):.2f} ms/batch",
        f"Latency per Image:    {eff_metrics.get('latency_per_image_ms', 0):.2f} ms/img",
        "",
        "--- Memory Usage ---",
        f"Peak VRAM:            {eff_metrics.get('peak_vram_mb', 0):.2f} MB",
        "",
        "--- Computational Complexity ---",
        f"GFLOPs:               {eff_metrics.get('gflops', 0):.2f}",
        f"Total Parameters:     {eff_metrics.get('total_params', 0):,}",
        f"Trainable Parameters: {eff_metrics.get('trainable_params', 0):,}",
        "",
        "--- Test Configuration ---",
        f"Batch Size:           {eff_metrics.get('batch_size', 'N/A')}",
        f"Batches Tested:       {eff_metrics.get('num_batches_tested', 'N/A')}",
        "=" * 60,
    ]

    report_path = os.path.join(output_dir, "efficiency_report.txt")
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))

    print('\n'.join(report_lines))
    return report_path


def create_results_report(ret_metrics, cls_metrics, output_dir):
    """Create a formatted results report text file."""
    class_names = [
        "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
        "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity",
        "No Finding", "Pleural Effusion", "Pleural Other", "Pneumonia",
        "Pneumothorax", "Support Devices"
    ]

    report_lines = [
        "=" * 60,
        "EVALUATION RESULTS REPORT",
        "=" * 60,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "--- Retrieval Metrics (Zero-Shot) ---",
        "",
        "Image-to-Text (I2T):",
        f"  R@1:   {ret_metrics.get('i2t_R@1', 0):.2f}%",
        f"  R@5:   {ret_metrics.get('i2t_R@5', 0):.2f}%",
        f"  R@10:  {ret_metrics.get('i2t_R@10', 0):.2f}%",
        "",
        "Text-to-Image (T2I):",
        f"  R@1:   {ret_metrics.get('t2i_R@1', 0):.2f}%",
        f"  R@5:   {ret_metrics.get('t2i_R@5', 0):.2f}%",
        f"  R@10:  {ret_metrics.get('t2i_R@10', 0):.2f}%",
        "",
        "--- Clustering Metrics (KMeans, PCA-50, Single-Label Test Samples) ---",
        "",
        f"NMI:      {cls_metrics.get('clustering_nmi', 0):.4f}",
        f"ARI:      {cls_metrics.get('clustering_ari', 0):.4f}",
        f"Purity:   {cls_metrics.get('clustering_purity', 0):.4f}",
        f"Samples:  {cls_metrics.get('clustering_num_single_label_samples', 0)}",
        f"Classes:  {cls_metrics.get('clustering_num_classes_present', 0)}",
        "",
        "Per-Class Sample Distribution:",
    ]

    class_dist = cls_metrics.get('clustering_class_distribution', {})
    for i, name in enumerate(class_names):
        count = class_dist.get(i, class_dist.get(str(i), 0))
        if count > 0:
            report_lines.append(f"  {name:30s}: {count} samples")

    report_lines.extend(["", "=" * 60])

    report_path = os.path.join(output_dir, "results_report.txt")
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))

    print('\n'.join(report_lines))
    return report_path


# --- ENTRY POINT ---
if __name__ == "__main__":
    def get_config(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--visualize", action="store_true", help="Generate attention visualizations")
    parser.add_argument("--num_vis_samples", type=int, default=10, help="Number of samples to visualize")
    parser.add_argument("--output_dir", type=str, default="logs", help="Output directory for results")
    parser.add_argument("--wandb", action="store_true", help="Upload results to Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="Thesis-PatchIB", help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name (auto-generated if not set)")
    parser.add_argument("--ms_cxr_csv", type=str, default=None, help="Path to MS-CXR CSV for phrase grounding eval")
    parser.add_argument("--ms_cxr_image_root", type=str, default="/datasets/MIMIC-CXR", help="Root dir for MIMIC-CXR images")
    args = parser.parse_args()

    # 1. Setup
    cfg = get_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    # Device setup
    config_device = cfg.get('training', {}).get('device', 'cpu')
    if config_device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    else:
        device = config_device

    # Initialize wandb if requested
    wandb_run = None
    if args.wandb and WANDB_AVAILABLE:
        run_name = args.wandb_run_name or f"eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=cfg,
            job_type="evaluation"
        )
        print(f"[WandB] Initialized run: {run_name}")
    elif args.wandb and not WANDB_AVAILABLE:
        print("[WandB] wandb not installed. Skipping upload.")

    # 2. Load Model
    model = ModelABaseline(cfg)

    if args.checkpoint:
        print(f"Running evaluation on {device} using checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        # Handle state dict vs full checkpoint
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"Running evaluation on {device} using pretrained weights")

    # 3. Load Data
    # We need train_loader to train the classification head!
    print("Loading Data...")
    train_loader, _, test_loader = create_dataloaders(cfg, batch_size=args.batch_size, return_labels=True)

    # 4. Run Evaluation
    # Pass train_loader to the evaluator class
    evaluator = Evaluator(model, train_loader, test_loader, device)

    # A. Efficiency
    eff_metrics = evaluator.benchmark_efficiency(batch_size=args.batch_size)

    # B. Retrieval (Zero-Shot)
    ret_metrics = evaluator.compute_retrieval()

    # C. Clustering Evaluation (KMeans)
    cls_metrics = evaluator.evaluate_clustering()

    # D. Pairwise Clustering (which pathologies are most confusable?)
    vis_dir = os.path.join(args.output_dir, "visualizations")
    pairwise_metrics = evaluator.evaluate_pairwise_clustering(output_dir=vis_dir)

    # E. MS-CXR Phrase Grounding (if CSV provided)
    grounding_metrics = {}
    if args.ms_cxr_csv and os.path.exists(args.ms_cxr_csv):
        use_amp = cfg.get('training', {}).get('use_amp', False)

        # E1. FILIP soft heatmap grounding (Models B/C/D with mid-fusion)
        grounding_metrics = evaluate_phrase_grounding(
            model=model,
            csv_path=args.ms_cxr_csv,
            image_root=args.ms_cxr_image_root,
            device=device,
            use_amp=use_amp,
        )

        # E2. Hard mask grounding (Models C and D with masking head)
        mask_grounding_metrics = evaluate_mask_grounding(
            model=model,
            csv_path=args.ms_cxr_csv,
            image_root=args.ms_cxr_image_root,
            device=device,
            use_amp=use_amp,
        )
        grounding_metrics.update(mask_grounding_metrics)

    # F. UMAP Visualization (always run, single-label test samples)
    umap_path = evaluator.generate_umap_visualization(output_dir=vis_dir)

    # F. Attention Visualizations (optional)
    if args.visualize:
        vis_dir = os.path.join(args.output_dir, "visualizations")
        use_amp = cfg.get('training', {}).get('use_amp', False)

        # Visualize attention masks and patch importance
        visualize_attention_samples(
            model=model,
            dataloader=test_loader,
            device=device,
            output_dir=vis_dir,
            num_samples=args.num_vis_samples,
            use_amp=use_amp
        )

        # Visualize per-token attention (if local alignment enabled)
        if cfg.get('model', {}).get('use_local_alignment', False):
            visualize_token_attention(
                model=model,
                dataloader=test_loader,
                device=device,
                output_dir=vis_dir,
                num_samples=min(5, args.num_vis_samples),
                use_amp=use_amp
            )

        # Visualize mid-fusion cross-attention maps (if mid-fusion enabled)
        if cfg.get('model', {}).get('use_mid_fusion', False):
            visualize_mid_fusion_attention(
                model=model,
                dataloader=test_loader,
                device=device,
                output_dir=vis_dir,
                num_samples=args.num_vis_samples,
                use_amp=use_amp
            )
            visualize_mid_fusion_token_attention(
                model=model,
                dataloader=test_loader,
                device=device,
                output_dir=vis_dir,
                num_samples=min(5, args.num_vis_samples),
                use_amp=use_amp
            )

            # FILIP alignment visualization (if FILIP local loss enabled)
            if cfg.get('model', {}).get('mid_fusion_loss_type') == 'filip':
                visualize_filip_alignment(
                    model=model,
                    dataloader=test_loader,
                    device=device,
                    output_dir=vis_dir,
                    num_samples=min(5, args.num_vis_samples),
                    use_amp=use_amp
                )

    # 5. Save Results
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    # Combine all metrics
    all_results = {
        **eff_metrics,
        **ret_metrics,
        **cls_metrics,
        **grounding_metrics,
        "checkpoint": args.checkpoint,
        "config": args.config,
        "device": device,
        "timestamp": datetime.now().isoformat()
    }

    # Save as JSON (machine-readable)
    json_path = os.path.join(args.output_dir, "eval_results.json")
    save_results_json(all_results, json_path)
    print(f"   >> JSON results:     {json_path}")

    # Save efficiency report (human-readable)
    eff_report_path = create_efficiency_report(eff_metrics, args.output_dir)
    print(f"   >> Efficiency report: {eff_report_path}")

    # Save results report (human-readable)
    results_report_path = create_results_report(ret_metrics, cls_metrics, args.output_dir)
    print(f"   >> Results report:    {results_report_path}")

    # 6. Upload to WandB
    if wandb_run:
        print("\n[WandB] Uploading results...")

        # Log metrics
        wandb.log({
            # Efficiency
            "eval/throughput_img_per_sec": eff_metrics.get('throughput_img_per_sec', 0),
            "eval/avg_step_time_ms": eff_metrics.get('avg_step_time_ms', 0),
            "eval/latency_per_image_ms": eff_metrics.get('latency_per_image_ms', 0),
            "eval/peak_vram_mb": eff_metrics.get('peak_vram_mb', 0),
            "eval/gflops": eff_metrics.get('gflops', 0),
            # Retrieval
            "eval/i2t_R@1": ret_metrics.get('i2t_R@1', 0),
            "eval/i2t_R@5": ret_metrics.get('i2t_R@5', 0),
            "eval/i2t_R@10": ret_metrics.get('i2t_R@10', 0),
            "eval/t2i_R@1": ret_metrics.get('t2i_R@1', 0),
            "eval/t2i_R@5": ret_metrics.get('t2i_R@5', 0),
            "eval/t2i_R@10": ret_metrics.get('t2i_R@10', 0),
            # Clustering
            "eval/clustering_nmi": cls_metrics.get('clustering_nmi', 0),
            "eval/clustering_ari": cls_metrics.get('clustering_ari', 0),
            "eval/clustering_purity": cls_metrics.get('clustering_purity', 0),
            "eval/clustering_num_samples": cls_metrics.get('clustering_num_single_label_samples', 0),
        })

        # Log UMAP image if generated
        if umap_path and os.path.exists(umap_path):
            wandb.log({"eval/umap_embeddings": wandb.Image(umap_path)})

        # Upload files as artifacts
        artifact = wandb.Artifact(
            name=f"eval-results-{wandb_run.id}",
            type="evaluation",
            description="Evaluation results including efficiency, retrieval, and classification metrics"
        )
        artifact.add_file(json_path)
        artifact.add_file(eff_report_path)
        artifact.add_file(results_report_path)

        # Add visualizations if they exist
        vis_dir = os.path.join(args.output_dir, "visualizations")
        if os.path.exists(vis_dir):
            artifact.add_dir(vis_dir, name="visualizations")

        wandb_run.log_artifact(artifact)
        print(f"   >> Uploaded artifact: eval-results-{wandb_run.id}")

        wandb.finish()
        print("[WandB] Run finished.")

    print(f"\n[Done] All results saved to {args.output_dir}/")
