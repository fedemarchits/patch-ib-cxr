import torch
import yaml
import argparse
import os

# Absolute imports (run from project root)
from models.full_model import ModelABaseline
from data.dataset import create_dataloaders
from engine.evaluator import Evaluator
from engine.visualizer import visualize_attention_samples, visualize_token_attention

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
    args = parser.parse_args()
    
    # 1. Setup
    cfg = get_config(args.config)
    
    # Device setup
    config_device = cfg.get('training', {}).get('device', 'cpu')
    if config_device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    else:
        device = config_device

    # 2. Load Data FIRST (before checkpoint to avoid tokenizer conflicts)
    # We need train_loader to train the classification head!
    print("Loading Data...")
    train_loader, _, test_loader = create_dataloaders(cfg, batch_size=args.batch_size, return_labels=True)

    # 3. Load Model
    model = ModelABaseline(cfg)

    if args.checkpoint:
        print(f"Running evaluation on {device} using checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        # Handle state dict vs full checkpoint
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"Running evaluation on {device} using pretrained weights")
    
    # 4. Run Evaluation (model is now step 3, data is step 2)
    # Pass train_loader to the evaluator class
    evaluator = Evaluator(model, train_loader, test_loader, device)
    
    # A. Efficiency
    eff_metrics = evaluator.benchmark_efficiency()
    
    # B. Retrieval (Zero-Shot)
    ret_metrics = evaluator.compute_retrieval()
    
    # C. Classification (Linear Probe) <--- NEW TASK
    cls_metrics = evaluator.evaluate_classification()

    # D. Attention Visualizations (optional)
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

    # 5. Save
    final_results = {**eff_metrics, **ret_metrics, **cls_metrics}

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "eval_results.txt"), "w") as f:
        f.write(str(final_results))

    print(f"\n[Done] Results saved to {args.output_dir}/eval_results.txt")
    print(final_results)