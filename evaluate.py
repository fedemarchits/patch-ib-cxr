import torch
import yaml
import argparse
import os

# Absolute imports (run from project root)
from models.full_model import ModelABaseline
from data.dataset import create_dataloaders
from engine.evaluator import Evaluator

# --- ENTRY POINT ---
if __name__ == "__main__":
    def get_config(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    
    # 1. Setup
    cfg = get_config(args.config)
    
    # Device setup
    config_device = cfg.get('training', {}).get('device', 'cpu')
    if config_device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    else:
        device = config_device

    # 2. Load Model
    model = ModelABaseline(cfg)

    if args.checkpoint:
        print(f"Running evaluation on {device} using checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        # Handle state dict vs full checkpoint
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"Running evaluation on {device} using pretrained weights")
    
    # 3. Load Data (UPDATED)
    # We need train_loader to train the classification head!
    print("Loading Data...")
    train_loader, _, test_loader = create_dataloaders(cfg, batch_size=args.batch_size, return_labels=True)
    
    # 4. Run Evaluation
    # Pass train_loader to the evaluator class
    evaluator = Evaluator(model, train_loader, test_loader, device)
    
    # A. Efficiency
    eff_metrics = evaluator.benchmark_efficiency()
    
    # B. Retrieval (Zero-Shot)
    ret_metrics = evaluator.compute_retrieval()
    
    # C. Classification (Linear Probe) <--- NEW TASK
    cls_metrics = evaluator.evaluate_classification()
    
    # 5. Save
    final_results = {**eff_metrics, **ret_metrics, **cls_metrics}
    
    os.makedirs("logs", exist_ok=True)
    with open("logs/eval_results.txt", "w") as f:
        f.write(str(final_results))
        
    print("\n[Done] Results saved to logs/eval_results.txt")
    print(final_results)