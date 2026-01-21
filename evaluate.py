import torch
import torch.nn.functional as F
import time
import yaml
import argparse
import os
from tqdm import tqdm

# Absolute imports (Assumes you run from project root)
from models.full_model import ModelABaseline
from data.dataset import create_dataloaders

class Evaluator:
    def __init__(self, model, test_loader, device):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.model.eval()
        self.model.to(device)

    def benchmark_efficiency(self, num_batches=50):
        """
        Measure Throughput (img/sec) and Peak VRAM.
        Works on CUDA, MPS, and CPU devices.
        """
        print(f"\n[Efficiency] Benchmarking over {num_batches} batches on {self.device}...")

        is_cuda = self.device == "cuda" or str(self.device).startswith("cuda")
        is_mps = self.device == "mps" or str(self.device).startswith("mps")

        # 1. Warmup
        with torch.no_grad():
            for i, (images, text) in enumerate(self.test_loader):
                if i >= 5: break
                images, text = images.to(self.device), text.to(self.device)
                _ = self.model(images, text)

        # 2. Reset Monitors & Sync
        if is_cuda:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        elif is_mps:
            torch.mps.synchronize()

        total_images = 0

        # 3. Timing Loop
        start_time = time.time()
        with torch.no_grad():
            # Only use autocast on CUDA
            autocast_enabled = is_cuda
            with torch.amp.autocast(device_type="cuda" if is_cuda else "cpu", enabled=autocast_enabled):
                for i, (images, text) in enumerate(self.test_loader):
                    if i >= num_batches: break
                    images, text = images.to(self.device), text.to(self.device)
                    _ = self.model(images, text)
                    total_images += images.size(0)

        # Sync before measuring time
        if is_cuda:
            torch.cuda.synchronize()
        elif is_mps:
            torch.mps.synchronize()

        elapsed_sec = time.time() - start_time

        # 4. Calculate Stats
        throughput = total_images / elapsed_sec  # img/sec

        # Memory stats (only available on CUDA)
        if is_cuda:
            peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        else:
            peak_mem = 0.0  # Not available on CPU/MPS

        print(f"   >> Throughput: {throughput:.2f} img/sec")
        if is_cuda:
            print(f"   >> Peak VRAM:  {peak_mem:.2f} MB")
        else:
            print(f"   >> Peak VRAM:  N/A (only measured on CUDA)")

        return {"throughput": throughput, "peak_vram_mb": peak_mem}

    def compute_retrieval(self):
        """
        Compute R@1, R@5, R@10 for Image-to-Text and Text-to-Image.
        """
        print("\n[Retrieval] Extracting embeddings...")
        
        img_embs = []
        txt_embs = []
        
        # 1. Extract Embeddings
        with torch.no_grad():
            for images, text in tqdm(self.test_loader, desc="Extracting"):
                images, text = images.to(self.device), text.to(self.device)
                
                # Model returns: img_emb, txt_emb, logits
                i_emb, t_emb, _ = self.model(images, text)
                
                # Normalize (Crucial for Cosine Similarity)
                i_emb = F.normalize(i_emb, dim=-1)
                t_emb = F.normalize(t_emb, dim=-1)
                
                img_embs.append(i_emb.cpu())
                txt_embs.append(t_emb.cpu())
                
        img_embs = torch.cat(img_embs, dim=0)
        txt_embs = torch.cat(txt_embs, dim=0)
        
        print(f"   >> Matrix Shape: {img_embs.shape[0]} images x {txt_embs.shape[0]} texts")

        # 2. Compute Similarity Matrix
        sim_matrix = img_embs @ txt_embs.T 
        
        metrics = {}
        
        # 3. Calculate R@K
        for mode in ['i2t', 't2i']:
            if mode == 't2i':
                sims = sim_matrix.T # Text query -> Image target
            else:
                sims = sim_matrix   # Image query -> Text target
                
            # Ground truth is diagonal (i-th image matches i-th text)
            ground_truth_scores = sims.diag()
            
            # Rank: Count how many scores > ground_truth
            rank_matrix = (sims > ground_truth_scores.unsqueeze(1)).sum(dim=1) + 1
            
            r1 = (rank_matrix <= 1).float().mean().item() * 100
            r5 = (rank_matrix <= 5).float().mean().item() * 100
            r10 = (rank_matrix <= 10).float().mean().item() * 100
            
            metrics[f"{mode}_R@1"] = r1
            metrics[f"{mode}_R@5"] = r5
            metrics[f"{mode}_R@10"] = r10
            
            print(f"   >> {mode.upper()}: R@1: {r1:.2f}% | R@5: {r5:.2f}% | R@10: {r10:.2f}%")
            
        return metrics

# --- ENTRY POINT ---
if __name__ == "__main__":
    def get_config(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (optional - uses pretrained weights if not provided)")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    
    # 1. Setup
    cfg = get_config(args.config)

    # Get device from config, with fallback logic
    config_device = cfg.get('training', {}).get('device', 'cpu')
    if config_device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
    elif config_device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        device = "cpu"
    else:
        device = config_device

    # 2. Load Model
    model = ModelABaseline(cfg)

    # Load Checkpoint (optional - uses pretrained weights if not provided)
    if args.checkpoint:
        print(f"Running evaluation on {device} using checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        print(f"Running evaluation on {device} using pretrained BiomedCLIP weights (no checkpoint)")
    
    # 3. Load Data
    # Only need test_loader
    _, _, test_loader = create_dataloaders(cfg, batch_size=args.batch_size)
    
    # 4. Run Evaluation
    evaluator = Evaluator(model, test_loader, device)
    
    eff_metrics = evaluator.benchmark_efficiency()
    ret_metrics = evaluator.compute_retrieval()
    
    # 5. Save
    final_results = {**eff_metrics, **ret_metrics}
    
    os.makedirs("logs", exist_ok=True)
    with open("logs/eval_results.txt", "w") as f:
        f.write(str(final_results))
    print("\n[Done] Results saved to logs/eval_results.txt")