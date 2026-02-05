import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np


class Evaluator:
    def __init__(self, model, train_loader, test_loader, device):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.model.eval()
        self.model.to(device)
        self.train_loader = train_loader

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
            for i, batch in enumerate(self.test_loader):
                if batch is None:
                    continue
                if i >= 5: break
                images, text = batch[0].to(self.device), batch[1].to(self.device)
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
                for i, batch in enumerate(self.test_loader):
                    if batch is None:
                        continue
                    if i >= num_batches: break
                    images, text = batch[0].to(self.device), batch[1].to(self.device)
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
            for batch in tqdm(self.test_loader, desc="Extracting"):
                
                images, text = batch[0].to(self.device), batch[1].to(self.device)
                
                # Model returns: img_emb, txt_emb, logits, local_features
                i_emb, t_emb, _, _, _ = self.model(images, text)
                
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
        
        # 3. Calculate R@K using sorting
        for mode in ['i2t', 't2i']:
            sims = sim_matrix.T if mode == 't2i' else sim_matrix
            num_queries = sims.size(0)
            
            # Get the indices of the top 10 matches for every query
            # topk_indices shape: [num_queries, 10]
            _, topk_indices = sims.topk(10, dim=1, largest=True, sorted=True)
            
            # Ground truth for query i is index i
            targets = torch.arange(num_queries).view(-1, 1)
            
            # Check if target index is in the top K
            # This will finally give you different numbers for R@1, 5, and 10
            r1 = (topk_indices[:, :1] == targets).any(dim=1).float().mean().item() * 100
            r5 = (topk_indices[:, :5] == targets).any(dim=1).float().mean().item() * 100
            r10 = (topk_indices[:, :10] == targets).any(dim=1).float().mean().item() * 100
            
            metrics[f"{mode}_R@1"] = r1
            metrics[f"{mode}_R@5"] = r5
            metrics[f"{mode}_R@10"] = r10
            
            print(f"   >> {mode.upper()}: R@1: {r1:.2f}% | R@5: {r5:.2f}% | R@10: {r10:.2f}%")

        return metrics

    def evaluate_classification(self, num_classes=14):
        """
        Linear Probe Classification: Train a linear classifier on frozen embeddings.
        Uses sklearn LogisticRegression for simplicity.
        """
        print("\n[Classification] Linear Probe Evaluation...")

        # 1. Extract embeddings and labels from train set
        print("   >> Extracting train embeddings...")
        train_embs = []
        train_labels = []

        with torch.no_grad():
            for batch in tqdm(self.train_loader, desc="Train embeddings"):
                if batch is None:
                    continue
                images, text, labels = batch[0].to(self.device), batch[1].to(self.device), batch[2]

                img_emb, _, _, _, _ = self.model(images, text)
                img_emb = F.normalize(img_emb, dim=-1)

                train_embs.append(img_emb.cpu().numpy())
                train_labels.append(labels.numpy())

        train_embs = np.concatenate(train_embs, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)

        # 2. Extract embeddings and labels from test set
        print("   >> Extracting test embeddings...")
        test_embs = []
        test_labels = []

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Test embeddings"):
                if batch is None:
                    continue
                images, text, labels = batch[0].to(self.device), batch[1].to(self.device), batch[2]

                img_emb, _, _, _, _ = self.model(images, text)
                img_emb = F.normalize(img_emb, dim=-1)

                test_embs.append(img_emb.cpu().numpy())
                test_labels.append(labels.numpy())

        test_embs = np.concatenate(test_embs, axis=0)
        test_labels = np.concatenate(test_labels, axis=0)

        print(f"   >> Train: {train_embs.shape}, Test: {test_embs.shape}")

        # 3. Train linear classifiers (one per class for multi-label)
        print("   >> Training linear classifiers...")

        aucs = []
        aps = []

        class_results = {}
        for i in range(num_classes):
            y_train = train_labels[:, i]
            y_test = test_labels[:, i]

            if y_train.sum() == 0 or y_test.sum() == 0:
                continue

            clf = LogisticRegression(max_iter=1000, solver='lbfgs', class_weight='balanced')
            clf.fit(train_embs, y_train)
            y_pred_proba = clf.predict_proba(test_embs)[:, 1]

            auc = roc_auc_score(y_test, y_pred_proba)
            ap = average_precision_score(y_test, y_pred_proba)

            aucs.append(auc)
            aps.append(ap)
            # Store individual class performance for your thesis table
            class_results[f"class_{i}_auc"] = auc

        mean_auc = np.mean(aucs) if aucs else 0.0
        mean_ap = np.mean(aps) if aps else 0.0
        
        # Merge dictionaries to return everything
        return {
            "classification_mean_auc": mean_auc,
            "classification_mean_ap": mean_ap,
            **class_results
        }