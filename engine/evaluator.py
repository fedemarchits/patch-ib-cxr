import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import json
import os
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.decomposition import PCA
import numpy as np

# Lazy imports for visualization (not required for clustering-only evaluation)
plt = None


def _ensure_matplotlib():
    global plt
    if plt is None:
        import matplotlib.pyplot as _plt
        plt = _plt


def _compute_purity(true_labels, cluster_assignments):
    """Cluster purity: fraction of samples assigned to majority class per cluster."""
    total = 0
    for cluster_id in np.unique(cluster_assignments):
        cluster_mask = cluster_assignments == cluster_id
        true_in_cluster = true_labels[cluster_mask]
        most_common_count = np.bincount(true_in_cluster).max()
        total += most_common_count
    return total / len(true_labels)


def estimate_model_flops(model, image_size=224, text_length=77):
    """
    Estimate FLOPs for the model using a simple forward pass counter.
    Returns GFLOPs (billions of floating point operations).
    """
    try:
        from fvcore.nn import FlopCountAnalysis

        # Create dummy inputs
        device = next(model.parameters()).device
        dummy_image = torch.randn(1, 3, image_size, image_size, device=device)
        dummy_text = torch.randint(0, 1000, (1, text_length), device=device)

        # Count FLOPs
        flops = FlopCountAnalysis(model, (dummy_image, dummy_text))
        total_flops = flops.total()
        gflops = total_flops / 1e9

        return gflops
    except ImportError:
        print("   >> [FLOPs] fvcore not installed. Using parameter-based estimate.")
        # Fallback: estimate based on parameter count
        # Rough estimate: 2 FLOPs per parameter per forward pass
        total_params = sum(p.numel() for p in model.parameters())
        gflops = (2 * total_params) / 1e9
        return gflops
    except Exception as e:
        print(f"   >> [FLOPs] Error computing FLOPs: {e}")
        return 0.0


class Evaluator:
    def __init__(self, model, train_loader, test_loader, device):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.model.eval()
        self.model.to(device)
        self.train_loader = train_loader

    def benchmark_efficiency(self, num_batches=50, batch_size=None):
        """
        Measure Throughput (img/sec), Peak VRAM, Step Time, and FLOPs.
        Works on CUDA, MPS, and CPU devices.
        """
        print(f"\n[Efficiency] Benchmarking over {num_batches} batches on {self.device}...")

        is_cuda = self.device == "cuda" or str(self.device).startswith("cuda")
        is_mps = self.device == "mps" or str(self.device).startswith("mps")

        # 0. Estimate FLOPs
        print("   >> Estimating FLOPs...")
        gflops = estimate_model_flops(self.model)
        print(f"   >> Estimated GFLOPs: {gflops:.2f}")

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
        step_times = []

        # 3. Timing Loop
        total_start = time.time()
        with torch.no_grad():
            # Only use autocast on CUDA
            autocast_enabled = is_cuda
            with torch.amp.autocast(device_type="cuda" if is_cuda else "cpu", enabled=autocast_enabled):
                for i, batch in enumerate(self.test_loader):
                    if batch is None:
                        continue
                    if i >= num_batches: break

                    # Per-step timing
                    if is_cuda:
                        torch.cuda.synchronize()
                    step_start = time.perf_counter()

                    images, text = batch[0].to(self.device), batch[1].to(self.device)
                    _ = self.model(images, text)

                    if is_cuda:
                        torch.cuda.synchronize()
                    step_time_ms = (time.perf_counter() - step_start) * 1000
                    step_times.append(step_time_ms)

                    total_images += images.size(0)
                    if batch_size is None:
                        batch_size = images.size(0)

        # Sync before measuring time
        if is_cuda:
            torch.cuda.synchronize()
        elif is_mps:
            torch.mps.synchronize()

        elapsed_sec = time.time() - total_start

        # 4. Calculate Stats
        throughput = total_images / elapsed_sec  # img/sec
        avg_step_time_ms = sum(step_times) / len(step_times) if step_times else 0
        latency_per_image_ms = avg_step_time_ms / batch_size if batch_size else 0

        # Memory stats (only available on CUDA)
        if is_cuda:
            peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        else:
            peak_mem = 0.0  # Not available on CPU/MPS

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"\n   === Efficiency Report ===")
        print(f"   >> Throughput:         {throughput:.2f} img/sec")
        print(f"   >> Avg Step Time:      {avg_step_time_ms:.2f} ms/batch")
        print(f"   >> Latency per Image:  {latency_per_image_ms:.2f} ms/img")
        print(f"   >> Peak VRAM:          {peak_mem:.2f} MB" if is_cuda else "   >> Peak VRAM:          N/A")
        print(f"   >> GFLOPs:             {gflops:.2f}")
        print(f"   >> Total Params:       {total_params:,}")
        print(f"   >> Trainable Params:   {trainable_params:,}")
        print(f"   ===========================\n")

        return {
            "throughput_img_per_sec": throughput,
            "avg_step_time_ms": avg_step_time_ms,
            "latency_per_image_ms": latency_per_image_ms,
            "peak_vram_mb": peak_mem,
            "gflops": gflops,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "batch_size": batch_size,
            "num_batches_tested": len(step_times)
        }

    def compute_retrieval(self):
        """
        Compute R@1, R@5, R@10 for Image-to-Text and Text-to-Image.

        For mid-fusion models, uses independent encoding (bypassing cross-attention)
        to avoid information leakage between modalities during evaluation.
        """
        # Detect mid-fusion: use independent encoding to avoid retrieval cheating
        use_independent = hasattr(self.model, 'use_mid_fusion') and self.model.use_mid_fusion
        if use_independent:
            print("\n[Retrieval] Mid-fusion detected: using independent encoding (no cross-attention)")
        else:
            print("\n[Retrieval] Extracting embeddings...")

        img_embs = []
        txt_embs = []

        # 1. Extract Embeddings
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Extracting"):
                if batch is None:
                    continue

                images, text = batch[0].to(self.device), batch[1].to(self.device)

                if use_independent:
                    i_emb, t_emb = self.model.encode_independent(images, text)
                else:
                    i_emb, t_emb, _, _, _ = self.model(images, text)
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

    def _extract_test_embeddings(self):
        """Extract L2-normalized image embeddings and labels from the test set."""
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

        return np.concatenate(test_embs, axis=0), np.concatenate(test_labels, axis=0)

    @staticmethod
    def _filter_single_label(embs, labels):
        """Filter to samples with exactly one active label. Returns embs, class_ids."""
        label_sums = labels.sum(axis=1)
        mask = (label_sums == 1)
        sl_embs = embs[mask]
        sl_class_ids = np.argmax(labels[mask], axis=1)
        return sl_embs, sl_class_ids

    def evaluate_clustering(self, num_classes=14):
        """
        K-Means clustering evaluation on frozen image embeddings.
        Uses cosine distance (embeddings are L2-normalized, so Euclidean K-Means
        is equivalent to cosine K-Means). Single-label test samples only.

        Returns:
            dict with NMI, ARI, purity, sample counts, class distribution
        """
        print("\n[Clustering] K-Means Evaluation...")

        print("   >> Extracting test embeddings...")
        test_embs, test_labels = self._extract_test_embeddings()

        # Filter to single-label samples
        sl_embs, sl_class_ids = self._filter_single_label(test_embs, test_labels)
        num_sl = sl_embs.shape[0]

        print(f"   >> Total test samples: {test_embs.shape[0]}")
        print(f"   >> Single-label samples: {num_sl}")
        print(f"   >> Multi-label excluded: {test_embs.shape[0] - num_sl}")

        if num_sl == 0:
            print("   >> WARNING: No single-label samples found!")
            return {"clustering_nmi": 0.0, "clustering_ari": 0.0,
                    "clustering_purity": 0.0, "clustering_num_single_label_samples": 0}

        unique_classes = np.unique(sl_class_ids)
        n_clusters = len(unique_classes)
        print(f"   >> Unique classes present: {n_clusters} out of {num_classes}")

        # K-Means on L2-normalized embeddings = cosine K-Means
        print(f"   >> Running K-Means (k={n_clusters}, cosine via L2-norm)...")
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42, max_iter=300)
        cluster_assignments = kmeans.fit_predict(sl_embs)

        nmi = normalized_mutual_info_score(sl_class_ids, cluster_assignments, average_method='arithmetic')
        ari = adjusted_rand_score(sl_class_ids, cluster_assignments)
        purity = _compute_purity(sl_class_ids, cluster_assignments)

        print(f"   >> NMI:    {nmi:.4f}")
        print(f"   >> ARI:    {ari:.4f}")
        print(f"   >> Purity: {purity:.4f}")

        class_names = [
            "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
            "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity",
            "No Finding", "Pleural Effusion", "Pleural Other", "Pneumonia",
            "Pneumothorax", "Support Devices"
        ]
        class_dist = {}
        print("   >> Class distribution:")
        for c in unique_classes:
            count = int((sl_class_ids == c).sum())
            class_dist[int(c)] = count
            print(f"      {class_names[c]:30s}: {count}")

        return {
            "clustering_nmi": nmi,
            "clustering_ari": ari,
            "clustering_purity": purity,
            "clustering_num_single_label_samples": num_sl,
            "clustering_num_classes_present": n_clusters,
            "clustering_class_distribution": class_dist,
        }

    def generate_umap_visualization(self, output_dir, max_samples=5000):
        """
        UMAP visualization of image embeddings colored by class.
        Pipeline: 512-d -> PCA(50) -> UMAP(2), cosine metric throughout.
        Single-label test samples only.

        Args:
            output_dir: Directory to save the figure
            max_samples: Random subsample cap for UMAP performance

        Returns:
            Path to saved figure, or None if not enough data
        """
        try:
            import umap
        except ImportError:
            print("   >> [UMAP] umap-learn not installed. Install with: pip install umap-learn")
            return None

        _ensure_matplotlib()
        print("\n[UMAP] Generating embedding visualization...")

        print("   >> Extracting test embeddings...")
        test_embs, test_labels = self._extract_test_embeddings()

        sl_embs, sl_class_ids = self._filter_single_label(test_embs, test_labels)
        num_sl = sl_embs.shape[0]
        print(f"   >> Single-label samples for UMAP: {num_sl}")

        if num_sl < 10:
            print("   >> Too few single-label samples for UMAP. Skipping.")
            return None

        # Subsample if too large
        if num_sl > max_samples:
            print(f"   >> Subsampling {num_sl} -> {max_samples}")
            rng = np.random.RandomState(42)
            indices = rng.choice(num_sl, size=max_samples, replace=False)
            sl_embs = sl_embs[indices]
            sl_class_ids = sl_class_ids[indices]
            num_sl = max_samples

        # PCA: 512-d -> 50-d (noise reduction before UMAP)
        n_components = min(50, sl_embs.shape[0], sl_embs.shape[1])
        print(f"   >> PCA: {sl_embs.shape[1]}-d -> {n_components}-d...")
        pca = PCA(n_components=n_components, random_state=42)
        embs_pca = pca.fit_transform(sl_embs)
        print(f"   >> PCA explained variance: {pca.explained_variance_ratio_.sum():.4f}")

        # UMAP: 50-d -> 2-d with cosine metric
        print("   >> UMAP: 50-d -> 2-d (cosine)...")
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric='cosine',
            random_state=42,
        )
        embs_2d = reducer.fit_transform(embs_pca)

        # Plot
        class_names = [
            "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
            "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity",
            "No Finding", "Pleural Effusion", "Pleural Other", "Pneumonia",
            "Pneumothorax", "Support Devices"
        ]

        os.makedirs(output_dir, exist_ok=True)

        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        unique_classes = np.unique(sl_class_ids)
        cmap = plt.cm.get_cmap('tab20', len(unique_classes))

        for idx, class_id in enumerate(unique_classes):
            mask = sl_class_ids == class_id
            ax.scatter(
                embs_2d[mask, 0], embs_2d[mask, 1],
                c=[cmap(idx)],
                label=class_names[class_id],
                alpha=0.5, s=8, edgecolors='none',
            )

        ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5),
                  fontsize=8, markerscale=3, frameon=True)
        ax.set_title(f"UMAP of Image Embeddings ({num_sl} single-label test samples)")
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.set_xticks([])
        ax.set_yticks([])

        save_path = os.path.join(output_dir, "umap_embeddings.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()

        print(f"   >> Saved to {save_path}")
        return save_path