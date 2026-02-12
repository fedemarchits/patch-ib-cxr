import torch
import torch.nn.functional as F
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def compute_validation_auc(model, train_loader, val_loader, device, use_amp=False,
                           max_train_samples=2000, max_val_samples=1000, num_classes=14,
                           cfg=None):
    """
    Compute a lightweight AUC estimate for early stopping during training.

    This is a faster version of full evaluation that:
    - Uses a subset of training data for fitting classifiers
    - Uses validation set (not test set) for scoring
    - Uses fewer LogisticRegression iterations

    Args:
        model: The model to evaluate
        train_loader: Training dataloader (may or may not have labels)
        val_loader: Validation dataloader (may or may not have labels)
        device: Device to use
        use_amp: Whether to use automatic mixed precision
        max_train_samples: Maximum training samples to use for fitting
        max_val_samples: Maximum validation samples for scoring
        num_classes: Number of classes for multi-label classification
        cfg: Config dict - if provided, will create new dataloaders with labels

    Returns:
        mean_auc: Mean AUC across all classes
    """
    model.eval()

    # Check if we need to create dataloaders with labels
    # Try to get a batch and check if it has labels
    sample_batch = None
    for batch in train_loader:
        if batch is not None:
            sample_batch = batch
            break

    needs_label_loaders = sample_batch is None or len(sample_batch) < 3

    if needs_label_loaders:
        if cfg is None:
            print("  [AUC] Warning: No labels in dataloader and no config provided to create new loaders")
            return 0.0

        # Import here to avoid circular imports
        from data.dataset import create_dataloaders

        print("  [AUC] Creating dataloaders with labels...")
        train_loader_with_labels, val_loader_with_labels, _ = create_dataloaders(
            cfg, return_labels=True
        )
    else:
        train_loader_with_labels = train_loader
        val_loader_with_labels = val_loader

    # Extract training embeddings and labels
    train_embs = []
    train_labels = []
    n_train = 0

    with torch.no_grad():
        for batch in train_loader_with_labels:
            if batch is None:
                continue
            if len(batch) < 3:
                continue

            images, text, labels = batch[0].to(device), batch[1].to(device), batch[2]

            with torch.amp.autocast(device_type=device, enabled=use_amp):
                img_emb, _, _, _, _ = model(images, text)
                img_emb = F.normalize(img_emb.float(), dim=-1)

            train_embs.append(img_emb.cpu().numpy())
            train_labels.append(labels.numpy())
            n_train += images.size(0)

            if n_train >= max_train_samples:
                break

    if not train_embs:
        print("  [AUC] Warning: No training samples with labels found")
        return 0.0

    train_embs = np.concatenate(train_embs, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)

    # Extract validation embeddings and labels
    val_embs = []
    val_labels = []
    n_val = 0

    with torch.no_grad():
        for batch in val_loader_with_labels:
            if batch is None:
                continue
            if len(batch) < 3:
                continue

            images, text, labels = batch[0].to(device), batch[1].to(device), batch[2]

            with torch.amp.autocast(device_type=device, enabled=use_amp):
                img_emb, _, _, _, _ = model(images, text)
                img_emb = F.normalize(img_emb.float(), dim=-1)

            val_embs.append(img_emb.cpu().numpy())
            val_labels.append(labels.numpy())
            n_val += images.size(0)

            if n_val >= max_val_samples:
                break

    if not val_embs:
        print("  [AUC] Warning: No validation samples with labels found")
        return 0.0

    val_embs = np.concatenate(val_embs, axis=0)
    val_labels = np.concatenate(val_labels, axis=0)

    print(f"  [AUC] Train samples: {train_embs.shape[0]}, Val samples: {val_embs.shape[0]}")

    # Train lightweight linear classifiers
    aucs = []
    for i in range(num_classes):
        y_train = train_labels[:, i]
        y_val = val_labels[:, i]

        # Skip if no positive samples in train or val
        if y_train.sum() == 0 or y_val.sum() == 0:
            continue

        # Use fewer iterations for speed
        clf = LogisticRegression(max_iter=200, solver='lbfgs', class_weight='balanced', n_jobs=-1)
        try:
            clf.fit(train_embs, y_train)
            y_pred_proba = clf.predict_proba(val_embs)[:, 1]
            auc = roc_auc_score(y_val, y_pred_proba)
            aucs.append(auc)
        except Exception:
            # Skip classes that fail (e.g., too few samples)
            continue

    mean_auc = np.mean(aucs) if aucs else 0.0
    print(f"  [AUC] Mean AUC: {mean_auc:.4f} (from {len(aucs)} classes)")

    return mean_auc


def compute_retrieval_metrics(img_emb, txt_emb, ks=[1, 5, 10], debug=True):
    """
    Compute image-to-text and text-to-image retrieval metrics.

    Args:
        img_emb: (N, D) normalized image embeddings
        txt_emb: (N, D) normalized text embeddings
        ks: list of K values for R@K

    Returns:
        dict with i2t_R@K and t2i_R@K metrics
    """
    # Compute similarity matrix (N x N)
    sim_matrix = img_emb @ txt_emb.T  # (N, N)

    N = sim_matrix.shape[0]
    metrics = {}

    if debug:
        # Debug: check embedding and similarity statistics
        diag_sim = sim_matrix.diag()  # Correct pair similarities
        off_diag_mask = ~torch.eye(N, dtype=torch.bool)
        off_diag_sim = sim_matrix[off_diag_mask]  # Incorrect pair similarities
        print(f"  [Retrieval Debug] N={N}, embed_dim={img_emb.shape[1]}")
        print(f"  [Retrieval Debug] Diagonal (correct pairs) sim: mean={diag_sim.mean():.4f}, std={diag_sim.std():.4f}")
        print(f"  [Retrieval Debug] Off-diagonal (wrong pairs) sim: mean={off_diag_sim.mean():.4f}, std={off_diag_sim.std():.4f}")

    # Image-to-text retrieval (each row is an image, find matching text)
    # Ground truth: diagonal (image i matches text i)
    i2t_ranks = []
    for i in range(N):
        sim_row = sim_matrix[i]
        # Rank of the correct text (index i) - lower is better
        rank = (sim_row > sim_row[i]).sum().item() + 1  # 1-indexed rank
        i2t_ranks.append(rank)

    # Text-to-image retrieval (each column is a text, find matching image)
    t2i_ranks = []
    for i in range(N):
        sim_col = sim_matrix[:, i]
        rank = (sim_col > sim_col[i]).sum().item() + 1
        t2i_ranks.append(rank)

    i2t_ranks = torch.tensor(i2t_ranks, dtype=torch.float)
    t2i_ranks = torch.tensor(t2i_ranks, dtype=torch.float)

    if debug:
        print(f"  [Retrieval Debug] i2t ranks: median={i2t_ranks.median():.0f}, mean={i2t_ranks.mean():.1f}")
        print(f"  [Retrieval Debug] t2i ranks: median={t2i_ranks.median():.0f}, mean={t2i_ranks.mean():.1f}")

    for k in ks:
        metrics[f'i2t_R@{k}'] = (i2t_ranks <= k).float().mean().item() * 100
        metrics[f't2i_R@{k}'] = (t2i_ranks <= k).float().mean().item() * 100

    # Mean recall (average of all R@K)
    metrics['mean_recall'] = sum(metrics.values()) / len(metrics)

    return metrics


@torch.no_grad()
def validate(model, dataloader, criterions, device, use_amp, compute_retrieval=False):
    """
    Validate model and optionally compute retrieval metrics.

    Args:
        compute_retrieval: If True, also compute retrieval metrics (slower but better for early stopping)

    Returns:
        If compute_retrieval=False: val_loss (float)
        If compute_retrieval=True: (val_loss, retrieval_metrics_dict)
    """
    model.eval()
    total_loss = 0
    num_batches = 0

    contrastive_criterion = criterions['contrastive']
    local_criterion = criterions.get('local_alignment', None)
    local_weight = criterions.get('local_weight', 0.1)

    # Mid-fusion local loss config
    mid_fusion_loss_weights = criterions.get('mid_fusion_loss_weights', None)

    # Check if using uncertainty weighting
    use_uncertainty = hasattr(model, 'use_uncertainty_weighting') and model.use_uncertainty_weighting

    # For retrieval metrics, collect all embeddings
    all_img_emb = []
    all_txt_emb = []

    for batch in dataloader:
        if batch is None:
            continue  # Skip empty batches

        images, text = batch[0].to(device), batch[1].to(device)

        with torch.amp.autocast(device_type=device, enabled=use_amp):
            img_emb, txt_emb, _, local_features, _ = model(images, text)
            loss_con_raw = contrastive_criterion(img_emb, txt_emb, model.logit_scale)

            if use_uncertainty:
                log_var_con = torch.clamp(model.log_var_contrastive, min=-2, max=2)
                loss = loss_con_raw * torch.exp(-log_var_con) + log_var_con
            else:
                loss = loss_con_raw

            # Include local alignment loss if enabled
            if local_criterion is not None and local_features is not None:
                if isinstance(local_features, list):
                    # Mid-fusion format: list of (patch_feat, token_feat, mask)
                    weights = mid_fusion_loss_weights or [1.0] * len(local_features)
                    loss_local_raw = sum(
                        weights[k] * local_criterion(pf, tf, am)
                        for k, (pf, tf, am) in enumerate(local_features)
                    )
                    loss = loss + loss_local_raw
                else:
                    # Standard format: single tuple
                    if len(local_features) == 5:
                        patch_feat, token_feat, attn_mask, aligned_feat, attn_weights = local_features
                        loss_local_raw = local_criterion(
                            patch_feat, token_feat, attn_mask,
                            aligned_features=aligned_feat, attn_weights=attn_weights
                        )
                    else:
                        patch_feat, token_feat, attn_mask = local_features
                        loss_local_raw = local_criterion(patch_feat, token_feat, attn_mask)

                    if use_uncertainty and hasattr(model, 'log_var_local'):
                        log_var_loc = torch.clamp(model.log_var_local, min=-2, max=2)
                        loss_local = loss_local_raw * torch.exp(-log_var_loc) + log_var_loc
                        loss = loss + loss_local
                    else:
                        loss = loss + local_weight * loss_local_raw

        total_loss += loss.item()
        num_batches += 1

        # Collect embeddings for retrieval
        if compute_retrieval:
            # Mid-fusion models: use independent encoding to avoid cross-attn leakage
            use_mid_fusion = hasattr(model, 'use_mid_fusion') and model.use_mid_fusion
            if use_mid_fusion:
                ind_img, ind_txt = model.encode_independent(images, text)
                all_img_emb.append(ind_img.float().cpu())
                all_txt_emb.append(ind_txt.float().cpu())
            else:
                all_img_emb.append(img_emb.float().cpu())
                all_txt_emb.append(txt_emb.float().cpu())

    avg_loss = total_loss / num_batches if num_batches > 0 else 0

    if compute_retrieval:
        # Concatenate all embeddings
        all_img_emb = torch.cat(all_img_emb, dim=0)
        all_txt_emb = torch.cat(all_txt_emb, dim=0)

        # Compute retrieval metrics
        retrieval_metrics = compute_retrieval_metrics(all_img_emb, all_txt_emb)
        return avg_loss, retrieval_metrics

    return avg_loss
