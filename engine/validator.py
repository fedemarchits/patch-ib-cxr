import torch


def compute_retrieval_metrics(img_emb, txt_emb, ks=[1, 5, 10]):
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
            img_emb, txt_emb, _, local_features = model(images, text)
            loss_con_raw = contrastive_criterion(img_emb, txt_emb, model.logit_scale)

            if use_uncertainty:
                log_var_con = torch.clamp(model.log_var_contrastive, min=-2, max=2)
                loss = loss_con_raw * torch.exp(-log_var_con) + log_var_con
            else:
                loss = loss_con_raw

            # Include local alignment loss if enabled
            if local_criterion is not None and local_features is not None:
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
