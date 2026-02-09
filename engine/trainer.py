import torch
import time
from tqdm import tqdm

def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, scaler, use_amp, wandb_run=None, log_every_n_steps=20, scheduler=None, global_step=0, accumulation_steps=1):
    model.train()
    total_loss = 0
    num_batches = 0

    # Timing tracking
    step_times = []
    is_cuda = device == "cuda" or str(device).startswith("cuda")
    sparsity_criterion = criterion['sparsity']
    contrastive_criterion = criterion['contrastive']
    local_criterion = criterion.get('local_alignment', None)
    local_weight_target = criterion.get('local_weight', 0.1)
    local_warmup_steps = criterion.get('local_warmup_steps', 0)

    # Patch-IB specific losses and weights
    consistency_criterion = criterion.get('consistency', None)
    consistency_weight = criterion.get('consistency_weight', 1.0)
    sparsity_weight = criterion.get('sparsity_weight', 10.0)
    sparsity_warmup_steps = criterion.get('sparsity_warmup_steps', 0)
    contrastive_mask_weight = criterion.get('contrastive_mask_weight', 1.0)
    contrastive_full_weight = criterion.get('contrastive_full_weight', 1.0)

    loop = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(loop):
        if batch is None:
            continue

        # Start timing
        if is_cuda:
            torch.cuda.synchronize()
        step_start = time.perf_counter()

        num_batches += 1
        images, text = batch[0].to(device), batch[1].to(device)

        # Only zero gradients at the start of accumulation cycle
        if batch_idx % accumulation_steps == 0:
            optimizer.zero_grad()

        with torch.amp.autocast(device_type=device, enabled=use_amp):
            # Model returns 5 values: img_emb (masked if use_masking), txt_emb, logits, local_features, img_emb_full
            img_emb, txt_emb, logits, local_features, img_emb_full = model(images, text)

            # Global contrastive loss (on masked embedding if Patch-IB, else on full)
            loss_con_raw = contrastive_criterion(img_emb, txt_emb, model.logit_scale)

            # Check if using uncertainty weighting (Kendall et al., 2018)
            use_uncertainty = hasattr(model, 'use_uncertainty_weighting') and model.use_uncertainty_weighting

            if use_uncertainty:
                # Uncertainty weighting: L * exp(-log_var) + log_var
                # Clamp log_var to [-6, 6] to prevent extreme weights
                log_var_con = torch.clamp(model.log_var_contrastive, min=-2, max=2)
                loss_con = loss_con_raw * torch.exp(-log_var_con) + log_var_con
            else:
                loss_con = loss_con_raw

            loss = loss_con * contrastive_mask_weight

            # ============ PATCH-IB LOSSES ============
            # InfoNCE on full embeddings (if Patch-IB enabled)
            loss_con_full_raw = None
            loss_con_full = None
            if img_emb_full is not None:
                loss_con_full_raw = contrastive_criterion(img_emb_full, txt_emb, model.logit_scale)
                if use_uncertainty:
                    loss_con_full = loss_con_full_raw * torch.exp(-log_var_con) + log_var_con
                else:
                    loss_con_full = loss_con_full_raw
                loss = loss + loss_con_full * contrastive_full_weight

            # Sparsity loss (if masking enabled)
            loss_sparse = None
            if logits is not None:
                if sparsity_warmup_steps > 0:
                    sparsity_warmup_factor = min(1.0, global_step / sparsity_warmup_steps)
                else:
                    sparsity_warmup_factor = 1.0
                loss_sparse = sparsity_criterion(logits) * sparsity_weight * sparsity_warmup_factor
                loss = loss + loss_sparse

            # Consistency loss (if Patch-IB enabled)
            loss_consistency = None
            if consistency_criterion is not None and img_emb_full is not None:
                loss_consistency = consistency_criterion(
                    img_emb_full, img_emb, txt_emb, model.logit_scale
                ) * consistency_weight
                loss = loss + loss_consistency
            # =========================================

            # Local alignment loss (if enabled)
            loss_local = None
            loss_local_raw = None
            local_weight = 0.0
            if local_criterion is not None and local_features is not None:
                # Unpack local features (supports both old 3-tuple and new 5-tuple format)
                if len(local_features) == 5:
                    patch_feat, token_feat, attn_mask, aligned_feat, attn_weights = local_features
                    loss_local_raw = local_criterion(
                        patch_feat, token_feat, attn_mask,
                        aligned_features=aligned_feat, attn_weights=attn_weights
                    )
                else:
                    # Legacy 3-tuple format (backward compatible)
                    patch_feat, token_feat, attn_mask = local_features
                    loss_local_raw = local_criterion(patch_feat, token_feat, attn_mask)

                if use_uncertainty and hasattr(model, 'log_var_local'):
                    # Uncertainty weighting for local loss (clamped to prevent extreme weights)
                    log_var_loc = torch.clamp(model.log_var_local, min=-2, max=2)
                    loss_local = loss_local_raw * torch.exp(-log_var_loc) + log_var_loc
                    loss = loss + loss_local
                else:
                    # Fixed weight with warmup (original behavior)
                    if local_warmup_steps > 0:
                        warmup_factor = min(1.0, global_step / local_warmup_steps)
                    else:
                        warmup_factor = 1.0
                    local_weight = local_weight_target * warmup_factor
                    loss_local = loss_local_raw
                    loss = loss + local_weight * loss_local

            # Scale loss by accumulation steps for gradient averaging
            loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        # Only update weights every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            # Track scale to detect if optimizer step was skipped (inf/nan gradients)
            old_scale = scaler.get_scale()

            scaler.step(optimizer)
            scaler.update()

            # Only step scheduler if optimizer actually stepped
            # If scale decreased, gradients had inf/nan and step was skipped
            if scheduler is not None and scaler.get_scale() >= old_scale:
                scheduler.step()

        # End timing
        if is_cuda:
            torch.cuda.synchronize()
        step_time_ms = (time.perf_counter() - step_start) * 1000
        step_times.append(step_time_ms)

        # Track unscaled loss for logging
        total_loss += loss.item() * accumulation_steps

        # Progress bar with sparsity info if masking enabled
        postfix_dict = {"loss": loss.item() * accumulation_steps, "ms/step": f"{step_time_ms:.0f}"}
        if logits is not None:
            with torch.no_grad():
                kept_ratio = (logits > 0).float().mean().item()
                postfix_dict["kept"] = f"{kept_ratio:.1%}"
        loop.set_postfix(**postfix_dict)

        global_step += 1

        if wandb_run and num_batches % log_every_n_steps == 0:
            # Compute average step time over recent steps
            recent_step_times = step_times[-log_every_n_steps:] if len(step_times) >= log_every_n_steps else step_times
            avg_step_time_ms = sum(recent_step_times) / len(recent_step_times) if recent_step_times else 0

            log_dict = {
                "train/step_loss": loss.item() * accumulation_steps,
                "train/contrastive_loss_raw": loss_con_raw.item(),
                "train/learning_rate": optimizer.param_groups[0]['lr'],
                "efficiency/step_time_ms": step_time_ms,
                "efficiency/avg_step_time_ms": avg_step_time_ms,
                "efficiency/throughput_img_per_sec": images.size(0) / (step_time_ms / 1000) if step_time_ms > 0 else 0
            }
            if loss_sparse is not None:
                log_dict["train/sparsity_loss"] = loss_sparse.item()
                log_dict["train/sparsity_weight_current"] = sparsity_weight * sparsity_warmup_factor

            # Sparsity tracking (actual mask statistics)
            if logits is not None:
                with torch.no_grad():
                    # Compute mask probabilities (sigmoid of logits)
                    mask_probs = torch.sigmoid(logits)
                    # Mean activation (average probability of keeping a patch)
                    mean_activation = mask_probs.mean().item()
                    # Sparsity ratio (% of patches with prob > 0.5, i.e., "kept")
                    hard_mask = (logits > 0).float()
                    patches_kept_ratio = hard_mask.mean().item()
                    # Number of patches kept per image (out of 196)
                    patches_kept_per_image = hard_mask.sum(dim=1).mean().item()

                    log_dict["mask/mean_activation"] = mean_activation
                    log_dict["mask/patches_kept_ratio"] = patches_kept_ratio
                    log_dict["mask/patches_kept_per_image"] = patches_kept_per_image
                    log_dict["mask/patches_dropped_ratio"] = 1.0 - patches_kept_ratio

            # Patch-IB specific logging
            if loss_con_full_raw is not None:
                log_dict["train/contrastive_full_loss_raw"] = loss_con_full_raw.item()
            if loss_consistency is not None:
                log_dict["train/consistency_loss"] = loss_consistency.item()

            # Log uncertainty weighting parameters
            if use_uncertainty:
                log_dict["uncertainty/log_var_contrastive"] = model.log_var_contrastive.item()
                log_dict["uncertainty/sigma_contrastive"] = torch.exp(0.5 * model.log_var_contrastive).item()
                log_dict["train/contrastive_loss_weighted"] = loss_con.item()

                if hasattr(model, 'log_var_local'):
                    log_dict["uncertainty/log_var_local"] = model.log_var_local.item()
                    log_dict["uncertainty/sigma_local"] = torch.exp(0.5 * model.log_var_local).item()

            if loss_local_raw is not None:
                # Raw local loss (before any weighting)
                log_dict["train/local_alignment_loss_raw"] = loss_local_raw.item()

                if use_uncertainty and hasattr(model, 'log_var_local'):
                    # Uncertainty-weighted local loss
                    log_dict["train/local_alignment_loss_weighted"] = loss_local.item()

                    # Loss balance metrics for uncertainty weighting
                    contrastive_contrib = loss_con.item()
                    local_contrib = loss_local.item()
                    total_contrib = contrastive_contrib + local_contrib
                    log_dict["loss_balance/local_contribution_pct"] = 100 * local_contrib / (total_contrib + 1e-8)
                    log_dict["loss_balance/contrastive_contribution_pct"] = 100 * contrastive_contrib / (total_contrib + 1e-8)
                else:
                    # Fixed weight metrics (original behavior)
                    weighted_local = local_weight * loss_local.item()
                    log_dict["train/local_alignment_loss_weighted"] = weighted_local

                    contrastive_val = loss_con_raw.item()
                    log_dict["loss_balance/contrastive_vs_local_ratio"] = contrastive_val / (weighted_local + 1e-8)
                    log_dict["loss_balance/local_contribution_pct"] = 100 * weighted_local / (contrastive_val + weighted_local + 1e-8)
                    log_dict["loss_balance/contrastive_contribution_pct"] = 100 * contrastive_val / (contrastive_val + weighted_local + 1e-8)
                    log_dict["train/local_weight_current"] = local_weight

            wandb_run.log(log_dict)

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss, global_step
