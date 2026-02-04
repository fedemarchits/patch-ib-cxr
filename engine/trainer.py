import torch
from tqdm import tqdm

def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, scaler, use_amp, wandb_run=None, log_every_n_steps=20, scheduler=None, global_step=0, accumulation_steps=1):
    model.train()
    total_loss = 0
    num_batches = 0
    sparsity_criterion = criterion['sparsity']
    contrastive_criterion = criterion['contrastive']
    local_criterion = criterion.get('local_alignment', None)
    local_weight_target = criterion.get('local_weight', 0.1)
    local_warmup_steps = criterion.get('local_warmup_steps', 0)

    loop = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(loop):
        if batch is None:
            continue

        num_batches += 1
        images, text = batch[0].to(device), batch[1].to(device)

        # Only zero gradients at the start of accumulation cycle
        if batch_idx % accumulation_steps == 0:
            optimizer.zero_grad()

        with torch.amp.autocast(device_type=device, enabled=use_amp):
            img_emb, txt_emb, logits, local_features = model(images, text)

            # Global contrastive loss
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

            loss = loss_con

            # Sparsity loss (if masking enabled)
            loss_sparse = None
            if logits is not None:
                loss_sparse = sparsity_criterion(logits) * 10.0
                loss = loss + loss_sparse

            # Local alignment loss (if enabled)
            loss_local = None
            loss_local_raw = None
            local_weight = 0.0
            if local_criterion is not None and local_features is not None:
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

        # Track unscaled loss for logging
        total_loss += loss.item() * accumulation_steps
        loop.set_postfix(loss=loss.item() * accumulation_steps)
        global_step += 1

        if wandb_run and num_batches % log_every_n_steps == 0:
            log_dict = {
                "train/step_loss": loss.item() * accumulation_steps,
                "train/contrastive_loss_raw": loss_con_raw.item(),
                "train/learning_rate": optimizer.param_groups[0]['lr']
            }
            if loss_sparse is not None:
                log_dict["train/sparsity_loss"] = loss_sparse.item()

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
