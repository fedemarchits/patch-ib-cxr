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
            loss_con = contrastive_criterion(img_emb, txt_emb, model.logit_scale)
            loss = loss_con

            # Sparsity loss (if masking enabled)
            loss_sparse = None
            if logits is not None:
                loss_sparse = sparsity_criterion(logits) * 10.0
                loss = loss + loss_sparse

            # Local alignment loss (if enabled)
            loss_local = None
            local_weight = 0.0
            if local_criterion is not None and local_features is not None:
                patch_feat, token_feat, attn_mask = local_features
                loss_local = local_criterion(patch_feat, token_feat, attn_mask)

                # Apply linear warmup to local weight
                if local_warmup_steps > 0:
                    warmup_factor = min(1.0, global_step / local_warmup_steps)
                else:
                    warmup_factor = 1.0
                local_weight = local_weight_target * warmup_factor

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
                "train/step_loss": loss.item(),
                "train/contrastive_loss": loss_con.item(),
                "train/learning_rate": optimizer.param_groups[0]['lr']
            }
            if loss_sparse is not None:
                log_dict["train/sparsity_loss"] = loss_sparse.item()
            if loss_local is not None:
                # Raw local loss
                log_dict["train/local_alignment_loss"] = loss_local.item()

                # Weighted local loss (actual contribution to total loss)
                weighted_local = local_weight * loss_local.item()
                log_dict["train/local_alignment_loss_weighted"] = weighted_local

                # Loss balance metrics
                contrastive_val = loss_con.item()
                log_dict["loss_balance/contrastive_vs_local_ratio"] = contrastive_val / (weighted_local + 1e-8)
                log_dict["loss_balance/local_contribution_pct"] = 100 * weighted_local / (contrastive_val + weighted_local + 1e-8)
                log_dict["loss_balance/contrastive_contribution_pct"] = 100 * contrastive_val / (contrastive_val + weighted_local + 1e-8)

                # Log current local weight (shows warmup progress)
                log_dict["train/local_weight_current"] = local_weight

            wandb_run.log(log_dict)

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss, global_step
