import torch
from tqdm import tqdm

def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, scaler, use_amp, wandb_run=None, log_every_n_steps=20, scheduler=None):
    model.train()
    total_loss = 0
    num_batches = 0
    sparsity_criterion = criterion['sparsity']
    contrastive_criterion = criterion['contrastive']
    local_criterion = criterion.get('local_alignment', None)
    local_weight = criterion.get('local_weight', 0.1)

    loop = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in loop:
        if batch is None:
            continue

        num_batches += 1
        images, text = batch[0].to(device), batch[1].to(device)

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
            if local_criterion is not None and local_features is not None:
                patch_feat, token_feat, attn_mask = local_features
                loss_local = local_criterion(patch_feat, token_feat, attn_mask)
                loss = loss + local_weight * loss_local

        scaler.scale(loss).backward()

        # Track optimizer step count to check if step was skipped (inf/nan gradients)
        old_step = optimizer._step_count

        scaler.step(optimizer)
        scaler.update()

        # Only step scheduler if optimizer actually stepped (not skipped due to inf/nan)
        if scheduler is not None and optimizer._step_count > old_step:
            scheduler.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

        if wandb_run and num_batches % log_every_n_steps == 0:
            log_dict = {
                "train/step_loss": loss.item(),
                "train/contrastive_loss": loss_con.item(),
                "train/learning_rate": optimizer.param_groups[0]['lr']
            }
            if loss_sparse is not None:
                log_dict["train/sparsity_loss"] = loss_sparse.item()
            if loss_local is not None:
                log_dict["train/local_alignment_loss"] = loss_local.item()

            wandb_run.log(log_dict)

    return total_loss / num_batches if num_batches > 0 else 0
