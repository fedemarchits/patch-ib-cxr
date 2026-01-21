import torch
from tqdm import tqdm

def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, scaler, use_amp, wandb_run=None, log_every_n_steps=20):
    model.train()
    total_loss = 0
    num_batches = 0
    sparsity_criterion = criterion['sparsity']
    contrastive_criterion = criterion['contrastive']

    loop = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in loop:
        if batch is None:
            continue

        num_batches += 1
        images, text = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device, enabled=use_amp):
            img_emb, txt_emb, logits = model(images, text)

            loss_con = contrastive_criterion(img_emb, txt_emb, model.logit_scale)
            loss = loss_con

            if logits is not None:
                loss_sparse = sparsity_criterion(logits) * 10.0
                loss = loss + loss_sparse

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

        if wandb_run and num_batches % log_every_n_steps == 0:
            log_dict = {
                "train/step_loss": loss.item(),
                "train/contrastive_loss": loss_con.item(),
                "train/learning_rate": optimizer.param_groups[0]['lr']
            }
            if logits is not None:
                log_dict["train/sparsity_loss"] = loss_sparse.item()

            wandb_run.log(log_dict)

    return total_loss / num_batches if num_batches > 0 else 0