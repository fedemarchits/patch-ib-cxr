import torch
from tqdm import tqdm

def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, scaler, use_amp, wandb_run=None):
    model.train()
    total_loss = 0
    
    # Optional: Sparsity regularizer
    sparsity_criterion = criterion['sparsity']
    contrastive_criterion = criterion['contrastive']
    
    loop = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch in loop:
        if batch is None:
            continue  # Skip empty batches
        images, text = batch[0].to(device), batch[1].to(device)
        # labels = batch[2] if len(batch) > 2 else None  # Available if needed
        
        optimizer.zero_grad()
        
        with torch.amp.autocast(device_type=device, enabled=use_amp):
            # Forward
            img_emb, txt_emb, logits = model(images, text)
            
            # Loss
            loss_con = contrastive_criterion(img_emb, txt_emb, model.logit_scale)
            loss = loss_con
            
            # Sparsity loss is only applied if masking is enabled (and logits are returned)
            if logits is not None:
                loss_sparse = sparsity_criterion(logits) * 10.0 # Weight for sparsity
                loss = loss + loss_sparse
        
        # Backward pass with scaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

        # Log to WandB if enabled
        if wandb_run:
            log_dict = {
                "train/step_loss": loss.item(),
                "train/contrastive_loss": loss_con.item(),
                "train/learning_rate": optimizer.param_groups[0]['lr']
            }
            if logits is not None:
                log_dict["train/sparsity_loss"] = loss_sparse.item()
            
            wandb_run.log(log_dict)
        
    return total_loss / len(dataloader)