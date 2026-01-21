import torch

@torch.no_grad()
def validate(model, dataloader, criterions, device, use_amp):
    model.eval()
    total_loss = 0
    num_batches = 0
    
    contrastive_criterion = criterions['contrastive']
    
    for batch in dataloader:
        if batch is None:
            continue  # Skip empty batches
            
        images, text = batch[0].to(device), batch[1].to(device)
        
        with torch.amp.autocast(device_type=device, enabled=use_amp):
            img_emb, txt_emb, _ = model(images, text)
            loss = contrastive_criterion(img_emb, txt_emb, model.logit_scale)
            
        total_loss += loss.item()
        num_batches += 1
        
    return total_loss / num_batches if num_batches > 0 else 0