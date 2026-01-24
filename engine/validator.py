import torch

@torch.no_grad()
def validate(model, dataloader, criterions, device, use_amp):
    model.eval()
    total_loss = 0
    num_batches = 0

    contrastive_criterion = criterions['contrastive']
    local_criterion = criterions.get('local_alignment', None)
    local_weight = criterions.get('local_weight', 0.1)

    for batch in dataloader:
        if batch is None:
            continue  # Skip empty batches

        images, text = batch[0].to(device), batch[1].to(device)

        with torch.amp.autocast(device_type=device, enabled=use_amp):
            img_emb, txt_emb, _, local_features = model(images, text)
            loss = contrastive_criterion(img_emb, txt_emb, model.logit_scale)

            # Include local alignment loss if enabled
            if local_criterion is not None and local_features is not None:
                patch_feat, token_feat, attn_mask = local_features
                loss_local = local_criterion(patch_feat, token_feat, attn_mask)
                loss = loss + local_weight * loss_local

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0
