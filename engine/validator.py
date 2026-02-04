import torch

@torch.no_grad()
def validate(model, dataloader, criterions, device, use_amp):
    model.eval()
    total_loss = 0
    num_batches = 0

    contrastive_criterion = criterions['contrastive']
    local_criterion = criterions.get('local_alignment', None)
    local_weight = criterions.get('local_weight', 0.1)

    # Check if using uncertainty weighting
    use_uncertainty = hasattr(model, 'use_uncertainty_weighting') and model.use_uncertainty_weighting

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

    return total_loss / num_batches if num_batches > 0 else 0
