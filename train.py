import yaml
import torch
import argparse
import wandb
import os
import math
from torch.amp import GradScaler
from torch.optim.lr_scheduler import LambdaLR

from models.full_model import ModelABaseline
from models.losses import ContrastiveLoss, SparsityLoss, LocalAlignmentLoss, ConsistencyLoss
from data.dataset import create_dataloaders
from engine.trainer import train_one_epoch
from engine.utils import EarlyStopping
from engine.validator import validate, compute_validation_auc
from engine.optimizer import create_optimizer, get_parameter_groups

def get_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """
    Linear warmup followed by cosine decay to 0.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def unfreeze_backbone(model):
    """
    Unfreeze all backbone parameters for fine-tuning phase.
    """
    for param in model.backbone.parameters():
        param.requires_grad = True

    # Count trainable params after unfreezing
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print("BACKBONE UNFROZEN - Starting Fine-tuning Phase")
    print(f"{'='*60}")
    print(f"  Trainable parameters: {trainable:,}")
    print(f"  Total parameters:     {total:,}")
    print(f"{'='*60}\n")

def main():
    parser = argparse.ArgumentParser(description="Main training script")
    parser.add_argument("--config", type=str, default="configs/model_a_baseline.yaml",
                        help="Path to the configuration file.")
    parser.add_argument("--output_dir", type=str, default="logs", help="Where to save checkpoints")
    parser.add_argument("--dry_run", action="store_true", help="Run 1 batch on CPU to test pipeline") # For DEBUGGING
    args = parser.parse_args()

    # 1. Load Config & Setup
    cfg = get_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)


    ######## DEBUG CODE ########
    if args.dry_run:
        print("⚠️  DRY RUN MODE: CPU + 1 Batch + WandB Check")
        cfg['training']['device'] = 'cpu'
        cfg['training']['use_amp'] = False
        cfg['data']['batch_size'] = 2
    ######## END DEBUG CODE ########

    # WandB Initialization
    wandb_cfg = cfg.get('wandb', {})
    if wandb_cfg.get('enable', False):
        wandb_run = wandb.init(
            project=wandb_cfg.get('project', 'my-thesis-project'),
            entity=wandb_cfg.get('entity', None),
            name=wandb_cfg.get('run_name', 'default-run'),
            config=cfg
        )
    else:
        wandb_run = None

    try:
        device = cfg['training']['device'] if torch.cuda.is_available() else "cpu"
        use_amp = cfg['training'].get('use_amp', False)
        
        print(f"Running on: {device}")

        # 2. Data Loading (THE NEW WAY)
        # This one line replaces your old 20 lines of dataset setup
        train_loader, val_loader, _ = create_dataloaders(cfg)
        print(f"Data Loaded: {len(train_loader)} train batches, {len(val_loader)} val batches.")

        # 3. Model
        model = ModelABaseline(cfg).to(device)
        
        if wandb_run:
            wandb.watch(model, log='all', log_freq=500)

        # 4. Optimization
        # Check for staged training configuration
        staged_training = cfg['training'].get('staged_training', False)
        warmup_epochs = cfg['training'].get('warmup_epochs', 3)
        warmup_lr = cfg['training'].get('warmup_lr', 1e-4)

        if staged_training:
            print(f"\n{'='*60}")
            print("STAGED TRAINING ENABLED")
            print(f"{'='*60}")
            print(f"  Phase 1: {warmup_epochs} epochs with frozen backbone (lr={warmup_lr:.2e})")
            print(f"  Phase 2: Fine-tuning with LLRD (lr={cfg['training']['lr']:.2e})")
            print(f"{'='*60}\n")

            # Phase 1: Create optimizer with frozen backbone
            # Temporarily override config for warmup phase
            warmup_cfg = cfg.copy()
            warmup_cfg['training'] = cfg['training'].copy()
            warmup_cfg['training']['freeze_backbone'] = True
            warmup_cfg['training']['lr'] = warmup_lr
            optimizer = create_optimizer(model, warmup_cfg)
            current_phase = 1
        else:
            # Standard training (with LLRD or frozen backbone as configured)
            optimizer = create_optimizer(model, cfg)
            current_phase = 0  # No staged training

        # Losses
        # Make sure these classes exist in your src/models/ directory
        weight_i2t = cfg['model'].get('contrastive_weight_i2t', 0.5)
        weight_t2i = cfg['model'].get('contrastive_weight_t2i', 0.5)
        criterions = {
            'contrastive': ContrastiveLoss(weight_i2t=weight_i2t, weight_t2i=weight_t2i),
            'sparsity': SparsityLoss(target_ratio=cfg['model']['mask_ratio'])
        }
        print(f"Contrastive loss weights: i2t={weight_i2t}, t2i={weight_t2i}")

        # Add local alignment loss if enabled
        if cfg['model'].get('use_local_alignment', False):
            criterions['local_alignment'] = LocalAlignmentLoss(
                temperature=cfg['model'].get('local_alignment_temperature', 1.0)
            )
            criterions['local_weight'] = cfg['model'].get('local_alignment_weight', 0.1)
            criterions['local_warmup_steps'] = cfg['model'].get('local_alignment_warmup_steps', 0)

        # Add Patch-IB losses if masking is enabled
        if cfg['model'].get('use_masking', False):
            criterions['consistency'] = ConsistencyLoss(
                include_negatives=cfg['model'].get('consistency_include_negatives', False)
            )
            criterions['consistency_weight'] = cfg['model'].get('consistency_weight', 1.0)
            criterions['sparsity_weight'] = cfg['model'].get('sparsity_weight', 10.0)

        # GradScaler for mixed precision
        scaler = torch.amp.GradScaler(device, enabled=use_amp)

        # Gradient accumulation
        accumulation_steps = cfg['training'].get('gradient_accumulation_steps', 1)

        # Learning rate scheduler with warmup
        # For staged training, we'll recreate the scheduler after phase transition
        if staged_training:
            # Phase 1 scheduler: just for warmup epochs
            warmup_phase_steps = (len(train_loader) * warmup_epochs) // accumulation_steps
            num_warmup_steps = min(cfg['training'].get('warmup_steps', 500), warmup_phase_steps // 2)
            scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, warmup_phase_steps)
            print(f"Phase 1 Scheduler: {num_warmup_steps} warmup steps, {warmup_phase_steps} total steps")
        else:
            num_training_steps = (len(train_loader) * cfg['training']['epochs']) // accumulation_steps
            num_warmup_steps = cfg['training'].get('warmup_steps', 500)
            scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
            print(f"Scheduler: {num_warmup_steps} warmup steps, {num_training_steps} total steps")
        print(f"Gradient accumulation: {accumulation_steps} steps (effective batch size: {cfg['data']['batch_size'] * accumulation_steps})")

        ######## DEBUG CODE ########
        if args.dry_run:
            print("   >> [Dry Run] Fetching 1 batch...")
            
            # Search for the first non-empty batch
            batch = None
            for b in train_loader:
                if b is not None:
                    batch = b
                    break
            
            if batch is None:
                print("❌ ERROR: Could not find any valid images in the dataset to test.")
                return

            # Now safe to subscript
            images, text = batch[0].to(cfg['training']['device']), batch[1].to(cfg['training']['device'])

            print("   >> [Dry Run] Testing Model Forward Pass...")
            img_emb, txt_emb, logits, local_features, img_emb_full = model(images, text)
            

            loss = criterions['contrastive'](img_emb, txt_emb, model.logit_scale)
            print(f"   >> [Dry Run] Success! Loss value: {loss.item():.4f}")
            train_loader = [b for i, b in enumerate(train_loader) if i < 2]
            val_loader = [b for i, b in enumerate(val_loader) if i < 2]
            #test_loader = [b for i, b in enumerate(test_loader) if i < 2]
        ######## DEBUG CODE ########

        # 5. Training Loop
        print("Starting Training...")
        best_loss = float('inf')
        global_step = 0

        # Early stopping configuration
        # Options: 'loss', 'recall', 'auc', 'combined'
        early_stopping_metric = cfg['training'].get('early_stopping_metric', 'loss')

        # For combined metrics, get weights (default: 70% recall, 30% AUC)
        combined_weights = cfg['training'].get('combined_weights', {'recall': 0.7, 'auc': 0.3})

        # AUC evaluation frequency (every N epochs, since it's slower)
        eval_auc_every = cfg['training'].get('eval_auc_every', 1)

        # Determine early stopping mode
        if early_stopping_metric == 'loss':
            early_stopping_mode = 'min'
        else:
            early_stopping_mode = 'max'  # recall, auc, combined are all "higher is better"

        early_stopping = EarlyStopping(
            patience=cfg['training'].get('early_stopping_patience', 5),
            checkpoint_path=os.path.join(args.output_dir, "best_model.pt"),
            verbose=True,
            mode=early_stopping_mode
        )

        print(f"Early stopping on: {early_stopping_metric} (mode: {early_stopping_mode})")
        if early_stopping_metric == 'combined':
            print(f"  Combined weights: recall={combined_weights['recall']}, auc={combined_weights['auc']}")
            print(f"  AUC evaluation frequency: every {eval_auc_every} epochs")

        for epoch in range(cfg['training']['epochs']):
            # Check for phase transition (staged training)
            if staged_training and current_phase == 1 and epoch == warmup_epochs:
                print(f"\n{'='*60}")
                print(f"PHASE TRANSITION: Epoch {epoch}")
                print(f"{'='*60}")

                # Unfreeze backbone
                unfreeze_backbone(model)

                # Create new optimizer with LLRD for fine-tuning
                optimizer = create_optimizer(model, cfg)

                # Create new scheduler for remaining epochs
                remaining_epochs = cfg['training']['epochs'] - warmup_epochs
                remaining_steps = (len(train_loader) * remaining_epochs) // accumulation_steps
                num_warmup_steps = cfg['training'].get('warmup_steps', 500)
                scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, remaining_steps)
                print(f"Phase 2 Scheduler: {num_warmup_steps} warmup steps, {remaining_steps} total steps")

                # Reset global step for the new scheduler
                global_step = 0
                current_phase = 2

                if wandb_run:
                    wandb_run.log({"training/phase": 2, "epoch": epoch})

            # Train
            avg_train_loss, global_step = train_one_epoch(model, train_loader, optimizer, criterions, device, epoch, scaler, use_amp, wandb_run, scheduler=scheduler, global_step=global_step, accumulation_steps=accumulation_steps)
            print(f"Epoch {epoch} Train Loss: {avg_train_loss}")

            # Validation with optional retrieval metrics and AUC
            use_retrieval = early_stopping_metric in ['recall', 'combined']
            use_auc = early_stopping_metric in ['auc', 'combined']

            # Compute validation loss and optionally retrieval metrics
            val_result = validate(model, val_loader, criterions, device, use_amp, compute_retrieval=use_retrieval)

            if use_retrieval:
                val_loss, retrieval_metrics = val_result
                mean_recall = retrieval_metrics['mean_recall']
            else:
                val_loss = val_result
                retrieval_metrics = None
                mean_recall = None

            # Compute AUC if needed (can be expensive, so optionally skip some epochs)
            val_auc = None
            if use_auc and (epoch % eval_auc_every == 0 or epoch == cfg['training']['epochs'] - 1):
                print(f"  Computing validation AUC...")
                val_auc = compute_validation_auc(model, train_loader, val_loader, device, use_amp, cfg=cfg)

            # Print results
            print_parts = [f"Epoch {epoch}", f"Train Loss: {avg_train_loss:.4f}", f"Val Loss: {val_loss:.4f}"]
            if mean_recall is not None:
                print_parts.append(f"Mean R@K: {mean_recall:.2f}%")
            if val_auc is not None:
                print_parts.append(f"Val AUC: {val_auc:.4f}")
            print(" | ".join(print_parts))

            if retrieval_metrics is not None:
                print(f"  i2t: R@1={retrieval_metrics['i2t_R@1']:.2f}%, R@5={retrieval_metrics['i2t_R@5']:.2f}%, R@10={retrieval_metrics['i2t_R@10']:.2f}%")
                print(f"  t2i: R@1={retrieval_metrics['t2i_R@1']:.2f}%, R@5={retrieval_metrics['t2i_R@5']:.2f}%, R@10={retrieval_metrics['t2i_R@10']:.2f}%")

            # Determine early stopping value based on metric
            if early_stopping_metric == 'loss':
                early_stopping_value = val_loss
            elif early_stopping_metric == 'recall':
                early_stopping_value = mean_recall
            elif early_stopping_metric == 'auc':
                if val_auc is not None:
                    early_stopping_value = val_auc * 100  # Scale to match recall range
                else:
                    # Skip early stopping check on epochs without AUC computation
                    early_stopping_value = None
            elif early_stopping_metric == 'combined':
                if val_auc is not None and mean_recall is not None:
                    # Combined metric: weighted average of recall and AUC
                    # Scale AUC to 0-100 range to match recall
                    auc_scaled = val_auc * 100
                    early_stopping_value = (
                        combined_weights['recall'] * mean_recall +
                        combined_weights['auc'] * auc_scaled
                    )
                    print(f"  Combined metric: {early_stopping_value:.2f} (recall={mean_recall:.2f}*{combined_weights['recall']} + AUC={auc_scaled:.2f}*{combined_weights['auc']})")
                else:
                    # If AUC wasn't computed this epoch, skip early stopping check
                    early_stopping_value = None

            # Only check early stopping if we have a valid metric value
            if early_stopping_value is not None:
                early_stopping(early_stopping_value, model, optimizer, epoch)

            # Logging
            if wandb_run:
                # Get current learning rate from optimizer
                current_lr = optimizer.param_groups[0]['lr']

                # Get the learned temperature (logit_scale)
                # CLIP uses exp(logit_scale) as the multiplier
                with torch.no_grad():
                    temp = model.logit_scale.exp().item()

                log_dict = {
                    "epoch": epoch,
                    "train/epoch_loss": avg_train_loss,
                    "val/loss": val_loss,
                    "learning_rate": current_lr,
                    "model/temperature": temp
                }

                # Log training phase for staged training
                if staged_training:
                    log_dict["training/phase"] = current_phase

                # Add retrieval metrics if computed
                if retrieval_metrics is not None:
                    log_dict["val/mean_recall"] = retrieval_metrics['mean_recall']
                    log_dict["val/i2t_R@1"] = retrieval_metrics['i2t_R@1']
                    log_dict["val/i2t_R@5"] = retrieval_metrics['i2t_R@5']
                    log_dict["val/i2t_R@10"] = retrieval_metrics['i2t_R@10']
                    log_dict["val/t2i_R@1"] = retrieval_metrics['t2i_R@1']
                    log_dict["val/t2i_R@5"] = retrieval_metrics['t2i_R@5']
                    log_dict["val/t2i_R@10"] = retrieval_metrics['t2i_R@10']

                # Add AUC if computed
                if val_auc is not None:
                    log_dict["val/auc"] = val_auc

                # Add combined metric if applicable
                if early_stopping_metric == 'combined' and early_stopping_value is not None:
                    log_dict["val/combined_metric"] = early_stopping_value

                # Log best score based on metric type
                if early_stopping.best_score is not None:
                    if early_stopping_metric == 'loss':
                        log_dict["best_val_loss"] = early_stopping.best_score
                    elif early_stopping_metric == 'recall':
                        log_dict["best_mean_recall"] = early_stopping.best_score
                    elif early_stopping_metric == 'auc':
                        log_dict["best_val_auc"] = early_stopping.best_score / 100  # Unscale
                    elif early_stopping_metric == 'combined':
                        log_dict["best_combined_metric"] = early_stopping.best_score

                wandb_run.log(log_dict)
            if early_stopping.early_stop:
                print(f"🛑 Early stopping triggered at epoch {epoch}")
                break

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving emergency checkpoint...")
        torch.save(model.state_dict(), os.path.join(args.output_dir, "emergency_checkpoint.pt"))
        
    finally:
        if wandb_run:
            wandb.finish()

if __name__ == "__main__":
    main()