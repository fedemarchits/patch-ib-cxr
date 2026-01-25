import yaml
import torch
import argparse
import wandb
import os
import math
from torch.amp import GradScaler
from torch.optim.lr_scheduler import LambdaLR

from models.full_model import ModelABaseline
from models.losses import ContrastiveLoss, SparsityLoss, LocalAlignmentLoss
from data.dataset import create_dataloaders
from engine.trainer import train_one_epoch
from engine.utils import EarlyStopping
from engine.validator import validate

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
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg['training']['lr']),
            weight_decay=float(cfg['training'].get('weight_decay', 0.01))
        )
        
        # Losses
        # Make sure these classes exist in your src/models/ directory
        criterions = {
            'contrastive': ContrastiveLoss(),
            'sparsity': SparsityLoss(target_ratio=cfg['model']['mask_ratio'])
        }

        # Add local alignment loss if enabled
        if cfg['model'].get('use_local_alignment', False):
            criterions['local_alignment'] = LocalAlignmentLoss(
                temperature=cfg['model'].get('local_alignment_temperature', 1.0)
            )
            criterions['local_weight'] = cfg['model'].get('local_alignment_weight', 0.1)
        
        # GradScaler for mixed precision
        scaler = torch.amp.GradScaler(device, enabled=use_amp)

        # Learning rate scheduler with warmup
        num_training_steps = len(train_loader) * cfg['training']['epochs']
        num_warmup_steps = cfg['training'].get('warmup_steps', 500)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
        print(f"Scheduler: {num_warmup_steps} warmup steps, {num_training_steps} total steps")

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
            img_emb, txt_emb, logits, local_features = model(images, text)
            

            loss = criterions['contrastive'](img_emb, txt_emb, model.logit_scale)
            print(f"   >> [Dry Run] Success! Loss value: {loss.item():.4f}")
            train_loader = [b for i, b in enumerate(train_loader) if i < 2]
            val_loader = [b for i, b in enumerate(val_loader) if i < 2]
            #test_loader = [b for i, b in enumerate(test_loader) if i < 2]
        ######## DEBUG CODE ########

        # 5. Training Loop
        print("Starting Training...")
        best_loss = float('inf')

        early_stopping = EarlyStopping(
            patience=cfg['training'].get('early_stopping_patience', 5),
            checkpoint_path=os.path.join(args.output_dir, "best_model.pt"),
            verbose=True
        )
        
        for epoch in range(cfg['training']['epochs']):
            # Train
            avg_train_loss = train_one_epoch(model, train_loader, optimizer, criterions, device, epoch, scaler, use_amp, wandb_run, scheduler=scheduler)
            print(f"Epoch {epoch} Train Loss: {avg_train_loss}") 
            
            # (Optional) Validation Loop could go here
            val_loss = validate(model, val_loader, criterions, device, use_amp)

            print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")

            early_stopping(val_loss, model, optimizer, epoch)

            # Logging
            if wandb_run:
                # Get current learning rate from optimizer
                current_lr = optimizer.param_groups[0]['lr']
                
                # Get the learned temperature (logit_scale)
                # CLIP uses exp(logit_scale) as the multiplier
                with torch.no_grad():
                    temp = model.logit_scale.exp().item()

                wandb_run.log({
                    "epoch": epoch, 
                    "train/epoch_loss": avg_train_loss,
                    "val/loss": val_loss,
                    "best_val_loss": early_stopping.best_loss,
                    "learning_rate": current_lr,
                    "model/temperature": temp
                })
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