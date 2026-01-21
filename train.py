import yaml
import torch
import argparse
import wandb
import os
from torch.cuda.amp import GradScaler

from models.full_model import ModelABaseline
from models.losses import ContrastiveLoss, SparsityLoss
from data.dataset import create_dataloaders
from engine.trainer import train_one_epoch

def get_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

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
        train_loader, val_loader, test_loader = create_dataloaders(cfg)
        print(f"Data Loaded: {len(train_loader)} train batches, {len(val_loader)} val batches.")

        # 3. Model
        model = ModelABaseline(cfg).to(device)
        
        if wandb_run:
            wandb.watch(model, log='all', log_freq=500)

        # 4. Optimization
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg['training']['lr']))
        
        # Losses
        # Make sure these classes exist in your src/models/ directory
        criterions = {
            'contrastive': ContrastiveLoss(),
            'sparsity': SparsityLoss(target_ratio=cfg['model']['mask_ratio'])
        }
        
        # GradScaler for mixed precision
        scaler = torch.amp.GradScaler(device, enabled=use_amp)

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
            img_emb, txt_emb, logits = model(images, text)
            

            loss = criterions['contrastive'](img_emb, txt_emb, model.logit_scale)
            print(f"   >> [Dry Run] Success! Loss value: {loss.item():.4f}")
            return
        ######## DEBUG CODE ########

        # 5. Training Loop
        print("Starting Training...")
        best_loss = float('inf')
        
        for epoch in range(cfg['training']['epochs']):
            # Train
            train_metrics = train_one_epoch(model, train_loader, optimizer, criterions, device, epoch, scaler, use_amp, wandb_run)
            print(f"Epoch {epoch} Train Loss: {train_metrics}") # Assuming train_one_epoch returns a float or dict
            
            # (Optional) Validation Loop could go here
            # val_loss = validate(model, val_loader...) 
            
            # Logging
            if wandb_run:
                # If train_one_epoch returns just a float:
                if isinstance(train_metrics, (float, int)):
                    wandb_run.log({'epoch': epoch, 'train_loss': train_metrics})
                else:
                    wandb_run.log(train_metrics)

            # Save Checkpoint (Every epoch)
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_ep{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_metrics,
            }, ckpt_path)
            
            # Save Best Model (Overwrite)
            # Logic: If current loss is lower than best, save as best_model.pt
            current_loss = train_metrics if isinstance(train_metrics, float) else train_metrics.get('total_loss', 0)
            if current_loss < best_loss:
                best_loss = current_loss
                best_path = os.path.join(args.output_dir, "best_model.pt")
                torch.save(model.state_dict(), best_path)
                print(f" >> Saved New Best Model (Loss: {best_loss:.4f})")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving emergency checkpoint...")
        torch.save(model.state_dict(), os.path.join(args.output_dir, "emergency_checkpoint.pt"))
        
    finally:
        if wandb_run:
            wandb.finish()

if __name__ == "__main__":
    main()