import yaml
import torch
import argparse
import wandb
from torch.utils.data import DataLoader
from models.full_model import ModelABaseline
from models.losses import ContrastiveLoss, SparsityLoss
from data.dataset import MedicalImageTextDataset, HuggingFaceMIMICDataset
from data.transforms import get_transforms
from torch.cuda.amp import GradScaler
from engine.trainer import train_one_epoch

def main():
    parser = argparse.ArgumentParser(description="Main training script")
    parser.add_argument("--config", type=str, default="configs/model_a_baseline.yaml",
                        help="Path to the configuration file.")
    args = parser.parse_args()

    # Load Config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    
    # WandB Initialization
    wandb_cfg = cfg.get('wandb', {})
    if wandb_cfg.get('enable', False):
        wandb_run = wandb.init(
            project=wandb_cfg.get('project', 'my-thesis-project'),
            entity=wandb_cfg.get('entity', None), # Replace with your wandb entity
            name=wandb_cfg.get('run_name', 'default-run'),
            config=cfg
        )
    else:
        wandb_run = None

    try:
        device = cfg['training']['device']
        use_amp = cfg['training'].get('use_amp', False)
        data_cfg = cfg['data']

        # Transforms
        train_tfm = get_transforms(cfg['model']['vision_backbone'], is_train=True)
        
        # Initialize Dataset based on config
        if data_cfg.get('dataset_type') == 'hf':
            dataset = HuggingFaceMIMICDataset(
                dataset_name=data_cfg['dataset_name'],
                split="train",
                transform=train_tfm
            )
        else:
            dataset = MedicalImageTextDataset(
                jsonl_path=data_cfg['jsonl_path'],   
                image_root=data_cfg['image_root'], 
                split="train",                        
                transform=train_tfm
            )

        dataloader = DataLoader(dataset, batch_size=data_cfg['batch_size'], shuffle=True, num_workers=data_cfg['num_workers'])
        
        # Model
        model = ModelABaseline(cfg).to(device)
        
        if wandb_run:
            wandb.watch(model, log='all', log_freq=500)

        # Optimization
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg['training']['lr']))
        
        # Losses
        criterions = {
            'contrastive': ContrastiveLoss(),
            'sparsity': SparsityLoss(target_ratio=cfg['model']['mask_ratio'])
        }
        
        # GradScaler for mixed precision
        scaler = torch.amp.GradScaler(device, enabled=use_amp)

        # Loop
        for epoch in range(cfg['training']['epochs']):
            loss = train_one_epoch(model, dataloader, optimizer, criterions, device, epoch, scaler, use_amp, wandb_run)
            print(f"Epoch {epoch}: Loss {loss}")
            if wandb_run:
                wandb_run.log({'epoch': epoch, 'avg_loss': loss})

            
            # Save checkpoint
            torch.save(model.state_dict(), f"checkpoint_ep{epoch}.pth")

    finally:
        if wandb_run:
            wandb.finish()

if __name__ == "__main__":
    main()

    
    