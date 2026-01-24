import os
import json
import torch
import open_clip
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

class MedicalImageTextDataset(Dataset):
    def __init__(self, jsonl_path, image_root, split="train", transform=None, tokenizer_name="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", return_labels=False):
        """
        Args:
            jsonl_path: Path to mimicxr_parsed_ds.jsonl
            image_root: Path to the folder containing MIMIC-CXR images (e.g., /datasets/MIMIC-CXR/files/)
            split: 'train', 'validate', or 'test'
        """
        self.image_root = image_root
        self.transform = transform
        self.tokenizer = open_clip.get_tokenizer(tokenizer_name)
        self.return_labels = return_labels
        self.data = []

        # Load JSONL and filter by split
        print(f"Loading dataset from {jsonl_path} for split: {split}...")
        with open(jsonl_path, 'r') as f:
            for line in f:
                entry = json.loads(line)
                if entry['split'] == split:
                    self.data.append(entry)
        
        print(f"Loaded {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        
        # 1. Text Parsing
        raw_text = entry.get('query', '')
        caption = raw_text.replace('|', ' ')
        
        # 2. Get verified image path
        img_path = entry.get('image_path')
        
        if img_path and os.path.exists(img_path):
            try:
                image = Image.open(img_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                
                text = self.tokenizer(caption).squeeze(0)
                
                if self.return_labels:
                    labels = torch.tensor(entry.get('labels', [0]*14), dtype=torch.float32)
                    return image, text, labels
                
                return image, text
                
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                return None
        else:
            return None

def get_transforms(is_train=True, img_size=224):
    """
    Standard ImageNet normalization + Augmentation for training.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.9, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

def collate_fn(batch):
    # Filter out None samples
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

def create_dataloaders(config, batch_size=None, return_labels=False):
    """
    Factory function to create Train/Val/Test loaders from config.
    """
    # 1. Parse Config
    data_cfg = config['data']
    model_cfg = config.get('model', {}) # Handle if model section is missing
    
    # Defaults
    bs = batch_size if batch_size is not None else data_cfg.get('batch_size', 32)
    num_workers = data_cfg.get('num_workers', 4)
    img_size = data_cfg.get('image_size', 224)
    tokenizer = model_cfg.get('tokenizer_name', "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")

    # 2. Instantiate Datasets
    jsonl_path = data_cfg.get('jsonl_path') or data_cfg.get('json_path')
    if not jsonl_path:
        raise ValueError("Config must contain 'jsonl_path' or 'json_path'.")

    print(f"[Data] Creating Datasets from {jsonl_path}")

    train_ds = MedicalImageTextDataset(
        jsonl_path=jsonl_path,
        image_root=data_cfg['image_root'],
        split="train",
        transform=get_transforms(True, img_size),
        tokenizer_name=tokenizer,
        return_labels=return_labels
    )

    val_ds = MedicalImageTextDataset(
        jsonl_path=jsonl_path,
        image_root=data_cfg['image_root'],
        split="validate",
        transform=get_transforms(False, img_size),
        tokenizer_name=tokenizer,
        return_labels=return_labels
    )

    test_ds = MedicalImageTextDataset(
        jsonl_path=jsonl_path,
        image_root=data_cfg['image_root'],
        split="test",
        transform=get_transforms(False, img_size),
        tokenizer_name=tokenizer,
        return_labels=return_labels
    )

    # 3. Create Loaders
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader