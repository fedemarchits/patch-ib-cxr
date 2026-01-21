import os
import json
import torch
import open_clip
import datasets
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

class MedicalImageTextDataset(Dataset):
    def __init__(self, jsonl_path, image_root, split="train", transform=None, tokenizer_name="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"):
        """
        Args:
            jsonl_path: Path to mimicxr_parsed_ds.jsonl
            image_root: Path to the folder containing MIMIC-CXR images (e.g., /datasets/MIMIC-CXR/files/)
            split: 'train', 'validate', or 'test'
        """
        self.image_root = image_root
        self.transform = transform
        self.tokenizer = open_clip.get_tokenizer(tokenizer_name)
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
        
        # 1. Parse Text (Query)
        # The query is "keyword|keyword|keyword". We replace pipes with spaces.
        # Check if 'query' exists, otherwise use a placeholder
        raw_text = entry.get('query', '')
        caption = raw_text.replace('|', ' ')
        
        # 2. Construct Image Path
        # MIMIC-CXR structure is usually: p{subject_id[:2]}/p{subject_id}/s{study_id}.jpg
        # WARNING: You must verify if the images on the server are .jpg or .dcm
        # and if they follow the standard folder structure.
        subj_id = str(entry['subject_id'])
        study_id = str(entry['study_id'])
        
        # Folder pXX (first two digits of subject_id)
        p_folder = f"p{subj_id[:2]}"
        # Folder pXXXXXXX
        subj_folder = f"p{subj_id}"
        
        # Construct path. Assuming standard MIMIC-CXR JPG layout:
        # If the server has a flat structure, change this line!
        img_filename = f"s{study_id}.jpg" 
        img_path = os.path.join(self.image_root, p_folder, subj_folder, img_filename)

        # 3. Load Image
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            # Fallback for debugging if paths are wrong
            print(f"MISSING IMAGE: {img_path}")
            # Return a black image to prevent crash during debugging
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)
            
        # 4. Tokenize
        text = self.tokenizer(caption).squeeze(0)

        # 5. Labels (if available in the JSONL)
        labels = torch.tensor(entry.get('labels', [0]*14))

        return image, text, labels

class HuggingFaceMIMICDataset(Dataset):
    def __init__(self, dataset_name, split="train", transform=None, tokenizer_name="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"):
        """
        Args:
            dataset_name: Name of the dataset on Hugging Face Hub (e.g., "itsanmolgupta/mimic-cxr-dataset")
            split: 'train', 'validation', or 'test'
        """
        self.transform = transform
        self.tokenizer = open_clip.get_tokenizer(tokenizer_name)
        
        print(f"Loading dataset {dataset_name} from Hugging Face Hub for split: {split}...")
        # Note: The split name in the HF dataset is 'validation', not 'validate'
        hf_split = 'validation' if split == 'validate' else split

        # Check available splits and fallback to 'train' if requested split doesn't exist
        available_splits = datasets.get_dataset_split_names(dataset_name)
        if hf_split not in available_splits:
            print(f"   >> Warning: Split '{hf_split}' not found. Available: {available_splits}. Using 'train' instead.")
            hf_split = 'train'

        self.dataset = datasets.load_dataset(dataset_name, split=hf_split)
        
        print(f"Loaded {len(self.dataset)} samples.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        entry = self.dataset[idx]
        
        # 1. Get Text
        caption = entry.get('text', '')
        
        # 2. Get Image
        image = entry['image'].convert('RGB')

        if self.transform:
            image = self.transform(image)
            
        # 3. Tokenize
        text = self.tokenizer(caption).squeeze(0)
        
        labels = torch.tensor(entry.get('labels', [0]*14))

        return image, text, labels

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

def create_dataloaders(config, batch_size=None):
    """
    Factory function to create Train/Val/Test loaders from config.
    """
    # 1. Parse Config
    data_cfg = config['data']
    model_cfg = config.get('model', {}) # Handle if model section is missing
    
    # Defaults
    bs = batch_size if batch_size is not None else data_cfg.get('batch_size', 32)
    num_workers = data_cfg.get('num_workers', 4)
    img_size = data_cfg.get('img_size', 224)
    tokenizer = model_cfg.get('tokenizer_name', "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")

    # 2. Instantiate Datasets
    # Check if we are using Local JSONL or HuggingFace based on config keys
    if 'jsonl_path' in data_cfg or 'json_path' in data_cfg:
        # --- LOCAL DATASET STRATEGY ---
        jsonl_path = data_cfg.get('jsonl_path') or data_cfg.get('json_path')
        print(f"[Data] Creating Local Datasets from {jsonl_path}")

        train_ds = MedicalImageTextDataset(
            jsonl_path=jsonl_path,
            image_root=data_cfg['image_root'],
            split="train",
            transform=get_transforms(True, img_size),
            tokenizer_name=tokenizer
        )
        
        val_ds = MedicalImageTextDataset(
            jsonl_path=jsonl_path,
            image_root=data_cfg['image_root'],
            split="validate",
            transform=get_transforms(False, img_size),
            tokenizer_name=tokenizer
        )

        test_ds = MedicalImageTextDataset(
            jsonl_path=jsonl_path,
            image_root=data_cfg['image_root'],
            split="test",
            transform=get_transforms(False, img_size),
            tokenizer_name=tokenizer
        )

    elif 'huggingface_dataset' in data_cfg or 'dataset_name' in data_cfg:
        # --- HUGGINGFACE STRATEGY ---
        ds_name = data_cfg.get('huggingface_dataset') or data_cfg.get('dataset_name')
        print(f"[Data] Creating HuggingFace Datasets: {ds_name}")
        
        train_ds = HuggingFaceMIMICDataset(ds_name, split="train", transform=get_transforms(True, img_size), tokenizer_name=tokenizer)
        val_ds = HuggingFaceMIMICDataset(ds_name, split="validation", transform=get_transforms(False, img_size), tokenizer_name=tokenizer)
        test_ds = HuggingFaceMIMICDataset(ds_name, split="test", transform=get_transforms(False, img_size), tokenizer_name=tokenizer)
    
    else:
        raise ValueError("Config must contain 'jsonl_path'/'json_path' (local) or 'huggingface_dataset'/'dataset_name' (HF).")

    # 3. Create Loaders
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader