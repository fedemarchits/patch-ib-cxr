import torch
import torch.nn as nn
from .backbone import BiomedCLIPBackbone
from .heads import PatchMaskingHead

class ModelABaseline(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = BiomedCLIPBackbone(
            model_name=cfg['model']['vision_backbone'],
            device=cfg['training']['device']
        )
        
        # Dim check: ViT-Base usually 768
        self.embed_dim = 768 

        self.use_masking = cfg['model'].get('use_masking', False)
        
        # The Information Bottleneck Masker
        if self.use_masking:
            self.mask_head = PatchMaskingHead(self.embed_dim)
        
        # Projectors (if needed, though BiomedCLIP has them built-in)
        # We might need a projector after masking if we change the flow
        self.logit_scale = self.backbone.clip_model.logit_scale

    def forward(self, images, text):
        # 1. Get Raw Features (Batch, 197, 768)
        features = self.backbone.encode_image_patches(images)
        
        importance_logits = None
        
        if self.use_masking:
            cls_token = features[:, 0, :]
            patch_tokens = features[:, 1:, :]
            
            # 2. Generate Mask (The "Bottleneck")
            # 1 = Keep, 0 = Drop
            mask, importance_logits = self.mask_head(patch_tokens)
            
            # 3. Apply Mask
            masked_patches = patch_tokens * mask.unsqueeze(-1)
            
            # 4. Global Pooling of Masked Patches
            mask_sum = mask.sum(dim=1, keepdim=True) + 1e-6
            img_embedding = masked_patches.sum(dim=1) / mask_sum
        else:
            # Model A: Take index 0 (CLS token) -> Shape: (Batch, 768)
            img_embedding = features[:, 0, :] 

        # ==================== FINAL FIX FOR BIOMEDCLIP ====================
        visual_model = self.backbone.clip_model.visual
        
        # CASE 1: Standard OpenAI CLIP (Matrix Multiplication)
        if hasattr(visual_model, 'proj') and visual_model.proj is not None:
            projection = visual_model.proj.to(img_embedding.dtype)
            img_embedding = img_embedding @ projection
            
        # CASE 2: Timm/BiomedCLIP (Linear Layer)
        elif hasattr(visual_model, 'head') and visual_model.head is not None:
            # .head is a nn.Linear(768, 512) layer. We just pass the data through it.
            img_embedding = visual_model.head(img_embedding)
            
        else:
            # FALLBACK DEBUGGING (If both fail, we need to see what exists)
            print("CRITICAL ERROR: Could not find projection layer.")
            print(f"Available attributes: {dir(visual_model)}")
            raise AttributeError("Cannot project 768->512. Check logs.")
        # ==================================================================

        # Normalize
        img_embedding = img_embedding / (img_embedding.norm(dim=-1, keepdim=True) + 1e-6)
        
        # Encode Text
        text_embedding = self.backbone.encode_text(text)
        text_embedding = text_embedding / (text_embedding.norm(dim=-1, keepdim=True) + 1e-6)
        
        return img_embedding, text_embedding, importance_logits