import torch
import torch.nn as nn
import open_clip


class BiomedCLIPBackbone(nn.Module):
    def __init__(self,
                 model_name="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
                 device="cuda"):
        super().__init__()

        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            model_name,
            device=device
        )

        self.visual = self.clip_model.visual
        self.text = self.clip_model.text

    def lock_backbones(self):
        for p in self.clip_model.parameters():
            p.requires_grad = False

    def encode_text(self, text_tokens):
        """
        text_tokens: already tokenized (B, L)
        """
        return self.clip_model.encode_text(text_tokens)

    def encode_image_patches(self, images):
        # Checks if the model has a "trunk" (common in some timm/clip backends)
        # or uses forward_features directly.
        if hasattr(self.visual, 'forward_features'):
            features = self.visual.forward_features(images)
        elif hasattr(self.visual, 'trunk'):
             features = self.visual.trunk.forward_features(images)
        else:
             # Fallback to your manual code if the method doesn't exist
             return self._manual_forward(images)
        
        return features

    def _manual_forward(self, x):
        """
        A manual forward pass for ViT models, used as a fallback.
        """
        x = self.visual.patch_embed(x)
        x = torch.cat([self.visual.cls_token.expand(x.shape[0], -1, -1), x], dim=1)
        x = x + self.visual.pos_embed
        x = self.visual.blocks(x)
        x = self.visual.norm(x)
        return x
