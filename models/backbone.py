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

    def encode_text_tokens(self, text_tokens):
        """
        Extract token-level embeddings from the text encoder.

        Args:
            text_tokens: (B, L) tokenized text

        Returns:
            token_embeddings: (B, L, 768) - raw token embeddings before pooling
            attention_mask: (B, L) - mask where 1=valid token, 0=padding
        """
        text_encoder = self.clip_model.text

        # Get the attention mask (1 for real tokens, 0 for padding)
        # BiomedCLIP/PubMedBERT uses 0 as pad_token_id
        attention_mask = (text_tokens != 0).long()

        # Forward through the transformer to get hidden states
        out = text_encoder.transformer(
            input_ids=text_tokens,
            attention_mask=attention_mask
        )

        # last_hidden_state: (B, L, 768)
        return out.last_hidden_state, attention_mask

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
