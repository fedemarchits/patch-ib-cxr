import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, img_feat, text_feat, logit_scale):
        # Scale logic from CLIP
        logit_scale = logit_scale.exp()
        
        # Cosine similarity
        logits_per_image = logit_scale * img_feat @ text_feat.t()
        logits_per_text = logits_per_image.t()
        
        # Targets are diagonal (0, 1, 2...)
        batch_size = img_feat.shape[0]
        labels = torch.arange(batch_size, device=img_feat.device)
        
        loss_i = self.cross_entropy(logits_per_image, labels)
        loss_t = self.cross_entropy(logits_per_text, labels)
        
        return (loss_i + loss_t) / 2

class SparsityLoss(nn.Module):
    """
    Encourages the mask to be sparse (Information Bottleneck).
    """
    def __init__(self, target_ratio=0.5):
        super().__init__()
        self.target_ratio = target_ratio

    def forward(self, importance_logits):
        # Sigmoid to approximate the binary decision for loss calculation
        probs = torch.sigmoid(importance_logits)
        mean_activation = probs.mean()
        # L2 penalty towards target ratio
        return (mean_activation - self.target_ratio) ** 2