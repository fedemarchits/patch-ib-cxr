import torch
import torch.nn as nn

class StraightThroughEstimator(torch.autograd.Function):
    """
    Passes binary mask forward, passes gradient back to the logits unchanged.
    """
    @staticmethod
    def forward(ctx, input_logits):
        # Hard thresholding (e.g., > 0.5)
        return (input_logits > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        # Identity gradient
        return grad_output

class PatchMaskingHead(nn.Module):
    def __init__(self, input_dim, reduction_ratio=0.5):
        super().__init__()
        # Simple MLP to predict "keep" importance score for each patch
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, 1) # Output: Logit for "Importance"
        )
        self.reduction_ratio = reduction_ratio

    def forward(self, patch_embeddings):
        """
        Args:
            patch_embeddings: (B, N_patches, Dim) - EXCLUDING CLS token usually
        """
        # Predict importance logits
        logits = self.predictor(patch_embeddings).squeeze(-1) # (B, N)
        
        # Option A: Dynamic Keep based on ratio (Top-K)
        # Option B: Learned Threshold with STE (Simpler for "Bottleneck")
        
        # Here we use a simple STE on the logits for binary masking
        # In a real IB, you might penalize the number of 1s.
        
        mask = StraightThroughEstimator.apply(logits) 
        
        # Alternatively: Gumbel Softmax for differentiable sampling
        # mask = F.gumbel_softmax(logits, hard=True)
        
        return mask, logits