import torch
import torch.nn as nn


class BidirectionalCrossAttention(nn.Module):
    """
    Bidirectional cross-attention module for mid-fusion between ViT and BERT.

    Two directions:
      - v2t: ViT queries BERT (image attends to text)
      - t2v: BERT queries ViT (text attends to image)

    Each direction: LayerNorm(query) -> MultiheadAttention -> out_proj -> residual add.
    Output projections are zero-initialized so the module starts as identity,
    preserving pretrained behavior at initialization.
    """

    def __init__(self, d_model=768, n_heads=12, dropout=0.1):
        super().__init__()

        # ViT queries BERT (image attends to text)
        self.v2t_norm = nn.LayerNorm(d_model)
        self.v2t_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.v2t_out_proj = nn.Linear(d_model, d_model)

        # BERT queries ViT (text attends to image)
        self.t2v_norm = nn.LayerNorm(d_model)
        self.t2v_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.t2v_out_proj = nn.Linear(d_model, d_model)

        # Zero-init output projections so module starts as identity
        nn.init.zeros_(self.v2t_out_proj.weight)
        nn.init.zeros_(self.v2t_out_proj.bias)
        nn.init.zeros_(self.t2v_out_proj.weight)
        nn.init.zeros_(self.t2v_out_proj.bias)

    def forward(self, x_vit, x_bert, bert_padding_mask=None, return_attention=False):
        """
        Args:
            x_vit:  (B, N_v, D) - ViT hidden states (CLS + patches)
            x_bert: (B, N_t, D) - BERT hidden states (CLS + tokens)
            bert_padding_mask: (B, N_t) - True for padding positions to ignore
            return_attention: If True, also return attention weights for visualization

        Returns:
            x_vit:  (B, N_v, D) - updated ViT hidden states
            x_bert: (B, N_t, D) - updated BERT hidden states
            (optional) v2t_weights: (B, N_v, N_t) - image-to-text attention weights
            (optional) t2v_weights: (B, N_t, N_v) - text-to-image attention weights
        """
        # Direction 1: ViT queries BERT (image attends to text)
        q_vit = self.v2t_norm(x_vit)
        v2t_out, v2t_weights = self.v2t_attn(
            query=q_vit,
            key=x_bert,
            value=x_bert,
            key_padding_mask=bert_padding_mask,
            need_weights=return_attention,
            average_attn_weights=True,
        )
        x_vit = x_vit + self.v2t_out_proj(v2t_out)

        # Direction 2: BERT queries ViT (text attends to image)
        q_bert = self.t2v_norm(x_bert)
        t2v_out, t2v_weights = self.t2v_attn(
            query=q_bert,
            key=x_vit,
            value=x_vit,
            need_weights=return_attention,
            average_attn_weights=True,
        )
        x_bert = x_bert + self.t2v_out_proj(t2v_out)

        if return_attention:
            return x_vit, x_bert, v2t_weights, t2v_weights

        return x_vit, x_bert
