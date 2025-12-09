# src/tpse_model.py
import torch
import torch.nn as nn

class TPSEModel(nn.Module):
    """
    Small Transformer-based TPSE.
    Input: (B, T, D_in)
    Output: time-dependent vectors (B, T, hidden_dim)
    Added LayerNorm (input_norm) so older checkpoints can load with strict=False.
    """
    def __init__(self, input_dim, hidden_dim=128, n_layers=2, n_heads=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        # normalization to stabilize scale
        self.input_norm = nn.LayerNorm(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        # optional final projection (keeps scale bounded)
        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, mask=None):
        # x: (B, T, D_in)
        h = self.input_proj(x)            # (B, T, H)
        h = self.input_norm(h)           # stabilize activations
        # src_key_padding_mask expects True where positions are padded
        src_key_padding_mask = None
        if mask is not None:
            # mask is (B, T) boolean True for valid; transformer wants True for padded positions
            src_key_padding_mask = ~mask
        h2 = self.transformer(h, src_key_padding_mask=src_key_padding_mask)  # (B, T, H)
        out = self.out_proj(h2)  # (B, T, H)
        return out  # (B, T, H)
