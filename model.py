"""Transformer model used to evaluate chess positions."""

from __future__ import annotations

import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    """Pre-norm transformer encoder block."""

    def __init__(self, embed_dim: int = 512, num_heads: int = 8, ff_dim: int = 2048, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention_input = self.norm1(x)
        attention_output, _ = self.attention(attention_input, attention_input, attention_input)
        x = x + attention_output

        feed_forward_input = self.norm2(x)
        return x + self.feed_forward(feed_forward_input)


class TransformerChessModel(nn.Module):
    """Evaluate a position encoded as ``[batch, 22, 8, 8]``.

    The scalar output is trained as a white-result score: 1.0 for a white win,
    0.5 for a draw, and 0.0 for a black win.
    """

    def __init__(
        self,
        in_channels: int = 22,
        embed_dim: int = 512,
        num_blocks: int = 4,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim // 2, kernel_size=3, padding=1),
            nn.LayerNorm([embed_dim // 2, 8, 8]),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, padding=1, stride=2),
            nn.LayerNorm([embed_dim, 4, 4]),
            nn.GELU(),
        )

        self.positional_embedding = nn.Parameter(torch.zeros(1, 16, embed_dim))
        nn.init.trunc_normal_(self.positional_embedding, std=0.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.transformer_blocks = nn.ModuleList(
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_blocks)
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.positional_embedding

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        for block in self.transformer_blocks:
            x = block(x)

        return self.head(self.norm(x[:, 0]))


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return total and trainable parameter counts."""

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable
