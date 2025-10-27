import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import math
from torch.utils.data import DataLoader, random_split


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, ff_dim=2048, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):

        attn_input = self.norm1(x)
        attn_output, _ = self.attention(attn_input, attn_input, attn_input)
        x = x + attn_output

        ff_input = self.norm2(x)
        ff_output = self.ff(ff_input)
        x = x + ff_output

        return x

class ChessPositionEncoder(nn.Module):

    def __init__(self, max_patches=64, embed_dim=512, dropout=0.1):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.zeros(1, max_patches + 1, embed_dim)) 
        self._init_positional_encoding(embed_dim, max_patches + 1)
        self.dropout = nn.Dropout(dropout)

    def _init_positional_encoding(self, embed_dim, length):
        position = torch.arange(length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(math.log(10000.0) / embed_dim))
        self.positional_embedding.data[0, :, 0::2] = torch.sin(position * div_term)
        self.positional_embedding.data[0, :, 1::2] = torch.cos(position * div_term)

    def forward(self, x):
        return self.dropout(x + self.positional_embedding[:, :x.size(1), :])

class TransformerChessModel(nn.Module):

    def __init__(self, in_channels=22, embed_dim=512, num_blocks=4, num_heads=8, ff_dim=2048, dropout=0.2):
        super().__init__()
        
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim//2, kernel_size=3, padding=1),
            nn.LayerNorm([embed_dim//2, 8, 8]),
            nn.GELU(),
            nn.Conv2d(embed_dim//2, embed_dim, kernel_size=3, padding=1, stride=2),
            nn.LayerNorm([embed_dim, 4, 4]),
            nn.GELU(),
            
        )
        
   
        self.pos_embed = nn.Parameter(torch.zeros(1, 16, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_blocks)
        ])
        

        self.norm = nn.LayerNorm(embed_dim)
     
        self.mlp_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Tanh() 
        )

        
    def forward(self, x):
  
        x = self.patch_embed(x)
        
        x = x.flatten(2).transpose(1, 2)  
        
        x = x + self.pos_embed

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        
        x = torch.cat((cls_token, x), dim=1)
        for block in self.transformer_blocks:
            x = block(x)
            
        x = x[:, 0]
        x = self.norm(x)
        x = self.mlp_head(x)
        return x