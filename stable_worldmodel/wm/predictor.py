# Make space-time attention module
# Make space-time transformer module
# Make Better predictor module

# kv-cache
# RoPE
# Grouped Query Attention
# SDPA
# BF16

# RMSNorm/SwiGLU


# https://github.com/myscience/open-genie/blob/main/genie/module/attention.py

import torch
from einops import rearrange
from torch import nn

from stable_worldmodel.wm.module import FeedForward, FusedSTAttention, STAttention


class CausalPredictor(nn.Module):
    def __init__(
        self,
        *,
        num_patches,
        num_frames,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        assert pool in {"cls", "mean"}, "pool type must be either cls (cls token) or mean (mean pooling)"

        self.num_patches = num_patches
        self.num_frames = num_frames

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames * (num_patches), dim))  # dim for the pos encodings
        self.dropout = nn.Dropout(emb_dropout)
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        FusedSTAttention(
                            dim,
                            heads=heads,
                            dim_head=dim_head,
                            dropout=dropout,
                            num_patches=num_patches,
                            num_frames=num_frames,
                        ),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )

        self.pool = pool

    def forward(self, x):  # x: (b, t, p, dim)
        b, t, p, _ = x.shape
        x = rearrange(x, "b t p d -> b (t p) d")
        x = x + self.pos_embedding[:, : t * p]
        x = rearrange(x, "b (t p) d -> b t p d", p=p)
        x = self.dropout(x)

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class FastPredictor(nn.Module):
    def __init__(
        self,
        *,
        num_patches,
        num_frames,
        dim,
        depth,
        heads,
        mlp_dim,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.num_frames = num_frames
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, num_patches, dim))
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(emb_dropout)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        STAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x):  # x: (b, window_size * H/patch_size * W/patch_size, 384)
        b, t, p, _ = x.shape
        x = x + self.pos_embedding[:, :t, :p]
        x = self.dropout(x)

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)
