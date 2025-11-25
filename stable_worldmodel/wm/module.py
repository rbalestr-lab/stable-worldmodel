import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F


class Attention(nn.Module):
    """Implements Scaled Dot-Product Attention"""

    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.0,
        is_causal=False,
        use_cache=False,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.causal = is_causal
        self.scale = dim_head**-0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()
        self.rope = nn.Identity()

    def forward(self, x, attn_mask=None):
        B, T, C = x.size()
        x = self.norm(x)

        # q, k, v: (B, heads, T, dim_head)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in qkv)

        q = self.rope(q)
        k = self.rope(k)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout.p if self.training else 0.0, is_causal=self.causal
        )

        out = rearrange(out, "b h n d -> b n (h d)")

        return self.to_out(out)


class SpatialAttention(Attention):
    """Implements Spatial Attention over patches within each frame (non causal)"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__(dim, heads, dim_head, dropout, is_causal=False)

        # todo add 2D RoPe + KV cache etc...

    def forward(self, x):
        # x : (B, T, P, C)

        # rearrange to (B*T, P, C)
        B, T, P, C = x.size()
        x = rearrange(x, "b t p c -> (b t) p c")
        out = super().forward(x)

        # rearrange back to (B, T, P, C)
        out = rearrange(out, "(b t) p c -> b t p c", b=B, t=T)

        return out


class TemporalAttention(Attention):
    """Implements Temporal Attention with a causal mask for future frames"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__(dim, heads, dim_head, dropout, is_causal=True)

        # todo add 1D RoPe + KV cache etc...

    def forward(self, x):
        # x : (B, T, P, C)

        # rearrange to (B*T, P, C)
        B, T, P, C = x.size()
        x = rearrange(x, "b t p c -> (b p) t c")
        out = super().forward(x)

        # rearrange back to (B, T, P, C)
        out = rearrange(out, "(b p) t c -> b t p c", b=B, p=P)

        return out


class STAttention(nn.Module):
    """Implements Spatio-Temporal Attention with separate Spatial and Temporal Attention layers"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()

        self.spatial_attn = SpatialAttention(dim, heads, dim_head, dropout)
        self.temporal_attn = TemporalAttention(dim, heads, dim_head, dropout)

    def forward(self, x):
        # x : (B, T, P, C)
        x = self.spatial_attn(x) + x
        x = self.temporal_attn(x) + x
        return x


class FusedSTAttention(Attention):
    """Implements Spatio-Temporal Attention with a single fused attention layer for space-time"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, num_patches=1, num_frames=1):
        super().__init__(dim, heads, dim_head, dropout, is_causal=False)
        self.register_buffer("bias", self.generate_mask_matrix(num_patches, num_frames))

    def forward(self, x):
        # x : (B, T, P, C)

        # rearrange to (B, T*P, C)
        B, T, P, C = x.size()
        x = rearrange(x, "b t p c -> b (t p) c")

        # compute attention with fused spatio-temporal mask
        attn_mask = self.bias[:, :, : (T * P), : (T * P)] == 1  # bool mask
        out = super().forward(x, attn_mask=attn_mask)

        # rearrange back to (B, T, P, C)
        out = rearrange(out, "b (t p) c -> b t p c", t=T, p=P)

        return out

    def generate_mask_matrix(self, npatch, nwindow):
        zeros = torch.zeros(npatch, npatch)
        ones = torch.ones(npatch, npatch)
        rows = []
        for i in range(nwindow):
            row = torch.cat([ones] * (i + 1) + [zeros] * (nwindow - i - 1), dim=1)
            rows.append(row)
        mask = torch.cat(rows, dim=0).unsqueeze(0).unsqueeze(0)
        return mask


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0, act_fn=nn.GELU, norm_fn=nn.LayerNorm):
        super().__init__()
        self.net = nn.Sequential(
            norm_fn(dim),
            nn.Linear(dim, hidden_dim),
            act_fn(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Embedder(torch.nn.Module):
    def __init__(
        self,
        num_frames=1,
        tubelet_size=1,
        in_chans=8,
        emb_dim=10,
    ):
        super().__init__()

        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.in_chans = in_chans
        self.emb_dim = emb_dim

        self.patch_embed = torch.nn.Conv1d(in_chans, emb_dim, kernel_size=tubelet_size, stride=tubelet_size)

    def forward(self, x):
        x = x.float()
        x = x.permute(0, 2, 1)  # (B, T, B) -> (B, D, T)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)  # (B, D, T) -> (B, T, D)
        return x


if __name__ == "__main__":
    # test fused spatio-temporal attention
    B, T, P, C = 4, 3, 196, 128
    x = torch.randn(B, T, P, C)
    attn = FusedSTAttention(dim=C, heads=4, dim_head=16, dropout=0.1, num_patches=P, num_frames=T)
    st_attn = STAttention(dim=C, heads=4, dim_head=16, dropout=0.1)
    out = attn(x)
    print(out.shape)  # should be (B, T, P, C)

    out2 = st_attn(x)
    print(out2.shape)  # should be (B, T, P, C)
