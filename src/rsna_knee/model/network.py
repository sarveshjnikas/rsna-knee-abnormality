"""KneeNet: one study (slots x slices x H x W) -> 12 logits.

1. 2.5D input: each slice is stacked with its two neighbours as 3 channels, giving the
   ImageNet-pretrained 2D encoder a little through-plane context.
2. A shared 2D encoder embeds every slice of every slot.
3. Per slot, learned attention pools the slices (findings sit in a few slices only).
4. Slot embeddings get a slot-type embedding; missing slots are masked out; a
   transformer layer lets views exchange information; masked mean -> head -> 12 logits.

fusion="label" instead gives every finding its own attention over all slices of all views
(an ACL tear shows on two slices, osteoarthritis on many; one shared pooling makes them compete).
"""

import timm
import torch
from torch import nn

from rsna_knee.config import LABELS


def neighbour_channels(volume: torch.Tensor) -> torch.Tensor:
    """(B, S, N, H, W) -> (B, S, N, 3, H, W): slices i-1, i, i+1 (edges repeated)."""
    prev = torch.cat([volume[:, :, :1], volume[:, :, :-1]], dim=2)
    nxt = torch.cat([volume[:, :, 1:], volume[:, :, -1:]], dim=2)
    return torch.stack([prev, volume, nxt], dim=3)


class AttentionPool(nn.Module):
    """Softmax-weighted average over the slice axis."""

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, dim // 4), nn.Tanh(), nn.Linear(dim // 4, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, N, D) -> (B, D)
        weights = self.score(x).softmax(dim=1)
        return (weights * x).sum(dim=1)


class LabelAttention(nn.Module):
    """One softmax attention per label over all (view, slice) tokens, then one linear output per label."""

    def __init__(self, dim: int, n_labels: int, dropout: float):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, dim // 4), nn.Tanh(), nn.Linear(dim // 4, n_labels))
        self.norm, self.drop = nn.LayerNorm(dim), nn.Dropout(dropout)
        self.weight = nn.Parameter(torch.randn(n_labels, dim) * dim ** -0.5)
        self.bias = nn.Parameter(torch.zeros(n_labels))

    def forward(self, tokens: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:  # (B, T, D), (B, T) -> (B, L)
        scores = self.score(tokens).masked_fill(~valid.unsqueeze(-1), float("-inf"))
        pooled = torch.einsum("btl,btd->bld", scores.softmax(dim=1), tokens)
        return (self.drop(self.norm(pooled)) * self.weight).sum(-1) + self.bias


class KneeNet(nn.Module):
    def __init__(self, n_slots: int, n_slices: int, backbone: str = "efficientnet_b0", pretrained: bool = True,
                 n_labels: int = len(LABELS), dropout: float = 0.2, fusion: str = "attention"):
        super().__init__()
        self.fusion_kind = fusion  # "attention": pools + transformer; "mean": plain masked means; "label": per-label attention
        self.encoder = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        dim = self.encoder.num_features
        self.slice_position = nn.Parameter(torch.zeros(1, n_slices, dim))
        self.pools = nn.ModuleList(AttentionPool(dim) for _ in range(n_slots))
        self.slot_type = nn.Parameter(torch.zeros(1, n_slots, dim))
        self.fusion = nn.TransformerEncoderLayer(dim, nhead=8, dim_feedforward=dim * 2, dropout=dropout, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Dropout(dropout), nn.Linear(dim, n_labels))
        if fusion == "label":
            self.label_attention = LabelAttention(dim, n_labels, dropout)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, image: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        """image: (B, S, N, H, W) float in [0, 1]; present: (B, S) bool -> logits (B, n_labels)."""
        b, s, n, h, w = image.shape
        x = neighbour_channels(image).reshape(b * s * n, 3, h, w)
        x = self.encoder((x - self.mean) / self.std).reshape(b, s, n, -1)
        weights = present.unsqueeze(-1).float()
        if self.fusion_kind == "mean":  # simple baseline: average slices, then average present views
            fused = (x.mean(dim=2) * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)
            return self.head(fused)
        x = x + self.slice_position
        if self.fusion_kind == "label":  # every (view, slice) is a token; missing views are masked out
            tokens = (x + self.slot_type.unsqueeze(2)).reshape(b, s * n, -1)
            return self.label_attention(tokens, present.repeat_interleave(n, dim=1))
        slots = torch.stack([pool(x[:, i]) for i, pool in enumerate(self.pools)], dim=1) + self.slot_type
        slots = self.fusion(slots, src_key_padding_mask=~present)
        fused = (slots * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)
        return self.head(fused)
