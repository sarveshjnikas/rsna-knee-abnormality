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


def neighbour_channels(volume: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
    """(B, S, N, H, W) -> (B, S, N, 3, H, W): slices i-1, i, i+1 (edges repeated).

    valid (B, S, N) marks real slices; a blank neighbour is replaced by the slice itself.
    """
    prev = torch.cat([volume[:, :, :1], volume[:, :, :-1]], dim=2)
    nxt = torch.cat([volume[:, :, 1:], volume[:, :, -1:]], dim=2)
    if valid is not None:
        edge = torch.zeros_like(valid[:, :, :1])
        prev_ok = torch.cat([edge, valid[:, :, :-1]], dim=2)[..., None, None]
        next_ok = torch.cat([valid[:, :, 1:], edge], dim=2)[..., None, None]
        prev, nxt = torch.where(prev_ok, prev, volume), torch.where(next_ok, nxt, volume)
    return torch.stack([prev, volume, nxt], dim=3)


class AttentionPool(nn.Module):
    """Softmax-weighted average over the slice axis."""

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, dim // 4), nn.Tanh(), nn.Linear(dim // 4, 1))

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:  # (B, N, D), (B, N) -> (B, D)
        weights = self.score(x).masked_fill(~valid.unsqueeze(-1), float("-inf")).softmax(dim=1)
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
        """image: (B, S, N, H, W) float in [0, 1]; present: (B, S) views, or (B, S, N) real slices
        per view (blank slices ignored) -> logits (B, n_labels)."""
        b, s, n, h, w = image.shape
        slices = present if present.dim() == 3 else None
        present = present.any(dim=-1) if slices is not None else present
        valid = slices if slices is not None else present.unsqueeze(-1).expand(b, s, n)
        x = neighbour_channels(image, slices).reshape(b * s * n, 3, h, w)
        x = self.encoder((x - self.mean) / self.std).reshape(b, s, n, -1)
        weights = present.unsqueeze(-1).float()
        pool_valid = valid | ~present.unsqueeze(-1)  # a missing view pools its blanks; it is masked out later
        if self.fusion_kind == "mean":  # simple baseline: average slices, then average present views
            per_view = (x * pool_valid[..., None]).sum(dim=2) / pool_valid.sum(dim=2, keepdim=True)
            fused = (per_view * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)
            return self.head(fused)
        x = x + self.slice_position
        if self.fusion_kind == "label":  # every real (view, slice) is a token; blanks and missing views masked
            tokens = (x + self.slot_type.unsqueeze(2)).reshape(b, s * n, -1)
            return self.label_attention(tokens, valid.reshape(b, s * n))
        slots = torch.stack([pool(x[:, i], pool_valid[:, i]) for i, pool in enumerate(self.pools)], dim=1) + self.slot_type
        slots = self.fusion(slots, src_key_padding_mask=~present)
        fused = (slots * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)
        return self.head(fused)


class WindowNet(nn.Module):
    """Window model: a set of 3-slice windows from all views -> 12 logits.

    1. A shared pretrained 2D backbone embeds every window.
    2. Each embedding gets its view's type embedding and a learned function of its position in the view.
    3. One attention per finding pools the windows (padding masked), then one linear output per finding.
    """

    def __init__(self, backbone: str = "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", pretrained: bool = True,
                 n_views: int = 5, n_labels: int = len(LABELS), dropout: float = 0.2, chunk: int = 32):
        super().__init__()
        self.encoder = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        dim = self.encoder.num_features
        self.view_type = nn.Embedding(n_views, dim)
        self.position = nn.Sequential(nn.Linear(1, dim // 4), nn.Tanh(), nn.Linear(dim // 4, dim))
        self.label_attention = LabelAttention(dim, n_labels, dropout)
        self.chunk = chunk  # windows per backbone call: bounds memory at 384 px
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, windows: torch.Tensor, view_ids: torch.Tensor, positions: torch.Tensor,
                valid: torch.Tensor) -> torch.Tensor:
        """windows (B, K, 3, H, W) float in [0, 1]; view_ids, positions, valid (B, K) -> logits (B, n_labels)."""
        b, k = windows.shape[:2]
        flat = (windows.flatten(0, 1) - self.mean) / self.std
        x = torch.cat([self.encoder(part) for part in flat.split(self.chunk)]).view(b, k, -1)
        x = x + self.view_type(view_ids) + self.position(positions.unsqueeze(-1))
        return self.label_attention(x, valid)
