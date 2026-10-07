"""Load preprocessed studies and augment them on the GPU."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from rsna_knee.config import LABELS


def unique_slices(image: np.ndarray, present: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop the repeats that slice picking adds to short series; real slices first, blanks after.

    image (S, N, H, W), present (S,) -> image with each view's distinct slices moved to the front and
    zeros after them, and valid (S, N): which positions hold a real slice. A repeat is a bit-for-bit
    copy of the slice before it, because both came from the same DICOM slice.
    """
    out, valid = np.zeros_like(image), np.zeros(image.shape[:2], dtype=bool)
    for s in np.flatnonzero(present):
        new = np.ones(image.shape[1], dtype=bool)
        new[1:] = (image[s, 1:] != image[s, :-1]).reshape(image.shape[1] - 1, -1).any(axis=1)
        out[s, : new.sum()], valid[s, : new.sum()] = image[s, new], True
    return out, valid


class StudyDataset(Dataset):
    """Items: (image uint8 (S, N, H, W), present bool (S,) or (S, N), target float (12,), uid).

    With unique=True, repeated slices become blanks and present marks real slices per view (S, N).
    """

    def __init__(self, uids: list[str], processed_dir: Path, labels: pd.DataFrame | None = None, unique: bool = False,
                 n_slots: int = 0):
        self.uids = list(uids)
        self.processed_dir = Path(processed_dir)
        self.targets = labels.set_index("StudyInstanceUID")[LABELS] if labels is not None else None
        self.unique = unique
        self.n_slots = n_slots  # 0: every view in the data; 4 on 5-view data reproduces the 4-view setup

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(self, i: int):
        uid = self.uids[i]
        with np.load(self.processed_dir / f"{uid}.npz") as data:
            image, present = data["image"], data["present"]
            valid = data["valid"] if "valid" in data else None  # v4+: real slices per view, no repeats
        if valid is not None:
            present = valid & present[:, None]
        elif self.unique:
            image, present = unique_slices(image, present)
        if self.n_slots:
            image, present = image[: self.n_slots], present[: self.n_slots]
        image, present = torch.from_numpy(image), torch.from_numpy(present)
        if self.targets is not None:
            target = torch.tensor(self.targets.loc[uid].to_numpy(dtype=np.float32))
        else:
            target = torch.zeros(len(LABELS))
        return image, present, target, uid


def augment(image: torch.Tensor, present: torch.Tensor, view_dropout: float = 0.15,
            min_scale: float = 0.85) -> tuple[torch.Tensor, torch.Tensor]:
    """image (B, S, N, H, W) float in [0, 1] on GPU; present (B, S) or per slice (B, S, N).
    Returns the augmented image and present mask.

    - horizontal flip per view: a mirrored left knee looks like a right knee, and the
      labels are anatomical (medial stays medial), so flips preserve every label;
    - random zoom crop per study, and intensity gain / offset per view (scanner differences);
    - view dropout: hide random present views (at least one kept) so the model does not
      rely on any single view, which test studies may lack.
    """
    b, s, n, h, w = image.shape
    flip = torch.rand(b, s, device=image.device) < 0.5
    image = torch.where(flip[:, :, None, None, None], image.flip(-1), image)

    out = torch.empty_like(image)
    for k in range(b):
        scale = float(torch.empty(1).uniform_(min_scale, 1.0))
        ch, cw = int(h * scale), int(w * scale)
        top, left = int(torch.randint(0, h - ch + 1, (1,))), int(torch.randint(0, w - cw + 1, (1,)))
        crop = image[k, :, :, top : top + ch, left : left + cw].reshape(s * n, 1, ch, cw)
        out[k] = F.interpolate(crop, size=(h, w), mode="bilinear", align_corners=False).reshape(s, n, h, w)

    gain = torch.empty(b, s, 1, 1, 1, device=image.device).uniform_(0.85, 1.15)
    offset = torch.empty(b, s, 1, 1, 1, device=image.device).uniform_(-0.05, 0.05)
    out = (out * gain + offset).clamp(0, 1)

    views = present.any(dim=-1) if present.dim() == 3 else present
    drop = (torch.rand(b, s, device=image.device) < view_dropout) & views
    keep_one = (views & ~drop).any(dim=1, keepdim=True)
    drop = drop & keep_one
    present = present & ~(drop[..., None] if present.dim() == 3 else drop)
    return out, present
