"""Load preprocessed studies and augment them on the GPU."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from rsna_knee.config import LABELS


class StudyDataset(Dataset):
    """Items: (image uint8 (S, N, H, W), present bool (S,), target float (12,), uid)."""

    def __init__(self, uids: list[str], processed_dir: Path, labels: pd.DataFrame | None = None):
        self.uids = list(uids)
        self.processed_dir = Path(processed_dir)
        self.targets = labels.set_index("StudyInstanceUID")[LABELS] if labels is not None else None

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(self, i: int):
        uid = self.uids[i]
        with np.load(self.processed_dir / f"{uid}.npz") as data:
            image, present = torch.from_numpy(data["image"]), torch.from_numpy(data["present"])
        if self.targets is not None:
            target = torch.tensor(self.targets.loc[uid].to_numpy(dtype=np.float32))
        else:
            target = torch.zeros(len(LABELS))
        return image, present, target, uid


def augment(image: torch.Tensor, present: torch.Tensor, view_dropout: float = 0.15,
            min_scale: float = 0.85) -> tuple[torch.Tensor, torch.Tensor]:
    """image (B, S, N, H, W) float in [0, 1] on GPU; returns augmented image and present mask.

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

    drop = (torch.rand(b, s, device=image.device) < view_dropout) & present
    keep_one = (present & ~drop).any(dim=1, keepdim=True)
    present = torch.where(keep_one, present & ~drop, present)
    return out, present
