"""Window data for the window model: every real slice of every view becomes one 3-slice window.

A window is (slice i-1, i, i+1) of one view, so the 2D backbone sees a little depth. Views are not
mixed inside a window; at the edges of a view the edge slice is repeated. Each window carries its
view index and its relative position in the view (0 = first slice, 1 = last), so the model knows
where it came from. Training samples k windows at random per study (a different subset every epoch,
which is also augmentation); evaluation uses every window.

The anatomical mirror turns a knee into its mirror image (a left knee into a right knee): sagittal
views run through the knee from one side to the other, so the slice order reverses (inside a window
the neighbours swap and the position becomes 1 - position); coronal and axial views flip left-right.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from rsna_knee.config import LABELS


def study_windows(views: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """views: per view, its real slices (k_v, H, W) uint8 (k_v may be 0).
    -> windows (W, 3, H, W) uint8, view index (W,), relative position (W,)."""
    windows, view_ids, positions = [], [], []
    for v, stack in enumerate(views):
        k = len(stack)
        for i in range(k):
            windows.append(np.stack([stack[max(i - 1, 0)], stack[i], stack[min(i + 1, k - 1)]]))
            view_ids.append(v)
            positions.append(i / max(k - 1, 1))
    if not windows:
        raise ValueError("study has no slices")
    return np.stack(windows), np.array(view_ids), np.array(positions, dtype=np.float32)


def mirror_windows(windows: torch.Tensor, view_ids: torch.Tensor, positions: torch.Tensor,
                   sagittal: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """Anatomical mirror of (..., W, 3, H, W) windows: sagittal -> neighbours swap and position flips;
    coronal / axial -> flip the width."""
    is_sag = torch.isin(view_ids, torch.tensor(sagittal, device=view_ids.device))
    flipped = torch.where(is_sag[..., None, None, None], windows.flip(-3), windows.flip(-1))
    return flipped, torch.where(is_sag, 1 - positions, positions)


class WindowDataset(Dataset):
    """Items: windows uint8 (K, 3, H, W), view ids (K,), positions (K,), valid (K,), target (12,), uid.

    k > 0: k windows sampled at random (all real ones when the study has fewer, the rest padded).
    k = 0: every window, padded to max_windows.
    """

    def __init__(self, uids: list[str], processed_dir: Path, labels: pd.DataFrame | None = None, k: int = 0,
                 max_windows: int = 96, n_views: int = 5):
        self.uids = list(uids)
        self.processed_dir = Path(processed_dir)
        self.targets = labels.set_index("StudyInstanceUID")[LABELS] if labels is not None else None
        self.k, self.max_windows, self.n_views = k, max_windows, n_views

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(self, i: int):
        uid = self.uids[i]
        with np.load(self.processed_dir / f"{uid}.npz") as data:
            views = [data[f"s{v}"] for v in range(self.n_views)]
        windows, view_ids, positions = study_windows(views)
        size = self.k or self.max_windows
        pick = np.random.permutation(len(windows))[:size] if self.k else np.arange(min(len(windows), size))
        n, h, w = len(pick), windows.shape[2], windows.shape[3]
        out = np.zeros((size, 3, h, w), dtype=np.uint8)
        ids, pos, valid = np.zeros(size, dtype=np.int64), np.zeros(size, dtype=np.float32), np.zeros(size, dtype=bool)
        out[:n], ids[:n], pos[:n], valid[:n] = windows[pick], view_ids[pick], positions[pick], True
        if self.targets is not None:
            target = torch.tensor(self.targets.loc[uid].to_numpy(dtype=np.float32))
        else:
            target = torch.zeros(len(LABELS))
        return torch.from_numpy(out), torch.from_numpy(ids), torch.from_numpy(pos), torch.from_numpy(valid), target, uid
