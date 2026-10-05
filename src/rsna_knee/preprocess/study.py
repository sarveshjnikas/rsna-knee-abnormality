"""One study -> fixed slots of volumes plus a mask of which slots exist.

Slots are plane x sequence type, chosen from the series flags (given at test time).
A series is "FS" when Fluid_Sensitive = 1 (fluid bright, fat suppressed); everything
else is "noFS". Unseen flag combinations therefore still map to a slot.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from rsna_knee.config import PREPROCESS, SLOTS
from rsna_knee.preprocess.volume import load_volume


def slot_of(plane: str, fluid_sensitive: int) -> str:
    return f"{plane[:3].lower()}_{'FS' if fluid_sensitive == 1 else 'noFS'}"


def choose_series(series: pd.DataFrame, study_dir: Path) -> dict[str, str]:
    """{slot: SeriesInstanceUID}. Duplicates: keep the series with the most slices (best coverage)."""
    series = series.assign(
        slot=[slot_of(p, f) for p, f in zip(series["Anatomical_Plane"], series["Fluid_Sensitive"])],
        n_files=[len(list((study_dir / uid).glob("*.dcm"))) for uid in series["SeriesInstanceUID"]],
    )
    series = series[series["slot"].isin(SLOTS) & (series["n_files"] > 0)]
    best = series.sort_values(["n_files", "SeriesInstanceUID"], ascending=[False, True]).drop_duplicates("slot")
    return dict(zip(best["slot"], best["SeriesInstanceUID"]))


def build_study(series: pd.DataFrame, study_dir: Path, version: str = "v1") -> tuple[np.ndarray, np.ndarray]:
    """image: (len(SLOTS), n_slices, image_size, image_size) uint8; present: (len(SLOTS),) bool."""
    cfg = PREPROCESS[version]
    image = np.zeros((len(SLOTS), cfg["n_slices"], cfg["image_size"], cfg["image_size"]), dtype=np.uint8)
    present = np.zeros(len(SLOTS), dtype=bool)
    for slot, series_uid in choose_series(series, study_dir).items():
        i = SLOTS.index(slot)
        image[i] = load_volume(study_dir / series_uid, cfg["n_slices"], cfg["image_size"], cfg["crop_mm"], cfg["slice_range"])
        present[i] = True
    return image, present
