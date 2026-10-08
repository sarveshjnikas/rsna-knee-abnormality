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


def choose_series(series: pd.DataFrame, study_dir: Path, slots: tuple[str, ...] = SLOTS) -> dict[str, str]:
    """{slot: SeriesInstanceUID}. Duplicates: keep the series with the most slices (best coverage)."""
    series = series.assign(
        slot=[slot_of(p, f) for p, f in zip(series["Anatomical_Plane"], series["Fluid_Sensitive"])],
        n_files=[len(list((study_dir / uid).glob("*.dcm"))) for uid in series["SeriesInstanceUID"]],
    )
    series = series[series["slot"].isin(slots) & (series["n_files"] > 0)]
    best = series.sort_values(["n_files", "SeriesInstanceUID"], ascending=[False, True]).drop_duplicates("slot")
    return dict(zip(best["slot"], best["SeriesInstanceUID"]))


def build_study(series: pd.DataFrame, study_dir: Path, version: str = "v1") -> dict[str, np.ndarray]:
    """image: (slots, n_slices, image_size, image_size) uint8; present: (slots,) bool; for versions that
    space slices by mm also valid: (slots, n_slices) bool, real slices first and blanks after."""
    cfg = PREPROCESS[version]
    slots = cfg.get("slots", SLOTS)
    if "slot_slices" in cfg:  # one array per view, its real slices only: s0 .. s4, each (k, size, size)
        out = {"present": np.zeros(len(slots), dtype=bool)}
        for i in range(len(slots)):
            out[f"s{i}"] = np.zeros((0, cfg["image_size"], cfg["image_size"]), dtype=np.uint8)
        for slot, series_uid in choose_series(series, study_dir, slots).items():
            i = slots.index(slot)
            out[f"s{i}"] = load_volume(study_dir / series_uid, cfg["slot_slices"][i], cfg["image_size"], cfg["crop_mm"],
                                       cfg["slice_range"], unique=True)
            out["present"][i] = True
        return out
    image = np.zeros((len(slots), cfg["n_slices"], cfg["image_size"], cfg["image_size"]), dtype=np.uint8)
    present = np.zeros(len(slots), dtype=bool)
    valid = np.zeros((len(slots), cfg["n_slices"]), dtype=bool)
    for slot, series_uid in choose_series(series, study_dir, slots).items():
        i = slots.index(slot)
        volume = load_volume(study_dir / series_uid, cfg["n_slices"], cfg["image_size"], cfg["crop_mm"],
                             cfg["slice_range"], cfg.get("spacing_mm"))
        image[i, : len(volume)], valid[i, : len(volume)], present[i] = volume, True, True
    out = {"image": image, "present": present}
    return out | ({"valid": valid} if cfg.get("spacing_mm") else {})
