import numpy as np
import pandas as pd

from rsna_knee.preprocess.study import SLOTS, choose_series, slot_of
from rsna_knee.preprocess.volume import crop_to_body, normalise, pick_indices, to_square


def test_pick_indices_cover_the_stack_evenly():
    idx = pick_indices(30, 24)
    assert idx[0] == 0 and idx[-1] == 29 and len(idx) == 24 and (np.diff(idx) >= 0).all()


def test_pick_indices_repeat_short_stacks():
    assert len(pick_indices(5, 24)) == 24


def test_to_square_keeps_aspect_by_padding():
    image = np.ones((100, 200), dtype=np.float32)
    out = to_square(image, 64)
    assert out.shape == (64, 64)
    assert out[0].sum() == 0 and out[32].sum() > 0  # padded top rows, content in the middle


def test_crop_removes_empty_field_of_view():
    volume = np.zeros((4, 100, 200), dtype=np.float32)
    volume[:, 40:60, 80:120] = 1.0
    assert crop_to_body(volume, margin=0).shape == (4, 20, 40)


def test_normalise_maps_to_uint8_range():
    out = normalise(np.random.default_rng(0).normal(500, 100, (4, 32, 32)))
    assert out.dtype == np.uint8 and out.min() == 0 and out.max() == 255


def test_slot_of_uses_fluid_sensitive_flag():
    assert slot_of("Sagittal", 1) == "sag_FS" and slot_of("Axial", 0) == "axi_noFS"


def test_choose_series_keeps_longest_duplicate(tmp_path):
    for uid, n in [("s_short", 10), ("s_long", 30), ("c", 20)]:
        (tmp_path / uid).mkdir()
        for i in range(n):
            (tmp_path / uid / f"{i}.dcm").touch()
    series = pd.DataFrame({
        "SeriesInstanceUID": ["s_short", "s_long", "c"],
        "Anatomical_Plane": ["Sagittal", "Sagittal", "Coronal"],
        "Fluid_Sensitive": [1, 1, 1],
    })
    chosen = choose_series(series, tmp_path)
    assert chosen == {"sag_FS": "s_long", "cor_FS": "c"} and set(chosen) <= set(SLOTS)
