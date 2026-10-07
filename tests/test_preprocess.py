import numpy as np
import pandas as pd

from rsna_knee.config import SLOTS
from rsna_knee.preprocess.study import choose_series, slot_of
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


def test_pick_by_mm_never_repeats_and_respects_spacing():
    import numpy as np

    from rsna_knee.preprocess.volume import pick_by_mm

    thick = np.arange(30) * 4.0  # 4 mm slices: every slice in the central range, none repeated
    idx = pick_by_mm(thick, 32, (0.06, 0.94), 3.0)
    assert len(idx) == len(set(idx)) and idx.tolist() == list(range(2, 28))
    thin = np.arange(200) * 0.5  # 0.5 mm slices over 100 mm: about 3 mm apart, at most 32
    idx = pick_by_mm(thin, 32, (0.06, 0.94), 3.0)
    gaps = np.diff(thin[idx])
    assert len(idx) <= 32 and gaps.min() >= 2.9 and len(set(idx)) == len(idx)
    long = np.arange(400) * 0.5  # 200 mm: 32 slices cannot cover it at 3 mm, so the gap widens
    assert len(pick_by_mm(long, 32, (0.06, 0.94), 3.0)) == 32


def test_dataset_reads_slice_mask_and_view_count(tmp_path):
    import numpy as np

    from rsna_knee.model.data import StudyDataset

    image = np.random.randint(1, 256, (5, 6, 8, 8), dtype=np.uint8)
    valid = np.zeros((5, 6), dtype=bool)
    valid[:, :4] = True
    present = np.array([True, True, False, True, True])
    np.savez(tmp_path / "s.npz", image=image, present=present, valid=valid)
    img, mask, _, _ = StudyDataset(["s"], tmp_path)[0]
    assert img.shape == (5, 6, 8, 8) and mask.shape == (5, 6) and not mask[2].any() and mask[0].sum() == 4
    img, mask, _, _ = StudyDataset(["s"], tmp_path, n_slots=4)[0]
    assert img.shape == (4, 6, 8, 8) and mask.shape == (4, 6)
