import numpy as np

from rsna_knee.infer import derivable, downsize


def test_only_same_crop_and_slices_are_derivable():
    assert derivable("v2", "v3") and not derivable("v3", "v2") and not derivable("v1", "v3")


def test_downsize_keeps_views_slices_and_mask(tmp_path):
    image = np.random.randint(0, 256, (4, 3, 40, 40), dtype=np.uint8)
    present = np.array([True, False, True, True])
    np.savez(tmp_path / "s.npz", image=image, present=present)
    downsize(tmp_path, tmp_path / "small", 32)
    with np.load(tmp_path / "small" / "s.npz") as out:
        assert out["image"].shape == (4, 3, 32, 32) and out["image"].dtype == np.uint8
        assert np.array_equal(out["present"], present)
