import pytest
import torch

from rsna_knee.model.data import augment
from rsna_knee.model.network import KneeNet, neighbour_channels

B, S, N, H = 2, 4, 6, 64


def _model() -> KneeNet:
    torch.manual_seed(0)
    return KneeNet(n_slots=S, n_slices=N, backbone="resnet18", pretrained=False).eval()


def test_forward_shape():
    logits = _model()(torch.rand(B, S, N, H, H), torch.ones(B, S, dtype=torch.bool))
    assert logits.shape == (B, 12)


def test_missing_view_content_is_ignored():
    model, image = _model(), torch.rand(1, S, N, H, H)
    present = torch.tensor([[False, True, True, True]])
    other = image.clone()
    other[:, 0] = torch.rand(N, H, H)
    with torch.no_grad():
        assert torch.allclose(model(image, present), model(other, present), atol=1e-5)


def test_neighbour_channels_centre_is_the_slice():
    volume = torch.rand(1, 1, N, 8, 8)
    stacked = neighbour_channels(volume)
    assert stacked.shape == (1, 1, N, 3, 8, 8) and torch.equal(stacked[:, :, :, 1], volume)
    assert torch.equal(stacked[0, 0, 3, 0], volume[0, 0, 2]) and torch.equal(stacked[0, 0, 3, 2], volume[0, 0, 4])


def test_augment_keeps_shape_range_and_one_view():
    image, present = torch.rand(B, S, N, H, H), torch.ones(B, S, dtype=torch.bool)
    out, kept = augment(image, present, view_dropout=1.0)
    assert out.shape == image.shape and out.min() >= 0 and out.max() <= 1
    assert (kept.sum(dim=1) >= 1).all()


@pytest.mark.parametrize("fusion", ["mean", "label"])
def test_simple_fusions_ignore_missing_views(fusion):
    torch.manual_seed(0)
    model = KneeNet(n_slots=S, n_slices=N, backbone="resnet18", pretrained=False, fusion=fusion).eval()
    image, present = torch.rand(1, S, N, H, H), torch.tensor([[False, True, True, True]])
    other = image.clone()
    other[:, 0] = torch.rand(N, H, H)
    with torch.no_grad():
        assert model(image, present).shape == (1, 12)
        assert torch.allclose(model(image, present), model(other, present), atol=1e-5)


def test_distilled_labels_blend_and_reject_duplicates():
    import pandas as pd
    import pytest

    from rsna_knee.config import LABELS
    from rsna_knee.model.distill import distilled_labels

    report = pd.DataFrame({"StudyInstanceUID": ["a", "b"], **{l: [1.0, 0.0] for l in LABELS}})
    oof = pd.DataFrame({"StudyInstanceUID": ["a", "b"], **{l: [0.0, 1.0] for l in LABELS}})
    out = distilled_labels(report, oof, alpha=0.5)
    assert (out[LABELS] == 0.5).all().all()
    with pytest.raises(ValueError):
        distilled_labels(report, pd.concat([oof, oof]), alpha=0.5)


def test_unique_slices_drops_repeats_and_keeps_order():
    import numpy as np

    from rsna_knee.model.data import unique_slices

    real = np.random.randint(1, 256, (3, 8, 8), dtype=np.uint8)
    image = np.zeros((2, 5, 8, 8), dtype=np.uint8)
    image[0] = real[[0, 0, 1, 2, 2]]  # a short series stretched to 5 positions
    out, valid = unique_slices(image, np.array([True, False]))
    assert valid.tolist() == [[True, True, True, False, False], [False] * 5]
    assert np.array_equal(out[0, :3], real) and not out[0, 3:].any()


@pytest.mark.parametrize("fusion", ["attention", "mean", "label"])
def test_blank_slices_are_ignored(fusion):
    torch.manual_seed(0)
    model = KneeNet(n_slots=S, n_slices=N, backbone="resnet18", pretrained=False, fusion=fusion).eval()
    image = torch.rand(1, S, N, H, H)
    present = torch.ones(1, S, N, dtype=torch.bool)
    present[0, 0, 4:] = False  # view 0 has 4 real slices
    present[0, 3] = False  # view 3 is missing
    other = image.clone()
    other[0, 0, 4:] = torch.rand(N - 4, H, H)
    other[0, 3] = torch.rand(N, H, H)
    with torch.no_grad():
        assert torch.allclose(model(image, present), model(other, present), atol=1e-5)


def test_full_slice_mask_matches_view_mask():
    model, image = _model(), torch.rand(1, S, N, H, H)
    views = torch.tensor([[True, True, False, True]])
    with torch.no_grad():
        assert torch.allclose(model(image, views), model(image, views[..., None].expand(1, S, N)), atol=1e-5)


def test_view_dropout_with_slice_mask_keeps_a_view():
    image, present = torch.rand(B, S, N, H, H), torch.ones(B, S, N, dtype=torch.bool)
    present[:, :, -2:] = False
    _, kept = augment(image, present, view_dropout=1.0)
    assert kept.shape == present.shape and (kept.any(dim=-1).sum(dim=1) >= 1).all()
    assert not kept[:, :, -2:].any()  # blanks stay blank


def test_masked_bce_ignores_unknown_targets():
    from rsna_knee.model.train import masked_bce

    logits = torch.tensor([[0.3, -1.0], [2.0, 0.5]])
    full = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    assert torch.allclose(masked_bce(logits, torch.tensor([[1.0, 0.0], [0.0, 1.0]])), full)
    partial = masked_bce(logits, torch.tensor([[1.0, float("nan")], [float("nan"), 1.0]]))
    expected = torch.nn.functional.binary_cross_entropy_with_logits(logits[[0, 1], [0, 1]], torch.tensor([1.0, 1.0]))
    assert torch.allclose(partial, expected)


def test_per_finding_report_weight():
    import pandas as pd

    from rsna_knee.config import LABELS
    from rsna_knee.model.distill import distilled_labels

    report = pd.DataFrame({"StudyInstanceUID": ["a"], **{l: [1.0] for l in LABELS}})
    oof = pd.DataFrame({"StudyInstanceUID": ["a"], **{l: [0.0] for l in LABELS}})
    out = distilled_labels(report, oof, alpha=0.5, alpha_by_label={"Effusion": 0.1}).set_index("StudyInstanceUID")
    assert out.at["a", "Effusion"] == pytest.approx(0.1) and out.at["a", "ACL"] == pytest.approx(0.5)


def test_mirror_right_knee():
    import numpy as np

    from rsna_knee.model.data import mirror_right_knee

    image = np.arange(3 * 4 * 2 * 2, dtype=np.uint8).reshape(3, 4, 2, 2)
    present = np.array([[True, True, True, False], [True] * 4, [True] * 4])
    out = mirror_right_knee(image, present, ["sag", "cor", "axi"])
    assert np.array_equal(out[0, :3], image[0, [2, 1, 0]]) and np.array_equal(out[0, 3], image[0, 3])
    assert np.array_equal(out[1], image[1, :, :, ::-1]) and np.array_equal(out[2], image[2, :, :, ::-1])
