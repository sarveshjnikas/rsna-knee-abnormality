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


def test_mean_fusion_ignores_missing_views():
    torch.manual_seed(0)
    model = KneeNet(n_slots=S, n_slices=N, backbone="resnet18", pretrained=False, fusion="mean").eval()
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
