import numpy as np
import pandas as pd
import pytest

from rsna_knee.validation.folds import make_folds

K = 5
LABELS = [f"label_{i}" for i in range(12)]


@pytest.fixture
def studies() -> pd.DataFrame:
    """Synthetic data shaped like ours: some large site groups plus many single studies."""
    rng = np.random.default_rng(42)
    group_sizes = rng.integers(20, 150, size=25)
    groups = [f"site_{g}" for g, size in enumerate(group_sizes) for _ in range(size)]
    groups += [f"study_{i}" for i in range(800)]
    prevalence = np.linspace(0.05, 0.6, len(LABELS))
    df = pd.DataFrame(rng.random((len(groups), len(LABELS))) < prevalence, columns=LABELS).astype(float)
    df["group"] = groups
    return df


def test_every_row_gets_one_valid_fold(studies):
    folds = make_folds(studies, LABELS, "group", k=K)
    assert folds.index.equals(studies.index)
    assert set(folds.unique()) == set(range(K))


def test_groups_never_span_folds(studies):
    folds = make_folds(studies, LABELS, "group", k=K)
    assert (folds.groupby(studies["group"]).nunique() == 1).all()


def test_fold_sizes_are_balanced(studies):
    sizes = make_folds(studies, LABELS, "group", k=K).value_counts()
    assert (abs(sizes - len(studies) / K) <= 0.10 * len(studies) / K).all()


def test_label_shares_are_balanced(studies):
    folds = make_folds(studies, LABELS, "group", k=K)
    shares = studies[LABELS].groupby(folds).mean()
    assert (abs(shares - studies[LABELS].mean()) < 0.03).all().all()


def test_same_seed_same_folds(studies):
    assert make_folds(studies, LABELS, "group", seed=7).equals(make_folds(studies, LABELS, "group", seed=7))


def test_soft_labels_are_accepted(studies):
    soft = studies.copy()
    soft[LABELS] = soft[LABELS] * 0.8 + 0.1
    assert make_folds(soft, LABELS, "group", k=K).between(0, K - 1).all()
