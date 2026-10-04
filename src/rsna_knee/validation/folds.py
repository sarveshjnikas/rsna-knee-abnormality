"""Cross-validation folds that keep site groups whole and balance all twelve labels.

No library does grouped + multilabel stratification, so this is a greedy
assignment: groups are placed largest first, each into the fold where it adds
the least imbalance. Imbalance compares each fold's label counts and size with
an equal 1/k share. Labels may be soft (probabilities): their sums are expected
counts, so the same code handles hard and soft labels.
"""

import numpy as np
import pandas as pd


def make_folds(
    df: pd.DataFrame,
    labels: list[str],
    group_col: str,
    k: int = 5,
    seed: int = 0,
) -> pd.Series:
    """Fold index 0..k-1 per row. Groups stay whole; label shares and fold sizes stay balanced."""
    rng = np.random.default_rng(seed)
    y = df[labels].to_numpy(dtype=float)
    group_ids, group_of_row = np.unique(df[group_col].to_numpy(), return_inverse=True)

    group_counts = np.zeros((len(group_ids), len(labels)))
    np.add.at(group_counts, group_of_row, y)
    group_size = np.bincount(group_of_row).astype(float)

    # Largest groups first (they are the hardest to place); ties in random order.
    order = rng.permutation(len(group_ids))
    order = order[np.argsort(-group_size[order], kind="stable")]

    target_counts = y.sum(axis=0) / k
    target_size = len(df) / k
    # Relative error per label, so a rare label weighs as much as a common one.
    label_scale = 1.0 / np.maximum(target_counts, 1e-9)
    size_weight = max(len(labels), 1)

    fold_counts = np.zeros((k, len(labels)))
    fold_size = np.zeros(k)
    fold_of_group = np.empty(len(group_ids), dtype=int)

    for g in order:
        label_cost = (
            (((fold_counts + group_counts[g]) - target_counts) * label_scale) ** 2
            - ((fold_counts - target_counts) * label_scale) ** 2
        ).sum(axis=1)
        size_cost = (
            ((fold_size + group_size[g] - target_size) / target_size) ** 2
            - ((fold_size - target_size) / target_size) ** 2
        )
        cost = label_cost + size_weight * size_cost
        best = np.flatnonzero(np.isclose(cost, cost.min()))
        fold = rng.choice(best)

        fold_of_group[g] = fold
        fold_counts[fold] += group_counts[g]
        fold_size[fold] += group_size[g]

    return pd.Series(fold_of_group[group_of_row], index=df.index, name="fold")


def main() -> None:
    """Freeze CV folds over the pseudo-labelled studies (gold studies excluded) -> data/splits/folds.csv."""
    import argparse

    from rsna_knee.config import INTERIM, LABEL_DIR, LABELS, SPLITS

    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default=str(LABEL_DIR / "labels_v2_gpt-6.1-sol.csv"))
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    labels = pd.read_csv(args.labels)
    sites = pd.read_csv(INTERIM / "site_proxy.csv")
    gold = set(pd.read_csv(SPLITS / "gold_split.csv")["StudyInstanceUID"])
    studies = labels.merge(sites[["StudyInstanceUID", "site_proxy"]], on="StudyInstanceUID")
    studies = studies[~studies["StudyInstanceUID"].isin(gold)].reset_index(drop=True)

    studies["fold"] = make_folds(studies, LABELS, "site_proxy", k=args.k)
    studies[["StudyInstanceUID", "site_proxy", "fold"]].to_csv(SPLITS / "folds.csv", index=False)
    print(f"wrote {SPLITS / 'folds.csv'}: {studies['fold'].value_counts().sort_index().tolist()}")
    print(studies.groupby("fold")[LABELS].mean().round(3).T.to_string())


if __name__ == "__main__":
    main()
