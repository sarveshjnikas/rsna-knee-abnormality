"""Split the 58 gold-labelled studies into dev (for tuning) and test (scored once).

Run as a module to write data/splits/gold_split.csv:
    python -m rsna_knee.validation.gold_split
"""

import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from rsna_knee.config import LABELS, RAW, SPLITS

SEED = 0


def gold_studies(train: pd.DataFrame) -> pd.DataFrame:
    """The studies with all twelve labels filled in."""
    return train.loc[train[LABELS].notna().all(axis=1)]


def split_gold(labelled: pd.DataFrame, seed: int = SEED) -> pd.Series:
    """'dev' or 'test' per study: two halves, stratified jointly over the twelve labels."""
    splitter = MultilabelStratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
    _, test_positions = next(splitter.split(labelled, labelled[LABELS].to_numpy()))
    split = pd.Series("dev", index=labelled.index, name="split")
    split.iloc[test_positions] = "test"
    return split


def main() -> None:
    labelled = gold_studies(pd.read_csv(RAW / "train.csv"))
    split = split_gold(labelled)

    SPLITS.mkdir(parents=True, exist_ok=True)
    out = SPLITS / "gold_split.csv"
    pd.concat([labelled["StudyInstanceUID"], split], axis=1).to_csv(out, index=False)

    positives = labelled[LABELS].groupby(split).sum().astype(int).T
    print(f"wrote {out}: {split.value_counts().to_dict()}")
    print(positives.to_string())


if __name__ == "__main__":
    main()
