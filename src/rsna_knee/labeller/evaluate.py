"""Score labels against the gold studies: per-label and macro AUC, plus status crosstabs."""

import pandas as pd
from sklearn.metrics import roc_auc_score

from rsna_knee.config import LABELS


def auc_table(gold: pd.DataFrame, predicted: pd.DataFrame) -> pd.Series:
    """AUC per label (and 'macro') on the studies present in both frames."""
    merged = gold[["StudyInstanceUID", *LABELS]].merge(predicted, on="StudyInstanceUID", suffixes=("_gold", "_pred"))
    aucs = {}
    for label in LABELS:
        y, p = merged[f"{label}_gold"], merged[f"{label}_pred"]
        aucs[label] = roc_auc_score(y, p) if y.nunique() == 2 else float("nan")
    result = pd.Series(aucs, name="auc")
    result["macro"] = result.mean()
    return result


def status_vs_gold(gold: pd.DataFrame, statuses: pd.DataFrame) -> pd.DataFrame:
    """For each label and status: how many studies, and the share that are gold-positive."""
    merged = gold[["StudyInstanceUID", *LABELS]].merge(statuses, on="StudyInstanceUID", suffixes=("_gold", "_status"))
    rows = []
    for label in LABELS:
        grouped = merged.groupby(f"{label}_status")[f"{label}_gold"]
        for status, values in grouped:
            rows.append({"label": label, "status": status, "n": len(values), "gold_positive_rate": values.mean()})
    return pd.DataFrame(rows)
