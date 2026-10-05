"""Self-training: blend report labels with out-of-fold image predictions into new soft labels.

Each study's prediction comes from the one fold model that never trained on it, so the
teacher cannot echo labels it memorised. Gold studies are never relabelled (they are not in
any fold). On gold dev, a 50/50 mix of LLM labels and image predictions scored 0.933 macro
AUC against 0.885 for the LLM labels alone and 0.916 for predictions alone.

    python -m rsna_knee.model.distill runs/b0_v3 --alpha 0.5 --out data/labels/labels_distill_r1.csv
    python -m rsna_knee.model.distill runs/b0_distill1 runs/b0v2_distill1 --alpha 0.3 --out data/labels/labels_distill_r2.csv

With several runs, the teacher is the mean of their out-of-fold predictions (every run uses the
same folds, so each is still out-of-fold for that study).
"""

import argparse
from pathlib import Path

import pandas as pd

from rsna_knee.config import LABEL_DIR, LABELS


def distilled_labels(report_labels: pd.DataFrame, oof: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """alpha * report label + (1 - alpha) * OOF prediction, for every study that has an OOF prediction."""
    report = report_labels.set_index("StudyInstanceUID")[LABELS]
    teacher = oof.set_index("StudyInstanceUID")[LABELS]
    if teacher.index.duplicated().any():
        raise ValueError("a study has more than one out-of-fold prediction")
    common = teacher.index.intersection(report.index)
    blended = alpha * report.loc[common] + (1 - alpha) * teacher.loc[common]
    return blended.reset_index()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("runs", type=Path, nargs="+")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--labels", type=Path, default=LABEL_DIR / "labels_v2_gpt-6.1-sol.csv")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    per_run = []
    for run in args.runs:
        run_oof = pd.concat([pd.read_csv(f) for f in sorted(run.glob("fold*/oof.csv"))], ignore_index=True)
        if run_oof["StudyInstanceUID"].duplicated().any():
            raise ValueError(f"{run}: a study has more than one out-of-fold prediction")
        per_run.append(run_oof.set_index("StudyInstanceUID")[LABELS])
    common = per_run[0].index
    for frame in per_run[1:]:
        common = common.intersection(frame.index)
    if any(len(frame) != len(common) for frame in per_run):
        raise ValueError("teacher runs cover different studies; finish every fold first")
    oof = (sum(frame.loc[common] for frame in per_run) / len(per_run)).reset_index()
    blended = distilled_labels(pd.read_csv(args.labels), oof, args.alpha)
    blended.to_csv(args.out, index=False)
    shift = (blended[LABELS] - pd.read_csv(args.labels).set_index("StudyInstanceUID").loc[blended["StudyInstanceUID"], LABELS].to_numpy()).abs()
    print(f"wrote {args.out}: {len(blended)} studies from {len(oof)} OOF predictions; mean |change| per label:")
    print(shift.mean().round(3).to_string())


if __name__ == "__main__":
    main()
