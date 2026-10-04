"""Score a finished run: CV from out-of-fold predictions, and the fold ensemble on the gold studies.

    python -m rsna_knee.model.evaluate runs/b0_v2
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from rsna_knee.config import LABEL_DIR, LABELS, RAW, SPLITS
from rsna_knee.labeller.evaluate import auc_table


def evaluate_run(run: Path, labels_path: Path) -> dict:
    oof = pd.concat([pd.read_csv(f) for f in sorted(run.glob("fold*/oof.csv"))], ignore_index=True)
    gold_preds = pd.concat([pd.read_csv(f) for f in sorted(run.glob("fold*/gold.csv"))])
    ensemble = gold_preds.groupby("StudyInstanceUID", as_index=False)[LABELS].mean()

    pseudo = pd.read_csv(labels_path)
    pseudo[LABELS] = (pseudo[LABELS] >= 0.5).astype(int)
    gold = pd.read_csv(RAW / "train.csv").merge(pd.read_csv(SPLITS / "gold_split.csv"), on="StudyInstanceUID")

    result = {
        "folds": len(list(run.glob("fold*/oof.csv"))),
        "cv_vs_pseudo": auc_table(pseudo, oof).round(4).to_dict(),
        "gold_all58": auc_table(gold, ensemble).round(4).to_dict(),
        "gold_dev29": auc_table(gold[gold["split"] == "dev"], ensemble)["macro"].round(4),
        "gold_test29": auc_table(gold[gold["split"] == "test"], ensemble)["macro"].round(4),
    }
    (run / "scores.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--labels", type=Path, default=LABEL_DIR / "labels_v2_gpt-6.1-sol.csv")
    args = p.parse_args()
    result = evaluate_run(args.run, args.labels)
    print(f"folds {result['folds']} | CV macro {result['cv_vs_pseudo']['macro']} | gold58 macro "
          f"{result['gold_all58']['macro']} (dev {result['gold_dev29']}, test {result['gold_test29']})")
    print(pd.DataFrame({"cv": result["cv_vs_pseudo"], "gold58": result["gold_all58"]}).to_string())


if __name__ == "__main__":
    main()
