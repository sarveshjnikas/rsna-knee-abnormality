"""Test-time pipeline: preprocess test studies with the training code, average fold models, write submission.csv.

    python -m rsna_knee.infer --comp-dir <competition dir> --models <dir with fold*/model.pt> \
        --work /kaggle/temp/test --out submission.csv
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from rsna_knee.config import LABELS, N_SLICES, SLOTS
from rsna_knee.model.data import StudyDataset
from rsna_knee.model.network import KneeNet
from rsna_knee.preprocess.run import preprocess_all


def fold_checkpoints(models_dir: Path) -> list[tuple[Path, str, str]]:
    """(model.pt, backbone, fusion) for every fold directory under models_dir."""
    found = []
    for checkpoint in sorted(models_dir.glob("**/fold*/model.pt")):
        config = json.loads((checkpoint.parent / "config.json").read_text())
        found.append((checkpoint, config.get("backbone", "efficientnet_b0"), config.get("fusion", "attention")))
    return found


@torch.no_grad()
def predict(processed: Path, uids: list[str], checkpoints: list[tuple[Path, str, str]], batch_size: int = 4,
            workers: int = 2, flip_tta: bool = True) -> pd.DataFrame:
    """Mean sigmoid probability over fold models (and a left-right flip of every view), one row per study."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader = DataLoader(StudyDataset(uids, processed), batch_size=batch_size, num_workers=workers)
    total = np.zeros((len(uids), len(LABELS)))
    for checkpoint, backbone, fusion in checkpoints:
        model = KneeNet(n_slots=len(SLOTS), n_slices=N_SLICES, backbone=backbone, pretrained=False, fusion=fusion)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        model.to(device).eval()
        rows = []
        for image, present, _, _ in loader:
            image = image.to(device).float() / 255
            present = present.to(device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
                prob = model(image, present).float().sigmoid()
                if flip_tta:  # trained with left-right flips, so the mirrored study is an equally valid view
                    prob = (prob + model(image.flip(-1), present).float().sigmoid()) / 2
            rows.append(prob.cpu().numpy())
        total += np.concatenate(rows)
    return pd.DataFrame(total / len(checkpoints), columns=LABELS).assign(StudyInstanceUID=uids)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--comp-dir", type=Path, required=True)
    p.add_argument("--models", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("submission.csv"))
    p.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    args = p.parse_args()

    test = pd.read_csv(args.comp_dir / "test.csv")
    errors = preprocess_all(args.comp_dir / "test_series.csv", args.comp_dir / "test_series", args.work,
                            workers=args.workers)
    for error in errors:
        print(error, flush=True)
    n_series_studies = pd.read_csv(args.comp_dir / "test_series.csv")["StudyInstanceUID"].nunique()
    if len(errors) > 0.05 * max(n_series_studies, 1):  # a broken decoder must fail the run, not hide in averages
        raise RuntimeError(f"{len(errors)} of {n_series_studies} test studies failed preprocessing")

    checkpoints = fold_checkpoints(args.models)
    print(f"{len(checkpoints)} fold models: {[str(c[0]) for c in checkpoints]}", flush=True)
    ready = [uid for uid in test["StudyInstanceUID"] if (args.work / f"{uid}.npz").exists()]
    predictions = predict(args.work, ready, checkpoints)

    # A study that failed preprocessing still needs a row: use the mean prediction (ranks it in the middle).
    submission = test[["StudyInstanceUID"]].merge(predictions, on="StudyInstanceUID", how="left")
    fill = predictions[LABELS].mean() if len(predictions) else pd.Series(0.5, index=LABELS)
    submission[LABELS] = submission[LABELS].fillna(fill)
    submission[["StudyInstanceUID", *LABELS]].to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(submission)} rows, {len(test) - len(ready)} filled", flush=True)


if __name__ == "__main__":
    main()
