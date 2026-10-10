"""Test-time pipeline: preprocess test studies with the training code, average fold models, write submission.csv.

    python -m rsna_knee.infer --comp-dir <competition dir> --models <dir with fold*/model.pt> \
        --work /kaggle/temp/test --out submission.csv
"""

import argparse
import json
import os
import shutil
import traceback
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from rsna_knee.config import LABELS, PREPROCESS, SLOTS
from rsna_knee.model.data import StudyDataset
from rsna_knee.model.network import KneeNet, WindowNet
from rsna_knee.model.windows import WindowDataset, mirror_windows
from rsna_knee.preprocess.run import preprocess_all
from rsna_knee.preprocess.side import side_table


def fold_checkpoints(models_dir: Path) -> list[dict]:
    """One entry per fold checkpoint under models_dir, with what is needed to rebuild and feed it."""
    found = []
    for checkpoint in sorted(models_dir.glob("**/model.pt")):  # fold<k>/ (CV models) or all_seed<s>/ (all data)
        if not checkpoint.parent.name.startswith(("fold", "all_seed")):
            continue
        config = json.loads((checkpoint.parent / "config.json").read_text())
        version = config.get("preprocess", "v1")
        found.append({
            "model": config.get("model", "knee"),
            "path": checkpoint,
            "family": checkpoint.parent.parent.name,  # one training run = one model family
            "backbone": config.get("backbone", "efficientnet_b0"),
            "head": config.get("head", "label"),
            "fusion": config.get("fusion", "attention"),
            "version": version,
            "n_slices": int(config.get("n_slices", PREPROCESS[version]["n_slices"])),
            "unique_slices": str(config.get("unique_slices", False)) == "True",
            "n_slots": int(config.get("n_slots", len(SLOTS))),
            "canonical_side": str(config.get("canonical_side", "None")) not in ("None", ""),
        })
    return found


def derivable(small: str, large: str) -> bool:
    """small is large at a lower resolution: same physical crop and the same slices."""
    a, b = PREPROCESS[small], PREPROCESS[large]
    return a["image_size"] < b["image_size"] and all(a[k] == b[k] for k in ("n_slices", "crop_mm", "slice_range"))


def downsize(src: Path, dst: Path, size: int) -> None:
    """Write src's arrays at a smaller size, so a second resolution needs no second pass over the DICOMs."""
    dst.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.glob("*.npz")):
        with np.load(f) as data:
            image, present = data["image"], data["present"]
        small = np.stack([[cv2.resize(s, (size, size), interpolation=cv2.INTER_AREA) for s in view] for view in image])
        np.savez_compressed(dst / f.name, image=small, present=present)  # uncompressed copies can fill the disk


@torch.no_grad()
def predict_windows(processed: Path, uids: list[str], checkpoints: list[dict], workers: int = 2) -> pd.DataFrame:
    """Window models: every window of the study, averaged with its anatomical mirror and over the models."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    version = checkpoints[0]["version"]
    slots, sagittal = PREPROCESS[version]["slots"], tuple(i for i, s in enumerate(PREPROCESS[version]["slots"]) if s.startswith("sag"))
    max_windows = sum(PREPROCESS[version]["slot_slices"])
    loader = DataLoader(WindowDataset(uids, processed, max_windows=max_windows, n_views=len(slots)), batch_size=2,
                        num_workers=workers)
    models = []
    for ck in checkpoints:
        model = WindowNet(ck["backbone"], pretrained=False, n_views=len(slots), head=ck["head"])
        model.load_state_dict(torch.load(ck["path"], map_location="cpu"))
        models.append(model.to(device).eval())
    rows = []
    for windows, ids, pos, valid, _, _ in loader:
        windows, ids, pos, valid = windows.to(device).float() / 255, ids.to(device), pos.to(device), valid.to(device)
        mirrored, mpos = mirror_windows(windows, ids, pos, sagittal)
        prob = 0
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
            for model in models:
                prob = prob + model(windows, ids, pos, valid).float().sigmoid() + model(mirrored, ids, mpos, valid).float().sigmoid()
        rows.append((prob / (2 * len(models))).cpu().numpy())
    return pd.DataFrame(np.concatenate(rows), columns=LABELS).assign(StudyInstanceUID=uids)


@torch.no_grad()
def predict(processed: Path, uids: list[str], checkpoints: list[dict], batch_size: int = 2,
            workers: int = 2, flip_tta: bool = True, sides: dict[str, str] | None = None) -> pd.DataFrame:
    """Mean sigmoid probability over fold models (and a left-right flip of every view), one row per study."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    modes = {(ck["unique_slices"], ck["n_slots"], ck["canonical_side"], ck["version"]) for ck in checkpoints}
    if len(modes) != 1:
        raise ValueError("one family must not mix slice modes, view counts or side handling")
    unique, n_slots, canonical, version = modes.pop()
    side_args = {}
    if canonical:  # right knees mirrored like in training; no flip TTA (the model never saw flipped knees)
        if sides is None:
            raise ValueError("this family needs the test studies' knee sides")
        side_args = dict(sides=sides, planes=[name[:3] for name in PREPROCESS[version].get("slots", SLOTS)][:n_slots])
        flip_tta = False
    loader = DataLoader(StudyDataset(uids, processed, unique=unique, n_slots=n_slots, **side_args),
                        batch_size=batch_size, num_workers=workers)
    models = []  # all fold models stay loaded, so each study is read and decompressed once, not once per model
    for ck in checkpoints:
        model = KneeNet(n_slots=ck["n_slots"], n_slices=ck["n_slices"], backbone=ck["backbone"], pretrained=False,
                        fusion=ck["fusion"])
        model.load_state_dict(torch.load(ck["path"], map_location="cpu"))
        models.append(model.to(device).eval())
    rows = []
    for image, present, _, _ in loader:
        image = image.to(device).float() / 255
        present = present.to(device)
        prob = 0
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
            for model in models:
                prob = prob + model(image, present).float().sigmoid()
                if flip_tta:  # trained with left-right flips, so the mirrored study is an equally valid view
                    prob = prob + model(image.flip(-1), present).float().sigmoid()
        rows.append((prob / (len(models) * (2 if flip_tta else 1))).cpu().numpy())
    return pd.DataFrame(np.concatenate(rows), columns=LABELS).assign(StudyInstanceUID=uids)

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--comp-dir", type=Path, required=True)
    p.add_argument("--models", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("submission.csv"))
    p.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    args = p.parse_args()

    test = pd.read_csv(args.comp_dir / "test.csv")
    checkpoints = fold_checkpoints(args.models)
    print(f"{len(checkpoints)} fold models: {[str(c['path']) for c in checkpoints]}", flush=True)
    n_series_studies = pd.read_csv(args.comp_dir / "test_series.csv")["StudyInstanceUID"].nunique()

    # Prepare each preprocessing version once, largest first; a smaller one with the same crop and slices
    # is resized from the larger arrays instead of decoding every DICOM again.
    prepared: dict[str, Path] = {}
    for version in sorted({c["version"] for c in checkpoints}, key=lambda v: -PREPROCESS[v]["image_size"]):
        work = args.work / version
        source = next((v for v in prepared if derivable(version, v)), None)
        if source:
            downsize(prepared[source], work, PREPROCESS[version]["image_size"])
            print(f"{version}: resized from {source}", flush=True)
        else:
            errors = preprocess_all(args.comp_dir / "test_series.csv", args.comp_dir / "test_series", work,
                                    workers=args.workers, version=version)
            for error in errors:
                print(error, flush=True)
            if len(errors) > 0.05 * max(n_series_studies, 1):  # a broken decoder must fail the run, not hide in averages
                raise RuntimeError(f"{len(errors)} of {n_series_studies} test studies failed preprocessing ({version})")
        prepared[version] = work
        print(f"{version} ready; disk free {shutil.disk_usage(args.work).free / 1e9:.1f} GB", flush=True)

    sides = None
    if any(c["canonical_side"] for c in checkpoints):  # headers only: one file per test series
        table = side_table(args.comp_dir / "test_series.csv", args.comp_dir / "test_series")
        sides = table.dropna().set_index("StudyInstanceUID")["side"].to_dict()
        print(f"knee sides: {table['side'].value_counts(dropna=False).to_dict()}", flush=True)

    family_predictions = []
    families = sorted({c["family"] for c in checkpoints})
    for family in families:
        group = [c for c in checkpoints if c["family"] == family]
        version = group[0]["version"]
        work = prepared[version]
        ready = [uid for uid in test["StudyInstanceUID"] if (work / f"{uid}.npz").exists()]
        try:
            if group[0]["model"] == "window":
                family_predictions.append(predict_windows(work, ready, group).set_index("StudyInstanceUID")[LABELS])
            else:
                family_predictions.append(predict(work, ready, group, sides=sides).set_index("StudyInstanceUID")[LABELS])
            print(f"family {family}: {len(group)} models, preprocessing {version}", flush=True)
        except Exception:  # one broken family must not cost the submission; the others still vote
            print(f"family {family} FAILED:\n{traceback.format_exc()}", flush=True)
        later = {c["version"] for c in checkpoints if c["family"] in families[families.index(family) + 1:]}
        if version not in later:  # free the disk as soon as no later family reads this version
            shutil.rmtree(work, ignore_errors=True)
    if not family_predictions:
        raise RuntimeError("every model family failed")
    # Average within a family (its folds), then give every family an equal vote.
    predictions = (sum(family_predictions) / len(family_predictions)).reset_index()

    # A study that failed preprocessing still needs a row: use the mean prediction (ranks it in the middle).
    submission = test[["StudyInstanceUID"]].merge(predictions, on="StudyInstanceUID", how="left")
    fill = predictions[LABELS].mean() if len(predictions) else pd.Series(0.5, index=LABELS)
    submission[LABELS] = submission[LABELS].fillna(fill)
    submission[["StudyInstanceUID", *LABELS]].to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(submission)} rows, {len(test) - len(ready)} filled", flush=True)


if __name__ == "__main__":
    main()
