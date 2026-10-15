"""Train one CV fold.

    python -m rsna_knee.model.train --fold 0 --processed data/processed/train --out runs/b0_v2

Validation each epoch: (1) the held-out fold vs pseudo-labels (targets >= 0.5 count as
positive), (2) all 58 gold studies, which no fold ever trains on. The checkpoint is
chosen on (1) only, so the gold score stays an honest check.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from rsna_knee.config import LABEL_DIR, LABELS, PREPROCESS, RAW, SLOTS, SPLITS
from rsna_knee.model.data import StudyDataset, augment
from rsna_knee.model.network import KneeNet


def masked_bce(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    """BCE over known targets only: a blank cell in the label file means "do not train this finding here".
    weight (same shape, optional) scales single cells, e.g. findings the report never mentions."""
    known = ~torch.isnan(target)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target.nan_to_num(0.0), reduction="none")
    if weight is not None:
        loss = loss * weight
    return (loss * known).sum() / known.sum().clamp(min=1)


def macro_auc(targets: np.ndarray, scores: np.ndarray, known: np.ndarray | None = None) -> float:
    """Mean AUC over labels with both classes; known (same shape) excludes unknown targets per label."""
    if known is not None:
        aucs = [roc_auc_score(t[k], s[k]) for t, s, k in zip(targets.T, scores.T, known.T) if len(np.unique(t[k])) == 2]
        return float(np.mean(aucs)) if aucs else float("nan")
    if len(targets) == 0:
        return float("nan")
    aucs = [roc_auc_score(t, s) for t, s in zip(targets.T, scores.T) if len(np.unique(t)) == 2]
    return float(np.mean(aucs)) if aucs else float("nan")


@torch.no_grad()
def predict(model: KneeNet, loader: DataLoader, device: str) -> tuple[np.ndarray, list[str]]:
    model.eval()
    scores, uids = [], []
    for image, present, _, batch_uids in loader:
        image = image.to(device, non_blocking=True).float() / 255
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = model(image, present.to(device))
        scores.append(logits.float().sigmoid().cpu().numpy())
        uids += list(batch_uids)
    return (np.concatenate(scores) if scores else np.zeros((0, len(LABELS)))), uids


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--processed", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--labels", type=Path, default=LABEL_DIR / "labels_v2_gpt-6.1-sol.csv")
    p.add_argument("--folds", type=Path, default=SPLITS / "folds.csv")
    p.add_argument("--backbone", default="efficientnet_b0")
    p.add_argument("--fusion", default="attention", choices=["attention", "mean", "label"])
    p.add_argument("--preprocess-version", default="v1", help="recorded so inference prepares test data the same way")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="debug: use only this many training studies")
    p.add_argument("--grad-checkpointing", action="store_true", help="trade compute for GPU memory")
    p.add_argument("--unique-slices", action="store_true", help="blank out repeated slices instead of feeding them twice")
    p.add_argument("--slots", type=int, default=0, help="use only the first N views of the data (0: all)")
    p.add_argument("--canonical-side", type=Path, default=None,
                   help="knee side table (StudyInstanceUID, side): mirror right knees and stop random left-right flips")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    out = args.out / f"fold{args.fold}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))

    labels = pd.read_csv(args.labels)
    folds = pd.read_csv(args.folds)
    gold = pd.read_csv(RAW / "train.csv").merge(pd.read_csv(SPLITS / "gold_split.csv"), on="StudyInstanceUID")
    available = {f.stem for f in args.processed.glob("*.npz")}

    folds = folds[folds["StudyInstanceUID"].isin(available) & folds["StudyInstanceUID"].isin(labels["StudyInstanceUID"])]
    train_uids = folds.loc[folds["fold"] != args.fold, "StudyInstanceUID"].tolist()
    val_uids = folds.loc[folds["fold"] == args.fold, "StudyInstanceUID"].tolist()
    gold_uids = [u for u in gold["StudyInstanceUID"] if u in available]
    if args.limit:
        train_uids, val_uids = train_uids[: args.limit], val_uids[: max(args.limit // 4, 2)]

    loader_args = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=device == "cuda")
    data = dict(processed_dir=args.processed, unique=args.unique_slices, n_slots=args.slots)
    if args.canonical_side:
        side = pd.read_csv(args.canonical_side).dropna().set_index("StudyInstanceUID")["side"].to_dict()
        slots = PREPROCESS[args.preprocess_version].get("slots", SLOTS)
        data |= dict(sides=side, planes=[name[:3] for name in slots][: args.slots or len(slots)])
    train_loader = DataLoader(StudyDataset(train_uids, labels=labels, **data), shuffle=True, drop_last=True, **loader_args)
    val_loader = DataLoader(StudyDataset(val_uids, labels=labels, **data), **loader_args)
    gold_loader = DataLoader(StudyDataset(gold_uids, **data), **loader_args)

    with np.load(args.processed / f"{train_uids[0]}.npz") as sample:
        n_views, n_slices = sample["image"].shape[:2]  # the data, not a constant, decides the shape
    n_slots = args.slots or n_views
    config = json.loads((out / "config.json").read_text()) | {"n_slices": str(n_slices), "n_slots": str(n_slots)}
    (out / "config.json").write_text(json.dumps(config, indent=2))
    model = KneeNet(n_slots=n_slots, n_slices=n_slices, backbone=args.backbone, fusion=args.fusion).to(device)
    if args.grad_checkpointing:
        model.encoder.set_grad_checkpointing(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    warmup = len(train_loader)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s / total_steps, 1.0)))
    )
    loss_fn = masked_bce

    val_raw = labels.set_index("StudyInstanceUID").loc[val_uids, LABELS].to_numpy()
    val_known, val_targets = ~np.isnan(val_raw), np.nan_to_num(val_raw) >= 0.5
    gold_targets = gold.set_index("StudyInstanceUID").loc[gold_uids, LABELS].to_numpy()
    best, history = -1.0, []
    for epoch in range(args.epochs):
        model.train()
        start, losses = time.time(), []
        for image, present, target, _ in train_loader:
            image = image.to(device, non_blocking=True).float() / 255
            image, present = augment(image, present.to(device), hflip=args.canonical_side is None)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                loss = loss_fn(model(image, present).float(), target.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            losses.append(loss.item())

        val_scores, _ = predict(model, val_loader, device)
        gold_scores, _ = predict(model, gold_loader, device)
        row = {
            "epoch": epoch, "train_loss": float(np.mean(losses)), "seconds": round(time.time() - start),
            "gpu_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1) if device == "cuda" else None,
            "val_auc": macro_auc(val_targets, val_scores, val_known), "gold_auc": macro_auc(gold_targets, gold_scores),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if row["val_auc"] > best or (math.isnan(row["val_auc"]) and best < 0):
            best = row["val_auc"]
            torch.save(model.state_dict(), out / "model.pt")
            pd.DataFrame(val_scores, columns=LABELS).assign(StudyInstanceUID=val_uids).to_csv(out / "oof.csv", index=False)
            pd.DataFrame(gold_scores, columns=LABELS).assign(StudyInstanceUID=gold_uids).to_csv(out / "gold.csv", index=False)
    (out / "history.json").write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
