"""Train the window model on one CV fold, or on all labelled studies (--fold -1).

    python -m rsna_knee.model.train_windows --fold 0 --processed data/processed/train_v5 --out runs/coat_v5

- Training: k random windows per study (a new subset every epoch), mild brightness jitter, no flips
  (left and right knees both occur; the anatomical mirror is used at test time instead).
- Two learning rates: the pretrained backbone moves slowly, the new head fast; warm-up then cosine.
- Weight averaging: the final weights are the mean of the best `--swa` epochs by fold validation
  (on --fold -1, with nothing held out, the last `--swa` epochs).
- Evaluation and the saved predictions use every window, averaged with the anatomical mirror.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from rsna_knee.config import LABEL_DIR, LABELS, PREPROCESS, RAW, SPLITS
from rsna_knee.model.network import WindowNet
from rsna_knee.model.train import macro_auc, masked_bce
from rsna_knee.model.windows import WindowDataset, mirror_windows


def sagittal_views(version: str) -> tuple[int, ...]:
    return tuple(i for i, name in enumerate(PREPROCESS[version]["slots"]) if name.startswith("sag"))


@torch.no_grad()
def predict(model: WindowNet, loader: DataLoader, device: str, sagittal: tuple[int, ...]) -> np.ndarray:
    """Probabilities averaged over the study and its anatomical mirror."""
    model.eval()
    out = []
    for windows, ids, pos, valid, _, _ in loader:
        windows, ids, pos, valid = windows.to(device).float() / 255, ids.to(device), pos.to(device), valid.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            p = model(windows, ids, pos, valid).float().sigmoid()
            mirrored, mpos = mirror_windows(windows, ids, pos, sagittal)
            p = (p + model(mirrored, ids, mpos, valid).float().sigmoid()) / 2
        out.append(p.cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, required=True, help="-1: train on every labelled study, no validation")
    p.add_argument("--processed", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--labels", type=Path, default=LABEL_DIR / "labels_distill_r2.csv")
    p.add_argument("--folds", type=Path, default=SPLITS / "folds.csv")
    p.add_argument("--backbone", default="coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k")
    p.add_argument("--preprocess-version", default="v5")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--k", type=int, default=16, help="windows per study per training step")
    p.add_argument("--backbone-lr", type=float, default=3e-5)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.02)
    p.add_argument("--swa", type=int, default=3, help="average the weights of this many epochs")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="debug: use only this many training studies")
    p.add_argument("--grad-checkpointing", action="store_true")
    p.add_argument("--head", default="label", choices=["label", "query"], help="query: learned query per label")
    p.add_argument("--view-quota", action="store_true", help="training windows: an equal share per view")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = args.out / (f"fold{args.fold}" if args.fold >= 0 else f"all_seed{args.seed}")
    out.mkdir(parents=True, exist_ok=True)
    cfg = PREPROCESS[args.preprocess_version]
    n_views, sagittal = len(cfg["slots"]), sagittal_views(args.preprocess_version)
    config = {k: str(v) for k, v in vars(args).items()} | {
        "model": "window", "preprocess": args.preprocess_version, "n_views": str(n_views),
        "max_windows": str(sum(cfg["slot_slices"]))}
    (out / "config.json").write_text(json.dumps(config, indent=2))

    labels = pd.read_csv(args.labels)
    folds = pd.read_csv(args.folds)
    gold = pd.read_csv(RAW / "train.csv").merge(pd.read_csv(SPLITS / "gold_split.csv"), on="StudyInstanceUID")
    available = {f.stem for f in args.processed.glob("*.npz")}
    folds = folds[folds["StudyInstanceUID"].isin(available) & folds["StudyInstanceUID"].isin(labels["StudyInstanceUID"])]
    train_uids = folds.loc[folds["fold"] != args.fold, "StudyInstanceUID"].tolist()
    val_uids = folds.loc[folds["fold"] == args.fold, "StudyInstanceUID"].tolist()  # empty for --fold -1
    gold_uids = [u for u in gold["StudyInstanceUID"] if u in available]
    if args.limit:
        train_uids, val_uids = train_uids[: args.limit], val_uids[: max(args.limit // 4, 2)]

    max_windows = sum(cfg["slot_slices"])
    loader_args = dict(num_workers=args.workers, pin_memory=device == "cuda")
    train_loader = DataLoader(WindowDataset(train_uids, args.processed, labels, k=args.k, max_windows=max_windows,
                                            n_views=n_views, view_quota=args.view_quota),
                              batch_size=args.batch_size, shuffle=True, drop_last=True, **loader_args)
    eval_args = dict(batch_size=2, **loader_args)
    val_loader = DataLoader(WindowDataset(val_uids, args.processed, labels, max_windows=max_windows, n_views=n_views),
                            **eval_args)
    gold_loader = DataLoader(WindowDataset(gold_uids, args.processed, max_windows=max_windows, n_views=n_views),
                             **eval_args)

    model = WindowNet(args.backbone, pretrained=True, n_views=n_views, head=args.head).to(device)
    if args.grad_checkpointing:
        model.encoder.set_grad_checkpointing(True)
    backbone = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    head = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    optimizer = torch.optim.AdamW([{"params": backbone, "lr": args.backbone_lr}, {"params": head, "lr": args.head_lr}],
                                  weight_decay=args.weight_decay)
    total_steps, warmup = args.epochs * len(train_loader), len(train_loader)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s / total_steps, 1.0))))

    val_raw = labels.set_index("StudyInstanceUID").loc[val_uids, LABELS].to_numpy() if val_uids else np.zeros((0, 12))
    val_known, val_targets = ~np.isnan(val_raw), np.nan_to_num(val_raw) >= 0.5
    gold_targets = gold.set_index("StudyInstanceUID").loc[gold_uids, LABELS].to_numpy()
    kept, history = [], []  # kept: (score, epoch, cpu state dict) of the epochs that will be averaged
    for epoch in range(args.epochs):
        model.train()
        start, losses = time.time(), []
        for windows, ids, pos, valid, target, _ in train_loader:
            windows = windows.to(device, non_blocking=True).float() / 255
            gain = torch.empty(windows.shape[0], 1, 1, 1, 1, device=device).uniform_(0.9, 1.1)
            windows = (windows * gain).clamp(0, 1)  # scanner brightness differences
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(windows, ids.to(device), pos.to(device), valid.to(device)).float()
            loss = masked_bce(logits, target.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            losses.append(loss.item())

        val_auc = macro_auc(val_targets, predict(model, val_loader, device, sagittal), val_known) if val_uids else float("nan")
        gold_auc = macro_auc(gold_targets, predict(model, gold_loader, device, sagittal))
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "seconds": round(time.time() - start),
               "gpu_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1) if device == "cuda" else None,
               "val_auc": val_auc, "gold_auc": gold_auc}
        history.append(row)
        print(json.dumps(row), flush=True)
        score = val_auc if val_uids else float(epoch)  # all data: the latest epochs
        if len(kept) < args.swa or score > min(k[0] for k in kept):
            kept.append((score, epoch, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}))
            kept = sorted(kept, key=lambda k: -k[0])[: args.swa]

    averaged = {k: sum(state[k].float() for _, _, state in kept) / len(kept) for k in kept[0][2]}
    model.load_state_dict({k: v.to(model.state_dict()[k].dtype) for k, v in averaged.items()})
    torch.save(model.state_dict(), out / "model.pt")
    gold_scores = predict(model, gold_loader, device, sagittal)
    pd.DataFrame(gold_scores, columns=LABELS).assign(StudyInstanceUID=gold_uids).to_csv(out / "gold.csv", index=False)
    summary = {"averaged_epochs": sorted(e for _, e, _ in kept), "gold_auc": macro_auc(gold_targets, gold_scores)}
    if val_uids:
        val_scores = predict(model, val_loader, device, sagittal)
        pd.DataFrame(val_scores, columns=LABELS).assign(StudyInstanceUID=val_uids).to_csv(out / "oof.csv", index=False)
        summary["val_auc"] = macro_auc(val_targets, val_scores, val_known)
    print(json.dumps({"averaged": summary}), flush=True)
    (out / "history.json").write_text(json.dumps(history + [{"averaged": summary}], indent=2))


if __name__ == "__main__":
    main()
