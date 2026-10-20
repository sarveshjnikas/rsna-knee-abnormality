# Kaggle GPU check: gold AUC of the side+v7 fold models with every window (inference) vs mean of random 43-window draws (training).
import glob, os, shutil, subprocess, sys, time
import pandas as pd
t0 = time.time()
CHECK = r'''
"""Gold AUC of fold models with every window (as at inference) vs the mean of random 43-window draws (as in training).

    python check.py <root> [draws]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "src")
from rsna_knee.config import INTERIM, LABELS, PREPROCESS, RAW, SPLITS  # noqa: E402
from rsna_knee.infer import fold_checkpoints  # noqa: E402
from rsna_knee.labeller.evaluate import auc_table  # noqa: E402
from rsna_knee.model.network import WindowNet  # noqa: E402
from rsna_knee.model.windows import WindowDataset, canonicalise  # noqa: E402

root = Path(sys.argv[1]); draws = int(sys.argv[2]) if len(sys.argv) > 2 else 4
device = "cuda"
gold = pd.read_csv(RAW / "train.csv").merge(pd.read_csv(SPLITS / "gold_split.csv"), on="StudyInstanceUID")
uids = gold.StudyInstanceUID.tolist()
sides = pd.read_csv(INTERIM / "knee_side.csv").set_index("StudyInstanceUID").iloc[:, 0].to_dict()
slots = PREPROCESS["v7"]["slots"]; sagittal = tuple(i for i, s in enumerate(slots) if s.startswith("sag"))
max_windows = sum(PREPROCESS["v7"]["slot_slices"])


@torch.no_grad()
def run(model, k, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    loader = DataLoader(WindowDataset(uids, root / "train_v7", k=k, max_windows=max_windows, n_views=len(slots)), batch_size=2)
    out = []
    for i, (windows, ids, pos, valid, _, batch) in enumerate(loader):
        windows, ids, pos, valid = windows.to(device).float() / 255, ids.to(device), pos.to(device), valid.to(device)
        left = torch.tensor([sides.get(u) == "L" for u in batch], device=device)
        windows, pos = canonicalise(windows, ids, pos, sagittal, left)
        out.append(model(windows, ids, pos, valid).float().sigmoid().cpu().numpy())
    return pd.DataFrame(np.concatenate(out), columns=LABELS).assign(StudyInstanceUID=uids)


full, sub = [], []
for ck in fold_checkpoints(root / "wside7"):
    model = WindowNet(ck["backbone"], pretrained=False, n_views=len(slots), head=ck["head"], mixer=ck["mixer"])
    model.load_state_dict(torch.load(ck["path"], map_location="cpu")); model = model.to(device).eval()
    full.append(run(model, 0))
    draw = [run(model, 43, s) for s in range(draws)]
    sub.append(draw[0].assign(**{L: np.mean([d[L] for d in draw], axis=0) for L in LABELS}))
    print(ck["path"].parent.name, f"all windows {auc_table(gold, full[-1])['macro']:.4f}",
          f"| 43-window x{draws} {auc_table(gold, sub[-1])['macro']:.4f}", flush=True)
mean = lambda fs: fs[0].assign(**{L: np.mean([f[L] for f in fs], axis=0) for L in LABELS})
a, b = auc_table(gold, mean(full)), auc_table(gold, mean(sub))
print(pd.DataFrame({"all": a, "k43": b, "diff": b - a}).round(4).to_string())
'''
src = os.path.dirname(os.path.dirname(glob.glob("/kaggle/input/**/rsna_knee/__init__.py", recursive=True)[0]))
meta = os.path.dirname(glob.glob("/kaggle/input/**/labels_x2_grouped.csv", recursive=True)[0])
comp = os.path.dirname(glob.glob("/kaggle/input/**/train_series.csv", recursive=True)[0])
mroot = glob.glob("/kaggle/input/**/rsna-knee-models", recursive=True)[0]
models = next(r for r, ds, _ in os.walk(mroot, followlinks=True) if os.path.basename(r) == "fold0")
print("models", models, os.listdir(os.path.dirname(models)), flush=True)
W = "/kaggle/temp/code"
shutil.copytree(src, f"{W}/src")
for d in ("raw", "splits", "labels", "interim", "processed/train_v7"):
    os.makedirs(f"{W}/data/{d}", exist_ok=True)
os.symlink(f"{comp}/train.csv", f"{W}/data/raw/train.csv")
for f, d in (("folds.csv", "splits"), ("gold_split.csv", "splits"), ("labels_x2_grouped.csv", "labels"), ("knee_side.csv", "interim")):
    os.symlink(f"{meta}/{f}", f"{W}/data/{d}/{f}")
gold = pd.read_csv(f"{meta}/gold_split.csv").StudyInstanceUID
names = {f"processed/{u}.npz" for u in gold}
for tar in sorted(glob.glob("/kaggle/input/**/train_v7_shard*.tar", recursive=True)):
    members = [m for m in subprocess.run(["tar", "-tf", tar], capture_output=True, text=True).stdout.split() if m in names]
    if members:
        subprocess.run(["tar", "-xf", tar, "-C", f"{W}/data/processed", "--transform", "s,^processed,train_v7,", *members], check=True)
print(f"gold studies: {len(os.listdir(f'{W}/data/processed/train_v7'))} in {time.time() - t0:.0f}s", flush=True)
os.makedirs(f"{W}/root", exist_ok=True)
os.symlink(f"{W}/data/processed/train_v7", f"{W}/root/train_v7"); os.symlink(os.path.dirname(models), f"{W}/root/wside7")
open(f"{W}/check.py", "w").write(CHECK)
subprocess.run([sys.executable, "-u", "check.py", f"{W}/root", "4"], cwd=W, check=True)
print(f"total {time.time() - t0:.0f}s")
