# Kaggle GPU notebook (2x T4): train window models on all data, one seed per GPU; models land in /kaggle/working.
import glob
import os
import shutil
import subprocess
import sys
import time

RUN, SEEDS, EXTRA, LIMIT = "{run}", "{seeds}".split("_"), "{extra}".split(), "{limit}"
t0 = time.time()
src = os.path.dirname(os.path.dirname(glob.glob("/kaggle/input/**/rsna_knee/__init__.py", recursive=True)[0]))
meta = os.path.dirname(glob.glob("/kaggle/input/**/labels_x2_grouped.csv", recursive=True)[0])
comp = os.path.dirname(glob.glob("/kaggle/input/**/train_series.csv", recursive=True)[0])

# A WRITABLE PROJECT LAYOUT: code + the small tables our config expects + the preprocessed v7 arrays
W = "/kaggle/temp/code"
shutil.copytree(src, f"{W}/src")
for d in ("raw", "splits", "labels", "interim", "processed"):
    os.makedirs(f"{W}/data/{d}", exist_ok=True)
os.symlink(f"{comp}/train.csv", f"{W}/data/raw/train.csv")
for f, d in (("folds.csv", "splits"), ("gold_split.csv", "splits"), ("labels_x2_grouped.csv", "labels"), ("knee_side.csv", "interim"),
             ("status_v2_gpt-6.1-sol.csv", "labels")):
    os.symlink(f"{meta}/{f}", f"{W}/data/{d}/{f}")
for tar in sorted(glob.glob("/kaggle/input/**/train_v7_shard*.tar", recursive=True)):
    subprocess.run(["tar", "-xf", tar, "-C", f"{W}/data/processed", "--transform", "s,^processed,train_v7,"], check=True)
print(f"unpacked {len(os.listdir(f'{W}/data/processed/train_v7'))} studies in {time.time() - t0:.0f}s", flush=True)
subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"])

# ONE TRAINING PROCESS PER GPU
out = f"/kaggle/working/runs/{RUN}"
os.makedirs(out, exist_ok=True)
env = dict(os.environ, PYTHONPATH=f"{W}/src")
procs = []
for gpu, seed in enumerate(SEEDS):
    cmd = [sys.executable, "-u", "-m", "rsna_knee.model.train_windows", "--fold", "-1", "--seed", seed,
           "--processed", f"{W}/data/processed/train_v7", "--out", out, "--labels", f"{W}/data/labels/labels_x2_grouped.csv",
           "--workers", "2", "--grad-checkpointing", "--amp", "fp16", "--limit", LIMIT, *EXTRA]
    log = open(f"{out}/seed{seed}.log", "w")
    procs.append(subprocess.Popen(cmd, cwd=W, env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu)), stdout=log, stderr=subprocess.STDOUT))
for p in procs:
    p.wait()
for seed in SEEDS:
    print(f"--- seed {seed} ---\n" + open(f"{out}/seed{seed}.log").read()[-1500:], flush=True)
print(f"total {time.time() - t0:.0f}s", flush=True)
