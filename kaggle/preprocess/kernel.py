# Kaggle CPU notebook: preprocess one shard of the training studies into a single tar.
import glob
import os
import subprocess
import sys

SHARD, NUM_SHARDS = {shard}, {num_shards}

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pylibjpeg", "pylibjpeg-openjpeg", "pylibjpeg-libjpeg"], check=True)
package = glob.glob("/kaggle/input/**/rsna_knee/__init__.py", recursive=True)[0]
sys.path.insert(0, os.path.dirname(os.path.dirname(package)))
print("python", sys.version, "| cpus", os.cpu_count(), flush=True)
comp = os.path.dirname(glob.glob("/kaggle/input/**/train_series.csv", recursive=True)[0])

from rsna_knee.preprocess import run  # noqa: E402

sys.argv = [
    "run", "--series-csv", f"{comp}/train_series.csv", "--image-root", f"{comp}/train_series",
    "--out", "/kaggle/temp/processed", "--shard", str(SHARD), "--num-shards", str(NUM_SHARDS),
    "--workers", str(os.cpu_count()),
]
run.main()
subprocess.run(["tar", "-cf", f"/kaggle/working/train_shard{SHARD}.tar", "-C", "/kaggle/temp", "processed"], check=True)
