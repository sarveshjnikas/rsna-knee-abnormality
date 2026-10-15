# Kaggle GPU notebook: inference parity audit. The 58 gold TRAIN studies are dressed up as a test set and run through
# exactly the submission path (preprocessing from raw DICOMs, knee side from headers, models); compare with training-time predictions.
import glob
import os
import subprocess
import sys

import pandas as pd

package = glob.glob("/kaggle/input/**/rsna_knee/__init__.py", recursive=True)[0]
src = os.path.dirname(os.path.dirname(package))
comp = os.path.dirname(glob.glob("/kaggle/input/**/train_series.csv", recursive=True)[0])
models = os.path.dirname(os.path.dirname(glob.glob("/kaggle/input/**/rsna-knee-models/**/model.pt", recursive=True)[0]))
for w in glob.glob("/kaggle/input/**/rsna-knee-wheels/*.whl", recursive=True):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", w])

fake = "/kaggle/temp/fake"
os.makedirs(f"{fake}/test_series", exist_ok=True)
gold = pd.read_csv(f"{comp}/train.csv").dropna(subset=["ACL"])["StudyInstanceUID"]
pd.DataFrame({"StudyInstanceUID": gold}).to_csv(f"{fake}/test.csv", index=False)
series = pd.read_csv(f"{comp}/train_series.csv")
series[series["StudyInstanceUID"].isin(gold)].to_csv(f"{fake}/test_series.csv", index=False)
for uid in gold:
    os.symlink(f"{comp}/train_series/{uid}", f"{fake}/test_series/{uid}")

subprocess.run([sys.executable, "-m", "rsna_knee.infer", "--comp-dir", fake, "--models", models,
                "--work", "/kaggle/temp/work", "--out", "/kaggle/working/audit_gold.csv"],
               env=dict(os.environ, PYTHONPATH=src), check=True)
print(open("/kaggle/working/audit_gold.csv").read()[:300])
