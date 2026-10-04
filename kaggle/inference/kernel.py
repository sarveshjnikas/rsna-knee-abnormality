# Kaggle GPU notebook, internet off: preprocess the test studies, average fold models, write submission.csv.
import glob
import os
import subprocess
import sys

wheels = glob.glob("/kaggle/input/**/rsna-knee-wheels", recursive=True)
if wheels:  # JPEG / JPEG 2000 decoders for pydicom, installed offline
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", "--find-links", wheels[0],
                    "pylibjpeg", "pylibjpeg-openjpeg", "pylibjpeg-libjpeg"], check=False)
import importlib  # noqa: E402

for module in ("pylibjpeg", "openjpeg", "libjpeg"):  # JPEG 2000 / JPEG lossless decoders for pydicom
    importlib.import_module(module)

package = glob.glob("/kaggle/input/**/rsna_knee/__init__.py", recursive=True)[0]
sys.path.insert(0, os.path.dirname(os.path.dirname(package)))
comp = os.path.dirname(glob.glob("/kaggle/input/**/test_series.csv", recursive=True)[0])
models = glob.glob("/kaggle/input/**/rsna-knee-models", recursive=True)[0]
print("python", sys.version, "| comp", comp, "| models", models, flush=True)

from rsna_knee import infer  # noqa: E402

sys.argv = ["infer", "--comp-dir", comp, "--models", models, "--work", "/kaggle/temp/test",
            "--out", "/kaggle/working/submission.csv"]
infer.main()
