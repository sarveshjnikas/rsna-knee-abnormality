"""Build the blend notebook: the public 0.943 community stack (unchanged) + our models as a 30% rank-blend leg.

    python kaggle/blend/build.py <public notebook .ipynb> <its kernel-metadata.json> <out dir>

Part A (the stack) is copied cell for cell from the public notebook. Our leg runs afterwards in a
subprocess with its own decoders and our rsna_knee code; if it fails for any reason, the stack's
submission is kept, so a broken leg can never cost the submission.
"""

import json
import sys
from pathlib import Path

OWN_W = 0.30  # fixed before submitting: a 0.929 reader scored 0.944 at 15% and 30%, 0.942 at 45%

HEADER = """# RSNA Knee: community stack + our 2.5D models

Part A is the public 0.943 community stack, unchanged
([skarin's reproduction](https://www.kaggle.com/code/skarin/rsna-knee-reproducing-the-0-943-public-stack),
via [goodpjw2008](https://www.kaggle.com/code/goodpjw2008/rsna-knee-stack-2-5d-convnext-mil-lb-0-944);
models by Mattia Angeli, Dread Development, Pilkwang Kim, prvsiyan, Sofia Anjenje, Antoine G., Marwan Mahmoud).

Part B adds our own model family (2.5D EfficientNet-B0, 5 folds, self-trained labels; 0.934 alone)
as a rank-blend leg at a fixed weight. If Part B fails, the stack's submission is kept."""

OWN_LEG = f'''# Part B: our models, rank-blended into the stack's submission. Any failure keeps the stack's submission.
import glob as _o_glob, os as _o_os, shutil as _o_shutil, subprocess as _o_sp, sys as _o_sys, time as _o_time
import pandas as _o_pd
OWN_W = {OWN_W}
_o_pub, _o_bak = "/kaggle/working/submission.csv", "/kaggle/working/_public_stack.csv"
_o_shutil.copy(_o_pub, _o_bak)
try:
    _o_first = lambda pat: sorted(_o_glob.glob(pat, recursive=True))[0]
    _o_code = _o_os.path.dirname(_o_os.path.dirname(_o_first("/kaggle/input/**/rsna_knee/__init__.py")))
    _o_wheels = _o_first("/kaggle/input/**/rsna-knee-wheels")
    _o_models = _o_first("/kaggle/input/**/rsna-knee-models")
    _o_comp = _o_os.path.dirname(_o_first("/kaggle/input/**/test_series.csv"))
    _o_target = "/kaggle/working/_own_env"
    _o_sp.run([_o_sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", "--target", _o_target,
               "--find-links", _o_wheels, "pylibjpeg", "pylibjpeg-openjpeg", "pylibjpeg-libjpeg"], check=False)
    _o_env = dict(_o_os.environ)
    _o_env["PYTHONPATH"] = _o_os.pathsep.join([_o_target, _o_code, _o_env.get("PYTHONPATH", "")])
    _o_check = "import cv2, pydicom, timm, pylibjpeg, openjpeg, libjpeg"
    if _o_sp.run([_o_sys.executable, "-c", _o_check], env=_o_env).returncode != 0:
        _o_cv = sorted(_o_glob.glob("/kaggle/input/**/opencv_python_headless-*.whl", recursive=True))
        if _o_cv:
            _o_sp.run([_o_sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", "--target", _o_target, _o_cv[0]], check=False)
        _o_sp.run([_o_sys.executable, "-c", _o_check], env=_o_env, check=True)
    try:
        import gc as _o_gc, torch as _o_torch
        _o_gc.collect(); _o_torch.cuda.empty_cache()
        _o_free = [_o_torch.cuda.mem_get_info(i)[0] for i in range(_o_torch.cuda.device_count())]
        _o_env["CUDA_VISIBLE_DEVICES"] = str(max(range(len(_o_free)), key=lambda i: _o_free[i]))
        print("own leg: free GPU memory (GB)", [round(f / 1e9, 1) for f in _o_free], flush=True)
    except Exception as _o_e:
        print("own leg: GPU pick skipped:", repr(_o_e))
    _o_t0 = _o_time.time()
    _o_sp.run([_o_sys.executable, "-m", "rsna_knee.infer", "--comp-dir", _o_comp, "--models", _o_models,
               "--work", "/kaggle/temp/own", "--out", "/kaggle/working/_own.csv", "--workers", "4"], check=True, env=_o_env)
    _o_stack = _o_pd.read_csv(_o_bak, dtype={{"StudyInstanceUID": str}})
    _o_own = _o_pd.read_csv("/kaggle/working/_own.csv", dtype={{"StudyInstanceUID": str}})
    _o_own = _o_own.set_index("StudyInstanceUID").loc[_o_stack.StudyInstanceUID].reset_index()
    _o_labels = [c for c in _o_stack.columns if c != "StudyInstanceUID"]
    assert _o_own[_o_labels].notna().all().all(), "own predictions contain NaN"
    assert len(_o_stack) < 3 or (_o_own[_o_labels].nunique() > 1).all(), "own predictions are constant"
    _o_rank = lambda d: d[_o_labels].rank(method="average", pct=True)
    _o_final = _o_stack.copy()
    _o_final[_o_labels] = ((1 - OWN_W) * _o_rank(_o_stack) + OWN_W * _o_rank(_o_own)).rank(method="average", pct=True)
    _o_final.to_csv(_o_pub, index=False)
    print(f"own leg: blended at weight {{OWN_W}} in {{_o_time.time() - _o_t0:.0f}}s, {{len(_o_final)}} studies", flush=True)
except Exception as _o_e:
    _o_shutil.copy(_o_bak, _o_pub)
    print("own leg FAILED, the stack's submission is kept:", repr(_o_e), flush=True)
'''


def cell(kind: str, source: str) -> dict:
    base = {"cell_type": kind, "metadata": {}, "source": source}
    return base | ({"execution_count": None, "outputs": []} if kind == "code" else {})


def main() -> None:
    notebook_path, metadata_path, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    nb = json.loads(notebook_path.read_text())
    sources = ["".join(c["source"]) for c in nb["cells"]]
    start = next(i for i, s in enumerate(sources) if s.startswith("# Set to False to skip the community stack"))
    end = next(i for i, s in enumerate(sources) if "# Part B" in s)  # first cell of the public reader we replace
    nb["cells"] = [cell("markdown", HEADER)] + nb["cells"][start:end] + [cell("markdown", "# Part B · Our models"), cell("code", OWN_LEG)]
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None

    meta = json.loads(metadata_path.read_text())
    out.mkdir(parents=True, exist_ok=True)
    (out / "rsna-knee-blend.ipynb").write_text(json.dumps(nb, indent=1))
    datasets = [d for d in meta["dataset_sources"] if not d.startswith("goodpjw2008/")]
    datasets += [f"{OWNER}/rsna-knee-code", f"{OWNER}/rsna-knee-wheels", f"{OWNER}/rsna-knee-models"]
    (out / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{OWNER}/rsna-knee-blend", "title": "rsna-knee-blend", "code_file": "rsna-knee-blend.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
        "enable_internet": False, "dataset_sources": datasets, "kernel_sources": meta["kernel_sources"],
        "competition_sources": meta["competition_sources"], "model_sources": meta["model_sources"],
        "docker_image": meta["docker_image"], "machine_shape": meta["machine_shape"]}, indent=1))
    print(f"wrote {out}: {len(nb['cells'])} cells, {len(datasets)} datasets")


OWNER = "sarveshnikas"

if __name__ == "__main__":
    main()
