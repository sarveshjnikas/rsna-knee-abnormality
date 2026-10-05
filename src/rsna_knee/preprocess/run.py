"""Preprocess every study listed in a series CSV into one compressed .npz per study.

    python -m rsna_knee.preprocess.run --series-csv data/raw/test_series.csv \
        --image-root data/raw/test_series --out data/processed/test [--shard 0 --num-shards 3]

The same entry point runs locally, on Kaggle (training shards) and in the inference notebook.
"""

import argparse
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from rsna_knee.preprocess.study import build_study


def _process(args: tuple[str, pd.DataFrame, Path, Path, str]) -> str | None:
    uid, series, image_root, out, version = args
    target = out / f"{uid}.npz"
    if target.exists():
        return None
    try:
        image, present = build_study(series, image_root / uid, version)
        np.savez_compressed(target, image=image, present=present)
        return None
    except Exception:  # one bad study must not stop the run; it is reported at the end
        return f"{uid}: {traceback.format_exc(limit=2)}"


def preprocess_all(series_csv: Path, image_root: Path, out: Path, shard: int = 0, num_shards: int = 1,
                   workers: int = 4, version: str = "v1") -> list[str]:
    """Write one .npz per study; returns error messages for studies that failed."""
    series = pd.read_csv(series_csv)
    studies = sorted(uid for uid in series["StudyInstanceUID"].unique() if (image_root / uid).is_dir())
    studies = studies[shard::num_shards]
    out.mkdir(parents=True, exist_ok=True)
    by_study = dict(tuple(series.groupby("StudyInstanceUID")))

    start, errors = time.time(), []
    jobs = [(uid, by_study[uid], image_root, out, version) for uid in studies]
    with ProcessPoolExecutor(workers) as pool:
        for i, error in enumerate(pool.map(_process, jobs, chunksize=4), 1):
            if error:
                errors.append(error)
            if i % 100 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} studies, {time.time() - start:.0f}s, {len(errors)} errors", flush=True)
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--series-csv", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--version", default="v1")
    args = parser.parse_args()
    errors = preprocess_all(args.series_csv, args.image_root, args.out, args.shard, args.num_shards, args.workers,
                            args.version)
    for error in errors:
        print(error)


if __name__ == "__main__":
    main()
