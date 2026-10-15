#!/usr/bin/env bash
# Push the GPU training notebook: push_train.sh <run> <seeds e.g. 4_5> "<extra train args>" [limit]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
USER_NAME="$(kaggle config view | sed -n 's/^- username: //p')"
RUN=$1; SEEDS=$2; EXTRA=$3; LIMIT=${4:-0}
DIR="$ROOT/kaggle/.build/train-$RUN" && rm -rf "$DIR" && mkdir -p "$DIR"
sed -e "s/{run}/$RUN/" -e "s/{seeds}/$SEEDS/" -e "s|{extra}|$EXTRA|" -e "s/{limit}/$LIMIT/" "$ROOT/kaggle/train/kernel.py" > "$DIR/kernel.py"
cat > "$DIR/kernel-metadata.json" <<JSON
{"id": "$USER_NAME/rsna-knee-train-$RUN", "title": "rsna-knee-train-$RUN", "code_file": "kernel.py",
 "language": "python", "kernel_type": "script", "is_private": true, "enable_gpu": true, "enable_internet": true,
 "dataset_sources": ["$USER_NAME/rsna-knee-code", "$USER_NAME/rsna-knee-train-meta"],
 "kernel_sources": ["$USER_NAME/rsna-knee-preprocess-v7-0", "$USER_NAME/rsna-knee-preprocess-v7-1", "$USER_NAME/rsna-knee-preprocess-v7-2", "$USER_NAME/rsna-knee-preprocess-v7-3", "$USER_NAME/rsna-knee-preprocess-v7-4"],
 "competition_sources": ["rsna-knee-abnormality-detection"]}
JSON
kaggle kernels push -p "$DIR"
