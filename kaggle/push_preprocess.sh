#!/usr/bin/env bash
# Push NUM_SHARDS private CPU notebooks, each preprocessing one shard of the training studies.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
USER_NAME="$(kaggle config view | sed -n 's/^- username: //p')"
NUM_SHARDS="${1:-3}"
for SHARD in $(seq 0 $((NUM_SHARDS - 1))); do
  DIR="$ROOT/kaggle/.build/preprocess_$SHARD" && rm -rf "$DIR" && mkdir -p "$DIR"
  sed -e "s/{shard}/$SHARD/" -e "s/{num_shards}/$NUM_SHARDS/" "$ROOT/kaggle/preprocess/kernel.py" > "$DIR/kernel.py"
  cat > "$DIR/kernel-metadata.json" <<JSON
{"id": "$USER_NAME/rsna-knee-preprocess-$SHARD", "title": "rsna-knee-preprocess-$SHARD", "code_file": "kernel.py",
 "language": "python", "kernel_type": "script", "is_private": true, "enable_gpu": false, "enable_internet": true,
 "dataset_sources": ["$USER_NAME/rsna-knee-code"], "competition_sources": ["rsna-knee-abnormality-detection"]}
JSON
  kaggle kernels push -p "$DIR"
done
