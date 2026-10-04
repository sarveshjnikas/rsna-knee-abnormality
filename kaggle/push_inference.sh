#!/usr/bin/env bash
# Push the private, internet-off GPU inference notebook (code + wheels + models datasets attached).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
USER_NAME="$(kaggle config view | sed -n 's/^- username: //p')"
DIR="$ROOT/kaggle/.build/inference" && rm -rf "$DIR" && mkdir -p "$DIR"
cp "$ROOT/kaggle/inference/kernel.py" "$DIR/kernel.py"
cat > "$DIR/kernel-metadata.json" <<JSON
{"id": "$USER_NAME/rsna-knee-inference", "title": "rsna-knee-inference", "code_file": "kernel.py",
 "language": "python", "kernel_type": "script", "is_private": true, "enable_gpu": true, "enable_internet": false,
 "dataset_sources": ["$USER_NAME/rsna-knee-code", "$USER_NAME/rsna-knee-wheels", "$USER_NAME/rsna-knee-models"],
 "competition_sources": ["rsna-knee-abnormality-detection"]}
JSON
kaggle kernels push -p "$DIR"
