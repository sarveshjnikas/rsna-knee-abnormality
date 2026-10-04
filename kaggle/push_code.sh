#!/usr/bin/env bash
# Upload src/ as the private Kaggle dataset <user>/rsna-knee-code (new version if it exists).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
USER_NAME="$(kaggle config view | sed -n 's/^- username: //p')"
BUILD="$ROOT/kaggle/.build/code"
rm -rf "$BUILD" && mkdir -p "$BUILD"
cp -r "$ROOT/src" "$BUILD/src"
find "$BUILD" -name __pycache__ -prune -exec rm -rf {} +
cat > "$BUILD/dataset-metadata.json" <<JSON
{"title": "rsna-knee-code", "id": "$USER_NAME/rsna-knee-code", "licenses": [{"name": "CC0-1.0"}]}
JSON
if kaggle datasets status "$USER_NAME/rsna-knee-code" >/dev/null 2>&1; then
  kaggle datasets version -p "$BUILD" -m "${1:-update}" --dir-mode zip -q
else
  kaggle datasets create -p "$BUILD" --dir-mode zip -q
fi
