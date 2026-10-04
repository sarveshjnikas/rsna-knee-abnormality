#!/usr/bin/env bash
# Create or version a private Kaggle dataset from a local folder: push_dataset.sh <slug> <folder> [message]
set -euo pipefail
USER_NAME="$(kaggle config view | sed -n 's/^- username: //p')"
SLUG="$1" FOLDER="$2"
cat > "$FOLDER/dataset-metadata.json" <<JSON
{"title": "$SLUG", "id": "$USER_NAME/$SLUG", "licenses": [{"name": "CC0-1.0"}]}
JSON
if kaggle datasets status "$USER_NAME/$SLUG" >/dev/null 2>&1; then
  kaggle datasets version -p "$FOLDER" -m "${3:-update}" --dir-mode zip -q
else
  kaggle datasets create -p "$FOLDER" --dir-mode zip -q
fi
