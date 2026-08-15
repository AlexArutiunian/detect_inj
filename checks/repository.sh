#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "[1/3] Compile Python files"
python -m py_compile src/*.py legacy/python/*.py

echo "[2/3] Check core repository files"
for path in \
  README.md \
  requirements.txt \
  src/train.py \
  src/predict.py \
  schema_joints.json \
  shema_joints_full_body.json \
  data/metadata/walk_data_meta_upd.csv \
  data/metadata/run_data_meta_upd.csv \
  data/processed/run_npy; do
  if [[ ! -e "$path" ]]; then
    echo "Missing: $path" >&2
    exit 1
  fi
done

echo "[3/3] Check retained artifacts"
for path in \
  artifacts/models \
  artifacts/outputs/run \
  artifacts/outputs/walk \
  docs/README.pdf; do
  if [[ ! -e "$path" ]]; then
    echo "Missing artifact: $path" >&2
    exit 1
  fi
done

echo "Repository check passed."
