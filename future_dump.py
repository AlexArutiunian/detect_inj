#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export engineered gait features from a directory of .npy files into a single table.

- Dynamically imports `load_schema` and `extract_features` from your feature script.
- Reads files either from a manifest CSV (column 'filename') or by scanning the folder.
- Saves to CSV and/or Parquet + writes a small meta.json with run parameters.
- Parallel processing supported.

Example:
python export_features.py \
  --feature_script path/to/your_train_xgb.py \
  --schema path/to/schema_joints.json \
  --data_dir /path/to/npy \
  --out_csv /path/to/features.csv \
  --out_parquet /path/to/features.parquet \
  --fps 30 --stride 2 --fast_axes --workers 4 --turn_curv_thr 0.15
"""

import os
import json
import argparse
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.util

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)


def _import_feature_module(py_path: str):
    """Dynamically import the user's script that defines `load_schema` and `extract_features`."""
    py_path = os.path.abspath(py_path)
    assert os.path.exists(py_path), f"feature_script not found: {py_path}"
    spec = importlib.util.spec_from_file_location("featmod", py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    # sanity checks
    assert hasattr(mod, "load_schema"), "feature_script must define load_schema(schema_json)->Schema"
    assert hasattr(mod, "extract_features"), "feature_script must define extract_features(path, schema, ...)->pd.DataFrame"
    return mod


def _list_npy_files(data_dir: str) -> list[str]:
    exts = (".npy",)
    files = []
    for p in Path(data_dir).rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(str(p))
    files.sort()
    return files


def _files_from_manifest(manifest_csv: str, data_dir: str) -> list[str]:
    df = pd.read_csv(manifest_csv)
    assert "filename" in df.columns, "manifest must have column 'filename'"
    files = []
    for fn in df["filename"].astype(str).tolist():
        p = os.path.join(data_dir, fn)
        if not p.endswith(".npy"):
            p = p + ".npy"
        if os.path.exists(p):
            files.append(p)
    files.sort()
    return files


def _process_one(args):
    """Worker: run extract_features on a single .npy and return dict of features + 'file' key."""
    (
        npy_path,
        schema_obj,
        fps,
        stride,
        fast_axes_flag,
        turn_curv_thr,
        feature_mod_path,
    ) = args

    # import module inside each process (safe for multiprocessing on all OS)
    mod = _import_feature_module(feature_mod_path)
    feats_df = mod.extract_features(
        npy_path,
        schema_obj,
        fps=fps,
        stride=stride,
        fast_axes_flag=fast_axes_flag,
        turn_curv_thr=turn_curv_thr,
    )
    # ensure single row
    if len(feats_df) != 1:
        feats_df = feats_df.iloc[:1]
    row = feats_df.iloc[0].to_dict()
    row = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in row.items()}
    row["file"] = npy_path
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_script", required=True, help="Path to your feature script (the one that contains load_schema & extract_features)")
    ap.add_argument("--schema", required=True, help="Path to schema JSON")
    ap.add_argument("--data_dir", required=True, help="Folder with .npy files")
    ap.add_argument("--manifest", help="Optional manifest.csv with column 'filename'")
    ap.add_argument("--out_csv", help="Where to save CSV with features (optional)")
    ap.add_argument("--out_parquet", help="Where to save Parquet with features (optional)")
    ap.add_argument("--meta_json", help="Where to save meta.json (optional)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--fast_axes", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--turn_curv_thr", type=float, default=0.15)
    ap.add_argument("--fillna", type=float, default=None, help="If set, fill NaNs with this value (e.g., 0.0)")

    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out_csv) if args.out_csv else ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_parquet) if args.out_parquet else ".", exist_ok=True)
    if args.meta_json:
        os.makedirs(os.path.dirname(args.meta_json), exist_ok=True)

    # import feature module & load schema once in the parent
    mod = _import_feature_module(args.feature_script)
    schema = mod.load_schema(args.schema)

    # pick files
    if args.manifest:
        files = _files_from_manifest(args.manifest, args.data_dir)
    else:
        files = _list_npy_files(args.data_dir)

    if not files:
        raise SystemExit("No .npy files found")

    print(f"[export] files: {len(files)} | workers={args.workers} | stride={args.stride} | fast_axes={args.fast_axes} | turn_thr={args.turn_curv_thr}")

    rows = []
    skipped = 0

    if args.workers <= 1:
        for p in tqdm(files, desc="Extract"):
            try:
                row = _process_one(
                    (
                        p,
                        schema,
                        args.fps,
                        args.stride,
                        args.fast_axes,
                        args.turn_curv_thr,
                        args.feature_script,
                    )
                )
                rows.append(row)
            except Exception as e:
                skipped += 1
                tqdm.write(f"[skip] {p} -> {type(e).__name__}: {e}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [
                ex.submit(
                    _process_one,
                    (
                        p,
                        schema,
                        args.fps,
                        args.stride,
                        args.fast_axes,
                        args.turn_curv_thr,
                        args.feature_script,
                    ),
                )
                for p in files
            ]
            for f in tqdm(as_completed(futs), total=len(futs), desc="Extract (parallel)"):
                try:
                    rows.append(f.result())
                except Exception as e:
                    skipped += 1
                    tqdm.write(f"[skip] -> {type(e).__name__}: {e}")

    if not rows:
        raise SystemExit("No rows extracted (all failed).")

    df = pd.DataFrame(rows)

    # put 'file' first
    cols = ["file"] + [c for c in df.columns if c != "file"]
    df = df[cols]

    # (optional) fill NaNs for downstream training
    if args.fillna is not None:
        df = df.fillna(args.fillna)

    # Save outputs
    if args.out_csv:
        df.to_csv(args.out_csv, index=False)
        print(f"[export] saved CSV: {args.out_csv} shape={df.shape}")

    if args.out_parquet:
        df.to_parquet(args.out_parquet, index=False)
        print(f"[export] saved Parquet: {args.out_parquet} shape={df.shape}")

    # Meta
    meta = {
        "feature_script": os.path.abspath(args.feature_script),
        "schema": os.path.abspath(args.schema),
        "data_dir": os.path.abspath(args.data_dir),
        "manifest": os.path.abspath(args.manifest) if args.manifest else None,
        "n_files": len(files),
        "n_rows": int(df.shape[0]),
        "n_features": int(df.shape[1] - 1),  # minus 'file'
        "fps": args.fps,
        "stride": args.stride,
        "fast_axes": bool(args.fast_axes),
        "turn_curv_thr": float(args.turn_curv_thr),
        "fillna": args.fillna,
        "skipped": int(skipped),
        "columns": [c for c in df.columns if c != "file"],
    }
    meta_path = args.meta_json or (
        (Path(args.out_parquet).with_suffix(".meta.json") if args.out_parquet
         else Path(args.out_csv).with_suffix(".meta.json"))
        if (args.out_csv or args.out_parquet) else "features.meta.json"
    )
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[export] saved meta: {meta_path}")
    print(f"[summary] ready: rows={df.shape[0]} cols={df.shape[1]} (incl. 'file'); skipped={skipped}")
    

if __name__ == "__main__":
    main()
