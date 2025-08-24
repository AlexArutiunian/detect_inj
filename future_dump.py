#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export engineered gait features from a directory of .npy files into one table.

- Динамически импортирует `load_schema` и `extract_features` из вашего feature-скрипта.
- Берёт список файлов либо из manifest CSV (колонка 'filename'), либо сканирует папку.
- Сохраняет CSV/Parquet + meta.json.
- Параллельная обработка через ProcessPoolExecutor с инициализацией воркера.
- Жёстко ограничивает потоки BLAS/FFT (MKL/OpenBLAS/OMP/NumExpr), чтобы избежать oversubscription.

Пример:
python export_features.py \
  --feature_script detect_inj/train_test2.py \
  --schema detect_inj/schema_joints.json \
  --data_dir 1/npy_run \
  --manifest detect_inj/run_data.csv \
  --out_csv /out/features_all.csv \
  --out_parquet /out/features_all.parquet \
  --fps 30 --stride 2 --fast_axes \
  --workers 6 --chunksize 16 --turn_curv_thr 0.15 --fillna 0.0
"""

import os
# ограничим потоки BLAS до импорта numpy/scipy (перезапишем из CLI ниже, если надо)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import json
import argparse
import warnings
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.util

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

# --------- worker globals ---------
_worker_state: Dict[str, Any] = {
    "mod": None,
    "schema": None,
    "fps": 30,
    "stride": 1,
    "fast_axes_flag": False,
    "turn_curv_thr": 0.15,
}

def _import_feature_module(py_path: str):
    """Dynamically import the user's script that defines load_schema & extract_features."""
    py_path = os.path.abspath(py_path)
    assert os.path.exists(py_path), f"feature_script not found: {py_path}"
    spec = importlib.util.spec_from_file_location("featmod", py_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore
    assert hasattr(mod, "load_schema"), "feature_script must define load_schema(schema_json)->Schema"
    assert hasattr(mod, "extract_features"), "feature_script must define extract_features(path, schema, ...)->pd.DataFrame"
    return mod

def _init_worker(feature_script: str,
                 schema_path: str,
                 fps: int,
                 stride: int,
                 fast_axes_flag: bool,
                 turn_curv_thr: float,
                 blas_threads: int):
    # продублируем лимиты потоков на всякий случай (инициализация процесса)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(blas_threads)
    mod = _import_feature_module(feature_script)
    schema = mod.load_schema(schema_path)
    _worker_state["mod"] = mod
    _worker_state["schema"] = schema
    _worker_state["fps"] = int(fps)
    _worker_state["stride"] = int(stride)
    _worker_state["fast_axes_flag"] = bool(fast_axes_flag)
    _worker_state["turn_curv_thr"] = float(turn_curv_thr)

def _process_one(npy_path: str) -> Dict[str, Any]:
    """Run extract_features in the worker using initialized globals."""
    mod = _worker_state["mod"]
    schema = _worker_state["schema"]
    feats_df = mod.extract_features(
        npy_path,
        schema,
        fps=_worker_state["fps"],
        stride=_worker_state["stride"],
        fast_axes_flag=_worker_state["fast_axes_flag"],
        turn_curv_thr=_worker_state["turn_curv_thr"],
    )
    if len(feats_df) != 1:
        feats_df = feats_df.iloc[:1]
    row = feats_df.iloc[0].to_dict()
    # нормализуем числа к float
    row = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in row.items()}
    row["file"] = npy_path
    return row

def _files_from_manifest(manifest_csv: str, data_dir: str) -> List[str]:
    df = pd.read_csv(manifest_csv)
    assert "filename" in df.columns, "manifest must have column 'filename'"
    files: List[str] = []
    for fn in df["filename"].astype(str).tolist():
        p = os.path.join(data_dir, fn)
        if not p.endswith(".npy"):
            p += ".npy"
        if os.path.exists(p):
            files.append(p)
    files.sort()
    return files

def _list_npy_files(data_dir: str) -> List[str]:
    files: List[str] = [str(p) for p in Path(data_dir).rglob("*.npy")]
    files.sort()
    return files

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_script", required=True, help="Path to script with load_schema & extract_features")
    ap.add_argument("--schema", required=True, help="Path to schema JSON")
    ap.add_argument("--data_dir", required=True, help="Folder with .npy files")
    ap.add_argument("--manifest", help="Optional manifest.csv with column 'filename'")
    ap.add_argument("--out_csv", help="Output CSV (optional)")
    ap.add_argument("--out_parquet", help="Output Parquet (optional)")
    ap.add_argument("--meta_json", help="Output meta.json (optional)")

    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--fast_axes", action="store_true")
    ap.add_argument("--turn_curv_thr", type=float, default=0.15)

    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--chunksize", type=int, default=16, help="How many files per task chunk for executor.map")
    ap.add_argument("--blas_threads", type=int, default=1, help="BLAS/FFT threads per process (OMP/MKL/OPENBLAS/NUMEXPR)")
    ap.add_argument("--fillna", type=float, default=None, help="Fill NaNs with this value (e.g., 0.0)")

    args = ap.parse_args()

    # подготовим папки
    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    if args.out_parquet:
        os.makedirs(os.path.dirname(args.out_parquet) or ".", exist_ok=True)
    if args.meta_json:
        os.makedirs(os.path.dirname(args.meta_json) or ".", exist_ok=True)

    # применим лимиты потоков BLAS из CLI и покажем
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(args.blas_threads)
    print(f"[env] BLAS threads per process = {args.blas_threads}")

    # какие файлы обрабатываем
    files = _files_from_manifest(args.manifest, args.data_dir) if args.manifest else _list_npy_files(args.data_dir)
    if not files:
        raise SystemExit("No .npy files found")
    print(f"[export] files: {len(files)} | workers={args.workers} | stride={args.stride} | fast_axes={args.fast_axes} | turn_thr={args.turn_curv_thr}")
    print(f"[cfg] chunksize={args.chunksize}")

    rows: List[Dict[str, Any]] = []
    skipped = 0

    if args.workers <= 1:
        # инициализация в главном процессе
        _init_worker(args.feature_script, args.schema, args.fps, args.stride, args.fast_axes, args.turn_curv_thr, args.blas_threads)
        for p in tqdm(files, desc="Extract"):
            try:
                rows.append(_process_one(p))
            except Exception as e:
                skipped += 1
                tqdm.write(f"[skip] {p} -> {type(e).__name__}: {e}")
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=(args.feature_script, args.schema, args.fps, args.stride, args.fast_axes, args.turn_curv_thr, args.blas_threads),
        ) as ex:
            def safe_process(path: str):
                try:
                    return _process_one(path)
                except Exception as e:
                    return ("__ERROR__", path, f"{type(e).__name__}: {e}")

            it = ex.map(safe_process, files, chunksize=max(1, args.chunksize))
            for res in tqdm(it, total=len(files), desc="Extract (parallel)"):
                if isinstance(res, tuple) and res and res[0] == "__ERROR__":
                    skipped += 1
                    _, p, msg = res
                    tqdm.write(f"[skip] {p} -> {msg}")
                else:
                    rows.append(res)

    if not rows:
        raise SystemExit("No rows extracted (all failed).")

    df = pd.DataFrame(rows)
    cols = ["file"] + [c for c in df.columns if c != "file"]
    df = df[cols]

    if args.fillna is not None:
        df = df.fillna(args.fillna)

    if args.out_csv:
        df.to_csv(args.out_csv, index=False)
        print(f"[export] saved CSV: {args.out_csv} shape={df.shape}")

    if args.out_parquet:
        df.to_parquet(args.out_parquet, index=False)
        print(f"[export] saved Parquet: {args.out_parquet} shape={df.shape}")

    meta = {
        "feature_script": os.path.abspath(args.feature_script),
        "schema": os.path.abspath(args.schema),
        "data_dir": os.path.abspath(args.data_dir),
        "manifest": os.path.abspath(args.manifest) if args.manifest else None,
        "n_files": len(files),
        "n_rows": int(df.shape[0]),
        "n_features": int(df.shape[1] - 1),
        "fps": args.fps,
        "stride": args.stride,
        "fast_axes": bool(args.fast_axes),
        "turn_curv_thr": float(args.turn_curv_thr),
        "fillna": args.fillna,
        "workers": args.workers,
        "chunksize": args.chunksize,
        "blas_threads": args.blas_threads,
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
