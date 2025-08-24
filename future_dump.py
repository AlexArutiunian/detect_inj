#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export engineered gait features from a directory of .npy files into a single table.

- Динамически импортирует `load_schema` и `extract_features` из заданного feature-скрипта.
- Берёт список файлов либо из манифеста (колонка 'filename'), либо рекурсивно сканирует папку.
- Пишет CSV и/или Parquet, а также meta.json с параметрами запуска и списком колонок.
- Параллелизация процессами через initializer (никаких PicklingError).

Пример:
python export_features.py \
  --feature_script detect_inj/train_test2.py \
  --schema detect_inj/schema_joints.json \
  --data_dir 1/npy_run \
  --manifest detect_inj/run_data.csv \
  --out_csv /out/features_all.csv \
  --out_parquet /out/features_all.parquet \
  --fps 30 --stride 2 --fast_axes --workers 4 --turn_curv_thr 0.15 --fillna 0.0
"""

import os
import json
import argparse
import warnings
from pathlib import Path
from typing import List, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.util

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

# -------- Worker globals (инициализируются один раз на процесс) --------
_WORKER: Dict[str, object] = {
    "mod": None,
    "schema": None,
    "fps": None,
    "stride": None,
    "fast_axes": None,
    "turn_thr": None,
}

def _init_worker(feature_script: str, schema_path: str,
                 fps: int, stride: int, fast_axes: bool, turn_thr: float):
    """Initializer вызывается один раз на процесс пула."""
    spec = importlib.util.spec_from_file_location("featmod", os.path.abspath(feature_script))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore

    assert hasattr(mod, "load_schema"), "feature_script должен иметь load_schema(schema_json)->Schema"
    assert hasattr(mod, "extract_features"), "feature_script должен иметь extract_features(path, schema, ...)->pd.DataFrame"

    schema = mod.load_schema(schema_path)

    _WORKER["mod"] = mod
    _WORKER["schema"] = schema
    _WORKER["fps"] = int(fps)
    _WORKER["stride"] = int(stride)
    _WORKER["fast_axes"] = bool(fast_axes)
    _WORKER["turn_thr"] = float(turn_thr)

def _process_one(npy_path: str) -> dict:
    """Запускается в процессе-воркере. Возвращает dict фич + 'file'."""
    mod = _WORKER["mod"]
    schema = _WORKER["schema"]
    fps = _WORKER["fps"]
    stride = _WORKER["stride"]
    fast_axes = _WORKER["fast_axes"]
    turn_thr = _WORKER["turn_thr"]

    feats_df = mod.extract_features(  # type: ignore[attr-defined]
        npy_path,
        schema,
        fps=fps,
        stride=stride,
        fast_axes_flag=fast_axes,
        turn_curv_thr=turn_thr,
    )
    if len(feats_df) != 1:
        feats_df = feats_df.iloc[:1]
    row = feats_df.iloc[0].to_dict()
    # привести numpy-скаляры к Python float/int
    row = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in row.items()}
    row["file"] = npy_path
    return row

# -------------------- Helpers --------------------
def _import_feature_module(py_path: str):
    """Preflight-import в главном процессе, чтобы рано словить ошибки."""
    py_path = os.path.abspath(py_path)
    assert os.path.exists(py_path), f"feature_script not found: {py_path}"
    spec = importlib.util.spec_from_file_location("featmod_preflight", py_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore
    assert hasattr(mod, "load_schema"), "feature_script должен иметь load_schema(schema_json)->Schema"
    assert hasattr(mod, "extract_features"), "feature_script должен иметь extract_features(path, schema, ...)->pd.DataFrame"
    return mod

def _list_npy_files(data_dir: str) -> List[str]:
    files: List[str] = []
    for p in Path(data_dir).rglob("*.npy"):
        if p.is_file():
            files.append(str(p))
    files.sort()
    return files

def _files_from_manifest(manifest_csv: str, data_dir: str) -> List[str]:
    df = pd.read_csv(manifest_csv)
    assert "filename" in df.columns, "manifest must have column 'filename'"
    files: List[str] = []
    for fn in df["filename"].astype(str).tolist():
        p = os.path.join(data_dir, fn)
        if not p.endswith(".npy"):
            p = p + ".npy"
        if os.path.exists(p):
            files.append(p)
    files.sort()
    return files

# -------------------- CLI --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_script", required=True, help="Путь к скрипту с load_schema & extract_features")
    ap.add_argument("--schema", required=True, help="schema_joints.json")
    ap.add_argument("--data_dir", required=True, help="Папка с .npy")
    ap.add_argument("--manifest", help="Опционально: CSV с колонкой 'filename'")
    ap.add_argument("--out_csv", help="Путь для CSV (опционально)")
    ap.add_argument("--out_parquet", help="Путь для Parquet (опционально)")
    ap.add_argument("--meta_json", help="Путь для meta.json (если не задан, будет рядом с CSV/Parquet)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--fast_axes", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--turn_curv_thr", type=float, default=0.15)
    ap.add_argument("--fillna", type=float, default=None, help="Если задано — заполнить NaN этим значением (например, 0.0)")
    args = ap.parse_args()

    # Создать папки вывода (если заданы)
    if args.out_csv:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    if args.out_parquet:
        Path(args.out_parquet).parent.mkdir(parents=True, exist_ok=True)
    if args.meta_json:
        Path(args.meta_json).parent.mkdir(parents=True, exist_ok=True)

    # Preflight: проверим модуль и схему (загрузим, но не будем использовать этот экземпляр в воркерах)
    mod = _import_feature_module(args.feature_script)
    # Проверим, что schema читается (создадим и сразу забудем)
    _ = mod.load_schema(args.schema)  # type: ignore[attr-defined]

    # Список файлов
    files = _files_from_manifest(args.manifest, args.data_dir) if args.manifest else _list_npy_files(args.data_dir)
    if not files:
        raise SystemExit("No .npy files found")

    print(f"[export] files: {len(files)} | workers={args.workers} | stride={args.stride} | fast_axes={args.fast_axes} | turn_thr={args.turn_curv_thr}")

    rows: List[dict] = []
    skipped = 0

    if args.workers <= 1:
        # Однопроцессный режим: инициализируем «воркер» в этом же процессе
        _init_worker(args.feature_script, args.schema, args.fps, args.stride, args.fast_axes, args.turn_curv_thr)
        for p in tqdm(files, desc="Extract"):
            try:
                rows.append(_process_one(p))
            except Exception as e:
                skipped += 1
                tqdm.write(f"[skip] {p} -> {type(e).__name__}: {e}")
    else:
        # Процессный пул с initializer
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=(args.feature_script, args.schema, args.fps, args.stride, args.fast_axes, args.turn_curv_thr),
        ) as ex:
            futs = [ex.submit(_process_one, p) for p in files]
            for f in tqdm(as_completed(futs), total=len(futs), desc="Extract (parallel)"):
                try:
                    rows.append(f.result())
                except Exception as e:
                    skipped += 1
                    tqdm.write(f"[skip] -> {type(e).__name__}: {e}")

    if not rows:
        raise SystemExit("No rows extracted (all failed).")

    df = pd.DataFrame(rows)
    # 'file' первой колонкой
    cols = ["file"] + [c for c in df.columns if c != "file"]
    df = df[cols]

    # Заполнить NaN при необходимости
    if args.fillna is not None:
        df = df.fillna(args.fillna)

    # Сохранение
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
        "n_features": int(df.shape[1] - 1),  # без 'file'
        "fps": int(args.fps),
        "stride": int(args.stride),
        "fast_axes": bool(args.fast_axes),
        "turn_curv_thr": float(args.turn_curv_thr),
        "fillna": args.fillna,
        "skipped": int(skipped),
        "columns": [c for c in df.columns if c != "file"],
    }
    meta_path = (
        args.meta_json
        or (Path(args.out_parquet).with_suffix(".meta.json") if args.out_parquet
            else Path(args.out_csv).with_suffix(".meta.json") if args.out_csv
            else Path("features.meta.json"))
    )
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[export] saved meta: {meta_path}")
    print(f"[summary] ready: rows={df.shape[0]} cols={df.shape[1]} (incl. 'file'); skipped={skipped}")

if __name__ == "__main__":
    # Рекоммендация против oversubscription (можно выставить в оболочке):
    # os.environ.setdefault("OMP_NUM_THREADS", "1")
    # os.environ.setdefault("MKL_NUM_THREADS", "1")
    # os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    main()
