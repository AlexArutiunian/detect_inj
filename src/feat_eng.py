#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Минимальный экстрактор фич из .npy c прогресс-баром и расширенной статистикой.

Считает на каждую координату {x,y,z} каждого сустава:
mean, std, min, max, ptp, median, q25, q75, iqr, mad, skew, kurt, rms, ac1

Сохраняет один CSV: file, basename, stem, n_frames, n_joints, ...фичи...
Зависимости: numpy, pandas, tqdm
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm


def _sanitize_names(names: List[str]) -> List[str]:
    out = []
    for i, s in enumerate(names):
        s = str(s).strip()
        s = s.replace(" ", "_").replace("/", "_")
        out.append(s if s else f"j{i}")
    return out

def load_schema(schema_path: Optional[str], n_joints_detected: int) -> List[str]:
    if schema_path is None:
        return [f"j{i}" for i in range(n_joints_detected)]
    names = json.load(open(schema_path, "r", encoding="utf-8"))
    if not isinstance(names, list) or len(names) != n_joints_detected:
        raise ValueError(f"Schema length ({len(names)}) != detected joints ({n_joints_detected})")
    return _sanitize_names(names)

def load_npy_to_TxNx3(path: str, n_from_schema: Optional[int]) -> np.ndarray:
    A = np.load(path, allow_pickle=False)
    if A.ndim == 3 and A.shape[2] == 3:
        pass
    elif A.ndim == 2 and A.shape[1] % 3 == 0:
        N = n_from_schema if n_from_schema is not None else (A.shape[1] // 3)
        if 3 * N != A.shape[1]:
            raise ValueError(f"Width {A.shape[1]} не делится на 3*N (N={N})")
        A = A.reshape(A.shape[0], N, 3)
    else:
        raise ValueError(f"Unexpected array shape {A.shape}. Ожидалось (T,N,3) или (T,3*N).")
    return np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

def stat_pack(x: np.ndarray) -> dict:
    x = x.astype(np.float32, copy=False)
    n = x.size
    m  = float(np.mean(x))
    sd = float(np.std(x))
    mn = float(np.min(x))
    mx = float(np.max(x))
    pt = float(np.ptp(x))
    med = float(np.median(x))
    q25 = float(np.percentile(x, 25))
    q75 = float(np.percentile(x, 75))
    iqr = float(q75 - q25)
    mad = float(np.median(np.abs(x - med)))
    rms = float(np.sqrt(np.mean(x * x)))
    if sd > 0:
        z = (x - m) / sd
        sk = float(np.mean(z ** 3))
        ku = float(np.mean(z ** 4) - 3.0)
    else:
        sk, ku = 0.0, 0.0
    if n > 1:
        ac1 = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    else:
        ac1 = np.nan
    return {
        "mean": m, "std": sd, "min": mn, "max": mx, "ptp": pt,
        "median": med, "q25": q25, "q75": q75, "iqr": iqr,
        "mad": mad, "skew": sk, "kurt": ku, "rms": rms, "ac1": ac1
    }

def extract_features(A: np.ndarray, joint_names: List[str]) -> dict:
    T, N, _ = A.shape
    out = {"n_frames": int(T), "n_joints": int(N)}
    axes = ["x", "y", "z"]
    for j in range(N):
        name = joint_names[j]
        for ai, ax in enumerate(axes):
            stats = stat_pack(A[:, j, ai])
            for k, v in stats.items():
                out[f"{name}_{ax}_{k}"] = float(v)
    return out

def main():
    ap = argparse.ArgumentParser(description="Экстрактор фич из .npy (расширенная статистика)")
    ap.add_argument("--data_dir", required=True, help="Папка с .npy (рекурсивно)")
    ap.add_argument("--out_csv",  required=True, help="Путь к итоговому CSV")
    ap.add_argument("--schema",   default=None,  help="schema_joints.json (опционально)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_csv  = Path(args.out_csv)
    schema_path = args.schema

    files = sorted(data_dir.rglob("*.npy"))
    if not files:
        raise SystemExit(f"[err] В {data_dir} не найдено .npy")

    rows = []
    for p in tqdm(files, desc="Extract"):  # ← прогресс-бар (одной строкой)
        try:
            A_raw = np.load(p, allow_pickle=False)
            if A_raw.ndim == 3 and A_raw.shape[2] == 3:
                N_detect = A_raw.shape[1]
            elif A_raw.ndim == 2 and A_raw.shape[1] % 3 == 0:
                N_detect = A_raw.shape[1] // 3
            else:
                raise ValueError(f"Unexpected shape {A_raw.shape}")

            names = load_schema(schema_path, N_detect)
            A = load_npy_to_TxNx3(str(p), len(names))
            feats = extract_features(A, names)
            feats.update({"file": str(p.resolve()), "basename": p.name, "stem": p.stem})
            rows.append(feats)
        except Exception as e:
            print(f"[skip] {p}: {type(e).__name__}: {e}")

    if not rows:
        raise SystemExit("[err] Не удалось извлечь ни одной строки фичей.")

    df = pd.DataFrame(rows)
    first = [c for c in ["file", "basename", "stem", "n_frames", "n_joints"] if c in df.columns]
    df = df[first + [c for c in df.columns if c not in first]]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[done] Saved: {out_csv}  |  rows={len(df)}  cols={len(df.columns)}")

if __name__ == "__main__":
    main()
