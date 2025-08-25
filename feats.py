#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Экстрактор фич из .npy c расширенными динамическими признаками.

На каждую координату {x,y,z} каждого сустава считаем базовые статфичи:
mean, std, min, max, ptp, median, q25, q75, iqr, mad, skew, kurt, rms, ac1
+ коэффициенты: cov=std/|mean|, iqr_over_range=iqr/(ptp+eps)

Дополнительно:
- magnitude по суставу: ||(x,y,z)|| и его статфичи
- скорость (первая разность) и ускорение (вторая разность) для осей и magnitude
- zero-crossing rate (zcr) для скорости (каждой оси и magnitude)
- межосевые корреляции corr(x,y), corr(y,z), corr(x,z)

Выход: CSV: file, basename, stem, n_frames, n_joints, ...фичи...
Зависимости: numpy, pandas, tqdm
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Optional, Tuple

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

def _basic_stats(x: np.ndarray) -> dict:
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
        c = np.corrcoef(x[:-1], x[1:])
        ac1 = float(c[0, 1]) if np.isfinite(c).all() else 0.0
    else:
        ac1 = 0.0
    cov = float(sd / (abs(m) + 1e-8))
    iqr_over_range = float(iqr / (pt + 1e-8))
    return {
        "mean": m, "std": sd, "min": mn, "max": mx, "ptp": pt,
        "median": med, "q25": q25, "q75": q75, "iqr": iqr,
        "mad": mad, "skew": sk, "kurt": ku, "rms": rms, "ac1": ac1,
        "cov": cov, "iqr_over_range": iqr_over_range
    }

def _zcr(x: np.ndarray) -> float:
    """Zero-crossing rate для времени: доля смен знака."""
    if x.size < 2:
        return 0.0
    s = np.sign(x)
    s[s == 0] = 1  # чтобы нули не портили смену знака
    return float(np.mean(s[1:] != s[:-1]))

def _diff(x: np.ndarray, order: int = 1) -> np.ndarray:
    if order == 1:
        return np.diff(x, n=1)
    elif order == 2:
        return np.diff(x, n=2)
    else:
        raise ValueError("order must be 1 or 2")

def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return 0.0
    sd_a, sd_b = np.std(a), np.std(b)
    if sd_a == 0 or sd_b == 0:
        return 0.0
    c = np.corrcoef(a, b)
    v = float(c[0, 1]) if np.isfinite(c).all() else 0.0
    return v

def _add_stats(out: dict, prefix: str, x: np.ndarray):
    """Добавить базовые статфичи + расширения."""
    st = _basic_stats(x)
    for k, v in st.items():
        out[f"{prefix}_{k}"] = float(v)

def extract_features(A: np.ndarray, joint_names: List[str]) -> dict:
    T, N, _ = A.shape
    out = {"n_frames": int(T), "n_joints": int(N)}
    axes = ["x", "y", "z"]

    for j in range(N):
        name = joint_names[j]
        x = A[:, j, 0]
        y = A[:, j, 1]
        z = A[:, j, 2]

        # Базовые по осям
        _add_stats(out, f"{name}_x", x)
        _add_stats(out, f"{name}_y", y)
        _add_stats(out, f"{name}_z", z)

        # Межосевые корреляции
        out[f"{name}_xy_corr"] = _safe_corr(x, y)
        out[f"{name}_yz_corr"] = _safe_corr(y, z)
        out[f"{name}_xz_corr"] = _safe_corr(x, z)

        # Magnitude положения
        mag = np.sqrt(x*x + y*y + z*z)
        _add_stats(out, f"{name}_mag", mag)

        # Скорость (первая разность) по осям и magnitude
        vx, vy, vz = _diff(x, 1), _diff(y, 1), _diff(z, 1)
        vmag = _diff(mag, 1)

        _add_stats(out, f"{name}_vx", vx)
        _add_stats(out, f"{name}_vy", vy)
        _add_stats(out, f"{name}_vz", vz)
        _add_stats(out, f"{name}_vmag", vmag)

        # ZCR для скорости (показатель «дёрганости»)
        out[f"{name}_vx_zcr"] = _zcr(vx)
        out[f"{name}_vy_zcr"] = _zcr(vy)
        out[f"{name}_vz_zcr"] = _zcr(vz)
        out[f"{name}_vmag_zcr"] = _zcr(vmag)

        # Ускорение (вторая разность) по осям и magnitude
        ax, ay, az = _diff(x, 2), _diff(y, 2), _diff(z, 2)
        amag = _diff(mag, 2)

        _add_stats(out, f"{name}_ax", ax)
        _add_stats(out, f"{name}_ay", ay)
        _add_stats(out, f"{name}_az", az)
        _add_stats(out, f"{name}_amag", amag)

    return out

def main():
    ap = argparse.ArgumentParser(description="Экстрактор фич из .npy (расширенная статистика + динамика)")
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
    for p in tqdm(files, desc="Extract"):
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
