#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Минималистичный экстрактор фич из .npy

Что делает:
- Рекурсивно обходит папку с .npy
- Поддерживает формы (T, N, 3) и (T, 3*N)
- (опц.) schema_joints.json: список имён суставов длины N
- Считает базовые статистики по каждой координате каждого сустава:
  mean, std, min, max, ptp (max-min)
- Пишет один общий CSV: file, basename, stem, n_frames, n_joints, ...фичи...

Зависимости: numpy, pandas
Пример:
  python mini_feats.py --data_dir /path/to/npy --out_csv /path/to/features.csv \
                       --schema /path/to/schema_joints.json
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


def load_schema(schema_path: Optional[str], n_joints_detected: int) -> List[str]:
    """
    Если передали schema_joints.json (список имён длиной N) — используем.
    Иначе генерируем имена j0..j{N-1}.
    """
    if schema_path is None:
        return [f"j{i}" for i in range(n_joints_detected)]
    names = json.load(open(schema_path, "r", encoding="utf-8"))
    if not isinstance(names, list) or len(names) != n_joints_detected:
        raise ValueError(f"Schema length ({len(names)}) != detected joints ({n_joints_detected})")
    # нормализуем пробелы/регистр минимально
    return [str(x).strip() if str(x).strip() else f"j{i}" for i, x in enumerate(names)]


def load_npy_to_TxNx3(path: str, schema_names_if_any: Optional[List[str]] = None) -> np.ndarray:
    """
    Загружает .npy и приводит к форме (T, N, 3).
    Если вход (T, 3*N) — N берём либо из schema (если дана), либо из деления на 3.
    """
    A = np.load(path, allow_pickle=False)
    if A.ndim == 3 and A.shape[2] == 3:
        # (T, N, 3)
        pass
    elif A.ndim == 2 and A.shape[1] % 3 == 0:
        N = len(schema_names_if_any) if schema_names_if_any is not None else (A.shape[1] // 3)
        if 3 * N != A.shape[1]:
            raise ValueError(f"Width {A.shape[1]} не делится на 3*N (N={N}). Уточните --schema.")
        A = A.reshape(A.shape[0], N, 3)
    else:
        raise ValueError(f"Unexpected array shape {A.shape}. Ожидалось (T, N, 3) или (T, 3*N).")
    # чистим NaN/Inf
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    return A


def basic_stats(x: np.ndarray):
    """
    Возвращает mean, std, min, max, ptp для 1D-вектора.
    """
    return (
        float(np.mean(x)),
        float(np.std(x)),
        float(np.min(x)),
        float(np.max(x)),
        float(np.ptp(x)),
    )


def extract_basic_features(A: np.ndarray, joint_names: List[str]) -> dict:
    """
    A: (T, N, 3), joint_names: длины N
    Возвращает словарь фич:
      { "<joint>_<axis>_<stat>": value, ... }
    где axis ∈ {x,y,z}, stat ∈ {mean,std,min,max,ptp}
    + глобальные: n_frames, n_joints
    """
    T, N, _ = A.shape
    out = {"n_frames": int(T), "n_joints": int(N)}
    axes = ["x", "y", "z"]

    for j in range(N):
        name = joint_names[j]
        for ax_i, ax in enumerate(axes):
            v = A[:, j, ax_i]
            m, s, mn, mx, r = basic_stats(v)
            out[f"{name}_{ax}_mean"] = m
            out[f"{name}_{ax}_std"]  = s
            out[f"{name}_{ax}_min"]  = mn
            out[f"{name}_{ax}_max"]  = mx
            out[f"{name}_{ax}_ptp"]  = r

    return out


def main():
    parser = argparse.ArgumentParser(description="Минимальный экстрактор фич из .npy")
    parser.add_argument("--data_dir", required=True, help="Папка с .npy (обход рекурсивный)")
    parser.add_argument("--out_csv",  required=True, help="Куда сохранить общий CSV")
    parser.add_argument("--schema",   default=None,  help="schema_joints.json (опционально)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_csv  = Path(args.out_csv)
    schema_path = args.schema

    files = sorted(data_dir.rglob("*.npy"))
    if not files:
        raise SystemExit(f"[err] В {data_dir} не найдено .npy")

    rows = []
    for p in files:
        try:
            # предварительно попробуем понять N для schema (если она указана)
            if schema_path is not None:
                # чтобы проверить длину, надо сначала загрузить как есть
                A_tmp = np.load(p, allow_pickle=False)
                if A_tmp.ndim == 3 and A_tmp.shape[2] == 3:
                    N_detect = A_tmp.shape[1]
                elif A_tmp.ndim == 2 and A_tmp.shape[1] % 3 == 0:
                    N_detect = A_tmp.shape[1] // 3
                else:
                    raise ValueError(f"Unexpected shape {A_tmp.shape}")

                names = load_schema(schema_path, N_detect)
                A = load_npy_to_TxNx3(str(p), names)
            else:
                # без schema
                A_tmp = np.load(p, allow_pickle=False)
                if A_tmp.ndim == 3 and A_tmp.shape[2] == 3:
                    N_detect = A_tmp.shape[1]
                    names = [f"j{i}" for i in range(N_detect)]
                elif A_tmp.ndim == 2 and A_tmp.shape[1] % 3 == 0:
                    N_detect = A_tmp.shape[1] // 3
                    names = [f"j{i}" for i in range(N_detect)]
                else:
                    raise ValueError(f"Unexpected shape {A_tmp.shape}")
                A = load_npy_to_TxNx3(str(p), names)

            feats = extract_basic_features(A, names)
            feats.update({
                "file": str(p.resolve()),
                "basename": p.name,
                "stem": p.stem,
            })
            rows.append(feats)

        except Exception as e:
            # максимально тихо: просто пропускаем битый файл
            print(f"[skip] {p}: {type(e).__name__}: {e}")

    if not rows:
        raise SystemExit("[err] Не удалось извлечь ни одной строки фичей.")

    # собираем датафрейм, ставим мета-колонки первыми
    df = pd.DataFrame(rows)
    cols_first = [c for c in ["file", "basename", "stem", "n_frames", "n_joints"] if c in df.columns]
    other = [c for c in df.columns if c not in cols_first]
    df = df[cols_first + other]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[done] Saved: {out_csv}  |  rows={len(df)}  cols={len(df.columns)}")


if __name__ == "__main__":
    main()
