#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, argparse, shutil
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

def map_label(v):
    s = str(v).strip()
    if s in ("0","no injury","no inj","no","n","false","f","0.0"): return 0
    if s in ("1","injury","inj","yes","y","true","t","1.0"): return 1
    try:
        f = float(s); 
        if f in (0.0, 1.0): return int(f)
    except: pass
    return None

def choose_col(df: pd.DataFrame, preferred: str|None, pool: tuple[str,...], err: str) -> str:
    if preferred and preferred in df.columns: return preferred
    for c in pool:
        if c in df.columns: return c
    raise SystemExit(err)

def build_ci_index(data_dir: Path) -> dict[str, Path]:
    """lower(stem) -> Path, регистронезависимое сопоставление."""
    idx = {}
    for p in data_dir.rglob("*.npy"):
        idx[p.stem] = p
    return idx

def load_len(npy_path: Path) -> int:
    A = np.load(npy_path, allow_pickle=False)
    return A.shape[0]  # и для (T,N,3), и для (T,3N)

def split_chunks(A: np.ndarray, k: int) -> list[np.ndarray]:
    T = A.shape[0]
    cuts = [round(i*T/k) for i in range(k+1)]
    return [A[cuts[i]:cuts[i+1]] for i in range(k) if cuts[i+1] > cuts[i]]

def main():
    ap = argparse.ArgumentParser(description="Копировать все .npy в итоговую директорию; длинные нули (T>min_frames) резать на chunks; CSV с именами файлов (без путей)")
    ap.add_argument("--data_dir", required=True, help="Где лежат исходные .npy (рекурсивно)")
    ap.add_argument("--run_csv",  required=True, help="Оригинальный run_data.csv")
    ap.add_argument("--out_dir",  default="/kaggle/working", help="Итоговая директория")
    ap.add_argument("--out_npy_subdir", default="npy_all", help="Подпапка с итоговыми .npy внутри out_dir")
    ap.add_argument("--label_col", default=None, help="Имя колонки метки (если нестандартное)")
    ap.add_argument("--fname_col",  default=None, help="Имя колонки файла (если нестандартное)")
    ap.add_argument("--only_label", type=int, default=0, help="Какой класс резать (по умолчанию 0)")
    ap.add_argument("--min_frames", type=int, default=9000, help="Порог длины T")
    ap.add_argument("--chunks", type=int, default=3, help="Сколько частей делать для длинных рядов")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_npy  = out_dir / args.out_npy_subdir
    out_npy.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.run_csv)
    label_col = choose_col(df, args.label_col,
                           ("label","No inj/ inj","injury","target","y","class"),
                           "[err] не нашёл колонку метки — укажите --label_col")
    fname_col = choose_col(df, args.fname_col,
                           ("filename","file","path","basename","stem"),
                           "[err] не нашёл колонку имени файла — укажите --fname_col")

    df["_label01"] = df[label_col].map(map_label)
    if df["_label01"].isna().all():
        raise SystemExit("[err] не удалось привести метки к 0/1")

    # регистронезависимый индекс исходных npy
    idx = build_ci_index(data_dir)
    if not idx:
        raise SystemExit(f"[err] в {data_dir} не найдено .npy")

    # чтобы не копировать один и тот же файл многократно
    copied = set()
    new_rows = []
    miss = 0
    split_cnt = 0
    copy_cnt = 0

    for row in tqdm(df.itertuples(index=False), total=len(df), desc="process"):
        r = dict(zip(df.columns, row))
        lab = r["_label01"]
        stem = Path(str(r[fname_col])).stem
        src = idx.get(stem)
        if src is None:
            miss += 1
            # как есть, только имя файла оставить (без путей)
            r[fname_col] = Path(str(r[fname_col])).name
            new_rows.append(r)
            continue

        try:
            T = load_len(src)
        except Exception:
            # если не читается — просто копируем как есть
            T = -1

        if (lab == args.only_label) and (T > args.min_frames > 0):
            # режем и сохраняем части; в CSV — заменяем одну строку тремя
            A = np.load(src, allow_pickle=False)
            chunks = split_chunks(A, args.chunks)
            base = src.stem
            for i, ch in enumerate(chunks, 1):
                out_name = f"{base}_{i}.npy"
                out_path = out_npy / out_name
                if not out_path.exists():
                    np.save(out_path, ch)
                rr = r.copy()
                rr[fname_col] = out_name         # ← только имя файла
                new_rows.append(rr)
                split_cnt += 1
        else:
            # просто копируем исходный файл в итоговую директорию, один раз
            out_name = src.name
            out_path = out_npy / out_name
            if out_name not in copied:
                if not out_path.exists():
                    shutil.copy2(src, out_path)
                copied.add(out_name)
                copy_cnt += 1
            r[fname_col] = out_name             # ← только имя файла
            new_rows.append(r)

    new_df = pd.DataFrame(new_rows)
    new_df.drop(columns=["_label01"], inplace=True, errors="ignore")

    out_csv = out_dir / "run_data_split.csv"
    new_df.to_csv(out_csv, index=False)

    print(f"\n[done] CSV: {out_csv}  | rows={len(new_df)}")
    print(f"[info] npy saved to: {out_npy}")
    print(f"[info] copied originals: {copy_cnt}  | generated chunks: {split_cnt}")
    if miss:
        print(f"[warn] не найдено файлов по CSV (регистр/имя): {miss}")

if __name__ == "__main__":
    main()
