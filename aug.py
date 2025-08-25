#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

def map_label(v):
    s = str(v).strip().lower()
    if s in ("0","no injury","no inj","no","n","false","f","0.0"): return 0
    if s in ("1","injury","inj","yes","y","true","t","1.0"): return 1
    try:
        f = float(s); 
        if f in (0.0, 1.0): return int(f)
    except: pass
    return None

def choose_label_col(df: pd.DataFrame, user_col: str|None):
    if user_col and user_col in df.columns: return user_col
    for c in df.columns:
        cl = c.strip().lower()
        if cl == "label" or "inj" in cl:  # покроет "No inj/ inj"
            return c
    raise SystemExit("[err] не нашёл колонку с меткой — укажите --label_col")

def choose_fname_col(df: pd.DataFrame, user_col: str|None):
    if user_col and user_col in df.columns: return user_col
    for c in ("filename","file","path","basename","stem"):
        if c in df.columns: return c
    raise SystemExit("[err] не нашёл колонку с именем файла — укажите --fname_col")

def build_ci_index(data_dir: Path):
    """Карта: lower(stem) -> абсолютный путь, чтобы игнорировать регистр ('T' vs 't')."""
    idx = {}
    for p in data_dir.rglob("*.npy"):
        idx[p.stem.lower()] = p
    return idx

def split_and_save(A: np.ndarray, k: int = 3):
    T = A.shape[0]
    # индексы ~равных кусков
    cuts = [round(i*T/k) for i in range(k+1)]
    chunks = []
    for i in range(k):
        a, b = cuts[i], cuts[i+1]
        if b > a:
            chunks.append(A[a:b])
    return chunks

def main():
    ap = argparse.ArgumentParser(description="Разрезать длинные (T>min_frames) .npy с label=0 на 3 части и пересохранить, собрать новый CSV")
    ap.add_argument("--data_dir", required=True, help="Где лежат исходные .npy (рекурсивно)")
    ap.add_argument("--run_csv",  required=True, help="Оригинальный run_data.csv")
    ap.add_argument("--out_dir",  default="/kaggle/working", help="Куда класть новые .npy и новый CSV")
    ap.add_argument("--label_col", default=None, help="Имя колонки метки (если не 'label' / 'No inj/ inj')")
    ap.add_argument("--fname_col",  default=None, help="Имя колонки файла (если не 'filename')")
    ap.add_argument("--only_label", type=int, default=0, help="Какой класс обрабатывать (по умолчанию 0)")
    ap.add_argument("--min_frames", type=int, default=9000, help="Порог T, выше которого режем на 3")
    ap.add_argument("--chunks", type=int, default=3, help="Сколько кусков делать")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_split = out_dir / "npy_split"
    out_split.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.run_csv)
    label_col = choose_label_col(df, args.label_col)
    fname_col = choose_fname_col(df, args.fname_col)

    # нормализуем метки в отдельную колонку для логики
    df["_label01"] = df[label_col].map(map_label)
    if df["_label01"].isna().all():
        raise SystemExit("[err] не удалось привести метки к 0/1")

    # индекс файлов по stem без регистра
    idx = build_ci_index(data_dir)
    if not idx:
        raise SystemExit(f"[err] в {data_dir} не найдено .npy")

    new_rows = []
    skipped = 0

    for row in tqdm(df.itertuples(index=False), total=len(df), desc="process"):
        r = dict(zip(df.columns, row))
        lab = r["_label01"]

        # берём stem из значения колонки файла (может быть путь или просто имя)
        raw = str(r[fname_col])
        stem = Path(raw).stem.lower()

        # если не нашлось — оставляем строку как есть
        path = idx.get(stem, None)

        if (lab == args.only_label) and (path is not None):
            # пробуем загрузить и проверить длину
            try:
                A = np.load(path, allow_pickle=False)
                T = A.shape[0]
            except Exception as e:
                skipped += 1
                new_rows.append(r)
                continue

            if T > args.min_frames:
                # режем и сохраняем
                chunks = split_and_save(A, k=args.chunks)
                for i, chunk in enumerate(chunks, 1):
                    new_name = f"{Path(path).stem}_{i}.npy"
                    out_path = out_split / new_name
                    np.save(out_path, chunk)
                    rr = r.copy()
                    # в CSV пишем полный путь нового файла, чтобы обучение точно нашло
                    rr[fname_col] = str(out_path.resolve())
                    new_rows.append(rr)
                continue  # оригинал НЕ добавляем (замена)
        
        # дефолт: оставляем строку без изменений
        new_rows.append(r)

    new_df = pd.DataFrame(new_rows)
    # подчистим служебную колонку
    new_df.drop(columns=["_label01"], inplace=True, errors="ignore")

    out_csv = out_dir / "run_data_split.csv"
    new_df.to_csv(out_csv, index=False)
    print(f"[done] saved CSV: {out_csv}  | rows={len(new_df)}")
    print(f"[info] new npy saved to: {out_split}")
    if skipped:
        print(f"[warn] пропущено файлов из-за ошибок чтения: {skipped}")

if __name__ == "__main__":
    main()
