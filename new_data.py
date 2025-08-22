#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, argparse, shutil, random, csv
from typing import List, Optional, Tuple
import pandas as pd

def possible_npy_paths(data_dir: str, rel: str) -> List[str]:
    """Подбираем разумные кандидаты .npy по строке из CSV (часто .json)."""
    rel = (rel or "").replace("\\", "/").lstrip("/")
    base = os.path.basename(rel)
    cands = []

    def add(p: str):
        if p and p not in cands:
            cands.append(p)

    # 1) Как есть
    add(os.path.join(data_dir, rel))

    # 2) Если .json → .npy
    if rel.endswith(".json"):
        add(os.path.join(data_dir, rel[:-5] + ".npy"))
        add(os.path.join(data_dir, base[:-5] + ".npy"))

    # 3) Если без .npy — добавим
    if not rel.endswith(".npy"):
        add(os.path.join(data_dir, rel + ".npy"))
        add(os.path.join(data_dir, base + ".npy"))

    # 4) json.npy варианты
    if rel.endswith(".json.npy"):
        add(os.path.join(data_dir, rel))  # уже npy
        add(os.path.join(data_dir, rel.replace(".json.npy", ".npy")))
        add(os.path.join(data_dir, base.replace(".json.npy", ".npy")))

    # 5) Просто имя без путей
    add(os.path.join(data_dir, base))
    if not base.endswith(".npy"):
        add(os.path.join(data_dir, base + ".npy"))

    # Уникальные в исходном порядке
    return list(dict.fromkeys(cands))

def find_existing_npy(data_dir: str, rel: str) -> Optional[str]:
    for p in possible_npy_paths(data_dir, rel):
        npy = os.path.normpath(p)
        if os.path.exists(npy) and npy.lower().endswith(".npy"):
            return npy
    return None

def load_split_from_csv(csv_path: str,
                        data_dir: str,
                        filename_col: str,
                        label_col: str,
                        injury_value: str,
                        noinju_value: str) -> Tuple[List[str], List[str], int]:
    """Возвращает списки существующих путей: (injury_paths, noinj_paths, missing_count)."""
    df = pd.read_csv(csv_path)
    # Попробуем найти реальные имена колонок без учёта регистра/пробелов
    cols_map = {c.lower().strip(): c for c in df.columns}
    filename_col = cols_map.get(filename_col.lower(), filename_col)
    label_col = cols_map.get(label_col.lower(), label_col)

    if filename_col not in df.columns:
        # эвристика
        for c in df.columns:
            lc = c.lower()
            if "file" in lc or "name" in lc:
                filename_col = c
                break
    if label_col not in df.columns:
        raise ValueError(f"Не найдена колонка с меткой: '{label_col}' в CSV.")

    inj, non, missing = [], [], 0
    for _, row in df.iterrows():
        rel = str(row[filename_col])
        lbl = str(row[label_col]).strip()
        p = find_existing_npy(data_dir, rel)
        if p is None:
            missing += 1
            continue
        if lbl == injury_value:
            inj.append(p)
        elif lbl == noinju_value:
            non.append(p)
        else:
            # игнорируем прочие значения
            continue
    return inj, non, missing

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def main():
    ap = argparse.ArgumentParser("Copy a stratified subset of NPY files (50% injury, 90% no-injury)")
    ap.add_argument("--data_dir", required=True, help="Корневая папка с данными (*.npy)")
    ap.add_argument("--csv", required=True, help="CSV с метаданными")
    ap.add_argument("--filename_col", default="filename", help="Колонка путей в CSV")
    ap.add_argument("--label_col", default="No inj/ inj", help="Колонка метки класса в CSV")
    ap.add_argument("--injury_value", default="Injury", help="Значение класса Injury в label_col")
    ap.add_argument("--noinj_value", default="No Injury", help="Значение класса No Injury в label_col")
    ap.add_argument("--injury_frac", type=float, default=0.5, help="Доля файлов Injury для копирования (0..1)")
    ap.add_argument("--noinj_frac", type=float, default=0.9, help="Доля файлов No Injury для копирования (0..1)")
    ap.add_argument("--out_dir", required=True, help="Куда копировать")
    ap.add_argument("--seed", type=int, default=42, help="Сид для рандома")
    ap.add_argument("--dry_run", action="store_true", help="Только показать, что было бы скопировано")
    args = ap.parse_args()

    random.seed(args.seed)

    injury_all, noinju_all, missing = load_split_from_csv(
        args.csv, args.data_dir, args.filename_col, args.label_col, args.injury_value, args.noinj_value
    )
    if missing:
        print(f"[warn] В CSV указано файлов, которых нет на диске: {missing}", file=sys.stderr)

    n_inj = len(injury_all)
    n_non = len(noinju_all)
    k_inj = int(round(args.injury_frac * n_inj))
    k_non = int(round(args.noinj_frac * n_non))
    k_inj = max(0, min(n_inj, k_inj))
    k_non = max(0, min(n_non, k_non))

    # Сэмплинг без повторов
    injury_sel = random.sample(injury_all, k_inj) if k_inj < n_inj else list(injury_all)
    noinju_sel = random.sample(noinju_all, k_non) if k_non < n_non else list(noinju_all)

    print(f"[info] Injury: всего={n_inj}, берём={len(injury_sel)} (frac={args.injury_frac})")
    print(f"[info] No Injury: всего={n_non}, берём={len(noinju_sel)} (frac={args.noinj_frac})")

    # Подготовка директорий
    dst_inj = os.path.join(args.out_dir, "injury")
    dst_non = os.path.join(args.out_dir, "no_injury")
    if not args.dry_run:
        ensure_dir(dst_inj); ensure_dir(dst_non)

    copied_rows = []
    def copy_list(files: List[str], dst_root: str, label: str):
        for src in files:
            # кладём «плоско» по basename; если нужны подпапки — можно сохранить структуру
            dst = os.path.join(dst_root, os.path.basename(src))
            if args.dry_run:
                print(f"[dry] {src} -> {dst}")
            else:
                shutil.copy2(src, dst)
            copied_rows.append({"src": src, "dst": dst, "label": label})

    copy_list(injury_sel, dst_inj, "Injury")
    copy_list(noinju_sel, dst_non, "No Injury")

    # Манифест
    if not args.dry_run:
        manifest = os.path.join(args.out_dir, "copied_manifest.csv")
        with open(manifest, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["src", "dst", "label"])
            w.writeheader()
            w.writerows(copied_rows)
        print(f"[ok] Скопировано файлов: {len(copied_rows)}")
        print(f"[ok] Манифест: {manifest}")

if __name__ == "__main__":
    main()
