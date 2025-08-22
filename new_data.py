#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, argparse, shutil, random, csv
from typing import List, Optional, Tuple
import pandas as pd

# -------- utils --------

def possible_npy_paths(data_dir: str, rel: str) -> List[str]:
    rel = (rel or "").replace("\\", "/").lstrip("/")
    base = os.path.basename(rel)
    cands = []

    def add(p: str):
        if p and p not in cands:
            cands.append(p)

    add(os.path.join(data_dir, rel))
    if rel.endswith(".json"):
        add(os.path.join(data_dir, rel[:-5] + ".npy"))
        add(os.path.join(data_dir, base[:-5] + ".npy"))
    if not rel.endswith(".npy"):
        add(os.path.join(data_dir, rel + ".npy"))
        add(os.path.join(data_dir, base + ".npy"))
    if rel.endswith(".json.npy"):
        add(os.path.join(data_dir, rel))
        add(os.path.join(data_dir, rel.replace(".json.npy", ".npy")))
        add(os.path.join(data_dir, base.replace(".json.npy", ".npy")))
    add(os.path.join(data_dir, base))
    if not base.endswith(".npy"):
        add(os.path.join(data_dir, base + ".npy"))
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
    df = pd.read_csv(csv_path)
    cols_map = {c.lower().strip(): c for c in df.columns}
    filename_col = cols_map.get(filename_col.lower(), filename_col)
    label_col = cols_map.get(label_col.lower(), label_col)

    if filename_col not in df.columns:
        for c in df.columns:
            lc = c.lower()
            if "file" in lc or "name" in lc:
                filename_col = c
                break
    if label_col not in df.columns:
        raise ValueError(f"Не найдена колонка '{label_col}' в CSV.")

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
            continue
    return inj, non, missing

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def copy_with_progress(files: List[str], dst_root: str, label: str, selected: bool, dry_run: bool, manifest_rows: list, desc: str):
    try:
        from tqdm import tqdm
        iterator = tqdm(files, desc=desc, unit="file")
    except Exception:
        iterator = files  # без tqdm

    for src in iterator:
        dst = os.path.join(dst_root, os.path.basename(src))
        if dry_run:
            print(f"[dry] {src} -> {dst}")
        else:
            shutil.copy2(src, dst)
        manifest_rows.append({"src": src, "dst": dst, "label": label, "selected": selected})

# -------- main --------

def main():
    ap = argparse.ArgumentParser("Copy subset of NPY files with progress (50% injury, 90% no-injury)")
    ap.add_argument("--data_dir", required=True, help="Корневая папка с данными (*.npy)")
    ap.add_argument("--csv", required=True, help="CSV с метаданными")
    ap.add_argument("--filename_col", default="filename", help="Колонка путей в CSV")
    ap.add_argument("--label_col", default="No inj/ inj", help="Колонка метки класса")
    ap.add_argument("--injury_value", default="Injury", help="Значение класса Injury")
    ap.add_argument("--noinj_value", default="No Injury", help="Значение класса No Injury")
    ap.add_argument("--injury_frac", type=float, default=0.5, help="Доля файлов Injury (0..1)")
    ap.add_argument("--noinj_frac", type=float, default=0.9, help="Доля файлов No Injury (0..1)")
    ap.add_argument("--out_dir", required=True, help="Куда копировать выбранные")
    ap.add_argument("--reject_dir", default=None, help="Куда класть НЕ выбранные (по умолчанию: <out_dir>/rejected)")
    ap.add_argument("--seed", type=int, default=42, help="Сид для рандома")
    ap.add_argument("--dry_run", action="store_true", help="Только показать, что было бы скопировано")
    args = ap.parse_args()

    random.seed(args.seed)

    injury_all, noinju_all, missing = load_split_from_csv(
        args.csv, args.data_dir, args.filename_col, args.label_col,
        args.injury_value, args.noinj_value
    )
    if missing:
        print(f"[warn] В CSV нет на диске: {missing}", file=sys.stderr)

    n_inj, n_non = len(injury_all), len(noinju_all)
    k_inj = int(round(args.injury_frac * n_inj))
    k_non = int(round(args.noinj_frac * n_non))

    injury_sel = random.sample(injury_all, k_inj) if k_inj < n_inj else list(injury_all)
    noinju_sel = random.sample(noinju_all, k_non) if k_non < n_non else list(noinju_all)
    injury_rej = sorted(set(injury_all) - set(injury_sel))
    noinju_rej = sorted(set(noinju_all) - set(noinju_sel))

    print(f"[info] Injury: всего={n_inj}, берём={len(injury_sel)}, отклоняем={len(injury_rej)}")
    print(f"[info] No Injury: всего={n_non}, берём={len(noinju_sel)}, отклоняем={len(noinju_rej)}")

    # directories
    dst_sel = os.path.join(args.out_dir, "selected")
    dst_rej = args.reject_dir or os.path.join(args.out_dir, "rejected")
    if not args.dry_run:
        ensure_dir(dst_sel)
        ensure_dir(dst_rej)

    manifest_rows = []
    copy_with_progress(injury_sel, dst_sel, "Injury", True, args.dry_run, manifest_rows, "Copy selected Injury")
    copy_with_progress(noinju_sel, dst_sel, "No Injury", True, args.dry_run, manifest_rows, "Copy selected NoInjury")
    copy_with_progress(injury_rej, dst_rej, "Injury", False, args.dry_run, manifest_rows, "Copy rejected Injury")
    copy_with_progress(noinju_rej, dst_rej, "No Injury", False, args.dry_run, manifest_rows, "Copy rejected NoInjury")

    if not args.dry_run:
        manifest = os.path.join(args.out_dir, "manifest.csv")
        with open(manifest, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["src", "dst", "label", "selected"])
            w.writeheader()
            w.writerows(manifest_rows)
        print(f"[ok] Манифест: {manifest}")

if __name__ == "__main__":
    main()
