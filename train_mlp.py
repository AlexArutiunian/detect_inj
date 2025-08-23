#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end обучение: npy -> фичи -> subject-wise сплиты -> PyTorch-MLP (GPU).
Поддерживается инференс для нового .npy (режим --mode predict).

Ожидается:
- manifest.csv с колонками: filename, No inj/ inj  (метки: "Injury"/"No Injury")
- .npy: форма (T, N*3) или (T, N, 3), где N = len(schema_joints.json)
- schema_joints.json: список имён маркеров (см. пример в вопросе)

Сохраняет в out_dir:
- model.pt, scaler.pkl, features_cols.json, threshold.txt
- metrics.json, curves.png, roc_pr.png
"""
from tqdm import tqdm

import os, json, math, argparse, random, re, warnings
from dataclasses import dataclass
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore", category=UserWarning)
schema_path_global = None
# --------------------- Utils ---------------------
def seed_all(s=42):
    random.seed(s); np.random.seed(s)
    try:
        import torch
        torch.manual_seed(s)
        torch.cuda.manual_seed_all(s)
    except Exception:
        pass

def label_to_int(v):
    if v is None: return None
    s = str(v).strip().lower()
    if s.startswith("inj"): return 1
    if s.startswith("no"): return 0
    if s in ("1","0"): return int(s)
    return None

def guess_subject_id(filename: str) -> str:
    b = Path(filename).stem
    m = re.match(r"(\d{8})", b)   # напр. 20120717...
    return m.group(1) if m else b.split("T")[0][:8]

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

# --------------------- I/O: schema + npy ---------------------
@dataclass
class Schema:
    names: List[str]
    groups: Dict[str, List[int]]

def load_schema(schema_json: str) -> Schema:
    names = json.load(open(schema_json, "r", encoding="utf-8"))
    # группируем индексы по префиксам
    def idxs(prefix):
        return [i for i, n in enumerate(names) if n.lower().startswith(prefix)]
    groups = {
        "pelvis": idxs("pelvis"),
        "L_foot": idxs("l_foot"),
        "L_shank": idxs("l_shank"),
        "L_thigh": idxs("l_thigh"),
        "R_foot": idxs("r_foot"),
        "R_shank": idxs("r_shank"),
        "R_thigh": idxs("r_thigh"),
    }
    # проверим наполнение
    for k, v in groups.items():
        if len(v) == 0:
            print(f"[warn] schema: group '{k}' пуст — некоторые фичи будут недоступны")
    return Schema(names=names, groups=groups)

def load_npy(path: str, schema: Schema) -> np.ndarray:
    arr = np.load(path, allow_pickle=False)
    if arr.ndim == 2 and arr.shape[1] == 3*len(schema.names):
        arr = arr.reshape(arr.shape[0], len(schema.names), 3)
    elif arr.ndim == 3 and arr.shape[2] == 3 and arr.shape[1] == len(schema.names):
        pass
    else:
        raise ValueError(f"Неожиданная форма массива {arr.shape} для N={len(schema.names)}")
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    return arr  # (T, N, 3)

# --------------------- Геометрия/биомех ---------------------
def pca_axis(points: np.ndarray) -> np.ndarray:
    """
    points: (K,3) -> 3D единичный вектор главной оси (через SVD).
    """
    P = points - points.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(P, full_matrices=False)
    v = Vt[0]  # первый ПК (длина -> S[0])
    n = v / (np.linalg.norm(v) + 1e-8)
    return n

def segment_axis_series(seg_xyz: np.ndarray) -> np.ndarray:
    """
    seg_xyz: (T, K, 3). Возвращает (T,3) ось сегмента (с непрерывным знаком).
    """
    T = seg_xyz.shape[0]
    out = np.zeros((T, 3), dtype=np.float32)
    prev = None
    for t in range(T):
        v = pca_axis(seg_xyz[t])
        if prev is not None and np.dot(v, prev) < 0:  # согласованность направления
            v = -v
        out[t] = v; prev = v
    return out

def angle_between(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """покомпонентно угол между векторами u,v формы (T,3) -> (T,) рад."""
    dot = (u*v).sum(axis=1)
    nu = np.linalg.norm(u, axis=1); nv = np.linalg.norm(v, axis=1)
    c = np.clip(dot/(np.maximum(nu*nv, 1e-8)), -1.0, 1.0)
    return np.arccos(c)

def centroid_series(x: np.ndarray) -> np.ndarray:
    """(T, K, 3) -> (T,3)"""
    return x.mean(axis=1)

def center_and_scale(all_xyz: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """центрируем по тазу и масштабируем на усреднённую длину бедро+голень"""
    pelvis = all_xyz.get("pelvis")
    if pelvis is not None:
        c = centroid_series(pelvis)
        for k in all_xyz:
            all_xyz[k] = all_xyz[k] - c[:, None, :]
    # оценим масштаб по среднему расстоянию thigh<->shank
    scale_list = []
    for side in ("L", "R"):
        th = all_xyz.get(f"{side}_thigh"); sh = all_xyz.get(f"{side}_shank")
        if th is not None and sh is not None:
            th_c = centroid_series(th); sh_c = centroid_series(sh)
            d = np.linalg.norm(th_c - sh_c, axis=1)
            scale_list.append(np.median(d[d>0]))
    scale = float(np.median(scale_list)) if scale_list else 1.0
    scale = max(scale, 1e-6)
    for k in all_xyz:
        all_xyz[k] = all_xyz[k] / scale
    return all_xyz

# ---- детектор событий шага (FS/FO) по центроидам стопы ----
def detect_gait_events(foot_centroid: np.ndarray, fps: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """
    Используем вертикальную координату и её производную:
    FS ~ локальный минимум (пятка низко), FO ~ локальный максимум (нога в подьёме).
    Это упрощённая kinematic-версия подходов Zeni (2008) и O'Connor FVA (2007).
    """
    y = foot_centroid[:, 1]
    vy = np.gradient(y)
    try:
        from scipy.signal import find_peaks
    except Exception:
        raise RuntimeError("Нужна scipy: pip install scipy")

    min_dist = max(3, int(0.20 * fps))  # не ближе 0.2с
    inv = -y
    mins, _ = find_peaks(inv, distance=min_dist)
    maxs, _ = find_peaks(y,   distance=min_dist)

    fs = [i for i in mins if i > 1 and vy[i-1] < 0 <= vy[i]]
    fo = [i for i in maxs if i > 1 and vy[i-1] > 0 >= vy[i]]
    return np.asarray(fs, dtype=int), np.asarray(fo, dtype=int)

# --------------------- Фичи из одного .npy ---------------------
def extract_features(path: str, schema: Schema, fps: int = 30, stride: int = 1, fast_axes: bool = False) -> pd.DataFrame:
    """
    Возвращает одну строку с числовыми фичами по треку.
    Поддерживает ускорения:
      - stride: берём каждый s-й кадр (fps_new = fps/stride)
      - fast_axes: без покадрового SVD; оси сегментов как вектор между центроидами соседних сегментов
    """
    A = load_npy(path, schema)              # (T,N,3)
    if stride > 1:
        A = A[::stride]
        fps = max(1, int(round(fps / stride)))

    T = A.shape[0]
    seg = {k: A[:, idx, :] for k, idx in schema.groups.items() if len(idx) > 0}
    if not seg:
        raise ValueError("По схеме не нашлось ни одного сегмента")

    seg = center_and_scale(seg)

    # центроиды сегментов (быстро)
    cogs = {k: centroid_series(v) for k, v in seg.items()}  # (T,3) на сегмент

    if fast_axes:
        # Быстрые оси: направим бедро к голени, голень к стопе, таз к среднему бедра
        axes = {}
        for side in ("L", "R"):
            th = cogs.get(f"{side}_thigh")
            sh = cogs.get(f"{side}_shank")
            ft = cogs.get(f"{side}_foot")
            if th is not None and sh is not None:
                v = sh - th
                axes[f"{side}_thigh"] = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
            if sh is not None and ft is not None:
                v = ft - sh
                axes[f"{side}_shank"] = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
            if ft is not None:
                # ось стопы: проекция направления движения сегмента (разность) или toe-heel недоступны => берём производную
                dv = np.vstack([ft[1:] - ft[:-1], ft[[-1]] - ft[[-2]]])
                axes[f"{side}_foot"] = dv / (np.linalg.norm(dv, axis=1, keepdims=True) + 1e-8)
        # таз: направим от таза к среднему центроиду бедер, если есть
        pel = cogs.get("pelvis")
        thL = cogs.get("L_thigh"); thR = cogs.get("R_thigh")
        if pel is not None and (thL is not None or thR is not None):
            tgt = pel.copy()
            if thL is not None and thR is not None:
                tgt = (thL + thR) / 2.0
            elif thL is not None:
                tgt = thL
            elif thR is not None:
                tgt = thR
            v = tgt - pel
            axes["pelvis"] = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
    else:
        # Оси через покадровый SVD (точнее, но медленнее)
        axes = {k: segment_axis_series(v) for k, v in seg.items()}

    rows = []
    for side in ("L", "R"):
        foot = cogs.get(f"{side}_foot")
        sh   = axes.get(f"{side}_shank")
        th   = axes.get(f"{side}_thigh")
        pel  = axes.get("pelvis")
        if foot is None or sh is None or th is None or pel is None:
            continue

        fs, fo = detect_gait_events(foot, fps=fps)
        if len(fs) < 2:
            fs = np.array([0, T-1], dtype=int)

        knee = angle_between(th, sh)
        hip  = angle_between(pel, th)
        foot_axis = axes.get(f"{side}_foot")
        ankle = angle_between(sh, foot_axis) if foot_axis is not None else np.zeros(T)

        for i in range(len(fs)-1):
            a, b = fs[i], fs[i+1]
            if b <= a+1: 
                continue
            segslice = slice(a, b)
            step_t = (b - a) / fps
            pelvis_y = cogs["pelvis"][:,1]
            vo = float(np.max(pelvis_y[segslice]) - np.min(pelvis_y[segslice]))
            p0 = cogs["pelvis"][a, [0,2]]; p1 = cogs["pelvis"][b, [0,2]]
            step_len = float(np.linalg.norm(p1 - p0))
            fo_in = fo[(fo > a) & (fo < b)]
            stance = float((fo_in[0]-a)/fps) if len(fo_in) else np.nan

            rows.append({
                "side": 0 if side=="L" else 1,   # числовая кодировка, если понадобится
                "step_time": step_t,
                "cadence": 60.0/step_t if step_t > 1e-6 else np.nan,
                "step_len": step_len,
                "pelvis_vert_osc": vo,
                "rom_knee": float(np.max(knee[segslice]) - np.min(knee[segslice])),
                "rom_hip":  float(np.max(hip[segslice])  - np.min(hip[segslice])),
                "rom_ankle": float(np.max(ankle[segslice]) - np.min(ankle[segslice])),
                "stance_time": stance,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        # fallback — одна строка с минимальным набором чисел
        return pd.DataFrame([{
            "step_time_median": np.nan, "step_time_mean": np.nan, "step_time_std": 0.0,
            "cadence_median": np.nan, "cadence_mean": np.nan, "cadence_std": 0.0,
            "step_len_median": 0.0, "step_len_mean": 0.0, "step_len_std": 0.0,
            "pelvis_vert_osc_median": 0.0, "pelvis_vert_osc_mean": 0.0, "pelvis_vert_osc_std": 0.0,
            "rom_knee_median": 0.0, "rom_knee_mean": 0.0, "rom_knee_std": 0.0,
            "rom_hip_median": 0.0,  "rom_hip_mean": 0.0,  "rom_hip_std": 0.0,
            "rom_ankle_median": 0.0, "rom_ankle_mean": 0.0, "rom_ankle_std": 0.0,
            "stance_time_median": np.nan, "stance_time_mean": np.nan, "stance_time_std": 0.0,
            "asym_step_time": np.nan, "asym_step_len": np.nan, "asym_rom_knee": np.nan,
            "asym_rom_hip": np.nan, "asym_rom_ankle": np.nan, "asym_stance_time": np.nan,
            "asym_cadence": np.nan, "asym_pelvis_vert_osc": np.nan,
            "cycles_count": 0
        }])

    # ---- агрегируем ТОЛЬКО ЧИСЛОВЫЕ колонки (без side как категории) ----

    num = df.select_dtypes(include=[np.number]).copy()
    side_col = num.pop("side")


# ---- агрегируем по шагам и плоско раскладываем в одну строку ----
    agg_tbl = num.agg(["median", "mean", "std"]).T  # index: features, columns: stats
    flat = {f"{feat}_{stat}": float(agg_tbl.loc[feat, stat])
            for feat in agg_tbl.index for stat in agg_tbl.columns}
    agg = pd.DataFrame([flat])
    # асимметрии по медианам L/R (если обе стороны есть)
    if (side_col==0).any() and (side_col==1).any():
        def med(side, col):
            v = num.loc[side_col==side, col]
            return float(np.median(v)) if len(v) else np.nan
        for col in ["step_time","step_len","rom_knee","rom_hip","rom_ankle","stance_time","cadence","pelvis_vert_osc"]:
            L = med(0, col); R = med(1, col)
            denom = (abs(L)+abs(R))/2 + 1e-6
            agg[f"asym_{col}"] = abs(L-R)/denom

    agg["cycles_count"] = len(num)
    return agg.reset_index(drop=True)



# --------------------- Dataset build ---------------------

from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def _extract_one(args):
    p, fn, y_val, schema_path, fps, stride, fast_axes = args
    # локальная загрузка схемы в процессе (дешёвая операция)
    schema = load_schema(schema_path)
    feats = extract_features(p, schema, fps=fps, stride=stride, fast_axes=fast_axes)
    # вернём как (features_dict, y, group, file)
    d = feats.iloc[0].to_dict()
    return d, int(y_val), guess_subject_id(fn), p

def build_table(manifest_csv: str, data_dir: str, schema: Schema, fps: int,
                workers: int = 1, stride: int = 1, fast_axes: bool = False
               ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    df = pd.read_csv(manifest_csv)
    assert "filename" in df.columns, "В CSV нужна колонка 'filename'"

    y_col = None
    for c in df.columns:
        if "inj" in c.lower():
            y_col = c; break
    assert y_col is not None, "Не найден столбец с меткой (например 'No inj/ inj')"

    f_idx = df.columns.get_loc("filename")
    y_idx = df.columns.get_loc(y_col)

    tasks = []
    for row in df.itertuples(index=False, name=None):
        fn = str(row[f_idx])
        p = os.path.join(data_dir, fn)
        if not p.endswith(".npy"): p += ".npy"
        if not os.path.exists(p): 
            continue
        y_val = label_to_int(row[y_idx])
        tasks.append((p, fn, y_val,  # данные
                      # передаём путь к schema, чтобы каждый процесс сам загрузил (избегаем больших pickles)
                      schema_path_global, fps, stride, fast_axes))

    if not tasks:
        raise RuntimeError("Нет валидных путей к .npy")

    print(f"[info] запуск извлечения фич: {len(tasks)} файлов | workers={workers} | stride={stride} | fast_axes={fast_axes}")
    X_rows = []; y_list = []; groups = []; files = []
    skipped = 0

    if workers <= 1:
        for t in tqdm(tasks, desc="Extract features"):
            try:
                d, yv, grp, fp = _extract_one(t)
                X_rows.append(d); y_list.append(yv); groups.append(grp); files.append(fp)
            except Exception as e:
                skipped += 1
                tqdm.write(f"[skip] {t[0]} -> {type(e).__name__} {e}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_extract_one, t) for t in tasks]
            for f in tqdm(as_completed(futs), total=len(futs), desc="Extract features (parallel)"):
                try:
                    d, yv, grp, fp = f.result()
                    X_rows.append(d); y_list.append(yv); groups.append(grp); files.append(fp)
                except Exception as e:
                    skipped += 1
                    # tqdm.write печатает, не ломая прогресс-бар
                    tqdm.write(f"[skip] -> {type(e).__name__} {e}")

    X = pd.DataFrame(X_rows).fillna(0.0)
    y = np.asarray(y_list, dtype=np.int64)
    groups = np.asarray(groups)
    files = np.asarray(files)

    ok = np.isin(y, [0,1])
    bad = (~ok).sum()
    if bad:
        print(f"[warn] удалены {bad} строк(и) из-за некорректных меток")
    X = X.loc[ok].reset_index(drop=True)
    y = y[ok]; groups = groups[ok]; files = files[ok]

    X = X.select_dtypes(include=[np.number]).copy()

    print(f"[info] Собрано примеров: {len(X)} (пропущено: {skipped})")
    print(f"[info] Позитивов (Injury): {(y==1).sum()} | Негативов (No Injury): {(y==0).sum()}")
    print(f"[info] Число признаков: {X.shape[1]}")
    return X, y, groups, files


def extract_and_save(manifest_csv: str, data_dir: str, schema_path: str, fps: int, out_dir: str,
                     workers: int = 1, stride: int = 1, fast_axes: bool = False):
    ensure_dir(out_dir)
    # установим глобальный путь для форков
    global schema_path_global
    schema_path_global = schema_path
    schema = load_schema(schema_path)
    X, y, groups, files = build_table(manifest_csv, data_dir, schema, fps=fps,
                                      workers=workers, stride=stride, fast_axes=fast_axes)
 
    # сохраняем
    X.to_parquet(os.path.join(out_dir, "features.parquet"))  # быстрая загрузка и без потерь типов
    np.save(os.path.join(out_dir, "labels.npy"), y)
    np.save(os.path.join(out_dir, "groups.npy"), groups)
    with open(os.path.join(out_dir, "files.txt"), "w", encoding="utf-8") as f:
        for p in files:
            f.write(str(p) + "\n")
    meta = {
        "manifest_csv": manifest_csv,
        "data_dir": data_dir,
        "fps": fps,
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "positives": int((y==1).sum()),
        "negatives": int((y==0).sum()),
        "schema": os.path.abspath(schema_path),
    }
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print(f"[extract] saved to: {out_dir}")
    print(f"[extract] X: {X.shape}, y: {y.shape}, groups: {groups.shape}")
    return out_dir


def load_features_cache(features_path: str, labels_path: str, groups_path: str, files_path: str):
    """Загружает кэш с фичами с диска."""
    X = pd.read_parquet(features_path)
    y = np.load(labels_path)
    groups = np.load(groups_path, allow_pickle=True)
    with open(files_path, "r", encoding="utf-8") as f:
        files = np.array([ln.strip() for ln in f if ln.strip()])
    # безопасность: оставим только числовые колонки
    X = X.select_dtypes(include=[np.number]).copy()
    return X, y, groups, files

# --------------------- Torch model ---------------------
def build_mlp(in_dim: int):
    import torch, torch.nn as nn
    return nn.Sequential(
        nn.Linear(in_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.25),
        nn.Linear(128, 64),     nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.25),
        nn.Linear(64, 1)  # logits
    )

def train_nn(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, files: np.ndarray,
             out_dir: str, epochs=50, batch_size=64, lr=1e-3, weight_decay=1e-4, seed=42):
    import torch, torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve, roc_curve
    import joblib
    import matplotlib.pyplot as plt
    ensure_dir(out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print("Device:", device)

    # --------- SUBJECT-WISE SPLIT: 60/20/20 ----------
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)   # test = 20%
    tr_idx, te_idx = next(gss1.split(X, y, groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)   # dev = 25% от train => 0.8*0.25=0.20
    tr2_idx, dv_idx = next(gss2.split(X.iloc[tr_idx], y[tr_idx], groups[tr_idx]))
    tr_idx = tr_idx[tr2_idx]

    def split_stats(name, idx):
        return {
            "n": int(len(idx)),
            "pos": int((y[idx]==1).sum()),
            "neg": int((y[idx]==0).sum()),
            "subjects": int(len(np.unique(groups[idx])))
        }

    stats = {"train": split_stats("train", tr_idx),
             "dev":   split_stats("dev",   dv_idx),
             "test":  split_stats("test",  te_idx)}
    print("[split] subject-wise 60/20/20")
    for k,v in stats.items():
        print(f"  {k:5s}: n={v['n']}, pos={v['pos']}, neg={v['neg']}, subjects={v['subjects']}")

    # сохраним разбиение (удобно для воспроизводимости/отладки)
    split_df = pd.DataFrame({
        "file": files,
        "group": groups,
        "y": y,
        "split": ["train"]*len(files)
    })
    split_df.loc[dv_idx, "split"] = "dev"
    split_df.loc[te_idx, "split"] = "test"
    split_df.to_csv(os.path.join(out_dir, "split_subjectwise.csv"), index=False)

    # --------- масштабирование по train ----------
    scaler = StandardScaler().fit(X.iloc[tr_idx])
    X_tr = scaler.transform(X.iloc[tr_idx]); X_dv = scaler.transform(X.iloc[dv_idx]); X_te = scaler.transform(X.iloc[te_idx])

    joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))
    json.dump(list(X.columns), open(os.path.join(out_dir, "features_cols.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # --------- DataLoaders ----------
    def mkdl(Xn, yn, bs, shuffle=False):
        tX = torch.tensor(Xn, dtype=torch.float32)
        ty = torch.tensor(yn.reshape(-1,1), dtype=torch.float32)
        ds = TensorDataset(tX, ty)
        return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=0, pin_memory=True)

    dl_tr = mkdl(X_tr, y[tr_idx], batch_size, True)
    dl_dv = mkdl(X_dv, y[dv_idx], batch_size, False)
    dl_te = mkdl(X_te, y[te_idx], batch_size, False)

    # --------- Модель/оптимизатор ----------
    model = build_mlp(X_tr.shape[1]).to(device)
    pos = max(1, int((y[tr_idx]==1).sum()))
    neg = max(1, int((y[tr_idx]==0).sum()))
    pos_weight = torch.tensor([neg/pos], dtype=torch.float32, device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best = {"auprc": -1, "state": None}
    hist = {"loss": [], "val_loss": [], "val_auroc": [], "val_auprc": []}

    # --------- Тренировка с прогрессом по батчам ----------
    for epoch in range(1, epochs+1):
        model.train(); run_loss = 0.0
        pbar = tqdm(dl_tr, total=len(dl_tr), desc=f"Epoch {epoch}/{epochs} [train]", leave=False)
        for bx, by in pbar:
            bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
            opt.zero_grad()
            loss = crit(model(bx), by)
            loss.backward(); opt.step()
            run_loss += float(loss.item())*len(bx)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        run_loss /= len(dl_tr.dataset)

        # val (быстрый прогон, без прогресс-бара чтобы не засорять)
        model.eval()
        with torch.no_grad():
            logits = []
            for bx, _ in dl_dv:
                bx = bx.to(device); logits.append(model(bx).cpu().numpy())
            logits = np.vstack(logits).ravel()
            probs = 1/(1+np.exp(-logits))
            auroc = roc_auc_score(y[dv_idx], probs) if len(np.unique(y[dv_idx]))==2 else float("nan")
            auprc = average_precision_score(y[dv_idx], probs)

        hist["loss"].append(run_loss); hist["val_loss"].append(float("nan"))
        hist["val_auroc"].append(auroc); hist["val_auprc"].append(auprc)
        print(f"[epoch {epoch:03d}] train_loss={run_loss:.4f} | val_AUROC={auroc:.3f} | val_AUPRC={auprc:.3f}")

        if auprc > best["auprc"]:
            best["auprc"] = auprc
            best["state"] = {k:v.cpu() for k,v in model.state_dict().items()}

    # --------- Лучшее состояние, подбор порога, тест ----------
    model.load_state_dict(best["state"])
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))

    model.eval()
    with torch.no_grad():
        dev_logits = np.vstack([model(torch.tensor(X_dv, dtype=torch.float32).to(device)).cpu().numpy()]).ravel()
    from sklearn.metrics import precision_recall_curve
    pr, rc, th = precision_recall_curve(y[dv_idx], 1/(1+np.exp(-dev_logits)))
    f1 = 2*pr[:-1]*rc[:-1]/np.clip(pr[:-1]+rc[:-1], 1e-12, None)
    thr = float(th[np.nanargmax(f1)])
    open(os.path.join(out_dir, "threshold.txt"), "w").write(str(thr))

    with torch.no_grad():
        te_logits = np.vstack([model(torch.tensor(X_te, dtype=torch.float32).to(device)).cpu().numpy()]).ravel()
    te_prob = 1/(1+np.exp(-te_logits))
    y_te = y[te_idx]; te_pred = (te_prob >= thr).astype(int)

    from sklearn.metrics import classification_report, confusion_matrix, roc_curve, average_precision_score
    rep = classification_report(y_te, te_pred, digits=3, output_dict=False)
    cm = confusion_matrix(y_te, te_pred).tolist()
    metrics = {"val_best_auprc": float(best["auprc"]), "test_cm": cm, "test_report": rep}
    json.dump(metrics, open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # графики
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,4))
    plt.plot(hist["loss"], label="train_loss")
    plt.title("Training loss"); plt.xlabel("epoch"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "curves.png"), dpi=140); plt.close()

    try:
        fpr, tpr, _ = roc_curve(y_te, te_prob)
        from sklearn.metrics import auc as _auc
        auroc = _auc(fpr, tpr)
        prc, rec, _ = precision_recall_curve(y_te, te_prob)
        auprc = average_precision_score(y_te, te_prob)
        plt.figure(figsize=(9,4))
        plt.subplot(1,2,1); plt.plot(fpr, tpr); plt.plot([0,1],[0,1],"--"); plt.title(f"ROC AUC={auroc:.3f}")
        plt.subplot(1,2,2); plt.plot(rec, prc); plt.title(f"PR AUC={auprc:.3f}")
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, "roc_pr.png"), dpi=140); plt.close()
    except Exception:
        pass

    print("\n=== TEST ===")
    print(rep)
    print("Saved to:", out_dir)
    return out_dir


# --------------------- Inference ---------------------
def predict_one(npy_path: str, schema: Schema, out_dir: str, fps: int = 30):
    import joblib, torch, torch.nn as nn
    X = extract_features(npy_path, schema, fps=fps)
    cols = json.load(open(os.path.join(out_dir, "features_cols.json"), "r", encoding="utf-8"))
    scaler = joblib.load(os.path.join(out_dir, "scaler.pkl"))
    thr = float(open(os.path.join(out_dir, "threshold.txt")).read().strip())
    model = build_mlp(len(cols))
    state = torch.load(os.path.join(out_dir, "model.pt"), map_location="cpu")
    model.load_state_dict(state); model.eval()

    # порядок колонок как при обучении
    X = X.reindex(columns=cols, fill_value=0.0)
    x = scaler.transform(X.values).astype(np.float32)
    with torch.no_grad():
        logits = model(torch.tensor(x)).numpy().ravel()
        prob = float(1/(1+np.exp(-logits[0])))
        pred = int(prob >= thr)
    return {"file": npy_path, "prob_injury": prob, "pred": pred, "threshold": thr}

# --------------------- CLI ---------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train","predict","extract","train_features"], default="train",
                    help="extract: только посчитать и сохранить фичи; "
                         "train_features: обучение из кэша фич")
    ap.add_argument("--csv", help="manifest.csv (для train/extract)")
    ap.add_argument("--data_dir", help="папка с .npy (для train/extract)")
    ap.add_argument("--schema", required=True, help="schema_joints.json (нужен для extract/train/predict)")
    ap.add_argument("--out_dir", default="out_nn")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--npy", help="путь к .npy (для predict)")
    
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                help="количество процессов для извлечения фич")
    ap.add_argument("--stride", type=int, default=2,  # 60 FPS -> 30 FPS по умолчанию
                    help="прореживание временной оси: берём каждый s-й кадр")
    ap.add_argument("--fast_axes", action="store_true",
                    help="быстрые оси сегментов без SVD (centroid-to-centroid)")


    # пути к кэшу (для train_features)
    ap.add_argument("--features", help="path to features.parquet (для train_features)")
    ap.add_argument("--labels", help="path to labels.npy (для train_features)")
    ap.add_argument("--groups", help="path to groups.npy (для train_features)")
    ap.add_argument("--files_list", help="path to files.txt (для train_features)")

    args = ap.parse_args()
    seed_all(42)
    
    global schema_path_global
    schema_path_global = args.schema


    if args.mode == "extract":
        assert args.csv and args.data_dir, "--csv и --data_dir обязательны в режиме extract"
        extract_and_save(args.csv, args.data_dir, args.schema, fps=args.fps,
                 out_dir=args.out_dir, workers=args.workers, stride=args.stride, fast_axes=args.fast_axes)

        return

    if args.mode == "train":
        assert args.csv and args.data_dir, "--csv и --data_dir обязательны в режиме train"
        schema = load_schema(args.schema)
        X, y, groups, files = build_table(
            args.csv, args.data_dir, schema, fps=args.fps,
            workers=args.workers, stride=args.stride, fast_axes=args.fast_axes
        )

        print(f"[info] samples={len(X)}, positives={(y==1).sum()}, features={X.shape[1]}")
        train_nn(X, y, groups, files, args.out_dir,
                 epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay)
        return

    if args.mode == "train_features":
        # если пути не заданы, берём по умолчанию из out_dir (там где делали extract)
        features = args.features or os.path.join(args.out_dir, "features.parquet")
        labels   = args.labels   or os.path.join(args.out_dir, "labels.npy")
        groups   = args.groups   or os.path.join(args.out_dir, "groups.npy")
        files_l  = args.files_list or os.path.join(args.out_dir, "files.txt")
        X, y, groups, files = load_features_cache(features, labels, groups, files_l)
        print(f"[info] loaded cache: {X.shape}, positives={(y==1).sum()}")
        train_nn(X, y, groups, files, args.out_dir,
                 epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay)
        return

    # predict
    assert args.npy, "--npy обязателен в режиме predict"
    schema = load_schema(args.schema)
    res = predict_one(args.npy, schema, args.out_dir, fps=args.fps)
    print(json.dumps(res, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
