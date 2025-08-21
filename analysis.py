#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analysis.py — Per-joint importance for classical models (RF/XGB/SVM).

- RF/XGB: feature_importances_
- SVM (linear): |coef_|
- SVM (non-linear, e.g. RBF) or any model w/o importances: permutation importance
  (needs raw data: pass --csv and --data_dir/--npy_dir)

NPY поддержка:
- Принимает (T, F), (T, J, 3) и произвольные перестановки осей.
- Самая длинная ось считается временем T, остальные оси сплющиваются в F.

Агрегация по суставам:
- 6 статистик × 3 оси = 18 фич на сустав.
- Если длина вектора важностей кратна 18, можно авто-сгенерировать схему суставов (--auto_joints).
"""

import os
import json
import argparse
import numpy as np
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from typing import List, Tuple, Optional

from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, roc_auc_score

# ---------- Константы для разметки фич ----------
STAT_NAMES = ["mean","std","min","max","dmean","dstd"]   # 6
AXES = ["x","y","z"]                                     # 3
FEATS_PER_JOINT = len(STAT_NAMES) * len(AXES)            # 18

# ---------- Дефолтная схема (можно переопределить --schema_json) ----------
DEFAULT_JOINTS = [
  "L_foot_1","L_foot_2","L_foot_3","L_foot_4",
  "L_shank_1","L_shank_2","L_shank_3","L_shank_4",
  "L_thigh_1","L_thigh_2","L_thigh_3","L_thigh_4",
  "R_foot_1","R_foot_2","R_foot_3","R_foot_4",
  "R_shank_1","R_shank_2","R_shank_3","R_shank_4",
  "R_thigh_1","R_thigh_2","R_thigh_3","R_thigh_4",
  "pelvis_1","pelvis_2","pelvis_3","pelvis_4"
]

# ---------- Метки ----------
def label_to_int(v):
    if v is None: return None
    s = str(v).strip().lower()
    if s == "injury": return 1
    if s == "no injury": return 0
    return None

# ---------- Резолвер путей к .npy ----------
def possible_npy_paths(data_dir: str, rel: str) -> List[str]:
    rel = (rel or "").strip().replace("\\", "/").lstrip("/")
    cands: List[str] = []
    def push(x: str):
        if x and x not in cands:
            cands.append(x)
    push(os.path.join(data_dir, rel))
    if not rel.endswith(".npy"):
        push(os.path.join(data_dir, rel + ".npy"))
    if rel.endswith(".json"):
        base_nojson = rel[:-5]
        push(os.path.join(data_dir, base_nojson + ".npy"))
        push(os.path.join(data_dir, rel + ".npy"))  # .json.npy
    if rel.endswith(".json.npy"):
        push(os.path.join(data_dir, rel.replace(".json.npy", ".npy")))
    b = os.path.basename(rel)
    push(os.path.join(data_dir, b))
    if not b.endswith(".npy"):
        push(os.path.join(data_dir, b + ".npy"))
    if b.endswith(".json"):
        push(os.path.join(data_dir, b[:-5] + ".npy"))
        push(os.path.join(data_dir, b + ".npy"))
    if b.endswith(".json.npy"):
        push(os.path.join(data_dir, b.replace(".json.npy", ".npy")))
    return cands

def pick_existing_path(cands: List[str]) -> Optional[str]:
    for p in cands:
        if os.path.exists(p):
            return p
    return None

# ---------- Универсальная нормализация формы (T,F) ----------
def sanitize_seq(a: np.ndarray) -> np.ndarray:
    """
    Привести массив к (T, F):
      - NaN/Inf -> 0
      - выбрать самую длинную ось как время (T)
      - остальные оси сплющить в F
    """
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    if a.ndim == 1:
        a = a[:, None]
    else:
        t_axis = int(np.argmax(a.shape))
        if t_axis != 0:
            a = np.moveaxis(a, t_axis, 0)
        if a.ndim > 2:
            a = a.reshape(a.shape[0], -1)
    return a.astype(np.float32, copy=False)

# ---------- Классические фичи (как в train3.py) ----------
def features_from_seq(seq_np: np.ndarray) -> np.ndarray:
    seq = seq_np.astype(np.float32, copy=False)
    dif = np.diff(seq, axis=0)
    stat = np.concatenate([
        np.nanmean(seq, axis=0), np.nanstd(seq, axis=0),
        np.nanmin(seq, axis=0),  np.nanmax(seq, axis=0),
        np.nanmean(dif, axis=0), np.nanstd(dif, axis=0)
    ]).astype(np.float32, copy=False)
    return np.nan_to_num(stat, nan=0.0, posinf=0.0, neginf=0.0)

# ---------- Загрузка схемы суставов ----------
def load_joints(schema_json: Optional[str], auto_joints: Optional[int]=None) -> List[str]:
    if schema_json and os.path.exists(schema_json):
        with open(schema_json, "r", encoding="utf-8") as f:
            return list(json.load(f))
    if auto_joints is not None and auto_joints > 0:
        return [f"joint_{i+1}" for i in range(int(auto_joints))]
    return list(DEFAULT_JOINTS)

# ---------- Загрузка модели ----------
def load_model_bundle(model_path: str):
    obj = joblib.load(model_path)
    if isinstance(obj, dict):
        return obj.get("model", obj), obj.get("scaler", None)
    return obj, None

# ---------- Построение фич из данных ----------
def build_features_from_files(
    csv_path: str,
    data_dir: str,
    input_format: str,
    joints: List[str],
    filename_col: str,
    label_col: str,
    downsample: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    meta = pd.read_csv(csv_path, usecols=[filename_col, label_col]).copy()
    meta["y"] = meta[label_col].map(label_to_int)
    meta = meta[pd.notnull(meta[filename_col]) & pd.notnull(meta["y"])]

    X_list, y_list = [], []
    if input_format == "npy":
        for _, row in meta.iterrows():
            fn = str(row[filename_col]).strip()
            p = pick_existing_path(possible_npy_paths(data_dir, fn))
            if not p: continue
            try:
                arr = np.load(p, allow_pickle=False, mmap_mode="r")
                x = sanitize_seq(arr)                # (T,F)
                if x.shape[0] < 2: continue
                if downsample > 1: x = x[::downsample]
                X_list.append(features_from_seq(x))
                y_list.append(int(row["y"]))
            except Exception:
                continue
    else:  # JSON
        def safe_json_load(path):
            try:
                import orjson
                with open(path, "rb") as f:
                    return orjson.loads(f.read())
            except Exception:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
        def stack_motion(md: dict, schema_joints: List[str]) -> Optional[np.ndarray]:
            present = [j for j in schema_joints if j in md]
            if not present: return None
            T = min(len(md[j]) for j in present)
            if T <= 1: return None
            cols = []
            for j in schema_joints:
                if j in md:
                    arr = np.asarray(md[j], dtype=np.float32)[:T]
                else:
                    arr = np.full((T, 3), np.nan, dtype=np.float32)
                cols.append(arr)
            return np.concatenate(cols, axis=1)  # (T, 3*|schema|)
        for _, row in meta.iterrows():
            base = os.path.join(data_dir, str(row[filename_col]).strip())
            p = base if os.path.exists(base) else (base + ".json" if os.path.exists(base + ".json") else "")
            if not p: continue
            try:
                data = safe_json_load(p)
                motion = None
                for k in ("running", "walking"):
                    if k in data and isinstance(data[k], dict):
                        motion = data[k]; break
                if motion is None: continue
                seq = stack_motion(motion, joints)
                if seq is None or seq.shape[0] < 2: continue
                if downsample > 1: seq = seq[::downsample]
                X_list.append(features_from_seq(seq))
                y_list.append(int(row["y"]))
            except Exception:
                continue

    if not X_list:
        raise RuntimeError("Не удалось построить ни одной строки фич из данных.")
    X = np.stack(X_list).astype(np.float32, copy=False)
    y = np.array(y_list, dtype=np.int32)
    return X, y

# ---------- Соответствие фич суставам ----------
def build_feature_name_map(joints: List[str]) -> List[str]:
    names = []
    for j in joints:
        for ax in AXES:
            for st in STAT_NAMES:
                names.append(f"{j}:{ax}:{st}")
    return names

def aggregate_joint_importance(importances: np.ndarray, joints: List[str]) -> pd.DataFrame:
    expected = FEATS_PER_JOINT * len(joints)
    if importances.size != expected:
        raise ValueError(
            f"Длина важностей = {importances.size}, ожидается {expected} "
            f"(18 фич × {len(joints)} суставов). Проверьте схему суставов и модель."
        )
    block = FEATS_PER_JOINT
    scores = []
    for j_idx, jname in enumerate(joints):
        s = j_idx * block
        e = s + block
        val = float(np.sum(importances[s:e]))
        scores.append((jname, val))
    df = pd.DataFrame(scores, columns=["joint", "importance_sum"])
    tot = df["importance_sum"].sum()
    df["importance_norm"] = df["importance_sum"] / (tot + 1e-12)
    return df.sort_values("importance_sum", ascending=False, ignore_index=True)

# ---------- Визуализация ----------
def plot_joint_importance(df: pd.DataFrame, topn: int, title: str, plot_path: str, show: bool=True, dpi: int=160):
    top = df.head(max(1, int(topn))).copy()
    top = top.iloc[::-1]
    top["importance_pct"] = 100.0 * top["importance_norm"]
    h = max(3.5, 0.35 * len(top))
    plt.figure(figsize=(9, h))
    plt.barh(top["joint"], top["importance_pct"])
    plt.xlabel("Importance (%) of total")
    plt.title(title)
    for i, v in enumerate(top["importance_pct"]):
        plt.text(v, i, f" {v:.1f}%", va="center")
    xmax = top["importance_pct"].max() if len(top) else 1.0
    plt.xlim(0, max(1.0, xmax * 1.1))
    plt.tight_layout()
    plt.savefig(plot_path, dpi=dpi)
    if show:
        try: plt.show()
        except Exception: pass
    plt.close()
    print(f"[OK] Plot saved: {plot_path}")

# ---------- Извлечение важностей ----------
def get_rf_xgb_importances(model) -> Optional[np.ndarray]:
    imp = getattr(model, "feature_importances_", None)
    return None if imp is None else np.asarray(imp, dtype=float)

def get_linear_svm_importances(model) -> Optional[np.ndarray]:
    coef = getattr(model, "coef_", None)
    if coef is None: return None
    return np.abs(np.asarray(coef, dtype=float).ravel())

def permutation_importances(model, scaler, X: np.ndarray, y: np.ndarray,
                            scoring: str = "f1", n_repeats: int = 10, random_state: int = 42) -> np.ndarray:
    Xs = scaler.transform(X) if scaler is not None else X
    if scoring == "roc_auc":
        def _proba_like(est, X_):
            if hasattr(est, "predict_proba"):
                return est.predict_proba(X_)[:, 1]
            if hasattr(est, "decision_function"):
                d = est.decision_function(X_)
                d = (d - d.min()) / (d.max() - d.min() + 1e-9)
                return d
            return est.predict(X_)
        scorer = lambda est, X_, y_: roc_auc_score(y_, _proba_like(est, X_))
    else:
        scorer = lambda est, X_, y_: f1_score(y_, est.predict(X_))
    r = permutation_importance(model, Xs, y, scoring=scorer,
                               n_repeats=n_repeats, random_state=random_state, n_jobs=-1)
    return np.maximum(r.importances_mean, 0.0)

# ---------- Автовывод количества суставов из длины вектора важностей ----------
def maybe_auto_joints_count(importances_len: int) -> Optional[int]:
    # importances_len = 18 * J
    if importances_len % FEATS_PER_JOINT == 0:
        return importances_len // FEATS_PER_JOINT
    return None

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Per-joint importance for RF/XGB/SVM (with optional permutation importance).")
    ap.add_argument("--model_joblib", required=True, help="Путь к model.joblib (из train3.py).")
    ap.add_argument("--schema_json", default="", help="JSON со списком суставов (если не задано — дефолт или авто).")
    ap.add_argument("--auto_joints", action="store_true",
                    help="Игнорировать схему и сгенерировать имена суставов из длины вектора важностей (если кратно 18).")

    # Данные (только если понадобится permutation importance)
    ap.add_argument("--csv", default="", help="CSV (filename, label). Нужно, если у модели нет importances/coef_.")
    ap.add_argument("--data_dir", default="", help="Папка с файлами (JSON/NPY).")
    ap.add_argument("--npy_dir", default="", help="Синоним data_dir для совместимости со старыми скриптами.")
    ap.add_argument("--input_format", choices=["npy","json"], default="npy")
    ap.add_argument("--filename_col", default="filename")
    ap.add_argument("--label_col", default="No inj/ inj")
    ap.add_argument("--downsample", type=int, default=1)

    # Вывод
    ap.add_argument("--topn", type=int, default=15, help="Сколько показать в консоли.")
    ap.add_argument("--topn_plot", type=int, default=20, help="Сколько нарисовать на графике.")
    ap.add_argument("--plot_path", default="top_joints.png")
    ap.add_argument("--no_show", action="store_true")
    ap.add_argument("--title", default="Joint Importance (sum over 18 features)")
    ap.add_argument("--save_csv", default="", help="Путь для сохранения CSV с важностями суставов.")

    # Permutation importance
    ap.add_argument("--pi_scoring", choices=["f1","roc_auc"], default="f1")
    ap.add_argument("--pi_repeats", type=int, default=10)

    args = ap.parse_args()

    model, scaler = load_model_bundle(args.model_joblib)

    # 1) Попробуем прямые важности
    importances = get_rf_xgb_importances(model)
    if importances is None:
        importances = get_linear_svm_importances(model)

    data_dir = args.data_dir or args.npy_dir

    # 2) Если нет — считаем permutation importance по данным
    if importances is None:
        if not args.csv or not data_dir:
            raise RuntimeError(
                "У модели нет feature_importances_/coef_. "
                "Для SVM с RBF (или др. моделей) укажите --csv и --data_dir (или --npy_dir) для permutation importance."
            )
        # joints предварительно: если есть schema_json — читаем, иначе дефолт (можно будет переопределить дальше)
        joints = load_joints(args.schema_json)
        X, y = build_features_from_files(
            csv_path=args.csv,
            data_dir=data_dir,
            input_format=args.input_format,
            joints=joints,
            filename_col=args.filename_col,
            label_col=args.label_col,
            downsample=max(1, int(args.downsample)),
        )
        print(f"[INFO] Permutation importance на {X.shape[0]} сэмплах × {X.shape[1]} фичах...")
        importances = permutation_importances(
            model, scaler, X, y, scoring=args.pi_scoring, n_repeats=int(args.pi_repeats)
        )
        # после PI мы точно знаем len(importances) => можно автоинферить joints, если просили
        auto_J = maybe_auto_joints_count(importances.size) if args.auto_joints else None
        joints = load_joints(args.schema_json, auto_joints=auto_J)
    else:
        # у нас есть вектор важностей напрямую от модели => можно автоинферить joints при желании
        auto_J = maybe_auto_joints_count(importances.size) if args.auto_joints else None
        joints = load_joints(args.schema_json, auto_joints=auto_J)

    # Проверим соответствие длины
    expected = FEATS_PER_JOINT * len(joints)
    if importances.size != expected:
        raise ValueError(
            f"Длина важностей = {importances.size}, ожидается {expected} (18×|joints|). "
            f"Либо передайте корректную --schema_json, либо используйте флаг --auto_joints."
        )

    # Агрегация по суставам
    df = aggregate_joint_importance(np.asarray(importances, dtype=float), joints)

    # Консольный топ
    topn = max(1, int(args.topn))
    print("\n=== TOP joints (sum of importances over 18 features per joint) ===")
    for i, row in df.head(topn).iterrows():
        print(f"{i+1:2d}. {row['joint']:>16s}  "
              f"sum={row['importance_sum']:.6f}  norm={row['importance_norm']:.4f}")

    # Плот
    plot_joint_importance(
        df=df,
        topn=args.topn_plot,
        title=args.title,
        plot_path=args.plot_path,
        show=not args.no_show,
        dpi=160
    )

    # CSV
    if args.save_csv:
        df.to_csv(args.save_csv, index=False, encoding="utf-8")
        print(f"[OK] CSV saved: {args.save_csv}")

if __name__ == "__main__":
    main()
