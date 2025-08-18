#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
top_joints_from_model.py
Aggregates per-joint importance for classical models (RF/XGB) and SVM.

- RF/XGB: use feature_importances_
- SVM (linear): use |coef_|
- SVM (non-linear, e.g. RBF) or any model without importances: use permutation importance
  (needs raw data to rebuild classical features; pass --csv/--data_dir/--input_format)

Features layout must match train3.py: 6 stats × 3 axes × |joints| = 18*|joints|
"""

import os
import json
import argparse
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from typing import List, Tuple, Optional

# Optional: permutation importance if needed
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, roc_auc_score

# ---------- Default joints schema (can be overridden via --schema_json)
DEFAULT_JOINTS = [
  "L_foot_1","L_foot_2","L_foot_3","L_foot_4",
  "L_shank_1","L_shank_2","L_shank_3","L_shank_4",
  "L_thigh_1","L_thigh_2","L_thigh_3","L_thigh_4",
  "R_foot_1","R_foot_2","R_foot_3","R_foot_4",
  "R_shank_1","R_shank_2","R_shank_3","R_shank_4",
  "R_thigh_1","R_thigh_2","R_thigh_3","R_thigh_4",
  "pelvis_1","pelvis_2","pelvis_3","pelvis_4"
]

STAT_NAMES = ["mean","std","min","max","dmean","dstd"]   # as in train3.py
AXES = ["x","y","z"]
FEATS_PER_JOINT = len(STAT_NAMES) * len(AXES)            # 6*3 = 18

# ---------- Utils to load schema / model ----------

def load_joints(schema_json: Optional[str]) -> List[str]:
    if schema_json and os.path.exists(schema_json):
        with open(schema_json, "r", encoding="utf-8") as f:
            return list(json.load(f))
    return list(DEFAULT_JOINTS)

def load_model_bundle(model_path: str):
    """Return (model, scaler_or_None). In train3.py, model.joblib stores dict {'model':..., 'scaler':...}."""
    obj = joblib.load(model_path)
    if isinstance(obj, dict):
        return obj.get("model", obj), obj.get("scaler", None)
    return obj, None

# ---------- Classical feature extraction (copy of logic from train3.py) ----------

def _features_from_seq(seq_np: np.ndarray) -> np.ndarray:
    seq = seq_np.astype(np.float32, copy=False)
    dif = np.diff(seq, axis=0)
    stat = np.concatenate([
        np.nanmean(seq, axis=0), np.nanstd(seq, axis=0),
        np.nanmin(seq, axis=0),  np.nanmax(seq, axis=0),
        np.nanmean(dif, axis=0), np.nanstd(dif, axis=0)
    ]).astype(np.float32, copy=False)
    return np.nan_to_num(stat, nan=0.0, posinf=0.0, neginf=0.0)

def _map_to_npy_path(data_dir: str, rel_path: str) -> str:
    base = rel_path[:-5] if rel_path.endswith(".json") else rel_path
    if not base.endswith(".npy"):
        base = base + ".npy"
    return os.path.join(data_dir, base)

def _safe_json_load(path):
    try:
        import orjson
        with open(path, "rb") as f:
            return orjson.loads(f.read())
    except Exception:
        import json as _json
        with open(path, "r", encoding="utf-8") as f:
            return _json.load(f)

def _stack_motion_frames_with_schema(md: dict, schema_joints: List[str]) -> Optional[np.ndarray]:
    present = [j for j in schema_joints if j in md]
    if not present:
        return None
    T = min(len(md[j]) for j in present)
    if T <= 1:
        return None
    cols = []
    for j in schema_joints:
        if j in md:
            arr = np.asarray(md[j], dtype=np.float32)[:T]   # (T,3)
        else:
            arr = np.full((T, 3), np.nan, dtype=np.float32)
        cols.append(arr)
    return np.concatenate(cols, axis=1)  # (T, 3*|schema|)

def _load_meta(csv_path: str, filename_col: str, label_col: str) -> pd.DataFrame:
    meta = pd.read_csv(csv_path, usecols=[filename_col, label_col])
    # map labels like in train3.py
    def label_to_int(v):
        if v is None: return None
        s = str(v).strip().lower()
        if s == "injury": return 1
        if s == "no injury": return 0
        return None
    meta["y"] = meta[label_col].map(label_to_int)
    meta = meta[pd.notnull(meta[filename_col]) & pd.notnull(meta["y"])].copy()
    return meta

def build_features(
    csv_path: str,
    data_dir: str,
    input_format: str,               # "npy" | "json"
    joints: List[str],
    filename_col: str,
    label_col: str,
    downsample: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    meta = _load_meta(csv_path, filename_col, label_col)
    X_list, y_list = [], []

    if input_format == "npy":
        for _, row in meta.iterrows():
            p = _map_to_npy_path(data_dir, str(row[filename_col]).strip())
            if not os.path.exists(p):
                continue
            try:
                arr = np.load(p, allow_pickle=False, mmap_mode="r")
                if arr.ndim != 2 or arr.shape[0] < 2:
                    continue
                seq = arr[::max(1, downsample)]
                X_list.append(_features_from_seq(np.asarray(seq)))
                y_list.append(int(row["y"]))
            except Exception:
                continue
    else:  # json
        for _, row in meta.iterrows():
            p = os.path.join(data_dir, str(row[filename_col]).strip())
            if not os.path.exists(p) and not p.endswith(".json"):
                p2 = p + ".json"
                if os.path.exists(p2):
                    p = p2
            if not os.path.exists(p):
                continue
            try:
                data = _safe_json_load(p)
                # motion key default "running" in train3; allow both 'running' and 'walking'
                motion = None
                for k in ("running", "walking"):
                    if k in data and isinstance(data[k], dict):
                        motion = data[k]; break
                if motion is None:
                    continue
                seq = _stack_motion_frames_with_schema(motion, joints)
                if seq is None or seq.shape[0] < 2:
                    continue
                if downsample > 1:
                    seq = seq[::downsample]
                X_list.append(_features_from_seq(np.asarray(seq)))
                y_list.append(int(row["y"]))
            except Exception:
                continue

    if not X_list:
        raise RuntimeError("Failed to build any feature rows for permutation importance.")
    X = np.stack(X_list).astype(np.float32, copy=False)
    y = np.array(y_list, dtype=np.int32)
    return X, y

# ---------- Mapping features -> joints ----------

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
            f"Len of importances = {importances.size}, expected by schema: {expected} "
            f"(18 features × {len(joints)} joints)."
        )
    joint_scores = []
    for j_idx, jname in enumerate(joints):
        start = j_idx * FEATS_PER_JOINT
        end = start + FEATS_PER_JOINT
        score = float(np.sum(importances[start:end]))
        joint_scores.append((jname, score))
    df = pd.DataFrame(joint_scores, columns=["joint", "importance_sum"])
    tot = df["importance_sum"].sum()
    df["importance_norm"] = df["importance_sum"] / tot if tot > 0 else 0.0
    return df.sort_values("importance_sum", ascending=False).reset_index(drop=True)

# ---------- Plot ----------

def plot_joint_importance(df: pd.DataFrame, topn: int, title: str, plot_path: str, show: bool=True, dpi: int=160):
    top = df.head(max(1, int(topn))).copy()
    top = top.iloc[::-1]
    top["importance_pct"] = 100 * top["importance_sum"]
    h = max(3.5, 0.3 * len(top))
    plt.figure(figsize=(8, h))
    plt.barh(top["joint"], top["importance_pct"])
    plt.xlabel("Importance (%)")
    plt.title(title)
    for i, v in enumerate(top["importance_pct"]):
        plt.text(v, i, f" {v:.1f}%", va="center")
    plt.xlim(0, top["importance_pct"].max() * 1.1 if len(top) else 1)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=dpi)
    if show:
        plt.show()
    plt.close()
    print(f"[OK] Plot saved: {plot_path}")

# ---------- Importance getters ----------

def get_rf_xgb_importances(model) -> Optional[np.ndarray]:
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        return None
    return np.asarray(imp, dtype=float)

def get_linear_svm_importances(model) -> Optional[np.ndarray]:
    # SVC(kernel='linear') or LinearSVC -> coef_ exists (shape: [1, n_features] for binary)
    coef = getattr(model, "coef_", None)
    if coef is None:
        return None
    coef = np.asarray(coef, dtype=float).ravel()
    return np.abs(coef)

def get_permutation_importances(model, scaler, X: np.ndarray, y: np.ndarray,
                                scoring: str = "f1", n_repeats: int = 10, random_state: int = 42) -> np.ndarray:
    # scale if scaler is provided (as saved in train3.py)
    Xs = scaler.transform(X) if scaler is not None else X
    # Use probability if available for roc_auc
    if scoring == "roc_auc":
        # ensure we have predict_proba or decision_function
        def _predict_proba_like(est, X_):
            if hasattr(est, "predict_proba"):
                return est.predict_proba(X_)[:, 1]
            if hasattr(est, "decision_function"):
                # map decision scores to [0,1] via rank or sigmoid-like scaling
                d = est.decision_function(X_)
                # simple normalization to (0,1)
                d = (d - d.min()) / (d.max() - d.min() + 1e-9)
                return d
            # fallback to predictions
            return est.predict(X_)
        scorer = lambda est, X_, y_: roc_auc_score(y_, _predict_proba_like(est, X_))
    else:
        scorer = lambda est, X_, y_: f1_score(y_, est.predict(X_))

    r = permutation_importance(
        model, Xs, y, scoring=scorer, n_repeats=n_repeats, random_state=random_state, n_jobs=-1
    )
    return np.maximum(r.importances_mean, 0.0)  # clip negatives to 0 for nicer chart

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(description="Per-joint importance for RF/XGB/SVM models.")
    ap.add_argument("--model_joblib", required=True, help="Path to model.joblib (as saved by train3.py).")
    ap.add_argument("--schema_json", default="", help="Optional JSON with joints list; defaults to built-in.")
    ap.add_argument("--topn_plot", type=int, default=20, help="How many top joints to plot.")
    ap.add_argument("--plot_path", default="top_joints_importance.png", help="Where to save the plot.")
    ap.add_argument("--no_show", action="store_true", help="Do not display plot, only save.")
    ap.add_argument("--title", default="Joint Importance (sum over 18 features per joint)", help="Plot title.")
    ap.add_argument("--topn", type=int, default=15, help="How many joints to print to console.")

    # For permutation importance on SVM RBF (or any model w/o importances)
    ap.add_argument("--csv", default="", help="CSV with metadata (filename, label). Required for permutation importance.")
    ap.add_argument("--data_dir", default="", help="Directory with JSON/NPY files.")
    ap.add_argument("--input_format", choices=["npy", "json"], default="npy", help="Source format for building features.")
    ap.add_argument("--filename_col", default="filename")
    ap.add_argument("--label_col", default="No inj/ inj")
    ap.add_argument("--downsample", type=int, default=1, help="Temporal downsample step (>=1).")
    ap.add_argument("--pi_scoring", choices=["f1", "roc_auc"], default="f1", help="Metric for permutation importance.")
    ap.add_argument("--pi_repeats", type=int, default=10, help="n_repeats for permutation importance.")

    args = ap.parse_args()

    joints = load_joints(args.schema_json)
    print(f"[INFO] Joints in schema: {len(joints)} (expected features = 18×|joints| = {FEATS_PER_JOINT*len(joints)})")

    model, scaler = load_model_bundle(args.model_joblib)

    # 1) Try tree importances (RF/XGB)
    importances = get_rf_xgb_importances(model)

    # 2) If none, try linear SVM coefs
    if importances is None:
        importances = get_linear_svm_importances(model)

    # 3) If still none — run permutation importance (needs data)
    if importances is None:
        if not args.csv or not args.data_dir:
            raise RuntimeError(
                "The model has no importances/coefficients. "
                "For SVM with non-linear kernel, please provide --csv and --data_dir to compute permutation importance."
            )
        # Rebuild classical features exactly like in train3.py
        X, y = build_features(
            csv_path=args.csv,
            data_dir=args.data_dir,
            input_format=args.input_format,
            joints=joints,
            filename_col=args.filename_col,
            label_col=args.label_col,
            downsample=max(1, int(args.downsample)),
        )
        # Permutation importance on the full set (or you can subsample here if it's huge)
        print(f"[INFO] Running permutation importance on {X.shape[0]} samples × {X.shape[1]} features...")
        importances = get_permutation_importances(
            model, scaler, X, y, scoring=args.pi_scoring, n_repeats=args.pi_repeats
        )

    # Sanity check: length must match features layout
    expected_len = FEATS_PER_JOINT * len(joints)
    if importances.size != expected_len:
        raise ValueError(
            f"Importances length = {importances.size}, but expected {expected_len} "
            f"(18 × {len(joints)} joints). Check that your model was trained on classical features "
            f"with the same joints schema."
        )

    # Aggregate to joints
    df = aggregate_joint_importance(importances, joints)

    # Print TOP-N
    topn = max(1, int(args.topn))
    print("\n=== TOP joints (sum of importances over 18 features per joint) ===")
    for i, row in df.head(topn).iterrows():
        print(f"{i+1:2d}. {row['joint']:>12s}  importance_sum={row['importance_sum']:.6f}  "
              f"(norm={row['importance_norm']:.4f})")

    # Plot
    plot_joint_importance(
        df=df,
        topn=args.topn_plot,
        title=args.title,
        plot_path=args.plot_path,
        show=not args.no_show,
        dpi=160
    )

if __name__ == "__main__":
    main()
