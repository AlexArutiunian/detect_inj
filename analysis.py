#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
top_joints_from_model.py
Aggregates feature importances by joints for classical models (RF/XGB),
trained on features from train3.py (6 statistics × 3 axes × |joints|),
+ builds a horizontal bar chart of importances (shown inline and saved as PNG).

Example:
    python top_joints_from_model.py \
        --npy_dir /path/to/npy \
        --model_joblib /path/to/outputs/rf/model.joblib \
        --topn 15 \
        --topn_plot 20 \
        --plot_path top_joints.png \
        --save_csv top_joints.csv
"""

import os
import json
import argparse
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# --- Default joints schema (can be overridden via --schema_json)
DEFAULT_JOINTS = [
  "L_foot_1","L_foot_2","L_foot_3","L_foot_4",
  "L_shank_1","L_shank_2","L_shank_3","L_shank_4",
  "L_thigh_1","L_thigh_2","L_thigh_3","L_thigh_4",
  "R_foot_1","R_foot_2","R_foot_3","R_foot_4",
  "R_shank_1","R_shank_2","R_shank_3","R_shank_4",
  "R_thigh_1","R_thigh_2","R_thigh_3","R_thigh_4",
  "pelvis_1","pelvis_2","pelvis_3","pelvis_4"
]

STAT_NAMES = ["mean","std","min","max","dmean","dstd"]  # as in train3.py
AXES = ["x","y","z"]                                   
FEATS_PER_JOINT = len(STAT_NAMES) * len(AXES)          # 6*3 = 18

def load_joints(schema_json: str | None):
    if schema_json and os.path.exists(schema_json):
        with open(schema_json, "r", encoding="utf-8") as f:
            return list(json.load(f))
    return list(DEFAULT_JOINTS)

def find_first_npy(npy_dir: str):
    for root, _, files in os.walk(npy_dir):
        for fn in files:
            if fn.lower().endswith(".npy"):
                return os.path.join(root, fn)
    return None

def load_model(model_path: str):
    obj = joblib.load(model_path)
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"]
    return obj

def get_importances(model):
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        raise RuntimeError(
            "The model has no feature_importances_. "
            "You need RandomForest/XGBoost trained on classical features."
        )
    return np.asarray(imp, dtype=float)

def sanity_check_with_npy(npy_dir: str, joints: list[str], verbose: bool=True):
    sample = find_first_npy(npy_dir)
    if not sample:
        if verbose:
            print("[WARN] No .npy file found in npy_dir — skipping shape check.")
        return
    try:
        arr = np.load(sample, allow_pickle=False, mmap_mode="r")
        if arr.ndim != 2 or arr.shape[0] < 2:
            print(f"[WARN] '{sample}' has shape {arr.shape}, expected (T, F) with T>=2.")
        else:
            F_seq = arr.shape[1]
            expect_coords = 3 * len(joints)
            if F_seq % 3 != 0:
                print(f"[WARN] Number of features per frame = {F_seq}, not divisible by 3.")
            if F_seq != expect_coords:
                print(f"[INFO] Sequence has F={F_seq} features per frame, expected 3*|joints|={expect_coords}. "
                      "This is not critical for importance aggregation, but check schema consistency.")
    except Exception as e:
        print(f"[WARN] Failed to open NPY for check: {e}")

def build_feature_name_map(joints: list[str]):
    names = []
    for j in joints:
        for ax in AXES:
            for st in STAT_NAMES:
                names.append(f"{j}:{ax}:{st}")
    return names

def aggregate_joint_importance(importances: np.ndarray, joints: list[str]) -> pd.DataFrame:
    expected = FEATS_PER_JOINT * len(joints)
    if importances.size != expected:
        raise ValueError(
            f"Len of importances = {importances.size}, expected by schema: {expected} "
            f"(18 features × {len(joints)} joints). Check if model was trained correctly."
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

def plot_joint_importance(df: pd.DataFrame, topn: int, title: str, plot_path: str, show: bool=True, dpi: int=160):
    """
    Builds a horizontal bar chart for topn joints.
    Displays inline (if show=True) and saves to PNG.
    """
    top = df.head(max(1, int(topn))).copy()
    # reverse order for nicer top-down display
    top = top.iloc[::-1]

    # normalize to percentages
    top["importance_pct"] = 100 * top["importance_sum"]

    h = max(3.5, 0.3 * len(top))       
    plt.figure(figsize=(8, h))
    plt.barh(top["joint"], top["importance_pct"])
    plt.xlabel("Importance (%)")
    plt.title(title)

    # add text labels to bars
    for i, v in enumerate(top["importance_pct"]):
        plt.text(v, i, f" {v:.1f}%", va="center")

    plt.xlim(0, top["importance_pct"].max() * 1.1)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=dpi)
    if show:
        plt.show()
    plt.close()

    print(f"[OK] Plot saved: {plot_path}")

def main():
    ap = argparse.ArgumentParser(description="Show top joints by importance and plot horizontal bar chart.")
    ap.add_argument("--npy_dir", required=True, help="Folder with *.npy (for quick shape check).")
    ap.add_argument("--model_joblib", required=True, help="Path to model.joblib (RF/XGB trained on classical features).")
    ap.add_argument("--schema_json", default="", help="JSON with list of joints (if not set — built-in default list).")
    ap.add_argument("--topn", type=int, default=15, help="How many joints to print in text output.")
    ap.add_argument("--save_csv", default="", help="Optional path to save CSV with importances.")

    # Plot options
    ap.add_argument("--topn_plot", type=int, default=20, help="How many top joints to plot in chart.")
    ap.add_argument("--plot_path", default="top_joints_importance.png", help="Where to save PNG chart.")
    ap.add_argument("--title", default="Joint Importance (sum over 18 features per joint)", help="Chart title.")
    ap.add_argument("--no_show", action="store_true", help="Do not display the chart, only save.")

    args = ap.parse_args()

    joints = load_joints(args.schema_json)
    print(f"[INFO] Joints in schema: {len(joints)} (expected features in model = 18×|joints| = {FEATS_PER_JOINT*len(joints)})")

    sanity_check_with_npy(args.npy_dir, joints, verbose=True)

    model = load_model(args.model_joblib)
    importances = get_importances(model)

    feat_names = build_feature_name_map(joints)
    if len(feat_names) != importances.size:
        raise ValueError(
            f"Shape mismatch: len(feat_names)={len(feat_names)} vs len(importances)={importances.size}. Check schema consistency."
        )

    df = aggregate_joint_importance(importances, joints)

    # Print TOP-N
    topn = max(1, int(args.topn))
    print("\n=== TOP joints (sum of importances over 18 features per joint) ===")
    for i, row in df.head(topn).iterrows():
        print(f"{i+1:2d}. {row['joint']:>12s}  importance_sum={row['importance_sum']:.6f}  "
              f"(norm={row['importance_norm']:.4f})")

    if args.save_csv:
        df.to_csv(args.save_csv, index=False)
        print(f"[OK] Saved CSV: {args.save_csv}")

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
