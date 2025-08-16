import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from train import load_features_streaming, label_to_int

# ==== утилиты ====

def classical_feature_names(schema_joints):
    """
    Генерация имён признаков (mean/std/min/max + дифы) для каждого сустава.
    """
    feats = []
    stats = ["mean", "std", "min", "max", "dmean", "dstd"]
    coords = ["x", "y", "z"]
    for stat in stats:
        for j in schema_joints:
            for c in coords:
                feats.append(f"{stat}_{c}_{j}")
    return feats

def aggregate_importance_by_joint(feat_names, importances):
    by_joint = {}
    by_joint_axis = {}
    for fn, im in zip(feat_names, importances):
        parts = fn.split("_")
        stat, axis, joint = parts[0], parts[1], "_".join(parts[2:])
        by_joint[joint] = by_joint.get(joint, 0.0) + im
        key = f"{joint}_{axis}"
        by_joint_axis[key] = by_joint_axis.get(key, 0.0) + im
    return by_joint, by_joint_axis

def save_importance_tables(out_dir, all_rows, by_joint, by_joint_axis, prefix="perm"):
    pd.DataFrame(all_rows).to_csv(os.path.join(out_dir, f"{prefix}_features.csv"), index=False)
    pd.DataFrame([{"joint": k, "importance": v} for k, v in by_joint.items()]) \
        .to_csv(os.path.join(out_dir, f"{prefix}_joints.csv"), index=False)
    pd.DataFrame([{"joint_axis": k, "importance": v} for k, v in by_joint_axis.items()]) \
        .to_csv(os.path.join(out_dir, f"{prefix}_joint_axes.csv"), index=False)

# ==== основной код ====

def main():
    out_dir = "outputs/rf"  # например
    csv_path = "walk_data_meta_upd.csv"
    data_dir = "walking"

    # загружаем модель + нормировщик
    bundle = joblib.load(os.path.join(out_dir, "model.joblib"))
    model, scaler = bundle["model"], bundle["scaler"]

    # схема суставов
    with open(os.path.join(out_dir, "schema_joints.json"), "r", encoding="utf-8") as f:
        schema_joints = json.load(f)

    # грузим фичи
    X_all, y_all, _ = load_features_streaming(
        csv_path, data_dir,
        motion_key="walking",
        use_joints=schema_joints,
        workers=4
    )
    X_all = scaler.transform(X_all)

    # считаем permutation importance
    feat_names = classical_feature_names(schema_joints)
    pi = permutation_importance(model, X_all, y_all, n_repeats=10, random_state=42, n_jobs=-1)
    importances_mean = pi.importances_mean

    # сохраняем
    all_rows = [{"feature": fn, "importance": float(im)} for fn, im in zip(feat_names, importances_mean)]
    by_joint, by_joint_axis = aggregate_importance_by_joint(feat_names, importances_mean)
    save_importance_tables(out_dir, all_rows, by_joint, by_joint_axis, prefix="perm")

    print("\n=== Важность суставов (по убыванию) ===")
    for j, v in sorted(by_joint.items(), key=lambda x: x[1], reverse=True):
        print(f"{j:20s} {v:.6f}")

if __name__ == "__main__":
    main()
