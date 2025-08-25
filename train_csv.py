#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train XGBoost on features.csv + labels CSV (filename -> Injury/No Injury).

- subject-wise split 60/20/20 (GroupShuffleSplit)
- removes leakage cols: file/basename/stem/subject/label/label-like
- robust label mapping: "No inj/ inj", "No Injury"/"Injury", 0/1
- saves: xgb.json, features_cols.json, metrics.json, split_subjectwise.csv, threshold.txt

Usage example (под твои пути в Kaggle):
python train_xgb_from_features_csv.py \
  --features_csv /kaggle/working/out_features/features.csv \
  --labels_csv   /kaggle/working/detect_inj/run_data.csv \
  --label_col    "No inj/ inj" \
  --fname_col    filename \
  --out_dir      /kaggle/working/out_model \
  --use_gpu
"""

import os, json, argparse, re
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, average_precision_score
)
import xgboost as xgb

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def norm_stem(x: str) -> str:
    b = os.path.basename(str(x))
    st = os.path.splitext(b)[0]
    return st.lower()

def map_label(v):
    s = str(v).strip().lower()
    if s in {"1","inj","injury","yes","y","true","t"}: return 1
    if s in {"0","no inj","no injury","no","n","false","f"}: return 0
    # иногда в таблицах "No inj/ inj" пишут как 0/1 — попробуем привести к int
    try:
        return int(float(s))
    except Exception:
        return np.nan

def choose_threshold(y, prob):
    # простой порог по 0.5; можно усложнить при желании
    return 0.5

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_csv", required=True)
    ap.add_argument("--labels_csv",   required=True)
    ap.add_argument("--label_col",    required=True, help='Колонка меток в labels CSV (например, "No inj/ inj")')
    ap.add_argument("--fname_col",    default="filename", help="Колонка с именем файла в labels CSV")
    ap.add_argument("--out_dir",      default="out_model")
    ap.add_argument("--use_gpu",      action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); ensure_dir(out_dir)

    # 1) read features (из твоего экстрактора)
    F = pd.read_csv(args.features_csv)
    # плавающее: иногда нет stem — сделаем
    if "stem" not in F.columns:
        if "basename" in F.columns:
            F["stem"] = F["basename"].astype(str).map(lambda s: os.path.splitext(s)[0])
        elif "file" in F.columns:
            F["stem"] = F["file"].astype(str).map(norm_stem)
        else:
            raise SystemExit("[err] features.csv must contain either stem/basename/file")

    if "subject" not in F.columns:
        # если вдруг нет — грубо из stem (первые 8 цифр)
        F["subject"] = F["stem"].astype(str).str.extract(r"(\d{8})", expand=False).fillna(F["stem"].astype(str))

    # 2) read labels CSV и сматчить по stem
    L = pd.read_csv(args.labels_csv)
    if args.fname_col not in L.columns:
        raise SystemExit(f"[err] labels CSV has no '{args.fname_col}' column")
    if args.label_col not in L.columns:
        raise SystemExit(f"[err] labels CSV has no '{args.label_col}' column")

    L["_key"] = L[args.fname_col].astype(str).map(norm_stem)
    F["_key"] = F["stem"].astype(str).map(lambda s: s.lower())

    lab = L[["_key", args.label_col]].drop_duplicates("_key", keep="last").copy()
    lab["label"] = lab[args.label_col].map(map_label)
    lab = lab.drop(columns=[args.label_col])

    DF = F.merge(lab, on="_key", how="left").drop(columns=["_key"])
    if DF["label"].notna().sum() == 0:
        ex = DF[["file","basename","stem"]].head(6) if "file" in DF else DF[["basename","stem"]].head(6)
        raise SystemExit("[err] no labels matched. Check fname_col vs real filenames. Examples:\n"+ex.to_string(index=False))

    # 3) build X, y, groups
    drop_cols = {"label", "file", "basename", "stem", "subject"}
    leak_like = [c for c in DF.columns if ("inj" in c.lower() or "label" in c.lower()) and c not in drop_cols]
    drop_cols.update(leak_like)
    feat_cols = [c for c in DF.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(DF[c])]
    X = DF[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = DF["label"].astype("Int64").astype("float").astype("int").values
    groups = DF["subject"].astype(str).values

    # sanity
    n0, n1 = int((y==0).sum()), int((y==1).sum())
    print(f"[info] samples={len(y)} | features={X.shape[1]} | class0={n0} class1={n1}")

    # 4) subject-wise split 60/20/20
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    tr_idx, te_idx = next(gss1.split(X, y, groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=43)  # 0.25 of 0.8 -> 0.2 итог
    tr_sub, dv_idx = next(gss2.split(X.iloc[tr_idx], y[tr_idx], groups[tr_idx]))
    tr_idx = tr_idx[tr_sub]

    assert set(groups[tr_idx]).isdisjoint(set(groups[dv_idx]))
    assert set(groups[tr_idx]).isdisjoint(set(groups[te_idx]))
    assert set(groups[dv_idx]).isdisjoint(set(groups[te_idx]))

    def stat(idx):
        return {"n":len(idx), "pos":int((y[idx]==1).sum()), "neg":int((y[idx]==0).sum()),
                "subjects": int(len(np.unique(groups[idx])))}
    print("[split] subject-wise 60/20/20")
    print("  train:", stat(tr_idx))
    print("  dev:  ", stat(dv_idx))
    print("  test: ", stat(te_idx))

    # 5) train XGBoost
    params = dict(
        objective="binary:logistic",
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=5,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        reg_alpha=0.0,
        tree_method="gpu_hist" if args.use_gpu else "hist",
        eval_metric="logloss",
        seed=42,
    )
    dtrain = xgb.DMatrix(X.iloc[tr_idx], label=y[tr_idx])
    dvalid = xgb.DMatrix(X.iloc[dv_idx], label=y[dv_idx])
    dtest  = xgb.DMatrix(X.iloc[te_idx], label=y[te_idx])

    print("[train] fitting XGBoost...")
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        evals=[(dtrain,"train"),(dvalid,"valid")],
        early_stopping_rounds=100,
        verbose_eval=50
    )

    prob_dv = bst.predict(dvalid, iteration_range=(0, bst.best_iteration+1))
    prob_te = bst.predict(dtest,  iteration_range=(0, bst.best_iteration+1))

    thr = choose_threshold(y[dv_idx], prob_dv)
    pred_te = (prob_te >= thr).astype(int)

    metrics = {
        "dev_auroc": float(roc_auc_score(y[dv_idx], prob_dv)) if len(np.unique(y[dv_idx]))>1 else None,
        "dev_auprc": float(average_precision_score(y[dv_idx], prob_dv)) if len(np.unique(y[dv_idx]))>1 else None,
        "test_auroc": float(roc_auc_score(y[te_idx], prob_te)) if len(np.unique(y[te_idx]))>1 else None,
        "test_auprc": float(average_precision_score(y[te_idx], prob_te)) if len(np.unique(y[te_idx]))>1 else None,
        "threshold": float(thr),
        "test_confusion_matrix": confusion_matrix(y[te_idx], pred_te).tolist(),
        "test_report": classification_report(y[te_idx], pred_te, digits=3),
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    # 6) save artifacts
    bst.save_model(str(out_dir / "xgb.json"))
    json.dump(feat_cols, open(out_dir / "features_cols.json","w",encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(metrics,   open(out_dir / "metrics.json","w",encoding="utf-8"), ensure_ascii=False, indent=2)
    (out_dir / "threshold.txt").write_text(str(thr), encoding="utf-8")

    split_df = pd.DataFrame({"idx": np.arange(len(y)),
                             "subject": groups,
                             "y": y, "split": "train"})
    split_df.loc[dv_idx, "split"] = "dev"
    split_df.loc[te_idx, "split"] = "test"
    split_df.to_csv(out_dir / "split_subjectwise.csv", index=False)

    print("Saved to:", out_dir)

if __name__ == "__main__":
    main()
