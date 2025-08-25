#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, argparse, re
from pathlib import Path
from typing import Optional, List
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix
)

import xgboost as xgb


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def norm_stem(p: str) -> str:
    b = os.path.basename(str(p))
    st, _ = os.path.splitext(b)
    return st.lower()


def guess_label_col(df: pd.DataFrame) -> Optional[str]:
    cands = ["No inj/ inj", "label", "injury", "target", "y", "class"]
    for c in cands:
        if c in df.columns:
            return c
    return None


def guess_fname_col(df: pd.DataFrame) -> Optional[str]:
    # имя файла в манифесте
    cands = ["filename", "file", "path", "basename", "stem"]
    for c in cands:
        if c in df.columns:
            return c
    # попробуем по подстроке
    for c in df.columns:
        if "file" in c.lower() or "name" in c.lower():
            return c
    return None


def label_to_int(v):
    if pd.isna(v): return None
    s = str(v).strip().lower()
    if s in {"0", "no", "no inj", "no-inj", "no injury", "healthy"}: return 0
    if s in {"1", "yes", "inj", "injury"}: return 1
    # иногда в csv бывают "No inj/ inj" как 0/1 уже в int/float
    try:
        f = float(s)
        if f in (0.0, 1.0): return int(f)
    except Exception:
        pass
    return None


def load_and_merge(features_csv: str, labels_csv: str,
                   label_col: Optional[str], fname_col: Optional[str]) -> pd.DataFrame:
    feats = pd.read_csv(features_csv)
    # убедимся, что есть чем мёрджить
    if "stem" not in feats.columns:
        if "basename" in feats.columns:
            feats["stem"] = feats["basename"].astype(str).map(norm_stem)
        elif "file" in feats.columns:
            feats["stem"] = feats["file"].astype(str).map(norm_stem)
        else:
            raise SystemExit("[err] features.csv must contain 'basename' or 'file' to derive stem")
    else:
        feats["stem"] = feats["stem"].astype(str).str.lower()

    labs = pd.read_csv(labels_csv)
    labs.columns = [c.strip() for c in labs.columns]

    if label_col is None:
        label_col = guess_label_col(labs)
    if label_col is None or label_col not in labs.columns:
        raise SystemExit("[err] label column not found in labels_csv. Pass --label_col")

    if fname_col is None:
        fname_col = guess_fname_col(labs)
    if fname_col is None or fname_col not in labs.columns:
        raise SystemExit("[err] filename column not found in labels_csv. Pass --fname_col")

    labs["_key"] = labs[fname_col].astype(str).map(norm_stem)
    labs_small = labs[["_key", label_col]].copy()
    labs_small.rename(columns={label_col: "label"}, inplace=True)

    # к числам 0/1
    labs_small["label"] = labs_small["label"].map(label_to_int)

    df = feats.merge(labs_small, left_on="stem", right_on="_key", how="left")
    df.drop(columns=["_key"], inplace=True, errors="ignore")
    return df


def split_subjectwise(df: pd.DataFrame, seed: int = 42):
    # группы — по subject если есть, иначе по stem
    if "subject" in df.columns:
        groups = df["subject"].astype(str).values
    else:
        groups = df["stem"].astype(str).values

    idx = np.arange(len(df))
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    tr_idx, te_idx = next(gss1.split(idx, groups=groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    tr2_idx, dv_idx = next(gss2.split(tr_idx, groups=groups[tr_idx]))
    tr_idx = tr_idx[tr2_idx]
    return tr_idx, dv_idx, te_idx


def main():
    ap = argparse.ArgumentParser(description="Train XGBoost on features.csv + run_data.csv")
    ap.add_argument("--features_csv", required=True, help="path to features.csv (with per-file features)")
    ap.add_argument("--labels_csv",   required=True, help="path to run_data.csv (with labels)")
    ap.add_argument("--label_col", default=None, help="label column in labels_csv (e.g., 'No inj/ inj')")
    ap.add_argument("--fname_col",  default=None, help="filename column in labels_csv")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--use_gpu", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); ensure_dir(out_dir)

    # 1) merge
    df = load_and_merge(args.features_csv, args.labels_csv, args.label_col, args.fname_col)

    # 2) оставить только валидные метки
    if "label" not in df.columns:
        raise SystemExit("[err] merge produced no 'label' column — check filename keys")

    valid = df["label"].isin([0, 1])
    dropped = int((~valid).sum())
    if dropped:
        print(f"[warn] dropped {dropped} rows without valid labels")

    df = df.loc[valid].reset_index(drop=True)
    if len(df) == 0:
        raise SystemExit("[err] no labeled rows after merge")

    # 3) построим X/y (+ вспомогательные столбцы)
    id_cols = [c for c in ["file", "basename", "subject", "stem", "label"] if c in df.columns]
    y = df["label"].astype(int).values
    # все числовые колонки минус label и, на всякий случай, subject если он числовой
    X = df.select_dtypes(include=[np.number]).copy()
    for c in ["label", "subject"]:
        if c in X.columns:
            X.drop(columns=[c], inplace=True)

    # sanity
    assert len(X) == len(y)
    print(f"[info] samples={len(y)} | features={X.shape[1]} | pos={int((y==1).sum())} neg={int((y==0).sum())}")

    # 4) split subject-wise
    tr_idx, dv_idx, te_idx = split_subjectwise(df, seed=args.seed)
    def stat(idx):
        subs = df.iloc[idx]
        return {"n": len(idx), "pos": int((subs.label==1).sum()), "neg": int((subs.label==0).sum()),
                "subjects": int(len(subs["subject"].astype(str).unique())) if "subject" in subs else None}
    print("[split] subject-wise 60/20/20")
    print("  train:", stat(tr_idx))
    print("  dev:  ", stat(dv_idx))
    print("  test: ", stat(te_idx))

    split_map = pd.Series("train", index=df.index)
    split_map.iloc[dv_idx] = "dev"
    split_map.iloc[te_idx] = "test"
    split_df = df[id_cols].copy()
    split_df["split"] = split_map.values
    split_df.to_csv(out_dir / "split_subjectwise.csv", index=False)

    X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
    X_dv, y_dv = X.iloc[dv_idx], y[dv_idx]
    X_te, y_te = X.iloc[te_idx], y[te_idx]

    # 5) class weights (balanced)
    classes = np.array([0, 1], dtype=np.int32)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_tr)
    w0, w1 = float(cw[0]), float(cw[1])
    sw_tr = np.where(y_tr == 0, w0, w1).astype(np.float32)

    # 6) XGBoost
    clf = xgb.XGBClassifier(
        objective="binary:logistic",
        tree_method="gpu_hist" if args.use_gpu else "hist",
        n_estimators=3000,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=6,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=args.seed,
        eval_metric="logloss",
        n_jobs=0,
    )

    clf.fit(
        X_tr, y_tr,
        sample_weight=sw_tr,
        eval_set=[(X_tr, y_tr), (X_dv, y_dv)],
        verbose=100,
        early_stopping_rounds=150,
    )

    # 7) метрики
    def safe_auc(y_true, prob):
        try:
            return float(roc_auc_score(y_true, prob))
        except Exception:
            return float("nan")

    def safe_ap(y_true, prob):
        try:
            return float(average_precision_score(y_true, prob))
        except Exception:
            return float("nan")

    prob_dv = clf.predict_proba(X_dv)[:, 1]
    prob_te = clf.predict_proba(X_te)[:, 1]
    thr = 0.5
    pred_te = (prob_te >= thr).astype(int)

    metrics = {
        "dev_auroc": safe_auc(y_dv, prob_dv),
        "dev_auprc": safe_ap(y_dv, prob_dv),
        "test_auroc": safe_auc(y_te, prob_te),
        "test_auprc": safe_ap(y_te, prob_te),
        "threshold": thr,
    }

    try:
        metrics["test_confusion_matrix"] = confusion_matrix(y_te, pred_te).tolist()
        metrics["test_report"] = classification_report(y_te, pred_te, digits=3)
    except Exception:
        pass

    json.dump(metrics, open(out_dir / "metrics.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    # 8) сохранить модель и мета
    clf.get_booster().save_model(str(out_dir / "xgb.json"))
    json.dump(list(X.columns), open(out_dir / "features_cols.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    # 9) сохранить прогнозы
    def dump_pred(idx, prob, name):
        tmp = df.iloc[idx][["file","basename","subject"]].copy() if "subject" in df.columns else df.iloc[idx][["file","basename"]].copy()
        tmp["y"] = df.iloc[idx]["label"].values
        tmp["prob_injury"] = prob
        tmp.to_csv(out_dir / f"pred_{name}.csv", index=False)

    dump_pred(dv_idx, prob_dv, "dev")
    dump_pred(te_idx, prob_te, "test")

    print("[done] saved to", out_dir)


if __name__ == "__main__":
    main()
