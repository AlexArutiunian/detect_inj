#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, argparse
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_SGKF = True
except Exception:
    HAS_SGKF = False
import xgboost as xgb

# plotting
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, ConfusionMatrixDisplay

def stem_lower(s: str) -> str:
    b = os.path.basename(str(s))
    return os.path.splitext(b)[0].lower()

def map_label(v):
    s = str(v).strip().lower()
    if s in ("1","injury","inj","yes","y","true","t","1.0"): return 1
    if s in ("0","no injury","no inj","no","n","false","f","0.0"): return 0
    try:
        f = float(s)
        if f in (0.0,1.0): return int(f)
    except: pass
    return np.nan

def main():
    ap = argparse.ArgumentParser(description="Train XGBoost on features.csv (+ labels CSV if needed)")
    ap.add_argument("--features_csv", required=True)
    ap.add_argument("--labels_csv", default=None)
    ap.add_argument("--label_col", default=None)
    ap.add_argument("--fname_in_labels", default=None)
    ap.add_argument("--out_dir", default="out_xgb_simple")
    ap.add_argument("--use_gpu", action="store_true")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--importance_csv", default=None)
    ap.add_argument("--min_total_gain", type=float, default=10.0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # --- load features + labels
    F = pd.read_csv(args.features_csv)
    fname_feat = next((c for c in ("stem","basename","file") if c in F.columns), None)
    if fname_feat is None:
        raise SystemExit("[err] features_csv должен содержать колонку stem/basename/file")

    if "label" in F.columns and F["label"].notna().any():
        DF = F.copy()
    else:
        if not args.labels_csv:
            raise SystemExit("[err] в features.csv нет 'label'; укажите --labels_csv")
        L = pd.read_csv(args.labels_csv)
        fname_lab = args.fname_in_labels or next((c for c in ("filename","file","path","basename","stem") if c in L.columns), None)
        if fname_lab is None:
            raise SystemExit("[err] не нашёл колонку имени файла в labels_csv")
        label_col = args.label_col or next((c for c in L.columns if (c.lower()=="label" or "inj" in c.lower() or c.strip()=="No inj/ inj")), None)
        if label_col is None:
            raise SystemExit("[err] не нашёл колонку меток. Укажите --label_col")

        F["_key"] = F[fname_feat].astype(str).map(stem_lower)
        L["_key"] = L[fname_lab].astype(str).map(stem_lower)
        lab_small = L[["_key", label_col]].drop_duplicates("_key", keep="last").rename(columns={label_col: "label"})
        DF = F.merge(lab_small, on="_key", how="left").drop(columns=["_key"])

    DF["label"] = DF["label"].map(map_label)
    DF = DF[DF["label"].isin([0,1])].reset_index(drop=True)
    if len(DF) == 0: raise SystemExit("[err] нет валидных 0/1 меток")

    # group key (без _<chunk> в конце)
    origin_source = "stem" if "stem" in DF.columns else fname_feat
    DF["origin"] = DF[origin_source].astype(str).map(lambda v: re.sub(r"_(\d+)$", "", os.path.splitext(os.path.basename(str(v)))[0].lower()))

    # X, y
    y = DF["label"].astype("int32").to_numpy()
    X_df = DF.select_dtypes(include=[np.number]).drop(columns=["label"], errors="ignore")
    X_df = X_df.drop(columns=["n_frames"], errors="ignore")

    if args.importance_csv and os.path.exists(args.importance_csv):
        imp = pd.read_csv(args.importance_csv)
        if not {"feature","total_gain"}.issubset(imp.columns):
            raise SystemExit("[err] importance_csv должен содержать 'feature' и 'total_gain'")
        selected = [f for f in imp.loc[imp["total_gain"] > args.min_total_gain, "feature"].tolist() if f in X_df.columns]
        if not selected:
            raise SystemExit("[err] по порогу total_gain ничего не осталось")
        X_df = X_df[selected]
        print(f"[info] Использую {len(selected)} фич по важности")

    feature_names = list(X_df.columns)
    X = X_df.to_numpy(dtype=np.float32)
    groups = DF["origin"].values

    # split by groups
    if HAS_SGKF:
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        train_idx, test_idx = next(sgkf.split(X, y, groups))
    else:
        print("[warn] нет StratifiedGroupKFold — GroupShuffleSplit")
        gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups))

    inter = set(DF.loc[train_idx, "origin"]) & set(DF.loc[test_idx, "origin"])
    if inter: raise SystemExit(f"[err] пересечение групп: {list(sorted(inter))[:5]} ...")

    Xtr, Xte, ytr, yte = X[train_idx], X[test_idx], y[train_idx], y[test_idx]

    # save split lists
    cols_keep = [c for c in [fname_feat, "basename", "stem"] if c in DF.columns]
    train_tbl = DF.iloc[train_idx][cols_keep + ["origin", "label"]].copy()
    test_tbl  = DF.iloc[test_idx ][cols_keep + ["origin", "label"]].copy()
    train_tbl["split"] = "train"; test_tbl["split"] = "test"
    pd.concat([train_tbl, test_tbl]).to_csv(os.path.join(args.out_dir, "split_all.csv"), index=False)
    train_tbl.to_csv(os.path.join(args.out_dir, "split_train.csv"), index=False)
    test_tbl.to_csv(os.path.join(args.out_dir, "split_test.csv"), index=False)

    print(f"[split] train={len(ytr)}  test={len(yte)}  groups_train={DF.loc[train_idx,'origin'].nunique()}  groups_test={DF.loc[test_idx,'origin'].nunique()}")

    # model
    clf = xgb.XGBClassifier(
        n_estimators=800, learning_rate=0.05, max_depth=6, min_child_weight=6,
        subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
        objective="binary:logistic", tree_method="gpu_hist" if args.use_gpu else "hist",
        eval_metric="logloss", random_state=42, n_jobs=0,
    )

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, dv_idx = next(gss2.split(Xtr, ytr, groups=DF.iloc[train_idx]["origin"].values))
    Xtr2, ytr2, Xdv, ydv = Xtr[tr_idx], ytr[tr_idx], Xtr[dv_idx], ytr[dv_idx]

    clf.fit(Xtr2, ytr2, eval_set=[(Xtr2, ytr2), (Xdv, ydv)], early_stopping_rounds=100, verbose=False)

    # TEST metrics
    prob = clf.predict_proba(Xte)[:, 1]
    pred = (prob >= 0.5).astype(int)
    auc  = roc_auc_score(yte, prob)
    cm   = confusion_matrix(yte, pred)
    print("\nAUC:", round(float(auc), 4))
    print("Confusion matrix:\n", cm)
    print("\nReport:\n", classification_report(yte, pred, digits=3))

    # === PLOTS ===
    # ROC
    try:
        fpr, tpr, _ = roc_curve(yte, prob)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        plt.plot([0, 1], [0, 1], "--", linewidth=1)
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title("ROC (test)"); plt.legend(loc="lower right"); plt.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, "roc_curve.png"), dpi=150); plt.close()
    except ValueError as e:
        print(f"[warn] ROC не построен: {e}")

    # Confusion Matrix
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Confusion Matrix (test)")
    fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, "confusion_matrix.png"), dpi=150); plt.close(fig)

    # --- Confidence buckets (как на твоём скрине)
    def save_confidence_buckets(prob, y_true, out_dir, model_name="xgb", thr=0.5):
        prob = np.asarray(prob).ravel()              # p(injury)
        pred = (prob >= thr).astype(int)             # предсказанный класс
        conf = np.maximum(prob, 1.0 - prob)          # УВЕРЕННОСТЬ ОТНОСИТЕЛЬНО ПРЕДСКАЗАННОГО КЛАССА

        bins  = [0.0, 0.5, 0.6, 0.8, 1.01]
        labels = ["conf <50%", "conf 50–60%", "conf 60–80%", "conf >80%"]

        def bucket_counts(mask):
            h, _ = np.histogram(conf[mask], bins=bins)
            return h

        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 3, figsize=(13.5, 3.2), dpi=150)
        panels = [
            (np.ones_like(conf, dtype=bool),     f"{model_name} — Confidence buckets (all)",     axs[0]),
            (pred == 1,                          f"{model_name} — Injury buckets (pred==1)",     axs[1]),
            (pred == 0,                          f"{model_name} — No-injury buckets (pred==0)",  axs[2]),
        ]

        for mask, title, ax in panels:
            cnts = bucket_counts(mask)
            bars = ax.barh(range(len(labels)), cnts, height=0.55)
            ax.set_yticks(range(len(labels)), labels)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Count")
            xmax = max(cnts.max(), 1)
            ax.set_xlim(0, xmax * 1.15)                 # запас справа, чтобы числа не выходили за график
            ax.bar_label(bars, labels=[str(int(v)) for v in cnts], padding=3, fontsize=9)  # подписи ВНУТРИ/на краю
            ax.grid(axis="x", alpha=0.2)

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "confidence_buckets.png"), dpi=150)
        plt.close(fig)

    save_confidence_buckets(prob, yte, args.out_dir, model_name="xgb", thr=0.5)  # или thr=твой_порог

    # save model + importances
    booster = clf.get_booster()
    booster.save_model(os.path.join(args.out_dir, "xgb.json"))
    json.dump(feature_names, open(os.path.join(args.out_dir, "features_cols.json"), "w"), ensure_ascii=False, indent=2)

    kinds = ["gain","total_gain","weight","cover","total_cover"]
    scores = {k: booster.get_score(importance_type=k) for k in kinds}
    imp = pd.DataFrame({
        "feature": feature_names,
        **{k: [scores[k].get(f"f{i}", 0.0) for i in range(len(feature_names))] for k in kinds}
    })
    (imp.assign(total_gain_pct=lambda d: d.total_gain/(d.total_gain.sum()+1e-12))
       .sort_values("total_gain", ascending=False)
       .to_csv(os.path.join(args.out_dir, "feature_importance.csv"), index=False))

    print(f"\n[done] модель и графики сохранены в {args.out_dir}")

if __name__ == "__main__":
    main()
