#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    accuracy_score, roc_auc_score,
)
import xgboost as xgb
# --- NEW: imports for plots ---
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize
import re
try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_SGKF = True
except Exception:
    from sklearn.model_selection import GroupShuffleSplit
    HAS_SGKF = False
import xgboost as xgb  # уже есть

# ========= plotting helpers =========
def save_confusion_matrix(cm, class_names, out_path, title="Confusion Matrix (test)"):
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def save_roc_ovr(y_true, prob, class_names, out_path, title="ROC (OVR)"):
    # y_true: int (0..K-1), prob: [N,K]
    K = prob.shape[1]
    y_bin = label_binarize(y_true, classes=list(range(K)))  # [N,K]
    fpr, tpr, roc_auc = {}, {}, {}
    valid = []
    for i in range(K):
        yi = y_bin[:, i]
        # пропускаем класс, если в тесте нет ни одного положительного
        if yi.sum() == 0 or yi.sum() == len(yi):
            continue
        fi, ti, _ = roc_curve(yi, prob[:, i])
        fpr[i], tpr[i] = fi, ti
        roc_auc[i] = auc(fi, ti)
        valid.append(i)
    if not valid:
        return  # нечего рисовать

    # macro average
    # интерполяция на общую сетку FPR
    all_fpr = np.unique(np.concatenate([fpr[i] for i in valid]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in valid:
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= len(valid)
    macro_auc = auc(all_fpr, mean_tpr)

    plt.figure(figsize=(6,5), dpi=150)
    for i in valid:
        plt.plot(fpr[i], tpr[i], lw=1, alpha=0.9, label=f"{class_names[i]} (AUC={roc_auc[i]:.3f})")
    plt.plot(all_fpr, mean_tpr, lw=2.0, label=f"macro-average (AUC={macro_auc:.3f})")
    plt.plot([0,1],[0,1],"--",lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(fontsize=8, loc="lower right", ncol=1, frameon=True)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def save_confidence_buckets_multiclass(prob, y_true, y_pred, out_path, model_name="xgb"):
    """
    Уверенность = top-1 вероятность (max по классам).
    Три панели: все примеры, корректные, ошибочные.
    """
    prob = np.asarray(prob, dtype=np.float32)
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    conf = prob.max(axis=1)  # top-1 confidence

    bins = [0.0, 0.5, 0.6, 0.8, 1.0001]
    labels = ["conf <50%", "50–60%", "60–80%", ">80%"]

    def bucket_counts(mask):
        if mask.sum() == 0:
            return np.zeros(len(labels), dtype=int)
        h, _ = np.histogram(conf[mask], bins=bins)
        return h

    panels = [
        (np.ones_like(conf, dtype=bool),        f"{model_name} — ALL",        None),
        (y_pred == y_true,                      f"{model_name} — CORRECT",    None),
        (y_pred != y_true,                      f"{model_name} — WRONG",      None),
    ]

    fig, axs = plt.subplots(1, 3, figsize=(13.5, 3.2), dpi=150)
    for ax, (mask, title, _) in zip(axs, panels):
        cnts = bucket_counts(mask)
        bars = ax.barh(range(len(labels)), cnts, height=0.55)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Count")
        xmax = max(int(cnts.max()), 1)
        ax.set_xlim(0, xmax * 1.15)
        ax.bar_label(bars, labels=[str(int(v)) for v in cnts], padding=3, fontsize=9)
        ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def stem_lower(s: str) -> str:
    b = os.path.basename(str(s))
    return os.path.splitext(b)[0].lower()

def main():
    ap = argparse.ArgumentParser(description="Train XGBoost (multiclass) with robust split")
    ap.add_argument("--features_csv", required=True)
    ap.add_argument("--labels_csv", default=None)
    ap.add_argument("--label_col", default=None)
    ap.add_argument("--fname_in_labels", default=None)
    ap.add_argument("--label_index_csv", default=None)
    ap.add_argument("--out_dir", default="out_xgb_multiclass")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_gpu", action="store_true")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # ===== 1) Load features
    F = pd.read_csv(args.features_csv)
    fname_feat = next((c for c in ("stem","basename","file") if c in F.columns), None)
    if fname_feat is None:
        raise SystemExit("[err] features_csv должен содержать колонку stem/basename/file для связи с labels_csv")

    # ===== 2) Merge labels (expect 1..n)
    if "label" in F.columns and F["label"].notna().any():
        DF = F.copy()
    else:
        if not args.labels_csv:
            raise SystemExit("[err] в features.csv нет 'label'; укажите --labels_csv (filename,label)")
        L = pd.read_csv(args.labels_csv)
        fname_lab = args.fname_in_labels or next((c for c in ("filename","file","path","basename","stem") if c in L.columns), None)
        print(fname_lab)
        if fname_lab is None:
            raise SystemExit("[err] не нашёл колонку имени файла в labels_csv (ожидал filename/file/path/basename/stem)")
        label_col = args.label_col or ("label" if "label" in L.columns else None)
        if label_col is None:
            raise SystemExit("[err] не нашёл колонку меток. Укажите --label_col")

        F["_key"] = F[fname_feat].astype(str).map(stem_lower)
        L["_key"] = L[fname_lab].astype(str).map(stem_lower)
        lab_small = L[["_key", label_col]].drop_duplicates("_key", keep="last").rename(columns={label_col:"label"})
        DF = F.merge(lab_small, on="_key", how="left").drop(columns=["_key"])

    DF["label"] = pd.to_numeric(DF["label"], errors="coerce")
    DF = DF[DF["label"].notna()].reset_index(drop=True)
    labels_sorted = sorted(DF["label"].astype(int).unique().tolist())
    if min(labels_sorted) < 0:
        raise SystemExit("[err] ожидаются метки 1..n или 0..n-1")

    # ===== 3) Map labels to 0..K-1 for XGBoost
    if 0 in labels_sorted:
        label_to_zero = {int(v): int(v) for v in labels_sorted}
    else:
        label_to_zero = {int(v): int(v-1) for v in labels_sorted}
    DF["_y"] = DF["label"].astype(int).map(label_to_zero).astype("int32")
    
        # --- origin для группового сплита: имя без суффиксов _chunk### / _### в конце
    origin_source = "stem" if "stem" in DF.columns else fname_feat  # fname_feat найден раньше из features_csv
    def to_origin(v: str) -> str:
        name = os.path.basename(str(v))
        st = os.path.splitext(name)[0].lower()
        # убираем ..._chunk000 / ..._chunk12 / ..._000
        st = re.sub(r"(?:_chunk\d+|_\d+)$", "", st)
        return st

    DF["origin"] = DF[origin_source].astype(str).map(to_origin)


    # ===== 4) X, y
    y_all = DF["_y"].to_numpy()
    X_all = DF.select_dtypes(include=[np.number]).drop(columns=["label","_y"], errors="ignore")
    feature_names = list(X_all.columns)
    X_all = X_all.to_numpy(dtype=np.float32)

    # ===== 5) Групповой стратифицированный split, без утечки кусков одного файла
    y_all = DF["_y"].to_numpy()
    X_all = DF.select_dtypes(include=[np.number]).drop(columns=["label","_y"], errors="ignore")
    feature_names = list(X_all.columns)
    X_all = X_all.to_numpy(dtype=np.float32)
    groups_all = DF["origin"].values
    idx_all = np.arange(len(y_all))

    # редкие классы (<2 объектов) полностью в train
    counts = pd.Series(y_all).value_counts().sort_index()
    rare_classes = counts[counts < 2].index.tolist()
    mask_rare = np.isin(y_all, rare_classes)
    idx_rare = idx_all[mask_rare]
    idx_rest = idx_all[~mask_rare]

    if len(idx_rest) == 0:
        raise SystemExit("[err] все классы слишком редкие (<2). Увеличьте данные или объедините классы.")

    X_rest, y_rest = X_all[idx_rest], y_all[idx_rest]
    groups_rest = groups_all[idx_rest]

    if HAS_SGKF:
        n_splits = max(2, min(10, int(round(1.0/args.test_size)) if 0.05 <= args.test_size <= 0.5 else 5))
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
        tr_r, te_r = next(sgkf.split(X_rest, y_rest, groups_rest))
    else:
        print("[warn] StratifiedGroupKFold недоступен — GroupShuffleSplit (без строгой стратификации).")
        gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
        tr_r, te_r = next(gss.split(X_rest, y_rest, groups_rest))

    # глобальные индексы
    g_tr = idx_rest[tr_r]
    g_te = idx_rest[te_r]

    # редкие классы — только в train
    if len(idx_rare):
        g_tr = np.concatenate([g_tr, idx_rare])

    # проверка отсутствия пересечения групп
    inter = set(DF.iloc[g_tr]["origin"]) & set(DF.iloc[g_te]["origin"])
    if inter:
        raise SystemExit(f"[err] Найдены пересечения групп между train/test: {sorted(list(inter))[:5]} ...")

    # в тесте не должно быть классов, отсутствующих в train
    if not set(np.unique(y_all[g_te])).issubset(set(np.unique(y_all[g_tr]))):
        keep_mask = np.isin(y_all[g_te], np.unique(y_all[g_tr]))
        g_te = g_te[keep_mask]

    Xtr, Xte = X_all[g_tr], X_all[g_te]
    ytr, yte = y_all[g_tr], y_all[g_te]

    print(f"[split] train={len(ytr)}  test={len(yte)}  "
        f"groups_train={DF.iloc[g_tr]['origin'].nunique()}  groups_test={DF.iloc[g_te]['origin'].nunique()}")

    # (опц.) сохраняем списки сплита для контроля
    cols_keep = [c for c in [fname_feat, "basename", "stem"] if c in DF.columns]
    train_tbl = DF.iloc[g_tr][cols_keep + ["origin", "_y", "label"]].copy(); train_tbl["split"] = "train"
    test_tbl  = DF.iloc[g_te][cols_keep + ["origin", "_y", "label"]].copy();  test_tbl["split"]  = "test"
    pd.concat([train_tbl, test_tbl], axis=0, ignore_index=True).to_csv(os.path.join(args.out_dir, "split_all.csv"), index=False)
    train_tbl.to_csv(os.path.join(args.out_dir, "split_train.csv"), index=False)
    test_tbl.to_csv(os.path.join(args.out_dir, "split_test.csv"), index=False)
    
    # ===== 5.1) Кол-во классов берём из train
    classes_in_train = np.unique(ytr)
    num_class = int(classes_in_train.size)

    # убеждаемся, что метки в трейне идут подряд 0..K-1
    assert classes_in_train.min() == 0 and classes_in_train.max() == num_class - 1, \
        f"Train labels must be 0..K-1, got {classes_in_train}"
    print(f"[info] num_class used: {num_class}")


    # ===== 6) XGBoost
    clf = xgb.XGBClassifier(
        n_estimators=1200,
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=4,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        objective="multi:softprob",
        num_class=num_class,
        tree_method="gpu_hist" if args.use_gpu else "hist",
        eval_metric=["mlogloss","merror"],
        random_state=args.seed,
        n_jobs=0,
    )

    monitor = xgb.callback.EvaluationMonitor(period=25)  # печать каждые 25 итераций
    clf.fit(
        Xtr, ytr,
        eval_set=[(Xtr, ytr), (Xte, yte)],
        verbose=False,
        early_stopping_rounds=100,
        callbacks=[monitor]
    )


    # ===== 7) Metrics
    prob = clf.predict_proba(Xte)
    pred = prob.argmax(axis=1)
    acc = accuracy_score(yte, pred)
    f1_macro = f1_score(yte, pred, average="macro")
    f1_micro = f1_score(yte, pred, average="micro")
    try:
        auc_macro_ovr = roc_auc_score(yte, prob, multi_class="ovr", average="macro")
    except Exception:
        auc_macro_ovr = np.nan

    print("\nAccuracy:", round(float(acc), 4))
    print("F1-macro:", round(float(f1_macro), 4))
    print("F1-micro:", round(float(f1_micro), 4))
    print("Macro AUC (OVR):", "nan" if np.isnan(auc_macro_ovr) else round(float(auc_macro_ovr), 4))
    print("\nConfusion matrix (rows=true, cols=pred):\n", confusion_matrix(yte, pred))

    # красивые имена классов (по желанию)
    target_names = None
    if args.label_index_csv and os.path.exists(args.label_index_csv):
        LI = pd.read_csv(args.label_index_csv)
        # label в справочнике ожидается 1..n, вернём к 0..K-1 через обратную маппу:
        # построим zero-based -> human label
        # восстановим исходный словарь
        unique_raw = sorted(DF["label"].astype(int).unique().tolist())
        if 0 in unique_raw:
            raw_to_zero = {int(v): int(v) for v in unique_raw}
        else:
            raw_to_zero = {int(v): int(v-1) for v in unique_raw}
        zero_to_raw = {z:r for r,z in raw_to_zero.items()}
        LI["_zero"] = LI["label"].astype(int).map({r:z for z,r in zero_to_raw.items()})
        LI = LI[LI["_zero"].notna()].sort_values("_zero")
        target_names = LI["InjuryClass"].astype(str).tolist()
    if target_names is None:
        target_names = [f"class_{i}" for i in range(num_class)]
    print("\nReport:\n", classification_report(yte, pred, target_names=target_names, digits=3))

    # ===== 8) Save artifacts
    booster = clf.get_booster()
    booster.save_model(os.path.join(args.out_dir, "xgb_multiclass.json"))
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

    # test predictions
    inv_map = {}  # zero->raw
    # построим по train-лейблам
    raw_labels = sorted(DF["label"].astype(int).unique().tolist())
    if 0 in raw_labels:
        label_to_zero_all = {int(v): int(v) for v in raw_labels}
    else:
        label_to_zero_all = {int(v): int(v-1) for v in raw_labels}
    inv_map = {z:r for r,z in label_to_zero_all.items()}

    test_df = pd.DataFrame({
        "y_true": [inv_map.get(int(v), int(v)) for v in yte],
        "y_pred": [inv_map.get(int(v), int(v)) for v in pred],
        "y_pred_topprob": prob.max(axis=1)
    })
    for j in range(prob.shape[1]):
        human_lbl = inv_map.get(j, j)
        test_df[f"proba_{human_lbl}"] = prob[:, j]
    test_df.to_csv(os.path.join(args.out_dir, "test_predictions.csv"), index=False)

    json.dump(
        {"label_to_zero_based": label_to_zero, "classes_in_train": sorted(map(int, np.unique(ytr)))},
        open(os.path.join(args.out_dir, "label_mapping.json"), "w"), ensure_ascii=False, indent=2
    )
    
        # ===== 7.1) Save metrics & plots =====
    cm = confusion_matrix(yte, pred)

    # JSON с метриками
    report_json = classification_report(yte, pred, target_names=target_names, digits=4, output_dict=True)
    metrics_payload = {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_micro": float(f1_micro),
        "auc_macro_ovr": None if np.isnan(auc_macro_ovr) else float(auc_macro_ovr),
        "n_test": int(len(yte)),
        "classes": target_names,
        "per_class": report_json  # precision/recall/f1/support по классам
    }
    with open(os.path.join(args.out_dir, "metrics_test.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

    # Текстовый отчёт и матрица в CSV
    with open(os.path.join(args.out_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(classification_report(yte, pred, target_names=target_names, digits=3))
    pd.DataFrame(cm, index=target_names, columns=target_names)\
      .to_csv(os.path.join(args.out_dir, "confusion_matrix.csv"), index=True)

    # Картинки
    save_confusion_matrix(cm, target_names, os.path.join(args.out_dir, "cm.png"))
    save_roc_ovr(yte, prob, target_names, os.path.join(args.out_dir, "roc_ovr.png"))
    save_confidence_buckets_multiclass(prob, yte, pred, os.path.join(args.out_dir, "confidence_buckets.png"), model_name="xgb-multiclass")


    print("\n[done] модель сохранена в", args.out_dir)

if __name__ == "__main__":
    main()
