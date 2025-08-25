#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix, recall_score
)
import xgboost as xgb

def stem_lower(s: str) -> str:
    b = os.path.basename(str(s))
    return os.path.splitext(b)[0].lower()

def map_label(v):
    s = str(v).strip().lower()
    if s in ("1","injury","inj","yes","y","true","t","1.0"): return 1
    if s in ("0","no injury","no inj","no","n","false","f","0.0"): return 0
    try:
        f = float(s)
        if f in (0.0, 1.0): return int(f)
    except: pass
    return np.nan

def pick_threshold_by_dev(prob_dev: np.ndarray, y_dev: np.ndarray,
                          min_recall_pos: float = 0.90) -> float:
    """Сканируем thr ∈ [0.05..0.95] и выбираем тот, который
       максимизирует среднюю recall (r0+r1)/2 при условии recall(1)≥min_recall_pos.
       Если DEV одно-классовый, возвращаем 0.5.
    """
    if len(np.unique(y_dev)) < 2:
        print("[thr] WARNING: DEV одно-классовый → thr=0.5")
        return 0.5
    thr_grid = np.linspace(0.05, 0.95, 181)
    best_thr, best_bal, best_r0, best_r1 = 0.5, -1.0, 0.0, 0.0
    for thr in thr_grid:
        pp = (prob_dev >= thr).astype(int)
        r0 = recall_score(y_dev, pp, pos_label=0)
        r1 = recall_score(y_dev, pp, pos_label=1)
        bal = 0.5 * (r0 + r1)
        if (r1 >= min_recall_pos) and (bal > best_bal):
            best_thr, best_bal, best_r0, best_r1 = float(thr), float(bal), float(r0), float(r1)
    if best_bal < 0:  # не нашлось с ограничением → возьмём просто максимальный bal
        for thr in thr_grid:
            pp = (prob_dev >= thr).astype(int)
            r0 = recall_score(y_dev, pp, pos_label=0)
            r1 = recall_score(y_dev, pp, pos_label=1)
            bal = 0.5 * (r0 + r1)
            if bal > best_bal:
                best_thr, best_bal, best_r0, best_r1 = float(thr), float(bal), float(r0), float(r1)
    print(f"[thr] chosen on DEV={best_thr:.3f} | recall0={best_r0:.3f} recall1={best_r1:.3f} bal={best_bal:.3f}")
    return best_thr

def main():
    ap = argparse.ArgumentParser(description="Train XGBoost with DEV-threshold tuning")
    ap.add_argument("--features_csv", required=True, help="features.csv из экстрактора")
    ap.add_argument("--labels_csv", default=None, help="CSV с метками (если в features.csv нет label)")
    ap.add_argument("--label_col", default=None, help="Имя колонки метки в labels_csv (напр. \"No inj/ inj\")")
    ap.add_argument("--fname_in_labels", default=None, help="Колонка с именем файла в labels_csv (auto: filename/file/path/basename/stem)")
    ap.add_argument("--out_dir", default="out_xgb_simple")
    ap.add_argument("--use_gpu", action="store_true")
    ap.add_argument("--min_recall_pos", type=float, default=0.90, help="Цель по recall(класс=1) при подборе thr на DEV")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) читаем фичи
    F = pd.read_csv(args.features_csv)
    # ключ для джоина по стему
    fname_feat = None
    for c in ("stem","basename","file"):
        if c in F.columns: fname_feat = c; break
    if fname_feat is None:
        raise SystemExit("[err] features_csv должен содержать колонку stem/basename/file для связи с labels_csv")

    # если метки уже есть в features.csv — используем их
    if "label" in F.columns and F["label"].notna().any():
        DF = F.copy()
    else:
        if not args.labels_csv:
            raise SystemExit("[err] в features.csv нет 'label'; укажите --labels_csv с метками")
        L = pd.read_csv(args.labels_csv)

        # колонка имени файла в labels_csv
        fname_lab = args.fname_in_labels
        if fname_lab is None:
            for c in ("filename","file","path","basename","stem"):
                if c in L.columns: fname_lab = c; break
        if fname_lab is None:
            raise SystemExit("[err] не нашёл колонку имени файла в labels_csv (ожидал filename/file/path/basename/stem)")

        # колонка метки
        label_col = args.label_col
        if label_col is None:
            for c in L.columns:
                cl = c.lower()
                if cl == "label" or "inj" in cl or c.strip() in ("No inj/ inj",):
                    label_col = c; break
        if label_col is None:
            raise SystemExit("[err] не нашёл колонку меток. Укажите --label_col")

        # джойн по стему
        F["_key"] = F[fname_feat].astype(str).map(stem_lower)
        L["_key"] = L[fname_lab].astype(str).map(stem_lower)
        lab_small = L[["_key", label_col]].drop_duplicates("_key", keep="last").rename(columns={label_col: "label"})
        DF = F.merge(lab_small, on="_key", how="left").drop(columns=["_key"])

    # 2) приводим метку к 0/1 и чистим
    DF["label"] = DF["label"].map(map_label)
    DF = DF[DF["label"].isin([0,1])].reset_index(drop=True)
    if len(DF) == 0:
        raise SystemExit("[err] после приведения меток ни одной валидной строки (0/1)")

    # 3) формируем X, y
    y = DF["label"].astype("int32").to_numpy()
    X = DF.select_dtypes(include=[np.number]).drop(columns=["label"], errors="ignore")
    feature_names = list(X.columns)
    X = X.to_numpy(dtype=np.float32)

    # 4) subject-agnostic сплит: Train/Dev/Test = 60/20/20
    X_tr_all, X_te, y_tr_all, y_te = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    X_tr, X_dv, y_tr, y_dv = train_test_split(X_tr_all, y_tr_all, test_size=0.25, random_state=42, stratify=y_tr_all)
    print(f"[split] train={len(y_tr)}  dev={len(y_dv)}  test={len(y_te)}")

    # 5) XGBoost (учим на TRAIN, early stopping по DEV)
    clf = xgb.XGBClassifier(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=6,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        objective="binary:logistic",
        tree_method="gpu_hist" if args.use_gpu else "hist",
        eval_metric="logloss",
        random_state=42,
        n_jobs=0,
    )

    clf.fit(
        X_tr, y_tr,
        eval_set=[(X_dv, y_dv)],
        verbose=False,
        early_stopping_rounds=100
    )

    # 6) подбор порога на DEV
    prob_dv = clf.predict_proba(X_dv)[:, 1]
    thr = pick_threshold_by_dev(prob_dv, y_dv, min_recall_pos=args.min_recall_pos)

    # 7) финальные метрики на TEST при выбранном thr
    prob_te = clf.predict_proba(X_te)[:, 1]
    pred_te = (prob_te >= thr).astype(int)
    auc_te = roc_auc_score(y_te, prob_te)
    cm_te  = confusion_matrix(y_te, pred_te)
    rep_te = classification_report(y_te, pred_te, digits=3)

    print("\n[TEST] AUC:", round(float(auc_te), 4))
    print("[TEST] Confusion matrix:\n", cm_te)
    print("\n[TEST] Report:\n", rep_te)

    # 8) сохранения: модель, список фич, выбранный порог, импорта́нсы
    booster = clf.get_booster()
    booster.save_model(os.path.join(args.out_dir, "xgb.json"))
    json.dump(feature_names, open(os.path.join(args.out_dir, "features_cols.json"), "w"), ensure_ascii=False, indent=2)
    open(os.path.join(args.out_dir, "threshold.txt"), "w").write(str(thr))

    bst = clf.get_booster()
    kinds = ["gain","total_gain","weight","cover","total_cover"]
    scores = {k: bst.get_score(importance_type=k) for k in kinds}
    imp = pd.DataFrame({
        "feature": feature_names,
        **{k: [scores[k].get(f"f{i}", 0.0) for i in range(len(feature_names))] for k in kinds}
    })
    (imp.assign(total_gain_pct=lambda d: d.total_gain/(d.total_gain.sum()+1e-12))
        .sort_values("total_gain", ascending=False)
        .to_csv(os.path.join(args.out_dir, "feature_importance.csv"), index=False))

    # 9) быстрый dump метрик
    metrics = {
        "dev_threshold": float(thr),
        "test_auc": float(auc_te),
        "test_confusion_matrix": cm_te.tolist(),
        "test_report": rep_te,
    }
    json.dump(metrics, open(os.path.join(args.out_dir, "metrics.json"), "w"), ensure_ascii=False, indent=2)

    print("\n[done] saved to", args.out_dir)

if __name__ == "__main__":
    main()
