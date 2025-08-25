#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import xgboost as xgb

def stem_lower(s: str) -> str:
    b = os.path.basename(str(s))
    return os.path.splitext(b)[0].lower()

def map_label(v):
    s = str(v).strip().lower()
    if s in ("1","injury","inj","yes","y","true","t","1.0"): return 1
    if s in ("0","no injury","no inj","no","n","false","f","0.0"): return 0
    # иногда в CSV уже числа/NaN
    try:
        f = float(s)
        if f in (0.0,1.0): return int(f)
    except: pass
    return np.nan

def main():
    ap = argparse.ArgumentParser(description="Train XGBoost on features.csv (+ labels CSV if needed)")
    ap.add_argument("--features_csv", required=True, help="features.csv из экстрактора")
    ap.add_argument("--labels_csv", default=None, help="CSV с метками (если в features.csv нет label)")
    ap.add_argument("--label_col", default=None, help="Имя колонки метки в labels_csv (напр. \"No inj/ inj\")")
    ap.add_argument("--fname_in_labels", default=None, help="Колонка с именем файла в labels_csv (auto: filename/file/path/basename/stem)")
    ap.add_argument("--out_dir", default="out_xgb_simple")
    ap.add_argument("--use_gpu", action="store_true")
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

        # определяем колонку с именем файла в labels.csv
        fname_lab = args.fname_in_labels
        if fname_lab is None:
            for c in ("filename","file","path","basename","stem"):
                if c in L.columns: fname_lab = c; break
        if fname_lab is None:
            raise SystemExit("[err] не нашёл колонку имени файла в labels_csv (ожидал filename/file/path/basename/stem)")

        # определяем колонку метки
        label_col = args.label_col
        if label_col is None:
            # авто-поиск
            for c in L.columns:
                cl = c.lower()
                if cl == "label" or "inj" in cl or c.strip() in ("No inj/ inj",):
                    label_col = c; break
        if label_col is None:
            raise SystemExit("[err] не нашёл колонку меток. Укажите --label_col")

        # готовим ключи
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
    # заберём только числовые колонки и уберём label
    X = DF.select_dtypes(include=[np.number]).drop(columns=["label"], errors="ignore")
    feature_names = list(X.columns)
    X = X.to_numpy(dtype=np.float32)

    # 4) трейн/тест (просто, стратифицировано)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 5) XGBoost
    clf = xgb.XGBClassifier(
        n_estimators=800,
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
        Xtr, ytr,
        eval_set=[(Xtr, ytr), (Xte, yte)],
        verbose=False,
        early_stopping_rounds=100
    )

    # 6) метрики
    prob = clf.predict_proba(Xte)[:,1]
    pred = (prob >= 0.5).astype(int)
    auc = roc_auc_score(yte, prob)
    cm = confusion_matrix(yte, pred)
    print("\nAUC:", round(float(auc), 4))
    print("Confusion matrix:\n", cm)
    print("\nReport:\n", classification_report(yte, pred, digits=3))

    # 7) сохранения
    booster = clf.get_booster()
    booster.save_model(os.path.join(args.out_dir, "xgb.json"))
    json.dump(feature_names, open(os.path.join(args.out_dir, "features_cols.json"), "w"), ensure_ascii=False, indent=2)
    pd.DataFrame(
        {"feature": feature_names, "gain": clf.feature_importances_}
    ).sort_values("gain", ascending=False).to_csv(
        os.path.join(args.out_dir, "feature_importance_gain.csv"), index=False
    )

    
    print("\n[done] модель сохранена в", args.out_dir)

if __name__ == "__main__":
    main()
