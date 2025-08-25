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
    try:
        f = float(s)
        if f in (0.0,1.0): return int(f)
    except: pass
    return np.nan

def main():
    ap = argparse.ArgumentParser(description="Train XGBoost on features.csv (+ labels CSV if needed)")
    ap.add_argument("--features_csv", required=True, help="features.csv из экстрактора")
    ap.add_argument("--labels_csv", default=None, help="CSV с метками (если в features.csv нет label)")
    ap.add_argument("--label_col", default=None, help="Имя колонки метки в labels_csv (напр. 'No inj/ inj')")
    ap.add_argument("--fname_in_labels", default=None, help="Колонка с именем файла в labels_csv (auto: filename/file/path/basename/stem)")
    ap.add_argument("--out_dir", default="out_xgb_simple")
    ap.add_argument("--use_gpu", action="store_true")
    ap.add_argument("--test_size", type=float, default=0.2, help="Доля данных в тесте (по умолчанию 0.2)")

    # >>> НОВОЕ: выбор фич по важности с прошлого запуска
    ap.add_argument("--importance_csv", default=None, help="Путь к feature_importance.csv с прошлого запуска")
    ap.add_argument("--min_total_gain", type=float, default=10.0, help="Порог total_gain для отбора фич")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 1) читаем фичи
    F = pd.read_csv(args.features_csv)
    fname_feat = None
    for c in ("stem","basename","file"):
        if c in F.columns: fname_feat = c; break
    if fname_feat is None:
        raise SystemExit("[err] features_csv должен содержать колонку stem/basename/file для связи с labels_csv")

    # 2) метки
    if "label" in F.columns and F["label"].notna().any():
        DF = F.copy()
    else:
        if not args.labels_csv:
            raise SystemExit("[err] в features.csv нет 'label'; укажите --labels_csv с метками")
        L = pd.read_csv(args.labels_csv)
        fname_lab = args.fname_in_labels
        if fname_lab is None:
            for c in ("filename","file","path","basename","stem"):
                if c in L.columns: fname_lab = c; break
        if fname_lab is None:
            raise SystemExit("[err] не нашёл колонку имени файла в labels_csv (ожидал filename/file/path/basename/stem)")
        label_col = args.label_col
        if label_col is None:
            for c in L.columns:
                cl = c.lower()
                if cl == "label" or "inj" in cl or c.strip() in ("No inj/ inj",):
                    label_col = c; break
        if label_col is None:
            raise SystemExit("[err] не нашёл колонку меток. Укажите --label_col")

        F["_key"] = F[fname_feat].astype(str).map(stem_lower)
        L["_key"] = L[fname_lab].astype(str).map(stem_lower)
        lab_small = L[["_key", label_col]].drop_duplicates("_key", keep="last").rename(columns={label_col: "label"})
        DF = F.merge(lab_small, on="_key", how="left").drop(columns=["_key"])

    # 3) привести метку к 0/1 и вычистить
    DF["label"] = DF["label"].map(map_label)
    DF = DF[DF["label"].isin([0,1])].reset_index(drop=True)
    if len(DF) == 0:
        raise SystemExit("[err] после приведения меток ни одной валидной строки (0/1)")

    # 4) сформировать X, y
    y = DF["label"].astype("int32").to_numpy()

    # --- базовый набор признаков: все числовые минус label
    X_df = DF.select_dtypes(include=[np.number]).drop(columns=["label"], errors="ignore")

    # >>> НОВОЕ: если указан importance_csv — сузить признаки по total_gain
    if args.importance_csv and os.path.exists(args.importance_csv):
        imp = pd.read_csv(args.importance_csv)
        if not {"feature","total_gain"}.issubset(imp.columns):
            raise SystemExit("[err] importance_csv должен содержать колонки 'feature' и 'total_gain'")
        selected = imp.loc[imp["total_gain"] > args.min_total_gain, "feature"].tolist()
        # Пересечение с доступными колонками (на случай расхождений)
        selected = [f for f in selected if f in X_df.columns]
        if len(selected) == 0:
            raise SystemExit(f"[err] по порогу total_gain > {args.min_total_gain} не осталось фич. Понизьте порог или не указывайте --importance_csv.")
        X_df = X_df[selected]
        print(f"[info] Использую {len(selected)} фич по важности из {args.importance_csv} (total_gain > {args.min_total_gain})")
    else:
        if args.importance_csv:
            print(f"[warn] importance_csv '{args.importance_csv}' не найден — используются все признаки.")

    feature_names = list(X_df.columns)
    X = X_df.to_numpy(dtype=np.float32)

    # 5) train/test split
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    # 6) XGBoost
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

    # 7) метрики
    prob = clf.predict_proba(Xte)[:, 1]
    pred = (prob >= 0.5).astype(int)
    auc = roc_auc_score(yte, prob)
    cm = confusion_matrix(yte, pred)
    print("\nAUC:", round(float(auc), 4))
    print("Confusion matrix:\n", cm)
    print("\nReport:\n", classification_report(yte, pred, digits=3))

    # 8) сохранения
    booster = clf.get_booster()
    booster.save_model(os.path.join(args.out_dir, "xgb.json"))
    json.dump(feature_names, open(os.path.join(args.out_dir, "features_cols.json"), "w"), ensure_ascii=False, indent=2)

    # важности
    bst = clf.get_booster(); kinds = ["gain","total_gain","weight","cover","total_cover"]
    scores = {k: bst.get_score(importance_type=k) for k in kinds}
    imp = pd.DataFrame({
        "feature": feature_names,
        **{k: [scores[k].get(f"f{i}", 0.0) for i in range(len(feature_names))] for k in kinds}
    })
    (imp.assign(total_gain_pct=lambda d: d.total_gain/(d.total_gain.sum()+1e-12))
       .sort_values("total_gain", ascending=False)
       .to_csv(os.path.join(args.out_dir, "feature_importance.csv"), index=False))

    print("\n[done] модель сохранена в", args.out_dir)

if __name__ == "__main__":
    main()
