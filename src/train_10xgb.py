#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
    roc_auc_score,
)
import xgboost as xgb

def stem_lower(s: str) -> str:
    b = os.path.basename(str(s))
    return os.path.splitext(b)[0].lower()

def main():
    ap = argparse.ArgumentParser(description="Train XGBoost (multiclass) on features.csv (+ labels CSV if needed)")
    ap.add_argument("--features_csv", required=True, help="features.csv из экстрактора")
    ap.add_argument("--labels_csv", default=None, help="CSV с метками (если в features.csv нет label)")
    ap.add_argument("--label_col", default=None, help='Имя колонки метки в labels_csv (по умолчанию "label")')
    ap.add_argument("--fname_in_labels", default=None, help="Колонка с именем файла в labels_csv (auto: filename/file/path/basename/stem)")
    ap.add_argument("--label_index_csv", default=None, help="(опц.) CSV со справочником label->InjuryClass для красивых имён в отчёте")
    ap.add_argument("--out_dir", default="out_xgb_multiclass")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
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

    # 2) получаем метки
    if "label" in F.columns and F["label"].notna().any():
        DF = F.copy()
    else:
        if not args.labels_csv:
            raise SystemExit("[err] в features.csv нет 'label'; укажите --labels_csv с метками (filename,label)")
        L = pd.read_csv(args.labels_csv)

        # имя файла в labels
        fname_lab = args.fname_in_labels
        if fname_lab is None:
            for c in ("filename","file","path","basename","stem"):
                if c in L.columns: fname_lab = c; break
        if fname_lab is None:
            raise SystemExit("[err] не нашёл колонку имени файла в labels_csv (ожидал filename/file/path/basename/stem)")

        # колонка метки
        label_col = args.label_col or ("label" if "label" in L.columns else None)
        if label_col is None:
            raise SystemExit("[err] не нашёл колонку меток. Укажите --label_col (обычно 'label')")

        # ключи
        F["_key"] = F[fname_feat].astype(str).map(stem_lower)
        L["_key"] = L[fname_lab].astype(str).map(stem_lower)

        lab_small = L[["_key", label_col]].drop_duplicates("_key", keep="last").rename(columns={label_col: "label"})
        DF = F.merge(lab_small, on="_key", how="left").drop(columns=["_key"])

    # 3) чистим метки и приводим к int
    if DF["label"].isna().any():
        print("[warn] найдены NaN в label после мерджа — строки будут отброшены")
    # ожидаем, что тут метки 1..n (как в training_labels.csv)
    # если попадутся строки вида " 3 " — приведём к числам
    DF["label"] = pd.to_numeric(DF["label"], errors="coerce")
    DF = DF[DF["label"].notna()].reset_index(drop=True)

    # уникальные классы
    labels_sorted = sorted(DF["label"].astype(int).unique().tolist())
    # проверим, начинаются ли с 1
    if min(labels_sorted) < 0:
        raise SystemExit("[err] обнаружены отрицательные метки, ожидаются 1..n или 0..n-1")
    # построим маппинг -> 0..K-1 для XGB
    # если метки уже 0..K-1 — оставим как есть; иначе сдвинем 1..K -> 0..K-1
    starts_at_zero = (0 in labels_sorted)
    if starts_at_zero:
        label_to_zero_based = {int(v): int(v) for v in labels_sorted}
    else:
        label_to_zero_based = {int(v): int(v-1) for v in labels_sorted}

    DF["_y"] = DF["label"].astype(int).map(label_to_zero_based).astype("int32")

    # 4) X и y
    y = DF["_y"].to_numpy()
    X = DF.select_dtypes(include=[np.number]).drop(columns=["label","_y"], errors="ignore")
    feature_names = list(X.columns)
    X = X.to_numpy(dtype=np.float32)

    # 5) train/test split (стратифицированно)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    num_class = len(labels_sorted)
    print(f"[info] classes: {num_class} (raw labels: {labels_sorted})")

    # 6) XGBoost Multiclass
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

    clf.fit(
        Xtr, ytr,
        eval_set=[(Xtr, ytr), (Xte, yte)],
        verbose=False,
        early_stopping_rounds=100
    )

    # 7) Метрики
    prob = clf.predict_proba(Xte)              # shape [N, K]
    pred = prob.argmax(axis=1)

    acc = accuracy_score(yte, pred)
    f1_macro = f1_score(yte, pred, average="macro")
    f1_micro = f1_score(yte, pred, average="micro")

    # многоклассовый AUC (OVR, macro) — требует one-vs-rest вероятности
    try:
        auc_macro_ovr = roc_auc_score(yte, prob, multi_class="ovr", average="macro")
    except Exception as e:
        auc_macro_ovr = np.nan
        print("[warn] roc_auc_score failed:", e)

    print("\nAccuracy:", round(float(acc), 4))
    print("F1-macro:", round(float(f1_macro), 4))
    print("F1-micro:", round(float(f1_micro), 4))
    print("Macro AUC (OVR):", "nan" if np.isnan(auc_macro_ovr) else round(float(auc_macro_ovr), 4))

    cm = confusion_matrix(yte, pred)
    print("\nConfusion matrix (rows=true, cols=pred):\n", cm)

    # красивые имена классов, если есть справочник
    target_names = None
    if args.label_index_csv and os.path.exists(args.label_index_csv):
        LI = pd.read_csv(args.label_index_csv)
        # ожидаем колонки: label, InjuryClass (label в исходной 1..n)
        if "label" in LI.columns:
            LI["_zero"] = LI["label"].astype(int).map(lambda v: label_to_zero_based.get(int(v), None))
            LI = LI[LI["_zero"].notna()].sort_values("_zero")
            target_names = LI["InjuryClass"].astype(str).tolist()
    if target_names is None:
        # просто 0..K-1
        target_names = [f"class_{i}" for i in range(num_class)]

    print("\nReport:\n", classification_report(yte, pred, target_names=target_names, digits=3))

    # 8) Сохранения
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # модель
    booster = clf.get_booster()
    booster.save_model(os.path.join(args.out_dir, "xgb_multiclass.json"))

    # список фич
    json.dump(feature_names, open(os.path.join(args.out_dir, "features_cols.json"), "w"), ensure_ascii=False, indent=2)

    # важность признаков
    bst = clf.get_booster()
    kinds = ["gain","total_gain","weight","cover","total_cover"]
    scores = {k: bst.get_score(importance_type=k) for k in kinds}
    imp = pd.DataFrame({"feature": feature_names, **{k: [scores[k].get(f"f{i}", 0.0) for i in range(len(feature_names))] for k in kinds}})
    (imp.assign(total_gain_pct=lambda d: d.total_gain/(d.total_gain.sum()+1e-12))
        .sort_values("total_gain", ascending=False)
        .to_csv(os.path.join(args.out_dir, "feature_importance.csv"), index=False))

    # предсказания на тесте (удобно для разборов)
    # вернём исходные «человеческие» метки 1..n, если они такими и были
    inv_map = {v:k for k,v in label_to_zero_based.items()}
    test_df = pd.DataFrame({
        "y_true": [inv_map.get(int(v), int(v)) for v in yte],
        "y_pred": [inv_map.get(int(v), int(v)) for v in pred],
        "y_pred_topprob": prob.max(axis=1)
    })
    # добавим по классам вероятности
    for j in range(prob.shape[1]):
        human_lbl = inv_map.get(j, j)  # вернём 1..n если были
        test_df[f"proba_{human_lbl}"] = prob[:, j]
    test_df.to_csv(os.path.join(args.out_dir, "test_predictions.csv"), index=False)

    # сохраним маппинги меток
    json.dump(
        {
            "labels_sorted_raw": labels_sorted,
            "label_to_zero_based": label_to_zero_based,
        },
        open(os.path.join(args.out_dir, "label_mapping.json"), "w"),
        ensure_ascii=False, indent=2
    )

    print("\n[done] модель и артефакты сохранены в", args.out_dir)

if __name__ == "__main__":
    main()
