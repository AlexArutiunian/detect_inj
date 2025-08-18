#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
top_joints_from_model.py
Агрегация важностей по суставам для классических моделей (RF/XGB),
натренированных на фичах из train3.py (6 статистик × 3 оси × |joints|),
+ построение горизонтальной диаграммы важностей (показывается в ячейке и сохраняется как PNG).

Пример:
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

# --- Встроенная схема суставов (можно заменить через --schema_json)
DEFAULT_JOINTS = [
  "L_foot_1","L_foot_2","L_foot_3","L_foot_4",
  "L_shank_1","L_shank_2","L_shank_3","L_shank_4",
  "L_thigh_1","L_thigh_2","L_thigh_3","L_thigh_4",
  "R_foot_1","R_foot_2","R_foot_3","R_foot_4",
  "R_shank_1","R_shank_2","R_shank_3","R_shank_4",
  "R_thigh_1","R_thigh_2","R_thigh_3","R_thigh_4",
  "pelvis_1","pelvis_2","pelvis_3","pelvis_4"
]

STAT_NAMES = ["mean","std","min","max","dmean","dstd"]  # как в train3.py
AXES = ["x","y","z"]                                   # 3 координаты
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
            "У модели нет feature_importances_. "
            "Нужен RandomForest/XGBoost, обученный на классических фичах."
        )
    return np.asarray(imp, dtype=float)

def sanity_check_with_npy(npy_dir: str, joints: list[str], verbose: bool=True):
    sample = find_first_npy(npy_dir)
    if not sample:
        if verbose:
            print("[WARN] В папке npy не найдено .npy — пропускаю проверку формы.")
        return
    try:
        arr = np.load(sample, allow_pickle=False, mmap_mode="r")
        if arr.ndim != 2 or arr.shape[0] < 2:
            print(f"[WARN] '{sample}' имеет форму {arr.shape}, ожидалась (T, F) с T>=2.")
        else:
            F_seq = arr.shape[1]
            expect_coords = 3 * len(joints)
            if F_seq % 3 != 0:
                print(f"[WARN] Число признаков в последовательности = {F_seq}, не кратно 3.")
            if F_seq != expect_coords:
                print(f"[INFO] В последовательности на кадр F={F_seq}, ожидалось 3*|joints|={expect_coords}. "
                      "Это не критично для агрегации важностей, но проверьте соответствие схемы.")
    except Exception as e:
        print(f"[WARN] Не удалось открыть NPY для проверки: {e}")

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
            f"Длина importances = {importances.size}, а по схеме ожидается {expected} "
            f"(18 признаков × {len(joints)} суставов). Проверьте, что модель обучалась "
            "на классических фичах из train3.py и схема совпадает."
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
    Строит горизонтальную диаграмму для topn суставов.
    Показывает в ячейке (если show=True) и сохраняет в PNG.
    """
    top = df.head(max(1, int(topn))).copy()
    # для красивого порядка сверху-вниз
    top = top.iloc[::-1]

       # нормируем в проценты
    top["importance_pct"] = 100 * top["importance_sum"] / top["importance_sum"].sum()

    h = max(3.5, 0.3 * len(top))       # делаем столбцы ниже (коэф. 0.3 вместо 0.4)
    plt.figure(figsize=(8, h))
    plt.barh(top["joint"], top["importance_pct"])
    plt.xlabel("Importance (%)")
    plt.title(title)

    # подписи значений справа от баров
    for i, v in enumerate(top["importance_pct"]):
        plt.text(v, i, f" {v:.1f}%", va="center")

    # добавляем запас справа, чтобы подписи не съезжали
    plt.xlim(0, top["importance_pct"].max() * 1.1)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=dpi)
    if show:
        plt.show()
    plt.close()

    print(f"[OK] Диаграмма сохранена: {plot_path}")

def main():
    ap = argparse.ArgumentParser(description="Вывод топ-значимых суставов и построение диаграммы важностей.")
    ap.add_argument("--npy_dir", required=True, help="Папка с *.npy (для быстрой проверки формы).")
    ap.add_argument("--model_joblib", required=True, help="Путь к model.joblib (RF/XGB, обученная на классических фичах).")
    ap.add_argument("--schema_json", default="", help="JSON со списком суставов (если не задан — встроенный список).")
    ap.add_argument("--topn", type=int, default=15, help="Сколько суставов показать текстом.")
    ap.add_argument("--save_csv", default="", help="Куда сохранить CSV с важностями (опционально).")

    # Параметры графика
    ap.add_argument("--topn_plot", type=int, default=20, help="Сколько верхних суставов рисовать на диаграмме.")
    ap.add_argument("--plot_path", default="top_joints_importance.png", help="Куда сохранить PNG диаграммы.")
    ap.add_argument("--title", default="Joint Importance (sum over 18 features per joint)", help="Заголовок диаграммы.")
    ap.add_argument("--no_show", action="store_true", help="Не показывать график в ячейке/окне, только сохранить.")

    args = ap.parse_args()

    joints = load_joints(args.schema_json)
    print(f"[INFO] Суставов в схеме: {len(joints)} (ожидается, что в модели фич = 18×|joints| = {FEATS_PER_JOINT*len(joints)})")

    # Быстрая проверка формата одного npy (не влияет на вычисление важностей, лишь sanity-check)
    sanity_check_with_npy(args.npy_dir, joints, verbose=True)

    # Модель и её важности
    model = load_model(args.model_joblib)
    importances = get_importances(model)

    # Карта имён фич (проверка размерности)
    feat_names = build_feature_name_map(joints)
    if len(feat_names) != importances.size:
        raise ValueError(
            f"Размерности не совпадают: len(feat_names)={len(feat_names)} "
            f"vs len(importances)={importances.size}. Проверьте схему суставов."
        )

    # Агрегация до суставов
    df = aggregate_joint_importance(importances, joints)

    # Текстовый вывод TOP-N
    topn = max(1, int(args.topn))
    print("\n=== TOP суставов (сумма важностей по 18 фичам каждого сустава) ===")
    for i, row in df.head(topn).iterrows():
        print(f"{i+1:2d}. {row['joint']:>12s}  importance_sum={row['importance_sum']:.6f}  "
              f"(norm={row['importance_norm']:.4f})")

    # Сохранить CSV (опционально)
    if args.save_csv:
        df.to_csv(args.save_csv, index=False)
        print(f"[OK] Сохранено: {args.save_csv}")

    # Диаграмма (показываем в ячейке и сохраняем)
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
