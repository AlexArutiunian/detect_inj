#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
import joblib
import numpy as np
import pandas as pd

# Matplotlib (headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

# ВАЖНО: функция загрузки фич из NPY из вашего проекта
# должна иметь сигнатуру: load_features_from_npy(csv_path, npy_dir, filename_col, label_col, downsample, workers)
from train_cl import load_features_from_npy


# ===================== УТИЛИТЫ =====================

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def classical_feature_names(schema_joints):
    """
    Имена признаков в порядке, совпадающем с _features_from_seq в train3:
    concat по статистикам [mean, std, min, max, dmean, dstd] поверх осей x,y,z и суставов.
    Итог: 6 * (3 * |joints|) признаков.
    """
    stats = ["mean", "std", "min", "max", "dmean", "dstd"]
    axes = ["x", "y", "z"]
    feats = []
    for st in stats:                 # порядок по статистикам
        for j in schema_joints:      # затем по суставам
            for ax in axes:          # затем по осям
                feats.append(f"{st}_{ax}_{j}")
    return feats


def aggregate_importance_by_joint(feat_names, importances):
    by_joint = {}
    by_joint_axis = {}
    for fn, im in zip(feat_names, importances):
        parts = fn.split("_")
        # "stat_axis_joint(with_underscores_inside)"
        stat, axis, joint = parts[0], parts[1], "_".join(parts[2:])
        by_joint[joint] = by_joint.get(joint, 0.0) + float(im)
        key = f"{joint}_{axis}"
        by_joint_axis[key] = by_joint_axis.get(key, 0.0) + float(im)
    return by_joint, by_joint_axis


def save_importance_tables(out_dir, all_rows, by_joint, by_joint_axis, prefix="perm"):
    ensure_dir(out_dir)
    pd.DataFrame(all_rows).to_csv(os.path.join(out_dir, f"{prefix}_features.csv"), index=False)
    pd.DataFrame([{"joint": k, "importance": v} for k, v in by_joint.items()]) \
        .to_csv(os.path.join(out_dir, f"{prefix}_joints.csv"), index=False)
    pd.DataFrame([{"joint_axis": k, "importance": v} for k, v in by_joint_axis.items()]) \
        .to_csv(os.path.join(out_dir, f"{prefix}_joint_axes.csv"), index=False)


def plot_top_barh(items, out_path, title, xlabel):
    if not items:
        return
    labels, vals = zip(*items)
    plt.figure(figsize=(10, max(3, 0.4 * len(labels))))
    y = np.arange(len(labels))
    plt.barh(y, vals)
    plt.yticks(y, labels)
    plt.gca().invert_yaxis()
    plt.xlabel(xlabel)
    plt.title(title)
    ax = plt.gca()
    vmax = max(abs(float(v)) for v in vals) if vals else 0.0
    offs = (vmax * 0.01) if vmax > 0 else 0.01
    for i, v in enumerate(vals):
        ax.text(float(v) + offs, i, f"{float(v):.4f}", va="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def stratified_subsample(X, y, max_samples, random_state=42):
    """Быстрая стратифицированная подвыборка для ускорения permutation_importance."""
    n = X.shape[0]
    if max_samples is None or max_samples <= 0 or max_samples >= n:
        return X, y
    test_size = 1.0 - float(max_samples) / float(n)
    if test_size <= 0:
        return X, y
    _, X_sub, _, y_sub = train_test_split(
        X, y, test_size=max(0.0001, test_size), random_state=random_state, stratify=y
    )
    return X_sub, y_sub


# ===================== ОСНОВНОЙ КОД =====================

def main(args):
    out_dir = args.out_dir
    prefix = args.prefix

    # Пути к модели и схеме
    model_path = args.model_path or os.path.join(out_dir, "model.joblib")
    schema_path = args.schema_path or os.path.join(out_dir, "schema_joints.json")

    # Загрузим модель и нормировщик
    bundle = joblib.load(model_path)
    model = bundle["model"]
    scaler = bundle["scaler"]

    # Схема суставов (список строк в нужном порядке)
    with open(schema_path, "r", encoding="utf-8") as f:
        schema_joints = json.load(f)

    # Загрузим фичи из NPY по CSV-индексу
    X_all, y_all = load_features_from_npy(
        args.csv_path,
        args.data_dir,
        filename_col=args.filename_col,
        label_col=args.label_col,
        downsample=args.downsample,
        workers=args.workers,
    )

    # Нормализация как при обучении
    X_all = scaler.transform(X_all)

    # Подвыборка для ускорения (опционально)
    if args.max_samples and args.max_samples > 0:
        X_all, y_all = stratified_subsample(X_all, y_all, args.max_samples, args.random_state)
        print(f"[subsample] using {X_all.shape[0]} samples")

    # Имена фич и sanity-check
    feat_names = classical_feature_names(schema_joints)
    if len(feat_names) != X_all.shape[1]:
        raise RuntimeError(
            f"Несовпадение размерностей: feature_names={len(feat_names)} vs X_all.shape[1]={X_all.shape[1]}.\n"
            f"Проверь порядок статистик/осей и schema_joints."
        )

    # Permutation importance
    pi = permutation_importance(
        model,
        X_all,
        y_all,
        n_repeats=args.n_repeats,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        scoring=args.scoring  # может быть None -> estimator.score
    )
    importances_mean = pi.importances_mean.astype(float)

    # Сохраним таблички
    all_rows = [{"feature": fn, "importance": float(im)} for fn, im in zip(feat_names, importances_mean)]
    by_joint, by_joint_axis = aggregate_importance_by_joint(feat_names, importances_mean)
    save_importance_tables(out_dir, all_rows, by_joint, by_joint_axis, prefix=prefix)

    # Топы и графики
    topJ = sorted(by_joint.items(), key=lambda x: x[1], reverse=True)[:args.top_k_joints]
    if args.plots and topJ:
        plot_path = os.path.join(out_dir, f"{prefix}_top{args.top_k_joints}_joints.png")
        plot_top_barh(
            topJ,
            out_path=plot_path,
            title=f"Top {len(topJ)} joints by importance",
            xlabel="Permutation importance (sum over features)"
        )
        print(f"Saved: {plot_path}")

    if args.plots and args.plot_joint_axes:
        topJA = sorted(by_joint_axis.items(), key=lambda x: x[1], reverse=True)[:args.top_k_joint_axes]
        if topJA:
            plot_path2 = os.path.join(out_dir, f"{prefix}_top{len(topJA)}_joint_axes.png")
            plot_top_barh(
                topJA,
                out_path=plot_path2,
                title=f"Top {len(topJA)} joint-axes by importance",
                xlabel="Permutation importance"
            )
            print(f"Saved: {plot_path2}")

    # Печать топа в stdout
    print("\n=== Топ суставов по значимости ===")
    for j, v in sorted(by_joint.items(), key=lambda x: x[1], reverse=True)[:args.top_k_joints]:
        print(f"{j:25s} {v:.6f}")

    if args.verbose_full:
        print("\n=== Полный список суставов (по убыванию) ===")
        for j, v in sorted(by_joint.items(), key=lambda x: x[1], reverse=True):
            print(f"{j:25s} {v:.6f}")


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Permutation importance: агрегирование по суставам (NPY-пайплайн) + бар-чарты."
    )

    # Источники данных
    p.add_argument("--csv-path", type=str, required=True, help="CSV с индексом файлов и метками.")
    p.add_argument("--data-dir", type=str, required=True, help="Папка с .npy последовательностями.")
    p.add_argument("--filename-col", type=str, default="filename", help="Имя столбца с путями/именами файлов.")
    p.add_argument("--label-col", type=str, default="No inj/ inj", help="Имя столбца с метками.")
    p.add_argument("--downsample", type=int, default=1, help="Шаг по времени (>=1) при чтении NPY.")
    p.add_argument("--workers", type=int, default=0, help="Воркеры при извлечении фич из NPY.")

    # Модель и схема
    p.add_argument("--out-dir", type=str, required=True, help="Директория эксперимента (где лежит model.joblib).")
    p.add_argument("--model-path", type=str, default=None, help="Путь к model.joblib (по умолчанию OUT_DIR/model.joblib).")
    p.add_argument("--schema-path", type=str, default=None, help="Путь к schema_joints.json (по умолчанию OUT_DIR/schema_joints.json).")

    # Permutation importance
    p.add_argument("--n-repeats", type=int, default=10, help="Повторов перемешивания на признак.")
    p.add_argument("--random-state", type=int, default=42, help="Сид случайности.")
    p.add_argument("--n-jobs", type=int, default=-1, help="Параллелизм (-1 = все ядра).")
    p.add_argument("--scoring", type=str, default=None,
                   help="Скора для permutation_importance (например, 'roc_auc'). По умолчанию estimator.score.")
    p.add_argument("--max-samples", type=int, default=0,
                   help="Ограничить число объектов для ускорения (0 = использовать все).")

    # Вывод/графики
    p.add_argument("--prefix", type=str, default="perm", help="Префикс имен файлов (CSV/PNG).")
    p.add_argument("--top-k-joints", type=int, default=20, help="Сколько суставов рисовать/печатать в топе.")
    p.add_argument("--plot-joint-axes", action="store_true", help="Строить доп. график по joint+axis.")
    p.add_argument("--top-k-joint-axes", type=int, default=30, help="Топ по joint+axis для графика.")
    p.add_argument("--no-plots", dest="plots", action="store_false", help="Отключить построение графиков.")
    p.add_argument("--verbose-full", action="store_true", help="Печатать полный список суставов в stdout.")
    p.set_defaults(plots=True)

    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    main(args)
