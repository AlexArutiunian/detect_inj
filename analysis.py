#!/usr/bin/env python
import os
import json
import joblib
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # чтобы скрипт работал в headless-окружениях
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

from train_cl import label_to_int, load_features_from_npy

# ==== утилиты ====


def classical_feature_names(schema_joints):
    """
    Генерация имён признаков (mean/std/min/max + дифы) для каждого сустава.
    """
    feats = []
    stats = ["mean", "std", "min", "max", "dmean", "dstd"]
    coords = ["x", "y", "z"]
    for stat in stats:
        for j in schema_joints:
            for c in coords:
                feats.append(f"{stat}_{c}_{j}")
    return feats


def aggregate_importance_by_joint(feat_names, importances):
    by_joint = {}
    by_joint_axis = {}
    for fn, im in zip(feat_names, importances):
        parts = fn.split("_")
        stat, axis, joint = parts[0], parts[1], "_".join(parts[2:])
        by_joint[joint] = by_joint.get(joint, 0.0) + im
        key = f"{joint}_{axis}"
        by_joint_axis[key] = by_joint_axis.get(key, 0.0) + im
    return by_joint, by_joint_axis


def save_importance_tables(out_dir, all_rows, by_joint, by_joint_axis, prefix="perm"):
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(os.path.join(out_dir, f"{prefix}_features.csv"), index=False)
    pd.DataFrame([{"joint": k, "importance": v} for k, v in by_joint.items()]) \
        .to_csv(os.path.join(out_dir, f"{prefix}_joints.csv"), index=False)
    pd.DataFrame([{"joint_axis": k, "importance": v} for k, v in by_joint_axis.items()]) \
        .to_csv(os.path.join(out_dir, f"{prefix}_joint_axes.csv"), index=False)


def plot_top_joints_barh(by_joint, top_k, out_path, title="Top joints by importance", xlabel="Importance"):
    """
    Строит горизонтальный бар-чарт для топ-K суставов по значимости и сохраняет в PNG.
    """
    if not by_joint:
        return
    # сортируем по значимости по убыванию и берём топ-K
    items = sorted(by_joint.items(), key=lambda x: x[1], reverse=True)[:top_k]
    joints, vals = zip(*items)

    plt.figure(figsize=(10, 6))
    y_pos = np.arange(len(joints))
    plt.barh(y_pos, vals)
    plt.yticks(y_pos, joints)
    plt.gca().invert_yaxis()  # самый важный сверху
    plt.xlabel(xlabel)
    plt.title(title)

    # аннотации значений на концах баров
    ax = plt.gca()
    for i, v in enumerate(vals):
        ax.text(v + (abs(max(vals)) * 0.01 if max(vals) != 0 else 0.01), i, f"{v:.4f}", va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


# ==== основной код ====


def main(args):
    # Базовые пути/параметры из аргументов
    out_dir = args.out_dir
    csv_path = args.csv_path
    data_dir = args.data_dir
    motion_key = args.motion_key
    workers = args.workers
    n_repeats = args.n_repeats
    random_state = args.random_state
    n_jobs = args.n_jobs
    prefix = args.prefix

    # Пути к модели и схеме (можно переопределить флагами)
    model_path = args.model_path or os.path.join(out_dir, "model.joblib")
    schema_path = args.schema_path or os.path.join(out_dir, "schema_joints.json")

    # загружаем модель + нормировщик
    bundle = joblib.load(model_path)
    model, scaler = bundle["model"], bundle["scaler"]

    # схема суставов
    with open(schema_path, "r", encoding="utf-8") as f:
        schema_joints = json.load(f)

    # грузим фичи
    X_all, y_all, _ = load_features_from_npy(
        csv_path,
        data_dir,
       
        workers=workers,
    )
    X_all = scaler.transform(X_all)

    # считаем permutation importance
    feat_names = classical_feature_names(schema_joints)
    pi = permutation_importance(
        model,
        X_all,
        y_all,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    importances_mean = pi.importances_mean

    # сохраняем таблицы
    all_rows = [{"feature": fn, "importance": float(im)} for fn, im in zip(feat_names, importances_mean)]
    by_joint, by_joint_axis = aggregate_importance_by_joint(feat_names, importances_mean)
    save_importance_tables(out_dir, all_rows, by_joint, by_joint_axis, prefix=prefix)

    # график топ суставов
    if args.plots:
        os.makedirs(out_dir, exist_ok=True)
        plot_path = os.path.join(out_dir, f"{prefix}_top{args.top_k_joints}_joints.png")
        plot_top_joints_barh(
            by_joint,
            top_k=args.top_k_joints,
            out_path=plot_path,
            title=f"Top {args.top_k_joints} joints by importance ({motion_key})",
            xlabel="Permutation importance (sum over features)",
        )
        print(f"Сохранён график: {plot_path}")

    print("\n=== Важность суставов (по убыванию) ===")
    for j, v in sorted(by_joint.items(), key=lambda x: x[1], reverse=True):
        print(f"{j:20s} {v:.6f}")


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Вычисление permutation importance с агрегацией по суставам и построением бар-чартов."
    )
    # Основные переменные
    p.add_argument("--out-dir", type=str, default="outputs/rf", help="Каталог для модели и вывода результатов.")
    p.add_argument("--csv-path", type=str, default="walk_data_meta_upd.csv", help="Путь к CSV с метаданными.")
    p.add_argument("--data-dir", type=str, default="walking", help="Каталог с данными (папка с треками).")
    p.add_argument(
        "--motion-key",
        type=str,
        choices=["walking", "running"],
        default="walking",
        help="Тип движения (используется при загрузке фич).",
    )

    # Кастомные пути (необязательные)
    p.add_argument("--model-path", type=str, default=None, help="Явный путь к model.joblib (если не в out_dir).")
    p.add_argument("--schema-path", type=str, default=None, help="Явный путь к schema_joints.json (если не в out_dir).")

    # Производительность и воспроизводимость
    p.add_argument("--workers", type=int, default=4, help="Количество воркеров для стриминговой загрузки фич.")
    p.add_argument("--n-repeats", type=int, default=10, help="Число повторов для permutation importance.")
    p.add_argument("--random-state", type=int, default=42, help="RandomState для воспроизводимости.")
    p.add_argument("--n-jobs", type=int, default=-1, help="Параллелизм в permutation_importance (-1 = все ядра).")

    # Графики
    p.add_argument("--top-k-joints", type=int, default=20, help="Сколько суставов показать на графике.")
    p.add_argument("--no-plots", dest="plots", action="store_false", help="Отключить построение графиков.")
    p.set_defaults(plots=True)

    # Прочее
    p.add_argument("--prefix", type=str, default="perm", help="Префикс для CSV и PNG файлов.")
    return p


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    main(args)
