#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRU inference with confidence (полный скрипт)

Что делает:
- Грузит модель (SavedModel/.keras/.h5) и норм-статы (mean, std, max_len из norm_stats.npz)
- Принимает входы тремя способами:
  (A) --scan_first: сначала сканирует --data_dir на *.npy (с --recursive), затем
      если есть --csv — берёт ПЕРЕСЕЧЕНИЕ с CSV (по относительным путям либо basename)
  (B) --csv + --data_dir: сопоставляет пути из CSV с файлами в data_dir
  (C) позиционные аргументы: явный список путей к .npy/.json.npy (с попыткой найти в --data_dir)
- Возвращает CSV с колонками:
    path, prob, pred, confidence, logit, entropy, (опц.) prob_std
  где:
    prob        — вероятность класса 1 (из сигмоиды)
    pred        — бинарный предсказанный класс по threshold
    confidence  — уверенность модели в предсказанном классе = max(p, 1-p)
    logit       — log(p/(1-p))
    entropy     — -[p ln p + (1-p) ln(1-p)], чем меньше — тем «увереннее»
    prob_std    — std по MC Dropout (если --mc_passes > 1)
- Совместимо с Dropout-MC: при --mc_passes > 1 делает несколько стохастических прогонов (training=True).

Замечания:
- Скрипт НЕ требует, чтобы все файлы из CSV существовали; отсутствующие просто пропускаются с предупреждением.
- Поддержаны различные размеры признаков: если x.shape[1] != feat_dim из norm_stats,
  то происходит усечение/допаддинг по признакам.
- Чтобы избежать предупреждений CUDA на машинах без GPU — можно добавить: --gpus cpu
"""

from __future__ import annotations

import os
import argparse
import sys
import math
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd


# ==========================
#      УТИЛИТЫ ДАННЫХ
# ==========================

def sanitize_seq(a: np.ndarray) -> np.ndarray:
    """Заменяет NaN/Inf на 0 и приводит к float32."""
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    return a.astype(np.float32, copy=False)


def possible_npy_paths(data_dir: str, rel: str) -> List[str]:
    """Формирует список возможных путей к .npy для разных суффиксов/вариантов, чтобы найти файл на диске."""
    rel = (rel or "").strip().replace("\\", "/").lstrip("/")
    cands: List[str] = []

    def push(x: str):
        if x not in cands:
            cands.append(x)

    push(os.path.join(data_dir, rel))
    if not rel.endswith(".npy"):
        push(os.path.join(data_dir, rel + ".npy"))
    if rel.endswith(".json"):
        base_nojson = rel[:-5]
        push(os.path.join(data_dir, base_nojson + ".npy"))
        push(os.path.join(data_dir, rel + ".npy"))
    if rel.endswith(".json.npy"):
        push(os.path.join(data_dir, rel.replace(".json.npy", ".npy")))

    b = os.path.basename(rel)
    push(os.path.join(data_dir, b))
    if not b.endswith(".npy"):
        push(os.path.join(data_dir, b + ".npy"))
    if b.endswith(".json"):
        push(os.path.join(data_dir, b[:-5] + ".npy"))
        push(os.path.join(data_dir, b + ".npy"))
    if b.endswith(".json.npy"):
        push(os.path.join(data_dir, b.replace(".json.npy", ".npy")))
    return cands


def pick_existing_path(cands: List[str]) -> Optional[str]:
    for p in cands:
        if os.path.exists(p):
            return p
    return None


def list_npy_files(root: str, recursive: bool = True) -> List[str]:
    """Возвращает список всех *.npy в каталоге (с рекурсией по желанию)."""
    out: List[str] = []
    root = root or "."
    if not os.path.isdir(root):
        return out
    if recursive:
        for dp, _, files in os.walk(root):
            for f in files:
                if f.endswith(".npy"):
                    out.append(os.path.join(dp, f))
    else:
        for f in os.listdir(root):
            p = os.path.join(root, f)
            if os.path.isfile(p) and p.endswith(".npy"):
                out.append(p)
    # уникализируем и сортируем для стабильности
    return sorted(set(out))


def _norm_rel_with_npy_suffix(s: str) -> str:
    s = (s or "").replace("\\", "/").lstrip("/")
    return s if s.endswith(".npy") else (s + ".npy")


def choose_paths_scan_then_csv(data_dir: str,
                               csv_path: Optional[str],
                               filename_col: str,
                               recursive: bool) -> List[str]:
    """
    1) Сканируем data_dir => список файлов на диске.
    2) Если CSV задан — берём пересечение:
       - сначала пытаемся сопоставить по относительному пути к data_dir (с .npy-суффиксом)
       - если нет — пробуем по basename (на случай, когда CSV хранит только имя файла)
    """
    disk = list_npy_files(data_dir, recursive=recursive)
    if not csv_path:
        return disk

    if not os.path.exists(csv_path):
        print(f"[warn] CSV '{csv_path}' не найден; используем только файлы на диске.", file=sys.stderr)
        return disk

    df = pd.read_csv(csv_path)
    cols = {c.lower().strip(): c for c in df.columns}
    fn_col = cols.get(filename_col.lower(), filename_col)
    if fn_col not in df.columns:
        # эвристика — выбрать первую колонку, где встречается 'file' или 'name'
        for c in df.columns:
            if "file" in c.lower() or "name" in c.lower():
                fn_col = c
                break
    wanted_raw = df[fn_col].astype(str).tolist()
    wanted = [_norm_rel_with_npy_suffix(x) for x in wanted_raw]

    # индекс на дисковые файлы
    by_rel = {}
    for p in disk:
        rel = os.path.relpath(p, data_dir).replace("\\", "/")
        by_rel[_norm_rel_with_npy_suffix(rel)] = p

    by_base = {}
    for p in disk:
        b = os.path.basename(p)
        if b not in by_base:  # первое вхождение
            by_base[b] = p

    chosen: List[str] = []
    missing = 0
    for w in wanted:
        p = by_rel.get(w) or by_base.get(os.path.basename(w))
        if p:
            chosen.append(p)
        else:
            missing += 1

    if missing:
        print(f"[warn] В CSV указано файлов, которых нет на диске: {missing}", file=sys.stderr)
    extra = len(disk) - len(chosen)
    if extra > 0:
        print(f"[info] На диске есть файлов вне CSV: {extra}", file=sys.stderr)

    return chosen


def resolve_inputs_from_csv(csv_path: str, data_dir: str, filename_col: str) -> List[str]:
    """Классический режим: сопоставить CSV с содержимым data_dir, пропуская отсутствующие файлы."""
    if not os.path.exists(csv_path):
        print(f"[warn] CSV '{csv_path}' не найден.", file=sys.stderr)
        return []
    df = pd.read_csv(csv_path)
    cols = {c.lower().strip(): c for c in df.columns}
    fn_col = cols.get(filename_col.lower(), filename_col)
    if fn_col not in df.columns:
        for c in df.columns:
            if "file" in c.lower() or "name" in c.lower():
                fn_col = c
                break
    paths: List[str] = []
    missing = 0
    for rel in df[fn_col].astype(str).tolist():
        p = pick_existing_path(possible_npy_paths(data_dir, rel))
        if p:
            paths.append(p)
        else:
            missing += 1
    if missing:
        print(f"[warn] Не найдено {missing} файлов из CSV", file=sys.stderr)
    return paths


def resolve_inputs_from_args(inputs: List[str], data_dir: Optional[str]) -> List[str]:
    """Режим: явные пути + попытка найти относительно data_dir."""
    out: List[str] = []
    for p in inputs:
        if os.path.exists(p):
            out.append(p)
        elif data_dir:
            r = pick_existing_path(possible_npy_paths(data_dir, p))
            if r:
                out.append(r)
    return out


def load_norm_stats(path: str) -> Tuple[np.ndarray, np.ndarray, int]:
    """Грузит mean, std, max_len из .npz."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"norm_stats '{path}' не найден")
    d = np.load(path, allow_pickle=False)
    if not all(k in d for k in ("mean", "std", "max_len")):
        raise ValueError(f"В '{path}' нет всех ключей (нужны mean, std, max_len)")
    mean = d["mean"].astype(np.float32)
    std = d["std"].astype(np.float32)
    max_len = int(d["max_len"])
    eps = 1e-8
    std = np.where(std < eps, 1.0, std)  # защита от деления на ~0
    return mean, std, max_len


def entropy_binary(p: np.ndarray) -> np.ndarray:
    """Бинарная энтропия в натах."""
    p = np.clip(p, 1e-8, 1 - 1e-8)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


# ==========================
#     ЗАГРУЗКА МОДЕЛИ
# ==========================

def load_any_model(model_path: str):
    """
    Пытается загрузить модель в следующем порядке:
    - tf.keras.models.load_model (SavedModel/.h5/.keras в новых TF)
    - keras.saving.load_model (Keras 3)
    - ещё раз tf.keras для каталогов SavedModel
    """
    import os
    try:
        import tensorflow as tf  # noqa
        return tf.keras.models.load_model(model_path)
    except Exception as e_tf:
        try:
            import keras  # noqa
            # Keras 3 (standalone)
            return keras.saving.load_model(model_path)
        except Exception as e_k:
            if os.path.isdir(model_path):
                # на случай SavedModel-каталога
                import tensorflow as tf  # noqa
                return tf.keras.models.load_model(model_path)
            raise RuntimeError(
                f"Не удалось загрузить модель '{model_path}'. "
                f"tf.keras: {type(e_tf).__name__}; keras: {type(e_k).__name__}"
            )


# ==========================
#        ИНФЕРЕНС
# ==========================

def make_dataset(paths: List[str],
                 mean: np.ndarray,
                 std: np.ndarray,
                 max_len: int,
                 downsample: int,
                 batch_size: int):
    """Создаёт tf.data.Dataset с паддингом до max_len и нормализацией."""
    import tensorflow as tf
    feat_dim = int(mean.shape[0])
    AUTOTUNE = tf.data.AUTOTUNE

    def gen():
        for p in paths:
            try:
                arr = np.load(p, allow_pickle=False, mmap_mode="r")
                if arr.ndim != 2 or arr.shape[0] < 1:
                    print(f"[warn] '{p}' пропущен: неправильная форма массива {arr.shape}", file=sys.stderr)
                    continue
                x = sanitize_seq(arr[::max(1, downsample)])

                # приведение размерности признаков к feat_dim
                if x.shape[1] != feat_dim:
                    if x.shape[1] > feat_dim:
                        x = x[:, :feat_dim]
                    else:
                        pad_w = feat_dim - x.shape[1]
                        x = np.pad(x, [[0, 0], [0, pad_w]], mode="constant", constant_values=0.0)

                # обрезка по времени
                if x.shape[0] > max_len:
                    x = x[:max_len]

                # нормализация
                x = (x - mean) / std

                yield x, p
            except Exception as e:
                print(f"[warn] skip '{p}': {type(e).__name__}: {e}", file=sys.stderr)

    out_sig = (
        # X
        (   # переменная длина по времени до паддинга
            # но мы зададим None, чтобы from_generator принял любую длину
        ),
        # путь (строка)
        None,
    )
    # Чёткая сигнатура (TensorSpec) нужна для TF 2.x
    out_sig = (
        # (None, feat_dim)
        tf.TensorSpec(shape=(None, feat_dim), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.string),
    )

    def pad_map(x, path):
        T = tf.shape(x)[0]
        pad_t = tf.maximum(0, max_len - T)
        x = tf.pad(x, [[0, pad_t], [0, 0]])
        x = x[:max_len]
        x.set_shape([max_len, feat_dim])
        return x, path

    ds = tf.data.Dataset.from_generator(gen, output_signature=out_sig)
    ds = ds.map(pad_map, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def predict(model, ds, mc_passes: int = 1):
    """
    Возвращает:
      paths_all: List[str]
      probs: np.ndarray shape (N,)
      probs_std (или None): np.ndarray shape (N,)
    """
    import tensorflow as tf

    if mc_passes <= 1:
        paths_all: List[str] = []
        probs_all: List[np.ndarray] = []
        for xb, pb in ds:
            p = model(xb, training=False)
            probs_all.append(tf.cast(p, tf.float32).numpy().reshape(-1))
            paths_all.extend([s.decode("utf-8") for s in pb.numpy().tolist()])
        probs = np.concatenate(probs_all, axis=0) if probs_all else np.zeros((0,), dtype=np.float32)
        return paths_all, probs, None

    # MC Dropout
    mc_stack: List[np.ndarray] = []
    paths_all: List[str] = []
    for t in range(mc_passes):
        probs_t: List[np.ndarray] = []
        paths_t: List[str] = []
        for xb, pb in ds:
            p = model(xb, training=True)  # включаем dropout
            probs_t.append(p.numpy().reshape(-1))
            if t == 0:
                paths_t.extend([s.decode("utf-8") for s in pb.numpy().tolist()])
        mc_stack.append(np.concatenate(probs_t, axis=0) if probs_t else np.zeros((0,), dtype=np.float32))
        if t == 0:
            paths_all = paths_t

    mc_arr = np.stack(mc_stack, axis=0) if mc_stack else np.zeros((0, 0), dtype=np.float32)
    probs_mean = mc_arr.mean(axis=0) if mc_arr.size else np.zeros((0,), dtype=np.float32)
    probs_std = mc_arr.std(axis=0) if mc_arr.size else np.zeros((0,), dtype=np.float32)
    return paths_all, probs_mean, probs_std


# ==========================
#          MAIN
# ==========================

def main():
    ap = argparse.ArgumentParser("GRU inference with confidence")
    # core
    ap.add_argument("--model",       required=True, help="Путь к model.keras / SavedModel / .h5")
    ap.add_argument("--norm_stats",  required=True, help="Путь к norm_stats.npz (mean,std,max_len)")
    ap.add_argument("--threshold",   default=None,  help="Файл с числом-порогом; если не задан, берём 0.5")
    ap.add_argument("--downsample",  type=int, default=1, help="Должен совпадать с train (если там меняли)")
    ap.add_argument("--batch_size",  type=int, default=32)
    ap.add_argument("--mc_passes",   type=int, default=1, help=">1 включает MC Dropout и даёт prob_std")

    # inputs
    ap.add_argument("--csv",         default=None, help="CSV с колонкой filename (альтернатива — передать пути позиционными аргументами)")
    ap.add_argument("--data_dir",    default=".", help="База для поиска файлов из CSV/относительных путей; также источник для --scan_first")
    ap.add_argument("--filename_col", default="filename")
    ap.add_argument("--out_csv",     required=True, help="Куда записать предсказания (.csv)")
    ap.add_argument("inputs", nargs="*", help="Список путей к .npy/.json.npy (если не используем --csv)")

    # scanning options
    ap.add_argument("--scan_first", action="store_true",
                    help="Сканировать data_dir на *.npy, затем (если задан CSV) брать пересечение")
    ap.add_argument("--recursive",  action="store_true",
                    help="Рекурсивно обходить подкаталоги data_dir при scan_first")

    # compute device
    ap.add_argument("--gpus", default="all", help='"all" | "cpu" | "0,1"')

    args = ap.parse_args()

    # GPU visibility до импорта TF
    if args.gpus.lower() == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif args.gpus.lower() != "all":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    import tensorflow as tf  # noqa

    # 1) входные пути
    if args.scan_first and args.data_dir:
        paths = choose_paths_scan_then_csv(args.data_dir, args.csv, args.filename_col, args.recursive)
    elif args.csv:
        paths = resolve_inputs_from_csv(args.csv, args.data_dir or ".", args.filename_col)
    else:
        paths = resolve_inputs_from_args(args.inputs, args.data_dir)

    if not paths:
        print("Нет входных файлов. Укажи --scan_first или --csv, или список путей.", file=sys.stderr)
        sys.exit(2)

    # 2) норм-статы и модель
    mean, std, max_len = load_norm_stats(args.norm_stats)
    model = load_any_model(args.model)

    # 3) датасет
    ds = make_dataset(paths, mean, std, max_len, args.downsample, args.batch_size)

    # 4) инференс (+опц. MC Dropout)
    pths, probs, probs_std = predict(model, ds, mc_passes=args.mc_passes)
    if len(pths) == 0:
        print("Нечего предсказывать — все примеры оказались битыми.", file=sys.stderr)
        sys.exit(3)

    # 5) порог
    thr = 0.5
    if args.threshold and os.path.exists(args.threshold):
        try:
            with open(args.threshold, "r", encoding="utf-8") as f:
                thr = float(f.read().strip())
        except Exception:
            print("[warn] не удалось прочитать threshold, используем 0.5", file=sys.stderr)

    # 6) метрики уверенности
    probs = probs.astype(np.float32)
    pred = (probs >= thr).astype(np.int32)
    confidence = np.where(pred == 1, probs, 1.0 - probs)
    p_clip = np.clip(probs, 1e-8, 1 - 1e-8)
    logit = np.log(p_clip / (1.0 - p_clip)).astype(np.float32)
    entr = entropy_binary(p_clip).astype(np.float32)

    # 7) сбор таблицы
    out = {
        "path": pths,
        "prob": probs,
        "pred": pred,
        "confidence": confidence,
        "logit": logit,
        "entropy": entr,
    }
    if probs_std is not None:
        out["prob_std"] = probs_std.astype(np.float32)

    df = pd.DataFrame(out)
    df.to_csv(args.out_csv, index=False, float_format="%.6f")

    # 8) краткий отчёт
    n = len(df)
    n_pos = int(df["pred"].sum())
    n_neg = n - n_pos
    print(f"Saved predictions to: {args.out_csv}")
    print(f"Used threshold: {thr}")
    print(f"Total: {n} | pred=1: {n_pos} | pred=0: {n_neg}")
    if "prob_std" in df.columns:
        print(f"MC Dropout: passes={args.mc_passes} | mean std={df['prob_std'].mean():.6f}")


if __name__ == "__main__":
    main()
