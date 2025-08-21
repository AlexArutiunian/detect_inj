#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified inference (NN + classical) with confidence — for NPY and JSON.
Теперь ещё рисует горизонтальные бары по всем группам:
- confidence buckets
- injury buckets (pred==1)
- no-injury buckets (pred==0)
"""

from __future__ import annotations
import os, sys, json, argparse
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

# --- Matplotlib для сохранения графиков без GUI ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# ================= численные утилиты =================

def sanitize_seq(a: np.ndarray) -> np.ndarray:
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    return a.astype(np.float32, copy=False)

def entropy_binary(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-8, 1 - 1e-8)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))

def load_norm_stats(path: str) -> Tuple[np.ndarray, np.ndarray, int]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"norm_stats '{path}' не найден")
    d = np.load(path, allow_pickle=False)
    if not all(k in d for k in ("mean", "std", "max_len")):
        raise ValueError(f"В '{path}' нет всех ключей (нужны mean, std, max_len)")
    mean = d["mean"].astype(np.float32)
    std  = d["std"].astype(np.float32)
    max_len = int(d["max_len"])
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std, max_len

# ==== ДОБАВЬ это где-нибудь рядом с численными утилитами ====
def as_TxF(a: np.ndarray) -> Optional[np.ndarray]:
    """Привести вход к (T, F). Поддерживает (T,F) и (T,V,C)."""
    if a is None:
        return None
    a = np.asarray(a)
    if a.ndim == 2:
        return a
    if a.ndim == 3:
        T, V, C = a.shape
        return a.reshape(T, V * C)
    if a.ndim == 1:
        return a.reshape(-1, 1)
    return None

# ================= пути/сканирование =================

def _norm_rel_with_suffix(s: str, input_format: str) -> str:
    s = (s or "").replace("\\", "/").lstrip("/")
    suf = ".npy" if input_format == "npy" else ".json"
    return s if s.endswith(suf) else (s + suf)

def list_files_with_ext(root: str, ext: str, recursive: bool=True) -> List[str]:
    out: List[str] = []
    if not root: root = "."
    if not os.path.isdir(root): return out
    ext = ext.lower()
    if recursive:
        for dp, _, files in os.walk(root):
            for f in files:
                if f.lower().endswith(ext):
                    out.append(os.path.join(dp, f))
    else:
        for f in os.listdir(root):
            p = os.path.join(root, f)
            if os.path.isfile(p) and p.lower().endswith(ext):
                out.append(p)
    return sorted(set(out))

def possible_paths(data_dir: str, rel: str, input_format: str) -> List[str]:
    rel = (rel or "").strip().replace("\\", "/").lstrip("/")
    cands: List[str] = []
    def push(x: str):
        if x and x not in cands: cands.append(x)

    push(os.path.join(data_dir, rel))

    if input_format == "npy":
        if not rel.endswith(".npy"): push(os.path.join(data_dir, rel + ".npy"))
        if rel.endswith(".json"):    push(os.path.join(data_dir, rel[:-5] + ".npy"))
        if rel.endswith(".json.npy"):push(os.path.join(data_dir, rel.replace(".json.npy", ".npy")))
        b = os.path.basename(rel)
        push(os.path.join(data_dir, b))
        if not b.endswith(".npy"): push(os.path.join(data_dir, b + ".npy"))
        if b.endswith(".json"):    push(os.path.join(data_dir, b[:-5] + ".npy")); push(os.path.join(data_dir, b + ".npy"))
        if b.endswith(".json.npy"):push(os.path.join(data_dir, b.replace(".json.npy", ".npy")))
    else:
        if not rel.endswith(".json"): push(os.path.join(data_dir, rel + ".json"))
        if rel.endswith(".npy"):      push(os.path.join(data_dir, rel[:-4] + ".json"))
        if rel.endswith(".json.npy"): push(os.path.join(data_dir, rel[:-4]))
        b = os.path.basename(rel)
        push(os.path.join(data_dir, b))
        if not b.endswith(".json"): push(os.path.join(data_dir, b + ".json"))
        if b.endswith(".npy"):      push(os.path.join(data_dir, b[:-4] + ".json"))
        if b.endswith(".json.npy"): push(os.path.join(data_dir, b[:-4]))
    return cands

def pick_existing_path(cands: List[str]) -> Optional[str]:
    for p in cands:
        if os.path.exists(p): return p
    return None
def choose_paths_scan_then_csv(data_dir: str,
                               csv_path: Optional[str],
                               filename_col: str,
                               recursive: bool,
                               input_format: str) -> List[str]:
    """
    1) Сканируем data_dir (по расширению из input_format) -> список файлов на диске.
    2) Если задан CSV — берём пересечение с CSV, НО c нормализацией:
       для каждой CSV-строки пробуем все кандидаты через possible_paths(...),
       совпадаем либо по абсолютному существующему файлу, либо по basename из скана.
    """
    # 1) что лежит на диске
    ext = ".npy" if input_format == "npy" else ".json"
    disk_files = list_files_with_ext(data_dir, ext, recursive=recursive)

    if not csv_path or not os.path.exists(csv_path):
        if csv_path and not os.path.exists(csv_path):
            print(f"[warn] CSV '{csv_path}' не найден; используем только файлы на диске.", file=sys.stderr)
        return disk_files

    # индексы для быстрых совпадений
    disk_set = set(os.path.normpath(p) for p in disk_files)
    by_base: dict[str, str] = {}
    for p in disk_files:
        b = os.path.basename(p)
        if b not in by_base:  # первое вхождение
            by_base[b] = p

    # 2) читаем CSV и сопоставляем с диском
    df = pd.read_csv(csv_path)
    cols = {c.lower().strip(): c for c in df.columns}
    fn_col = cols.get(filename_col.lower(), filename_col)
    if fn_col not in df.columns:
        for c in df.columns:
            if "file" in c.lower() or "name" in c.lower():
                fn_col = c; break

    chosen: list[str] = []
    missing = 0
    for raw in df[fn_col].astype(str).tolist():
        # переберём все разумные кандидаты путей (json<->npy тоже)
        cands = possible_paths(data_dir, raw, input_format)
        hit = None
        for c in cands:
            nc = os.path.normpath(c)
            if os.path.exists(nc):
                hit = nc; break
            # попробуем совпасть по basename со сканом
            b = os.path.basename(nc)
            if b in by_base:
                hit = by_base[b]; break
        if hit:
            chosen.append(hit)
        else:
            missing += 1

    if missing:
        print(f"[warn] В CSV указано файлов, которых нет на диске: {missing}", file=sys.stderr)
    extra = len(disk_files) - len(set(chosen))
    if extra > 0:
        print(f"[info] На диске есть файлов вне CSV: {extra}", file=sys.stderr)

    # уберём дубликаты и вернём в стабильном порядке
    seen, out = set(), []
    for p in chosen:
        if p not in seen:
            seen.add(p); out.append(p)
    return out


def resolve_inputs_from_csv(csv_path: str, data_dir: str, filename_col: str, input_format: str) -> List[str]:
    if not os.path.exists(csv_path):
        print(f"[warn] CSV '{csv_path}' не найден.", file=sys.stderr)
        return []
    df = pd.read_csv(csv_path)
    cols = {c.lower().strip(): c for c in df.columns}
    fn_col = cols.get(filename_col.lower(), filename_col)
    if fn_col not in df.columns:
        for c in df.columns:
            if "file" in c.lower() or "name" in c.lower():
                fn_col = c; break
    paths, missing = [], 0
    for rel in df[fn_col].astype(str).tolist():
        p = pick_existing_path(possible_paths(data_dir, rel, input_format))
        if p: paths.append(p)
        else: missing += 1
    if missing:
        print(f"[warn] Не найдено {missing} файлов из CSV", file=sys.stderr)
    return paths

def resolve_inputs_from_args(inputs: List[str], data_dir: Optional[str], input_format: str) -> List[str]:
    out: List[str] = []
    for p in inputs:
        if os.path.exists(p): out.append(p)
        elif data_dir:
            r = pick_existing_path(possible_paths(data_dir, p, input_format))
            if r: out.append(r)
    return out


# ================= JSON-помощники =================

def _safe_json_load(path: str):
    try:
        import orjson
        with open(path, "rb") as f: return orjson.loads(f.read())
    except Exception:
        with open(path, "r", encoding="utf-8") as f: return json.load(f)

def _stack_motion_frames_with_schema(md: dict, schema_joints: List[str]) -> Optional[np.ndarray]:
    present = [j for j in schema_joints if j in md]
    if not present: return None
    T = min(len(md[j]) for j in present)
    if T <= 0: return None
    cols = []
    for j in schema_joints:
        if j in md: arr = np.asarray(md[j], dtype=np.float32)[:T]
        else:       arr = np.full((T,3), np.nan, dtype=np.float32)
        cols.append(arr)
    return np.concatenate(cols, axis=1)  # (T, 3*|schema|)

def _infer_schema_from_first(paths: List[str], motion_keys=("running","walking")) -> Optional[List[str]]:
    for p in paths:
        try:
            d = _safe_json_load(p)
            md = None
            for k in motion_keys:
                if k in d and isinstance(d[k], dict):
                    md = d[k]; break
            if not md: continue
            joints = sorted(list(md.keys()))
            if joints: return joints
        except Exception:
            continue
    return None


# ================= классические фичи (train3.py) =================

def _features_from_seq(seq_np: np.ndarray) -> np.ndarray:
    seq = seq_np.astype(np.float32, copy=False)
    dif = np.diff(seq, axis=0)
    stat = np.concatenate([
        np.nanmean(seq, axis=0), np.nanstd(seq, axis=0),
        np.nanmin(seq, axis=0),  np.nanmax(seq, axis=0),
        np.nanmean(dif, axis=0), np.nanstd(dif, axis=0)
    ]).astype(np.float32, copy=False)
    return np.nan_to_num(stat, nan=0.0, posinf=0.0, neginf=0.0)

def build_features_for_paths(paths: List[str],
                             input_format: str,
                             downsample: int,
                             schema_joints: Optional[List[str]],
                             motion_key: str="running") -> Tuple[np.ndarray, List[str]]:
    X_list, keep_paths = [], []
    for p in paths:
        try:
            if input_format == "npy":
                arr = np.load(p, allow_pickle=False, mmap_mode="r")
                arr = as_TxF(arr)               # <-- добавили
                if arr is None or arr.shape[0] < 2:
                    continue
                seq = arr[::max(1, downsample)]

            else:
                d = _safe_json_load(p)
                md = d.get(motion_key)
                if not isinstance(md, dict):
                    for k in ("running","walking"):
                        if k in d and isinstance(d[k], dict):
                            md = d[k]; break
                if not isinstance(md, dict): continue
                if not schema_joints:
                    continue
                seq = _stack_motion_frames_with_schema(md, schema_joints)
                if seq is None or seq.shape[0] < 2: continue
                if downsample > 1: seq = seq[::downsample]
            X_list.append(_features_from_seq(np.asarray(seq)))
            keep_paths.append(p)
        except Exception:
            continue
    if not X_list:
        raise RuntimeError("Не удалось собрать ни одной строки фич для классической модели.")
    return np.stack(X_list).astype(np.float32, copy=False), keep_paths


# ================= загрузка моделей =================

def is_classical_path(p: str) -> bool:
    low = p.lower()
    return low.endswith(".joblib") or low.endswith(".pkl")

def load_any_model(model_path: str, unsafe_deser: bool = False):
    import os
    try:
        import keras
        if unsafe_deser:
            try: keras.config.enable_unsafe_deserialization()
            except Exception: pass
        import tensorflow as tf  # noqa
        try:    return tf.keras.models.load_model(model_path, safe_mode=not unsafe_deser)
        except TypeError: return tf.keras.models.load_model(model_path)
    except Exception as e_tf:
        try:
            import keras  # noqa
            try:    return keras.saving.load_model(model_path, safe_mode=not unsafe_deser)
            except TypeError: return keras.saving.load_model(model_path)
        except Exception as e_k:
            if os.path.isdir(model_path):
                import tensorflow as tf  # noqa
                try:    return tf.keras.models.load_model(model_path, safe_mode=not unsafe_deser)
                except TypeError: return tf.keras.models.load_model(model_path)
            raise RuntimeError(
                f"Не удалось загрузить модель '{model_path}'. "
                f"tf.keras: {type(e_tf).__name__}; keras: {type(e_k).__name__}"
            )

def load_classical_bundle(path: str):
    import joblib
    obj = joblib.load(path)
    if isinstance(obj, dict):
        return obj.get("model", obj), obj.get("scaler", None)
    return obj, None


# ================= датасет для NN =================

def make_dataset_nn(paths: List[str],
                    mean: np.ndarray,
                    std: np.ndarray,
                    max_len: int,
                    downsample: int,
                    batch_size: int,
                    input_format: str,
                    schema_joints: Optional[List[str]] = None,
                    motion_key: str = "running"):
    import tensorflow as tf
    AUTOTUNE = tf.data.AUTOTUNE
    feat_dim = int(mean.shape[0])

    def _load_seq(p: str) -> Optional[np.ndarray]:
        if input_format == "npy":
            arr = np.load(p, allow_pickle=False, mmap_mode="r")
            arr = as_TxF(arr)               # <-- добавили
            if arr is None or arr.shape[0] < 1:
                return None
            return arr

        d = _safe_json_load(p)
        md = d.get(motion_key)
        if not isinstance(md, dict):
            for k in ("running","walking"):
                if k in d and isinstance(d[k], dict):
                    md = d[k]; break
        if not isinstance(md, dict): return None
        if not schema_joints: return None
        return _stack_motion_frames_with_schema(md, schema_joints)

    def gen():
        for p in paths:
            try:
                x0 = _load_seq(p)
                if x0 is None:
                    print(f"[warn] пропуск '{p}': пустой/неправильный вход", file=sys.stderr)
                    continue
                x = sanitize_seq(x0[::max(1, downsample)])
                if x.shape[1] != feat_dim:
                    if x.shape[1] > feat_dim: x = x[:, :feat_dim]
                    else: x = np.pad(x, [[0,0],[0, feat_dim - x.shape[1]]], mode="constant")
                if x.shape[0] > max_len: x = x[:max_len]
                x = (x - mean) / std
                yield x, p
            except Exception as e:
                print(f"[warn] skip '{p}': {type(e).__name__}: {e}", file=sys.stderr)

    import tensorflow as tf  # noqa
    out_sig = (
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
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    return ds

def predict_nn(model, ds, mc_passes: int = 1):
    import tensorflow as tf
    if mc_passes <= 1:
        paths_all, probs_all = [], []
        for xb, pb in ds:
            p = model(xb, training=False)
            probs_all.append(tf.cast(p, tf.float32).numpy().reshape(-1))
            paths_all.extend([s.decode("utf-8") for s in pb.numpy().tolist()])
        probs = np.concatenate(probs_all, axis=0) if probs_all else np.zeros((0,), dtype=np.float32)
        return paths_all, probs, None

    mc_stack, paths_all = [], []
    for t in range(mc_passes):
        probs_t, paths_t = [], []
        for xb, pb in ds:
            p = model(xb, training=True)
            probs_t.append(p.numpy().reshape(-1))
            if t == 0:
                paths_t.extend([s.decode("utf-8") for s in pb.numpy().tolist()])
        mc_stack.append(np.concatenate(probs_t, axis=0) if probs_t else np.zeros((0,), dtype=np.float32))
        if t == 0: paths_all = paths_t
    mc_arr = np.stack(mc_stack, axis=0) if mc_stack else np.zeros((0,0), dtype=np.float32)
    return paths_all, mc_arr.mean(axis=0), mc_arr.std(axis=0)


# ================= отрисовка суммарных групп =================

def plot_group_bars(df: pd.DataFrame, out_path: str, title_prefix: str = "", dpi: int = 180):
    """Рисует 3 блока: confidence / injury(pred=1) / no-injury(pred=0) — горизонтальные бары.
       Подписи внутри/снаружи подбираются автоматически, чтобы не вылезали.
    """
    # Подготовка данных
    conf_labels = ["conf <50%", "conf 50–60%", "conf 60–80%", "conf >80%"]
    inj_labels  = ["inj <50%", "inj 50–60%", "inj 60–80%", "inj >80%"]
    noi_labels  = ["no-injury <50%", "no-injury 50–60%", "no-injury 60–80%", "no-injury >80%"]

    conf_counts = [int((df["confidence_group"] == g).sum()) for g in conf_labels]
    inj_counts  = [int((df["inj_group"] == g).sum()) for g in inj_labels]
    noi_counts  = [int((df["noinj_group"] == g).sum()) for g in noi_labels]

    # Фигура
    # высота под число категорий; ширину возьмём побольше, чтобы подписи точно влезли
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    blocks = [
        (axes[0], conf_labels, conf_counts, "Confidence buckets (all)"),
        (axes[1], inj_labels,  inj_counts,  "Injury buckets (pred==1)"),
        (axes[2], noi_labels,  noi_counts,  "No-injury buckets (pred==0)"),
    ]

    for ax, labels, counts, ttl in blocks:
        y = np.arange(len(labels))
        ax.barh(y, counts)
        ax.set_yticks(y, labels=labels)
        ax.invert_yaxis()
        ax.set_xlabel("Count")
        tt = f"{title_prefix} — {ttl}" if title_prefix else ttl
        ax.set_title(tt)
        maxc = max([0] + counts)
        ax.set_xlim(0, max(1, int(maxc * 1.18)))  # запас справа, чтобы текст не резался

        # подписи: если столбик широкий — пишем внутри, иначе чуть правее бара
        for i, v in enumerate(counts):
            if maxc == 0:
                ax.text(0.02, i, "0", va="center", ha="left")
                continue
            if v >= 0.18 * maxc:
                ax.text(v - 0.02 * maxc, i, f"{v}", va="center", ha="right", color="white", fontsize=9)
            else:
                ax.text(v + 0.02 * maxc, i, f"{v}", va="center", ha="left", fontsize=9)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Group plot saved: {out_path}")


# ================= main =================

def main():
    ap = argparse.ArgumentParser("Unified inference (NN + classical) for NPY/JSON with group bars")
    # формат данных
    ap.add_argument("--input_format", choices=["npy","json"], default="npy")
    ap.add_argument("--motion_key", default="running", help="для JSON: running|walking")
    ap.add_argument("--schema_json", default="", help="для JSON: путь к schema_joints.json (порядок суставов)")
    ap.add_argument("--use_joints", default="", help="для JSON: список через запятую (приоритет над schema_json)")

    # модель
    ap.add_argument("--model", required=True, help=".keras/.h5/SavedModel ИЛИ .joblib/.pkl (классика)")
    ap.add_argument("--norm_stats", default="", help="для NN: norm_stats.npz (mean,std,max_len)")
    ap.add_argument("--threshold",  default=None, help="порог; если файл, читается из файла, иначе число; дефолт 0.5")
    ap.add_argument("--downsample", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--mc_passes",  type=int, default=1, help="только для NN (MC Dropout)")

    # входы
    ap.add_argument("--csv",         default=None)
    ap.add_argument("--data_dir",    default=".")
    ap.add_argument("--filename_col", default="filename")
    ap.add_argument("--out_csv",     required=True)
    ap.add_argument("inputs", nargs="*")

    # scan
    ap.add_argument("--scan_first", action="store_true")
    ap.add_argument("--recursive",  action="store_true")

    # device / keras
    ap.add_argument("--gpus", default="all")
    ap.add_argument("--unsafe_deser", action="store_true")

    # plot
    ap.add_argument("--plot_summary", default="", help="PNG для групповых баров; по умолчанию: <out_csv>_groups.png")
    ap.add_argument("--plot_title",   default="", help="Префикс к заголовкам графиков")
    ap.add_argument("--plot_dpi",     type=int, default=180)

    args = ap.parse_args()

    # определить тип модели
    is_classical = is_classical_path(args.model)

    # GPU только для NN
    if not is_classical:
        if args.gpus.lower() == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        elif args.gpus.lower() != "all":
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    # входные пути
    if args.scan_first and args.data_dir:
        paths = choose_paths_scan_then_csv(args.data_dir, args.csv, args.filename_col, args.recursive, args.input_format)
    elif args.csv:
        paths = resolve_inputs_from_csv(args.csv, args.data_dir or ".", args.filename_col, args.input_format)
    else:
        paths = resolve_inputs_from_args(args.inputs, args.data_dir, args.input_format)

    if not paths:
        print("Нет входных файлов. Укажи --scan_first или --csv, или список путей.", file=sys.stderr)
        sys.exit(2)

    # схема суставов для JSON (и для классики, и для NN)
    schema_joints: Optional[List[str]] = None
    if args.input_format == "json":
        if args.use_joints.strip():
            schema_joints = [s.strip() for s in args.use_joints.split(",") if s.strip()]
        elif args.schema_json and os.path.exists(args.schema_json):
            with open(args.schema_json, "r", encoding="utf-8") as f:
                schema_joints = list(json.load(f))
        else:
            schema_joints = _infer_schema_from_first(paths)
            if schema_joints:
                print(f"[info] auto schema from JSON keys: {len(schema_joints)} joints")
        if not schema_joints:
            raise RuntimeError("Для JSON не удалось определить схему суставов. Передайте --schema_json или --use_joints.")

    # порог
    thr = 0.5
    if args.threshold:
        if os.path.exists(args.threshold):
            try:
                with open(args.threshold, "r", encoding="utf-8") as f: thr = float(f.read().strip())
            except Exception:
                print("[warn] не удалось прочитать threshold, используем 0.5", file=sys.stderr)
        else:
            try: thr = float(args.threshold)
            except Exception: pass

    # === Ветка КЛАССИЧЕСКОЙ модели ===
    if is_classical:
        model, scaler = load_classical_bundle(args.model)
        X, keep_paths = build_features_for_paths(paths, args.input_format, max(1, int(args.downsample)), schema_joints, motion_key=args.motion_key)
        Xs = scaler.transform(X) if scaler is not None else X

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(Xs)[:, 1]
        elif hasattr(model, "decision_function"):
            d = model.decision_function(Xs)
            d = (d - d.min()) / (d.max() - d.min() + 1e-9)
            prob = d
        else:
            prob = model.predict(Xs).astype(np.float32)

        probs = np.asarray(prob, dtype=np.float32)
        pred = (probs >= thr).astype(np.int32)
        p_clip = np.clip(probs, 1e-8, 1 - 1e-8)
        logit = np.log(p_clip / (1.0 - p_clip)).astype(np.float32)
        confidence = np.where(pred == 1, p_clip, 1.0 - p_clip).astype(np.float32)
        entr = entropy_binary(p_clip).astype(np.float32)

        def _conf_group(c: float) -> str:
            if c >= 0.8: return "conf >80%"
            if c >= 0.6: return "conf 60–80%"
            if c >= 0.5: return "conf 50–60%"
            return "conf <50%"

        def _inj_group(p: float) -> str:
            if p >= 0.8: return "inj >80%"
            if p >= 0.6: return "inj 60–80%"
            if p >= 0.5: return "inj 50–60%"
            return "inj <50%"

        def _noinj_group(p0: float) -> str:
            if p0 >= 0.8: return "no-injury >80%"
            if p0 >= 0.6: return "no-injury 60–80%"
            if p0 >= 0.5: return "no-injury 50–60%"
            return "no-injury <50%"

        confidence_group = [_conf_group(float(c)) for c in confidence.tolist()]
        inj_group  = [_inj_group(float(p))  if y == 1 else "" for p, y in zip(p_clip.tolist(), pred.tolist())]
        noin_group = [_noinj_group(float(1.0 - p)) if y == 0 else "" for p, y in zip(p_clip.tolist(), pred.tolist())]
        pred_group = [inj_group[i] if pred[i] == 1 else noin_group[i] for i in range(len(pred))]

        df = pd.DataFrame({
            "path": keep_paths,
            "prob": probs,
            "pred": pred,
            "confidence": confidence,
            "logit": logit,
            "entropy": entr,
            "confidence_group": confidence_group,
            "inj_group": inj_group,
            "noinj_group": noin_group,
            "pred_group": pred_group,
        }).sort_values(["confidence"], ascending=[False]).reset_index(drop=True)

        df.to_csv(args.out_csv, index=False, float_format="%.6f")

        # график групп
        plot_path = args.plot_summary or (os.path.splitext(args.out_csv)[0] + "_groups.png")
        plot_group_bars(df, plot_path, title_prefix=args.plot_title, dpi=args.plot_dpi)

        # отчёт
        n = len(df); n_pos = int(df["pred"].sum()); n_neg = n - n_pos
        c_lt50 = int((df["confidence_group"] == "conf <50%").sum())
        c_50_60 = int((df["confidence_group"] == "conf 50–60%").sum())
        c_60_80 = int((df["confidence_group"] == "conf 60–80%").sum())
        c_80p   = int((df["confidence_group"] == "conf >80%").sum())
        print(f"Predicted: injury={n_pos} | no-injury={n_neg}")
        print(f"Confidence buckets: <50%={c_lt50} | 50–60%={c_50_60} | 60–80%={c_60_80} | >80%={c_80p}")
        return

    # === Ветка НЕЙРОСЕТИ ===
    if not args.norm_stats:
        raise RuntimeError("Для NN необходимо указать --norm_stats (mean,std,max_len).")

    mean, std, max_len = load_norm_stats(args.norm_stats)

    # предупреждение для JSON
    if args.input_format == "json":
        # ожидаем F=3*|joints| — проверка только для информации
        pass

    model = load_any_model(args.model, unsafe_deser=args.unsafe_deser)

    ds = make_dataset_nn(paths, mean, std, max_len, args.downsample, args.batch_size,
                         input_format=args.input_format, schema_joints=schema_joints, motion_key=args.motion_key)
    pths, probs, probs_std = predict_nn(model, ds, mc_passes=args.mc_passes)
    if len(pths) == 0:
        print("Нечего предсказывать — все примеры оказались битыми.", file=sys.stderr)
        sys.exit(3)

    probs = probs.astype(np.float32)
    pred = (probs >= thr).astype(np.int32)
    p_clip = np.clip(probs, 1e-8, 1 - 1e-8)
    logit = np.log(p_clip / (1.0 - p_clip)).astype(np.float32)
    confidence = np.where(pred == 1, p_clip, 1.0 - p_clip).astype(np.float32)
    entr = entropy_binary(p_clip).astype(np.float32)

    def _conf_group(c: float) -> str:
        if c >= 0.8: return "conf >80%"
        if c >= 0.6: return "conf 60–80%"
        if c >= 0.5: return "conf 50–60%"
        return "conf <50%"

    def _inj_group(p: float) -> str:
        if p >= 0.8: return "inj >80%"
        if p >= 0.6: return "inj 60–80%"
        if p >= 0.5: return "inj 50–60%"
        return "inj <50%"

    def _noinj_group(p0: float) -> str:
        if p0 >= 0.8: return "no-injury >80%"
        if p0 >= 0.6: return "no-injury 60–80%"
        if p0 >= 0.5: return "no-injury 50–60%"
        return "no-injury <50%"

    confidence_group = [_conf_group(float(c)) for c in confidence.tolist()]
    inj_group  = [_inj_group(float(p))  if y == 1 else "" for p, y in zip(p_clip.tolist(), pred.tolist())]
    noin_group = [_noinj_group(float(1.0 - p)) if y == 0 else "" for p, y in zip(p_clip.tolist(), pred.tolist())]
    pred_group = [inj_group[i] if pred[i] == 1 else noin_group[i] for i in range(len(pred))]

    out = {
        "path": pths,
        "prob": probs,
        "pred": pred,
        "confidence": confidence,
        "logit": logit,
        "entropy": entr,
        "confidence_group": confidence_group,
        "inj_group": inj_group,
        "noinj_group": noin_group,
        "pred_group": pred_group,
    }
    if probs_std is not None:
        out["prob_std"] = probs_std.astype(np.float32)

    df = pd.DataFrame(out).sort_values(["confidence"], ascending=[False]).reset_index(drop=True)
    df.to_csv(args.out_csv, index=False, float_format="%.6f")

    # график групп
    plot_path = args.plot_summary or (os.path.splitext(args.out_csv)[0] + "_groups.png")
    plot_group_bars(df, plot_path, title_prefix=args.plot_title, dpi=args.plot_dpi)

    n = len(df); n_pos = int(df["pred"].sum()); n_neg = n - n_pos
    c_lt50 = int((df["confidence_group"] == "conf <50%").sum())
    c_50_60 = int((df["confidence_group"] == "conf 50–60%").sum())
    c_60_80 = int((df["confidence_group"] == "conf 60–80%").sum())
    c_80p   = int((df["confidence_group"] == "conf >80%").sum())
    print(f"Predicted: injury={n_pos} | no-injury={n_neg}")
    print(f"Confidence buckets: <50%={c_lt50} | 50–60%={c_50_60} | 60–80%={c_60_80} | >80%={c_80p}")
    if "prob_std" in df.columns:
        print(f"MC Dropout: passes={args.mc_passes} | mean std={df['prob_std'].mean():.6f}")

if __name__ == "__main__":
    main()
