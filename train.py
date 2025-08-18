#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ===================== detect_inj/train.py =====================

import os
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight
import joblib

# ===================== УТИЛИТЫ =====================

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def label_to_int(v):
    if v is None: return None
    s = str(v).strip().lower()
    if s == "injury": return 1
    if s == "no injury": return 0
    return None

def compute_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else float("nan"),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "report": classification_report(y_true, y_pred, digits=3),
    }

def best_threshold_by_f1(y_true, proba):
    from sklearn.metrics import precision_recall_curve
    pr, rc, th = precision_recall_curve(y_true, proba)
    f1_vals = 2 * pr[:-1] * rc[:-1] / np.clip(pr[:-1] + rc[:-1], 1e-12, None)
    if len(f1_vals) == 0 or np.all(np.isnan(f1_vals)): return 0.5
    best_idx = int(np.nanargmax(f1_vals))
    return float(th[max(0, min(best_idx, len(th) - 1))])

def print_split_stats(name, y):
    n = len(y); n1 = int(np.sum(y == 1)); n0 = n - n1
    p1 = (n1 / n * 100) if n else 0.0; p0 = (n0 / n * 100) if n else 0.0
    print(f"[{name}] total={n} | Injury=1: {n1} ({p1:.1f}%) | No Injury=0: {n0} ({p0:.1f}%)")

def _sanitize_seq(a: np.ndarray) -> np.ndarray:
    # заменяем nan/inf и ограничиваем экстремумы (на всякий случай)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    return a.astype(np.float32, copy=False)

# ===================== JSON utils (для input_format=json) =====================

def _safe_json_load(path):
    try:
        import orjson
        with open(path, "rb") as f:
            return orjson.loads(f.read())
    except Exception:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

def _stack_motion_frames_with_schema(md, schema_joints):
    present = [j for j in schema_joints if j in md]
    if not present: return None
    T = min(len(md[j]) for j in present)
    if T <= 1: return None
    cols = []
    for j in schema_joints:
        if j in md:
            arr = np.asarray(md[j], dtype=np.float32)[:T]   # (T,3)
        else:
            arr = np.full((T, 3), np.nan, dtype=np.float32) # отсутствующий сустав
        cols.append(arr)
    return np.concatenate(cols, axis=1)  # (T, 3*|schema|)

def discover_joint_schema(csv_path, data_dir, motion_key,
                          filename_col="filename", label_col="No inj/ inj",
                          sample_max=500, freq_threshold=0.9, use_joints=None):
    if use_joints: return list(use_joints)
    meta = pd.read_csv(csv_path, usecols=[filename_col, label_col])
    counts, seen = {}, 0
    for _, row in meta.iterrows():
        if seen >= sample_max: break
        fn = str(row[filename_col]).strip() if pd.notnull(row.get(filename_col)) else ""
        y = label_to_int(row.get(label_col))
        if not fn or y is None: continue
        p = os.path.join(data_dir, fn)
        if not os.path.exists(p) and not fn.endswith(".json"):
            p2 = p + ".json"
            if os.path.exists(p2): p = p2
        if not os.path.exists(p): continue
        try:
            data = _safe_json_load(p)
            if motion_key not in data or not isinstance(data[motion_key], dict): continue
            for j in list(data[motion_key].keys()):
                counts[j] = counts.get(j, 0) + 1
            seen += 1
        except Exception:
            continue
    if seen == 0:
        raise RuntimeError("discover_joint_schema: не удалось прочитать ни одного файла.")
    freq = {j: c / seen for j, c in counts.items()}
    schema = [j for j, f in freq.items() if f >= freq_threshold] or list(counts.keys())
    return sorted(schema)

# ===================== ФИЧИ ДЛЯ "КЛАССИКИ" =====================

def _features_from_seq(seq_np):
    """seq_np: (T, F=3*|schema|) -> агрегированные статистики для классики."""
    seq = _sanitize_seq(seq_np)
    dif = np.diff(seq, axis=0)
    stat = np.concatenate([
        np.nanmean(seq, axis=0), np.nanstd(seq, axis=0),
        np.nanmin(seq, axis=0),  np.nanmax(seq, axis=0),
        np.nanmean(dif, axis=0), np.nanstd(dif, axis=0)
    ]).astype(np.float32, copy=False)
    return np.nan_to_num(stat, nan=0.0, posinf=0.0, neginf=0.0)

# ---- JSON -> фичи ----

def _feats_from_json_task(args):
    json_path, y, motion_key, schema_joints = args
    try:
        data = _safe_json_load(json_path)
        if motion_key not in data or not isinstance(data[motion_key], dict): return None
        md = data[motion_key]
        seq = _stack_motion_frames_with_schema(md, schema_joints)
        if seq is None or seq.shape[0] < 2: return None
        return (_features_from_seq(seq), int(y))
    except Exception:
        return None

def load_features_from_json(csv_path, data_dir, motion_key="running",
                            filename_col="filename", label_col="No inj/ inj",
                            schema_joints=None, workers=0):
    assert schema_joints is not None, "schema_joints не заданы"
    meta = pd.read_csv(csv_path, usecols=[filename_col, label_col])
    tasks = []
    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Indexing(JSON)", dynamic_ncols=True, mininterval=0.2):
        fn = str(row[filename_col]).strip() if pd.notnull(row.get(filename_col)) else ""
        y = label_to_int(row.get(label_col))
        if not fn or y is None: continue
        p = os.path.join(data_dir, fn)
        if not os.path.exists(p) and not fn.endswith(".json"):
            p2 = p + ".json"
            if os.path.exists(p2): p = p2
        if os.path.exists(p): tasks.append((p, y, motion_key, schema_joints))

    results = []
    if workers and workers > 0:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for r in tqdm(ex.map(_feats_from_json_task, tasks, chunksize=16),
                          total=len(tasks), desc="Feats(JSON)", dynamic_ncols=True, mininterval=0.2):
                if r is not None: results.append(r)
    else:
        for t in tqdm(tasks, desc="Feats(JSON)", dynamic_ncols=True, mininterval=0.2):
            r = _feats_from_json_task(t)
            if r is not None: results.append(r)

    if not results:
        raise RuntimeError("Не удалось получить ни одной строки фич из JSON.")
    X = np.stack([r[0] for r in results]).astype(np.float32, copy=False)
    y = np.array([r[1] for r in results], dtype=np.int32)
    return X, y

# ---- NPY -> фичи ----

def _map_to_npy_path(data_dir, rel_path):
    base = rel_path[:-5] if rel_path.endswith(".json") else rel_path
    if not base.endswith(".npy"):
        base = base + ".npy"
    return os.path.join(data_dir, base)

def _feats_from_npy_task(args):
    npy_path, y, downsample = args
    try:
        arr = np.load(npy_path, allow_pickle=False, mmap_mode="r")
        if arr.ndim != 2 or arr.shape[0] < 2: return None
        seq = arr[::downsample] if downsample > 1 else arr
        return (_features_from_seq(np.asarray(seq)), int(y))
    except Exception:
        return None

def load_features_from_npy(csv_path, npy_dir, filename_col="filename", label_col="No inj/ inj",
                           downsample=1, workers=0):
    meta = pd.read_csv(csv_path, usecols=[filename_col, label_col])
    tasks = []
    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Indexing(NPY)", dynamic_ncols=True, mininterval=0.2):
        fn = str(row[filename_col]).strip() if pd.notnull(row.get(filename_col)) else ""
        y = label_to_int(row.get(label_col))
        if not fn or y is None: continue
        p = _map_to_npy_path(npy_dir, fn)
        if os.path.exists(p): tasks.append((p, y, int(max(1, downsample))))

    results = []
    if workers and workers > 0:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for r in tqdm(ex.map(_feats_from_npy_task, tasks, chunksize=32),
                          total=len(tasks), desc="Feats(NPY)", dynamic_ncols=True, mininterval=0.2):
                if r is not None: results.append(r)
    else:
        for t in tqdm(tasks, desc="Feats(NPY)", dynamic_ncols=True, mininterval=0.2):
            r = _feats_from_npy_task(t)
            if r is not None: results.append(r)

    if not results:
        raise RuntimeError("Не удалось получить ни одной строки фич из NPY.")
    X = np.stack([r[0] for r in results]).astype(np.float32, copy=False)
    y = np.array([r[1] for r in results], dtype=np.int32)
    return X, y

# ===================== NN: индекс и оценка длины =====================

def _build_index(csv_path, data_dir_or_npy_dir, filename_col, label_col, input_format="npy"):
    meta = pd.read_csv(csv_path, usecols=[filename_col, label_col])
    items = []
    for _, row in meta.iterrows():
        fn = str(row[filename_col]).strip() if pd.notnull(row.get(filename_col)) else ""
        y = label_to_int(row.get(label_col))
        if not fn or y is None: continue
        if input_format == "json":
            p = os.path.join(data_dir_or_npy_dir, fn)
            if not os.path.exists(p) and not fn.endswith(".json"):
                p2 = p + ".json"
                if os.path.exists(p2): p = p2
        else:
            p = _map_to_npy_path(data_dir_or_npy_dir, fn)
        if os.path.exists(p): items.append((p, int(y)))
    return items

def _probe_max_len_json(file_label_pairs, motion_key, schema_joints, pctl=95, sample_max=None):
    L = []
    it = file_label_pairs if sample_max is None else file_label_pairs[:sample_max]
    for p, _ in tqdm(it, desc="Probe(JSON)", dynamic_ncols=True, mininterval=0.2):
        try:
            data = _safe_json_load(p)
            if motion_key not in data or not isinstance(data[motion_key], dict): continue
            md = data[motion_key]
            present = [j for j in schema_joints if j in md]
            if not present: continue
            lengths = [len(md[j]) for j in present]
            if lengths: L.append(min(lengths))
        except Exception:
            continue
    if not L: return None
    return int(np.percentile(L, pctl))

def _probe_max_len_npy(file_label_pairs, pctl=95, sample_max=None):
    L = []
    it = file_label_pairs if sample_max is None else file_label_pairs[:sample_max]
    for p, _ in tqdm(it, desc="Probe(NPY)", dynamic_ncols=True, mininterval=0.2):
        try:
            arr = np.load(p, allow_pickle=False, mmap_mode="r")
            if arr.ndim == 2 and arr.shape[0] > 1:
                L.append(int(arr.shape[0]))
        except Exception:
            continue
    if not L: return None
    return int(np.percentile(L, pctl))

# ===================== МОДЕЛИ (TF/Keras) — ленивый импорт =====================

def _lazy_tf():
    """Импортирует TF только при необходимости (для нейросетей)."""
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import (
            LSTM, Dense, Conv1D, BatchNormalization, Dropout,
            GlobalAveragePooling1D, Masking
        )
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        return {
            "tf": tf,
            "Sequential": Sequential,
            "LSTM": LSTM,
            "Dense": Dense,
            "Conv1D": Conv1D,
            "BatchNormalization": BatchNormalization,
            "Dropout": Dropout,
            "GlobalAveragePooling1D": GlobalAveragePooling1D,
            "Masking": Masking,
            "EarlyStopping": EarlyStopping,
            "pad_sequences": pad_sequences,
        }
    except Exception as e:
        raise RuntimeError(f"TensorFlow недоступен: {e}")

def _compile_with_adam(model, lr=1e-3):
    k = _lazy_tf()
    tf = k["tf"]
    opt = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
    return model

def build_lstm(input_shape):
    k = _lazy_tf()
    model = k["Sequential"]([
        k["Masking"](mask_value=0.0, input_shape=input_shape),
        k["LSTM"](128, return_sequences=False),
        k["Dense"](64, activation='relu'),
        k["Dropout"](0.3),
        k["Dense"](1, activation='sigmoid', dtype="float32")
    ])
    return _compile_with_adam(model)

def build_tcn(input_shape, dilations=(1, 2, 4, 8)):
    k = _lazy_tf()
    from tensorflow.keras.layers import Conv1D
    model = k["Sequential"]()
    for i, d in enumerate(dilations):
        if i == 0:
            model.add(Conv1D(64, 3, padding='causal', dilation_rate=d, activation='relu', input_shape=input_shape))
        else:
            model.add(Conv1D(64, 3, padding='causal', dilation_rate=d, activation='relu'))
        model.add(k["BatchNormalization"]()); model.add(k["Dropout"](0.2))
    model.add(k["GlobalAveragePooling1D"]()); model.add(k["Dense"](64, activation='relu')); model.add(k["Dropout"](0.3))
    model.add(k["Dense"](1, activation='sigmoid', dtype="float32"))
    return _compile_with_adam(model)

def build_cnn_lstm(input_shape):
    k = _lazy_tf()
    from tensorflow.keras import layers, models
    inp = layers.Input(shape=input_shape)
    x = layers.Masking(mask_value=0.0)(inp)
    x = layers.Conv1D(128, 5, padding="same", activation="relu")(x); x = layers.BatchNormalization()(x); x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 3, padding="same", activation="relu")(x); x = layers.BatchNormalization()(x); x = layers.MaxPooling1D(2)(x)
    x = layers.LSTM(128)(x); x = layers.Dropout(0.3)(x); x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid", dtype="float32")(x)
    model = models.Model(inp, out)
    return _compile_with_adam(model)

def build_gru(input_shape):
    k = _lazy_tf()
    from tensorflow.keras import layers, models
    inp = layers.Input(shape=input_shape)
    x = layers.Masking(mask_value=0.0)(inp)
    x = layers.GRU(128, return_sequences=True)(x)
    x = layers.GRU(64)(x)
    x = layers.Dropout(0.3)(x); x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid", dtype="float32")(x)
    model = models.Model(inp, out)
    return _compile_with_adam(model)

def build_transformer(input_shape, d_model=128, num_heads=4, ff_dim=256, num_layers=2, dropout=0.1):
    _ = _lazy_tf()
    from tensorflow.keras import layers, models
    inp = layers.Input(shape=input_shape)
    x = layers.Masking(mask_value=0.0)(inp)
    x = layers.Dense(d_model)(x)
    def add_positional_encoding(x_):
        import tensorflow as tf
        T = tf.shape(x_)[1]; d = int(x_.shape[-1])
        pos = tf.cast(tf.range(T)[:, None], tf.float32)
        i   = tf.cast(tf.range(d)[None, :], tf.float32)
        angle = pos / tf.pow(10000.0, (2*(i//2))/d)
        pe = tf.where(tf.equal(tf.cast(i % 2, tf.int32), 0), tf.sin(angle), tf.cos(angle))
        pe = tf.expand_dims(pe, 0)
        return x_ + pe
    x = layers.Lambda(add_positional_encoding)(x)
    for _ in range(num_layers):
        a = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads, dropout=dropout)(x, x)
        x = layers.LayerNormalization(epsilon=1e-6)(x + a)
        f = layers.Dense(ff_dim, activation="relu")(x); f = layers.Dropout(dropout)(f); f = layers.Dense(d_model)(f)
        x = layers.LayerNormalization(epsilon=1e-6)(x + f)
    x = layers.GlobalAveragePooling1D()(x); x = layers.Dropout(dropout)(x); x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid", dtype="float32")(x)
    model = models.Model(inp, out)
    return _compile_with_adam(model)

def build_timesnet(input_shape):
    _ = _lazy_tf()
    from tensorflow.keras import layers, models
    inp = layers.Input(shape=input_shape); x = layers.Masking(mask_value=0.0)(inp)
    b1 = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    b2 = layers.Conv1D(64, 5, padding="same", activation="relu")(x)
    b3 = layers.Conv1D(64, 7, padding="same", activation="relu")(x)
    x = layers.Concatenate()([b1, b2, b3]); x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling1D()(x); x = layers.Dropout(0.3)(x); x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid", dtype="float32")(x)
    model = models.Model(inp, out)
    return _compile_with_adam(model)

def build_patchtst(input_shape, patch_len=16, stride=8, d_model=128, num_layers=2, num_heads=4, ff_dim=256):
    _ = _lazy_tf()
    from tensorflow.keras import layers, models
    inp = layers.Input(shape=input_shape); x = layers.Masking(mask_value=0.0)(inp)
    x = layers.Conv1D(filters=d_model, kernel_size=patch_len, strides=stride, padding="valid")(x)
    for _ in range(num_layers):
        a = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads)(x, x)
        x = layers.LayerNormalization(epsilon=1e-6)(x + a)
        f = layers.Dense(ff_dim, activation="relu")(x); f = layers.Dense(d_model)(f)
        x = layers.LayerNormalization(epsilon=1e-6)(x + f)
    x = layers.GlobalAveragePooling1D()(x); x = layers.Dropout(0.3)(x); x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid", dtype="float32")(x)
    model = models.Model(inp, out)
    return _compile_with_adam(model)

def build_informer(input_shape, d_model=128, num_heads=4, ff_dim=256, num_layers=2, dropout=0.1):
    _ = _lazy_tf()
    from tensorflow.keras import layers, models
    inp = layers.Input(shape=input_shape); x = layers.Masking(mask_value=0.0)(inp); x = layers.Dense(d_model)(x)
    for l in range(num_layers):
        if l > 0: x = layers.Conv1D(d_model, 3, strides=2, padding="same", activation="relu")(x)
        a = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads, dropout=dropout)(x, x)
        x = layers.LayerNormalization(epsilon=1e-6)(x + a)
        f = layers.Dense(ff_dim, activation="relu")(x); f = layers.Dropout(dropout)(f); f = layers.Dense(d_model)(f)
        x = layers.LayerNormalization(epsilon=1e-6)(x + f)
    x = layers.GlobalAveragePooling1D()(x); x = layers.Dropout(dropout)(x); x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid", dtype="float32")(x)
    model = models.Model(inp, out)
    return _compile_with_adam(model)

# ===================== ОБУЧЕНИЕ (классика) =====================

def train_classical(X_train, X_dev, X_test, y_train, y_dev, y_test, model_type, out_dir):
    if model_type == "rf":
        model = RandomForestClassifier(n_estimators=400, n_jobs=-1, class_weight="balanced", random_state=42)
    elif model_type == "svm":
        model = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)
    elif model_type == "xgb":
        pos, neg = int(np.sum(y_train == 1)), int(np.sum(y_train == 0))
        spw = (neg / max(pos, 1)) if pos > 0 else 1.0
        model = XGBClassifier(
            n_estimators=600, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            objective="binary:logistic", eval_metric="auc", tree_method="hist",
            random_state=42, scale_pos_weight=spw
        )
    else:
        raise ValueError("Unknown classical model type")

    model.fit(X_train, y_train)

    prob_dev = model.predict_proba(X_dev)[:, 1]
    thr = best_threshold_by_f1(y_dev, prob_dev)
    pred_dev = (prob_dev >= thr).astype(int)
    dev_metrics = compute_metrics(y_dev, pred_dev, prob_dev)

    prob_test = model.predict_proba(X_test)[:, 1]
    pred_test = (prob_test >= thr).astype(int)
    test_metrics = compute_metrics(y_test, pred_test, prob_test)

    with open(os.path.join(out_dir, "threshold.txt"), "w") as f:
        f.write(str(thr))
    return dev_metrics, test_metrics, thr, model

# ===================== MAIN =====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Binary injury classifier from 3D joint sequences (70/10/20 split)")
    parser.add_argument("--model", type=str,
                        choices=["rf","svm","xgb","lstm","tcn","gru","cnn_lstm","transformer","timesnet","patchtst","informer"],
                        required=True)
    parser.add_argument("--csv", type=str, required=True, help="Путь к CSV с метаданными (filename, label)")
    parser.add_argument("--data_dir", type=str, required=True, help="Папка с файлами (JSON или NPY)")
    parser.add_argument("--input_format", type=str, choices=["json","npy"], default="npy",
                        help="Источник данных: json (старый) или npy (быстрый)")
    parser.add_argument("--schema_json", type=str, default="", help="schema_joints.json (для JSON / имен фич)")
    parser.add_argument("--motion_key", type=str, default="running", help="running | walking (только для JSON)")
    parser.add_argument("--filename_col", type=str, default="filename")
    parser.add_argument("--label_col", type=str, default="No inj/ inj")
    parser.add_argument("--use_joints", type=str, default="", help="через запятую: pelvis_1,hip_r,... (пусто = авто-схема)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)   # локальный BS на одну реплику
    parser.add_argument("--out_dir", type=str, default="outputs", help="куда сохранять модель/метрики")
    parser.add_argument("--max_len", type=str, default="auto", help="'auto' (95-перцентиль по train) или число для NN")
    parser.add_argument("--loader_workers", type=int, default=0, help="(для классики/пре-вычислений)")
    parser.add_argument("--downsample", type=int, default=1, help="шаг по времени для ускорения (>=1)")
    # ускорение/мульти-GPU
    parser.add_argument("--gpus", type=str, default="all", help="all | cpu | список индексов через запятую, напр. '0,1'")
    parser.add_argument("--mixed_precision", action="store_true", help="float16 на GPU (T4/Ampere и новее)")
    parser.add_argument("--xla", action="store_true", help="включить XLA JIT")
    args = parser.parse_args()

    # --- Настроим видимость GPU ДО импорта TensorFlow ---
    if args.gpus.lower() == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif args.gpus.lower() != "all":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    # joints schema
    use_joints = None
    if args.use_joints.strip():
        use_joints = [s.strip() for s in args.use_joints.split(",") if s.strip()]
    schema_joints = None
    if args.input_format == "npy":
        if args.schema_json and os.path.exists(args.schema_json):
            with open(args.schema_json, "r", encoding="utf-8") as f:
                schema_joints = json.load(f)
        elif use_joints:
            schema_joints = list(use_joints)
    else:
        schema_joints = use_joints or discover_joint_schema(
            args.csv, args.data_dir, args.motion_key,
            filename_col=args.filename_col, label_col=args.label_col,
            sample_max=500, freq_threshold=0.9, use_joints=None
        )

    out_dir = os.path.join(args.out_dir, args.model); ensure_dir(out_dir)

    # ========= ВЕТКА КЛАССИЧЕСКИХ МОДЕЛЕЙ =========
    if args.model in ["rf","svm","xgb"]:
        if args.input_format == "npy":
            X_all, y_all = load_features_from_npy(
                args.csv, args.data_dir,
                filename_col=args.filename_col, label_col=args.label_col,
                downsample=args.downsample, workers=args.loader_workers
            )
        else:
            X_all, y_all = load_features_from_json(
                args.csv, args.data_dir, motion_key=args.motion_key,
                filename_col=args.filename_col, label_col=args.label_col,
                schema_joints=schema_joints, workers=args.loader_workers
            )

        if schema_joints is not None:
            with open(os.path.join(out_dir, "schema_joints.json"), "w", encoding="utf-8") as f:
                json.dump(schema_joints, f, ensure_ascii=False, indent=2)

        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X_all, y_all, test_size=0.20, random_state=42, stratify=y_all
        )
        X_train_feat, X_dev_feat, y_train, y_dev = train_test_split(
            X_train_full, y_train_full, test_size=0.125, random_state=42, stratify=y_train_full
        )

        print("\n=== Split stats (classical) ===")
        print_split_stats("TRAIN (≈70%)", y_train)
        print_split_stats("DEV   (≈10%)", y_dev)
        print_split_stats("TEST  (≈20%)", y_test)

        scaler = StandardScaler()
        X_train_feat = scaler.fit_transform(X_train_feat)
        X_dev_feat   = scaler.transform(X_dev_feat)
        X_test_feat  = scaler.transform(X_test)

        dev_metrics, test_metrics, thr, model = train_classical(
            X_train_feat, X_dev_feat, X_test_feat, y_train, y_dev, y_test, args.model, out_dir
        )

        joblib.dump({"model": model, "scaler": scaler}, os.path.join(out_dir, "model.joblib"))

    # ========= ВЕТКА НЕЙРОСЕТЕЙ (с поддержкой нескольких GPU) =========
    else:
        import contextlib
        k_tf = _lazy_tf()
        tf = k_tf["tf"]
        pad_sequences = k_tf["pad_sequences"]
        SequenceBase = tf.keras.utils.Sequence

        # опции производительности
        gpus = tf.config.list_physical_devices("GPU")
        for g in gpus:
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except Exception:
                pass

        if args.mixed_precision:
            from tensorflow.keras import mixed_precision as mp
            mp.set_global_policy("mixed_float16")

        if args.xla:
            tf.config.optimizer.set_jit(True)

        # стратегия для нескольких GPU
        strategy = tf.distribute.MirroredStrategy() if len(gpus) > 1 else None
        replicas = strategy.num_replicas_in_sync if strategy else 1
        global_bs = args.batch_size * replicas  # итоговый размер батча из генератора

        # --- Локальные датасеты-генераторы ---
        class _BaseSeq(SequenceBase):
            def __init__(self, items, max_len, batch_size, replicas, shuffle=True, downsample=1):
                super().__init__()
                self.items = list(items)
                self.max_len = int(max_len)
                self.bs = int(batch_size)             # это GLOBAL batch size
                self.replicas = max(1, int(replicas))
                self.shuffle = bool(shuffle)
                self.downsample = max(1, int(downsample))
                self.indices = np.arange(len(self.items))
                self.feature_dim = None  # заполним при первом удачном чтении
                self.on_epoch_end()

            def __len__(self):
                # важнo: отбрасываем хвост, чтобы всегда возвращать полный батч
                return len(self.items) // self.bs

            def on_epoch_end(self):
                if self.shuffle:
                    np.random.shuffle(self.indices)

            def _finalize_batch(self, seq_list, y_list):
                # если ничего не загрузили — сгенерируем «пустышки»
                if not seq_list:
                    if self.feature_dim is None:
                        # худший случай — пока не знаем размер, примем 3
                        self.feature_dim = 3
                    dummy = np.zeros((min(self.max_len, 2), self.feature_dim), dtype=np.float32)
                    seq_list = [dummy]

                # добить до размера батча (и кратности репликам) дубликатами последнего
                need = self.bs - len(seq_list)
                if need > 0:
                    last = seq_list[-1]; last_y = y_list[-1] if y_list else 0
                    for _ in range(need):
                        seq_list.append(last)
                        y_list.append(last_y)

                # выравнивание на случай редких несостыковок
                extra = (-len(seq_list)) % self.replicas
                for _ in range(extra):
                    seq_list.append(seq_list[-1]); y_list.append(y_list[-1])

                Xp = pad_sequences(seq_list, maxlen=self.max_len, dtype='float32',
                                   padding='post', truncating='post')
                y  = np.array(y_list[:len(Xp)], dtype=np.int32)
                return Xp, y

        class JsonSeq(_BaseSeq):
            def __init__(self, items, motion_key, schema_joints, max_len, batch_size, replicas,
                         shuffle=True, downsample=1):
                super().__init__(items, max_len, batch_size, replicas, shuffle, downsample)
                self.motion_key = motion_key
                self.schema_joints = schema_joints

            def __getitem__(self, idx):
                sl = self.indices[idx*self.bs:(idx+1)*self.bs]
                X_batch, y_batch = [], []
                for i in sl:
                    p, y = self.items[i]
                    try:
                        data = _safe_json_load(p)
                        if self.motion_key in data and isinstance(data[self.motion_key], dict):
                            md = data[self.motion_key]
                            seq = _stack_motion_frames_with_schema(md, self.schema_joints)
                            if seq is None or seq.shape[0] < 2: 
                                continue
                            if self.downsample > 1:
                                seq = seq[::self.downsample]
                            seq = _sanitize_seq(seq)
                            if self.feature_dim is None:
                                self.feature_dim = seq.shape[1]
                            X_batch.append(seq); y_batch.append(y)
                    except Exception:
                        continue
                return self._finalize_batch(X_batch, y_batch)

        class NpySeq(_BaseSeq):
            def __getitem__(self, idx):
                sl = self.indices[idx*self.bs:(idx+1)*self.bs]
                X_batch, y_batch = [], []
                for i in sl:
                    p, y = self.items[i]
                    try:
                        arr = np.load(p, allow_pickle=False, mmap_mode="r")
                        if arr.ndim != 2 or arr.shape[0] < 2:
                            continue
                        seq = arr[::self.downsample] if self.downsample > 1 else arr
                        seq = _sanitize_seq(np.asarray(seq))
                        if self.feature_dim is None:
                            self.feature_dim = seq.shape[1]
                        X_batch.append(seq); y_batch.append(y)
                    except Exception:
                        continue
                return self._finalize_batch(X_batch, y_batch)

        # Построим индекс
        items = _build_index(args.csv, args.data_dir, args.filename_col, args.label_col, input_format=args.input_format)
        assert items, "Нет валидных файлов/меток."

        # Оценим max_len
        if args.max_len.strip().lower() == "auto":
            if args.input_format == "npy":
                max_len = _probe_max_len_npy(items, pctl=95, sample_max=None)
            else:
                max_len = _probe_max_len_json(items, args.motion_key, schema_joints, pctl=95, sample_max=None)
            if not max_len or max_len <= 1:
                raise ValueError("Не удалось оценить max_len. Укажи --max_len вручную.")
        else:
            max_len = int(args.max_len)
        print("max_len (train) =", max_len)

        files = np.array([p for p, _ in items], dtype=object)
        labels = np.array([y for _, y in items], dtype=np.int32)

        idx_train_full, idx_test = train_test_split(
            np.arange(len(items)), test_size=0.20, random_state=42, stratify=labels
        )
        idx_train, idx_dev = train_test_split(
            idx_train_full, test_size=0.125, random_state=42, stratify=labels[idx_train_full]
        )

        print("\n=== Split stats (deep) ===")
        print_split_stats("TRAIN (≈70%)", labels[idx_train])
        print_split_stats("DEV   (≈10%)", labels[idx_dev])
        print_split_stats("TEST  (≈20%)", labels[idx_test])

        if args.input_format == "npy":
            train_seq = NpySeq([items[i] for i in idx_train], max_len, global_bs, replicas, shuffle=True,  downsample=args.downsample)
            dev_seq   = NpySeq([items[i] for i in idx_dev],   max_len, global_bs, replicas, shuffle=False, downsample=args.downsample)
            test_seq  = NpySeq([items[i] for i in idx_test],  max_len, global_bs, replicas, shuffle=False, downsample=args.downsample)
        else:
            train_seq = JsonSeq([items[i] for i in idx_train], args.motion_key, schema_joints, max_len, global_bs, replicas, shuffle=True,  downsample=args.downsample)
            dev_seq   = JsonSeq([items[i] for i in idx_dev],   args.motion_key, schema_joints, max_len, global_bs, replicas, shuffle=False, downsample=args.downsample)
            test_seq  = JsonSeq([items[i] for i in idx_test],  args.motion_key, schema_joints, max_len, global_bs, replicas, shuffle=False, downsample=args.downsample)

        # пример батча -> input_shape
        X_sample, _ = train_seq[0]
        if X_sample.shape[0] == 0:
            raise RuntimeError("Пустой первый батч. Проверь данные.")
        input_shape = (X_sample.shape[1], X_sample.shape[2])

        # веса классов по train
        classes = np.array([0, 1], dtype=np.int32)
        w = compute_class_weight(class_weight="balanced", classes=classes, y=labels[idx_train])
        class_weight = {0: float(w[0]), 1: float(w[1])}
        print("Class weights (train):", class_weight)
        print(f"GPUs: {len(gpus)} | replicas_in_sync: {replicas} | global_batch: {global_bs}")

        # строим модель под стратегией и обучаем
        ctx = strategy.scope() if strategy else contextlib.nullcontext()
        with ctx:
            if   args.model == "lstm":        model = build_lstm(input_shape)
            elif args.model == "tcn":         model = build_tcn(input_shape)
            elif args.model == "gru":         model = build_gru(input_shape)
            elif args.model == "cnn_lstm":    model = build_cnn_lstm(input_shape)
            elif args.model == "transformer": model = build_transformer(input_shape)
            elif args.model == "timesnet":    model = build_timesnet(input_shape)
            elif args.model == "patchtst":    model = build_patchtst(input_shape)
            elif args.model == "informer":    model = build_informer(input_shape)
            else: raise ValueError("Unknown deep model")

        cb = [k_tf["EarlyStopping"](monitor="val_loss", patience=5, restore_best_weights=True)]
        # ВАЖНО: без workers/use_multiprocessing — совместимо с Keras 3
        model.fit(
            train_seq,
            validation_data=dev_seq,
            epochs=args.epochs,
            class_weight=class_weight,
            callbacks=cb,
            verbose=1,
        )

        # DEV
        prob_dev, y_dev = [], []
        for Xb, yb in dev_seq:
            if len(Xb) == 0: continue
            pb = model.predict(Xb, verbose=0).flatten().astype(np.float32)
            prob_dev.append(pb); y_dev.append(yb)
        prob_dev = np.concatenate(prob_dev); y_dev = np.concatenate(y_dev)
        thr = best_threshold_by_f1(y_dev, prob_dev)
        pred_dev = (prob_dev >= thr).astype(int)
        dev_metrics = compute_metrics(y_dev, pred_dev, prob_dev)

        # TEST
        prob_test, y_test = [], []
        for Xb, yb in test_seq:
            if len(Xb) == 0: continue
            pb = model.predict(Xb, verbose=0).flatten().astype(np.float32)
            prob_test.append(pb); y_test.append(yb)
        prob_test = np.concatenate(prob_test); y_test = np.concatenate(y_test)
        pred_test = (prob_test >= thr).astype(int)
        test_metrics = compute_metrics(y_test, pred_test, prob_test)

        model.save(os.path.join(out_dir, "model.keras"))  # формат Keras 3
        with open(os.path.join(out_dir, "threshold.txt"), "w") as f:
            f.write(str(thr))
        np.savez(os.path.join(out_dir, "norm_stats.npz"), max_len=input_shape[0])

        if schema_joints is not None:
            with open(os.path.join(out_dir, "schema_joints.json"), "w", encoding="utf-8") as f:
                json.dump(schema_joints, f, ensure_ascii=False, indent=2)

    # ===== Сохранение отчётов =====
    print("\n=== DEV METRICS (threshold tuned here) ===")
    for k in ["accuracy", "f1", "roc_auc", "confusion_matrix"]:
        print(f"{k}: {dev_metrics[k]}")
    print("\nDev classification report:\n", dev_metrics["report"])

    print("\n=== TEST METRICS (using dev-tuned threshold) ===")
    for k in ["accuracy", "f1", "roc_auc", "confusion_matrix"]:
        print(f"{k}: {test_metrics[k]}")
    print("\nTest classification report:\n", test_metrics["report"])

    with open(os.path.join(out_dir, "metrics_dev.json"), "w", encoding="utf-8") as f:
        json.dump(dev_metrics, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "metrics_test.json"), "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to: {out_dir}")
