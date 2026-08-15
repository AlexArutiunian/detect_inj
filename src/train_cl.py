#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ===================== detect_inj/train3.py =====================
# Классика (RF/SVM/XGB) и нейросети (LSTM/TCN/и т.п.) для бинарной классификации.
# Визуализации: confusion matrix, ROC, PR, F1(threshold), feature importance (+ HTML отчёт).

import os
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Optional

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.utils.class_weight import compute_class_weight
import joblib

# ---- Matplotlib (без GUI) ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    pr, rc, th = precision_recall_curve(y_true, proba)
    f1_vals = 2 * pr[:-1] * rc[:-1] / np.clip(pr[:-1] + rc[:-1], 1e-12, None)
    if len(f1_vals) == 0 or np.all(np.isnan(f1_vals)): return 0.5
    best_idx = int(np.nanargmax(f1_vals))
    return float(th[max(0, min(best_idx, len(th) - 1))])

def print_split_stats(name, y):
    n = len(y); n1 = int(np.sum(y == 1)); n0 = n - n1
    p1 = (n1 / n * 100) if n else 0.0; p0 = (n0 / n * 100) if n else 0.0
    print(f"[{name}] total={n} | Injury=1: {n1} ({p1:.1f}%) | No Injury=0: {n0} ({p0:.1f}%)")

# ---------- Универсальный нормализатор формы ----------
def sanitize_seq(a: np.ndarray) -> np.ndarray:
    """
    Приводит массив к (T, F):
    - NaN/Inf -> 0
    - выбирает самую длинную ось как время и переносит её на 0
    - остальные оси сплющивает в признаки
    """
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

    if a.ndim == 1:
        a = a[:, None]  # (T,) -> (T,1)
    elif a.ndim >= 2:
        t_axis = int(np.argmax(a.shape))  # самая длинная ось — время
        if t_axis != 0:
            a = np.moveaxis(a, t_axis, 0)  # время -> ось 0
        if a.ndim > 2:
            a = a.reshape(a.shape[0], -1)  # (T, ..., ...) -> (T, F)

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
    """seq_np: (T, F=3*|schema| или просто F) -> агрегированные статистики для классики."""
    seq = seq_np.astype(np.float32, copy=False)
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

def possible_npy_paths(data_dir: str, rel: str) -> List[str]:
    """Генерирует разумные варианты путей (.npy, .json.npy и базовые имена)."""
    rel = (rel or "").strip().replace("\\", "/").lstrip("/")
    cands: List[str] = []

    def push(x: str):
        if x and x not in cands:
            cands.append(x)

    # как есть
    push(os.path.join(data_dir, rel))
    if not rel.endswith(".npy"):
        push(os.path.join(data_dir, rel + ".npy"))

    if rel.endswith(".json"):
        base_nojson = rel[:-5]
        push(os.path.join(data_dir, base_nojson + ".npy"))
        push(os.path.join(data_dir, rel + ".npy"))  # .json.npy

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

def _feats_from_npy_task(args):
    npy_path, y, downsample = args
    try:
        arr = np.load(npy_path, allow_pickle=False, mmap_mode="r")
        x = sanitize_seq(arr)                       # приведение к (T,F)
        if x.shape[0] < 2:
            return None
        if downsample > 1:
            x = x[::downsample]
        return (_features_from_seq(np.asarray(x)), int(y))
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
        p = pick_existing_path(possible_npy_paths(npy_dir, fn))
        if p:
            tasks.append((p, y, int(max(1, downsample))))

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
            p = pick_existing_path(possible_npy_paths(data_dir_or_npy_dir, fn))
        if p and os.path.exists(p):
            items.append((p, int(y)))
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
            x = sanitize_seq(arr)                  # привести к (T,F)
            if x.ndim == 2 and x.shape[0] > 1:
                L.append(int(x.shape[0]))
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

def build_lstm(input_shape):
    k = _lazy_tf()
    model = k["Sequential"]([
        k["Masking"](mask_value=0.0, input_shape=input_shape),
        k["LSTM"](128, return_sequences=False),
        k["Dense"](64, activation='relu'),
        k["Dropout"](0.3),
        k["Dense"](1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

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
    model.add(k["Dense"](1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_cnn_lstm(input_shape):
    k = _lazy_tf()
    from tensorflow.keras import layers, models
    inp = layers.Input(shape=input_shape)
    x = layers.Masking(mask_value=0.0)(inp)
    x = layers.Conv1D(128, 5, padding="same", activation="relu")(x); x = layers.BatchNormalization()(x); x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 3, padding="same", activation="relu")(x); x = layers.BatchNormalization()(x); x = layers.MaxPooling1D(2)(x)
    x = layers.LSTM(128)(x); x = layers.Dropout(0.3)(x); x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inp, out)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def build_gru(input_shape):
    k = _lazy_tf()
    from tensorflow.keras import layers, models
    inp = layers.Input(shape=input_shape)
    x = layers.Masking(mask_value=0.0)(inp)
    x = layers.GRU(128, return_sequences=True)(x)
    x = layers.GRU(64)(x)
    x = layers.Dropout(0.3)(x); x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inp, out)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def build_transformer(input_shape, d_model=128, num_heads=4, ff_dim=256, num_layers=2, dropout=0.1):
    _ = _lazy_tf()  # гарантируем наличие TF
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
    out = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inp, out)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

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
    out = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inp, out); model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

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
    out = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inp, out); model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

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
    out = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inp, out); model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# ===================== ОБУЧЕНИЕ =====================

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

def train_deep(name, train_seq, dev_seq, test_seq, class_weight, epochs, out_dir, input_shape):
    k = _lazy_tf()
    if   name == "lstm":        model = build_lstm(input_shape)
    elif name == "tcn":         model = build_tcn(input_shape)
    elif name == "gru":         model = build_gru(input_shape)
    elif name == "cnn_lstm":    model = build_cnn_lstm(input_shape)
    elif name == "transformer": model = build_transformer(input_shape)
    elif name == "timesnet":    model = build_timesnet(input_shape)
    elif name == "patchtst":    model = build_patchtst(input_shape)
    elif name == "informer":    model = build_informer(input_shape)
    else: raise ValueError("Unknown deep model")

    cb = [k["EarlyStopping"](monitor="val_loss", patience=5, restore_best_weights=True)]
    model.fit(train_seq, validation_data=dev_seq, epochs=epochs,
              class_weight=class_weight, callbacks=cb, verbose=1)

    # DEV
    prob_dev, y_dev = [], []
    for Xb, yb in dev_seq:
        if len(Xb) == 0: continue
        pb = model.predict(Xb, verbose=0).flatten()
        prob_dev.append(pb); y_dev.append(yb)
    prob_dev = np.concatenate(prob_dev); y_dev = np.concatenate(y_dev)
    thr = best_threshold_by_f1(y_dev, prob_dev)
    pred_dev = (prob_dev >= thr).astype(int)
    dev_metrics = compute_metrics(y_dev, pred_dev, prob_dev)

    # TEST
    prob_test, y_test = [], []
    for Xb, yb in test_seq:
        if len(Xb) == 0: continue
        pb = model.predict(Xb, verbose=0).flatten()
        prob_test.append(pb); y_test.append(yb)
    prob_test = np.concatenate(prob_test); y_test = np.concatenate(y_test)
    pred_test = (prob_test >= thr).astype(int)
    test_metrics = compute_metrics(y_test, pred_test, prob_test)

    model.save(os.path.join(out_dir, "model.h5"))
    with open(os.path.join(out_dir, "threshold.txt"), "w") as f:
        f.write(str(thr))
    np.savez(os.path.join(out_dir, "norm_stats.npz"), max_len=input_shape[0])
    return dev_metrics, test_metrics, thr, model

# ===================== VIZ / REPORT =====================

CLASS_NAMES = ["No Injury (0)", "Injury (1)"]

def _plot_confusion(cm, title, path):
    cm = np.asarray(cm, dtype=np.int32)
    cmn = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cmn, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xlabel('Predicted label'); ax.set_ylabel('True label')
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(CLASS_NAMES, rotation=15, ha='right'); ax.set_yticklabels(CLASS_NAMES)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]}\n{cmn[i,j]:.2f}", ha="center", va="center", fontsize=10,
                    color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

def _plot_roc(y_true, y_prob, title, path):
    if len(np.unique(y_true)) < 2:
        return None
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5,4))
    ax.plot(fpr, tpr, label=f"AUC = {auc_val:.3f}")
    ax.plot([0,1],[0,1], linestyle='--')
    ax.set_xlim([0,1]); ax.set_ylim([0,1])
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate"); ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return auc_val

def _plot_pr(y_true, y_prob, title, path):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5,4))
    ax.plot(recall, precision, label=f"AP = {ap:.3f}")
    ax.set_xlim([0,1]); ax.set_ylim([0,1])
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title(title)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return ap

def _plot_f1_threshold(y_true, y_prob, best_thr, title, path):
    pr, rc, th = precision_recall_curve(y_true, y_prob)
    f1_vals = 2 * pr[:-1] * rc[:-1] / np.clip(pr[:-1] + rc[:-1], 1e-12, None)
    th = np.asarray(th)
    fig, ax = plt.subplots(figsize=(5.5,4))
    ax.plot(th, f1_vals)
    ax.axvline(best_thr, linestyle='--', label=f"best thr = {best_thr:.3f}")
    ax.set_xlabel("Threshold"); ax.set_ylabel("F1"); ax.set_title(title); ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

def _plot_feature_importance(model, feature_names, topn, title, path):
    if not hasattr(model, "feature_importances_"):
        return False
    imp = np.asarray(model.feature_importances_, dtype=float)
    idx = np.argsort(-imp)[:topn]
    names = [feature_names[i] for i in idx]
    vals = imp[idx]
    fig, ax = plt.subplots(figsize=(7, max(3, 0.25*len(idx)+1)))
    ax.barh(range(len(idx)), vals)
    ax.set_yticks(range(len(idx))); ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Importance"); ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True

# имена фич для классики (если известна схема)
_STAT_NAMES = ["mean","std","min","max","dmean","dstd"]
_AXIS = ["x","y","z"]
def classical_feature_names(schema_joints, d_dim):
    if schema_joints:
        names = []
        for j in schema_joints:
            for ax in _AXIS:
                for st in _STAT_NAMES:
                    names.append(f"{j}:{ax}:{st}")
        return names
    # фолбэк, если схемы нет
    return [f"f_{i}" for i in range(d_dim)]

def _write_html_report(out_dir, dev_metrics, test_metrics, images, thr):
    html = []
    html.append("<html><head><meta charset='utf-8'><title>Training Report</title></head><body>")
    html.append("<h1>Training report</h1>")
    html.append(f"<p><b>Best threshold (from DEV)</b>: {thr:.6f}</p>")
    def sec(name, m):
        html.append(f"<h2>{name}</h2>")
        html.append("<ul>")
        html.append(f"<li>accuracy: {m['accuracy']:.4f}</li>")
        html.append(f"<li>f1: {m['f1']:.4f}</li>")
        html.append(f"<li>roc_auc: {m['roc_auc']:.4f}</li>")
        html.append("</ul>")
    sec("DEV metrics", dev_metrics); sec("TEST metrics", test_metrics)
    def img(title, key):
        if key in images:
            html.append(f"<h3>{title}</h3><img src='{os.path.basename(images[key])}' style='max-width:700px'>")
    img("DEV: Confusion Matrix", "cm_dev")
    img("DEV: ROC", "roc_dev"); img("DEV: Precision-Recall", "pr_dev"); img("DEV: F1 vs threshold", "f1_dev")
    img("TEST: Confusion Matrix", "cm_test")
    img("TEST: ROC", "roc_test"); img("TEST: Precision-Recall", "pr_test")
    img("Feature Importance (top)", "fi")
    html.append("</body></html>")
    with open(os.path.join(out_dir, "report.html"), "w", encoding="utf-8") as f:
        f.write("\n".join(html))


def _display_inline(images: dict, show: bool):
    """Показать картинки внутри ячейки (работает при запуске через %run)."""
    if not show:
        return
    try:
        from IPython.display import display, HTML, Image as IPyImage
        sections = [
            ("DEV",  ["cm_dev", "roc_dev", "pr_dev", "f1_dev"]),
            ("TEST", ["cm_test", "roc_test", "pr_test"]),
            ("FEATURES", ["fi"]),
        ]
        for title, keys in sections:
            have = [k for k in keys if k in images]
            if not have:
                continue
            display(HTML(f"<h3>{title}</h3>"))
            for k in have:
                display(IPyImage(filename=images[k]))
    except Exception as e:
        print(f"[inline-display disabled] {e}")

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
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--out_dir", type=str, default="outputs", help="куда сохранять модель/метрики")
    parser.add_argument("--max_len", type=str, default="auto", help="'auto' (95-перцентиль по train) или число для NN")
    parser.add_argument("--loader_workers", type=int, default=0, help="воркеры для извлечения фич / датасетов")
    parser.add_argument("--downsample", type=int, default=1, help="шаг по времени для ускорения (>=1)")
    parser.add_argument("--top_features", type=int, default=30, help="сколько лучших фич показывать на графике важности (RF/XGB)")
    parser.add_argument("--show_plots", action="store_true",
                    help="показывать графики прямо в ячейке (используй %run)")

    args = parser.parse_args()

    # joints schema
    use_joints = None
    if args.use_joints.strip():
        use_joints = [s.strip() for s in args.use_joints.split(",") if s.strip()]
    schema_joints = None
    if args.input_format == "npy":
        # имена суставов нужны только для отчётов/совместимости
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

        # сохраним схему, если есть
        if schema_joints is not None:
            with open(os.path.join(out_dir, "schema_joints.json"), "w", encoding="utf-8") as f:
                json.dump(schema_joints, f, ensure_ascii=False, indent=2)

        # Сплит 70/20/10
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

        # ---- Визуализации (классика) ----
        prob_dev  = model.predict_proba(X_dev_feat)[:,1]
        prob_test = model.predict_proba(X_test_feat)[:,1]
        pred_dev  = (prob_dev  >= thr).astype(int)
        pred_test = (prob_test >= thr).astype(int)

        images = {}
        # Confusion matrices
        _plot_confusion(np.array(dev_metrics["confusion_matrix"]),  "DEV Confusion Matrix",  os.path.join(out_dir, "cm_dev.png"));  images["cm_dev"]  = os.path.join(out_dir, "cm_dev.png")
        _plot_confusion(np.array(test_metrics["confusion_matrix"]), "TEST Confusion Matrix", os.path.join(out_dir, "cm_test.png")); images["cm_test"] = os.path.join(out_dir, "cm_test.png")
        # ROC/PR/F1
        if len(np.unique(y_dev)) == 2:
            _plot_roc(y_dev, prob_dev,  "DEV ROC",  os.path.join(out_dir, "roc_dev.png"));  images["roc_dev"]  = os.path.join(out_dir, "roc_dev.png")
            _plot_pr(y_dev,  prob_dev,  "DEV PR",   os.path.join(out_dir, "pr_dev.png"));   images["pr_dev"]   = os.path.join(out_dir, "pr_dev.png")
            _plot_f1_threshold(y_dev, prob_dev, thr, "DEV F1 vs threshold", os.path.join(out_dir, "f1_dev.png")); images["f1_dev"] = os.path.join(out_dir, "f1_dev.png")
        if len(np.unique(y_test)) == 2:
            _plot_roc(y_test, prob_test, "TEST ROC", os.path.join(out_dir, "roc_test.png")); images["roc_test"] = os.path.join(out_dir, "roc_test.png")
            _plot_pr(y_test,  prob_test, "TEST PR",  os.path.join(out_dir, "pr_test.png"));  images["pr_test"]  = os.path.join(out_dir, "pr_test.png")

        # Feature importance (RF/XGB)
        feat_names = classical_feature_names(schema_joints, X_train_feat.shape[1])
        if _plot_feature_importance(model, feat_names, args.top_features, "Feature Importance (top)", os.path.join(out_dir, "feature_importance_top.png")):
            images["fi"] = os.path.join(out_dir, "feature_importance_top.png")

        _write_html_report(out_dir, dev_metrics, test_metrics, images, thr)
        
        _display_inline(images, args.show_plots)


    # ========= ВЕТКА НЕЙРОСЕТЕЙ =========
    else:
        # Локально определяем Keras Sequence на базе ленивого TF
        k_tf = _lazy_tf()
        tf = k_tf["tf"]
        pad_sequences = k_tf["pad_sequences"]
        SequenceBase = tf.keras.utils.Sequence

        class JsonSeq(SequenceBase):
            def __init__(self, items, motion_key, schema_joints, max_len, batch_size,
                         shuffle=True, downsample=1, **kwargs):
                super().__init__(**kwargs)
                self.items = list(items)
                self.motion_key = motion_key
                self.schema_joints = schema_joints
                self.max_len = int(max_len)
                self.bs = int(batch_size)
                self.shuffle = bool(shuffle)
                self.downsample = max(1, int(downsample))
                self.indices = np.arange(len(self.items))
                self.on_epoch_end()
            def __len__(self): return (len(self.items) + self.bs - 1) // self.bs
            def on_epoch_end(self):
                if self.shuffle: np.random.shuffle(self.indices)
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
                            if seq is None: continue
                            if self.downsample > 1: seq = seq[::self.downsample]
                            seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
                            X_batch.append(seq); y_batch.append(y)
                    except Exception:
                        continue
                if not X_batch:
                    return np.zeros((0, self.max_len, 1), dtype=np.float32), np.zeros((0,), dtype=np.int32)
                Xp = pad_sequences(X_batch, maxlen=self.max_len, dtype='float32', padding='post', truncating='post')
                return Xp, np.array(y_batch, dtype=np.int32)

        class NpySeq(SequenceBase):
            def __init__(self, items, max_len, batch_size, shuffle=True, downsample=1, **kwargs):
                super().__init__(**kwargs)
                self.items = list(items)
                self.max_len = int(max_len)
                self.bs = int(batch_size)
                self.shuffle = bool(shuffle)
                self.downsample = max(1, int(downsample))
                self.indices = np.arange(len(self.items))
                self.on_epoch_end()
            def __len__(self): return (len(self.items) + self.bs - 1) // self.bs
            def on_epoch_end(self):
                if self.shuffle: np.random.shuffle(self.indices)
            def __getitem__(self, idx):
                sl = self.indices[idx*self.bs:(idx+1)*self.bs]
                X_batch, y_batch = [], []
                for i in sl:
                    p, y = self.items[i]
                    try:
                        arr = np.load(p, allow_pickle=False, mmap_mode="r")
                        x = sanitize_seq(arr)                 # (T,F)
                        if x.shape[0] < 2:
                            continue
                        if self.downsample > 1:
                            x = x[::self.downsample]
                        X_batch.append(x.astype(np.float32))
                        y_batch.append(y)
                    except Exception:
                        continue
                if not X_batch:
                    return np.zeros((0, self.max_len, 1), dtype=np.float32), np.zeros((0,), dtype=np.int32)
                Xp = pad_sequences(X_batch, maxlen=self.max_len, dtype='float32', padding='post', truncating='post')
                return Xp, np.array(y_batch, dtype=np.int32)

        # Построим индекс
        items = _build_index(args.csv, args.data_dir, args.filename_col, args.label_col, input_format=args.input_format)
        assert items, "Нет валидных файлов/меток."

        # оценим max_len
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
            train_seq = NpySeq([items[i] for i in idx_train], max_len, args.batch_size, shuffle=True,  downsample=args.downsample)
            dev_seq   = NpySeq([items[i] for i in idx_dev],   max_len, args.batch_size, shuffle=False, downsample=args.downsample)
            test_seq  = NpySeq([items[i] for i in idx_test],  max_len, args.batch_size, shuffle=False, downsample=args.downsample)
        else:
            train_seq = JsonSeq([items[i] for i in idx_train], args.motion_key, schema_joints, max_len, args.batch_size, shuffle=True,  downsample=args.downsample)
            dev_seq   = JsonSeq([items[i] for i in idx_dev],   args.motion_key, schema_joints, max_len, args.batch_size, shuffle=False, downsample=args.downsample)
            test_seq  = JsonSeq([items[i] for i in idx_test],  args.motion_key, schema_joints, max_len, args.batch_size, shuffle=False, downsample=args.downsample)

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

        dev_metrics, test_metrics, thr, model = train_deep(
            args.model, train_seq, dev_seq, test_seq, class_weight, args.epochs, out_dir, input_shape
        )

        # ---- Визуализации (нейросеть) ----
        def _collect(seq):
            ys, ps = [], []
            for Xb, yb in seq:
                if len(Xb) == 0: continue
                pb = model.predict(Xb, verbose=0).flatten()
                ys.append(yb); ps.append(pb)
            return (np.concatenate(ys), np.concatenate(ps)) if ys else (np.array([]), np.array([]))

        y_dev,  prob_dev  = _collect(dev_seq)
        y_test, prob_test = _collect(test_seq)

        images = {}
        _plot_confusion(np.array(dev_metrics["confusion_matrix"]),  "DEV Confusion Matrix",  os.path.join(out_dir, "cm_dev.png"));  images["cm_dev"]  = os.path.join(out_dir, "cm_dev.png")
        _plot_confusion(np.array(test_metrics["confusion_matrix"]), "TEST Confusion Matrix", os.path.join(out_dir, "cm_test.png")); images["cm_test"] = os.path.join(out_dir, "cm_test.png")
        if y_dev.size:
            _plot_roc(y_dev, prob_dev,  "DEV ROC",  os.path.join(out_dir, "roc_dev.png"));  images["roc_dev"]  = os.path.join(out_dir, "roc_dev.png")
            _plot_pr(y_dev,  prob_dev,  "DEV PR",   os.path.join(out_dir, "pr_dev.png"));   images["pr_dev"]   = os.path.join(out_dir, "pr_dev.png")
            _plot_f1_threshold(y_dev, prob_dev, thr, "DEV F1 vs threshold", os.path.join(out_dir, "f1_dev.png")); images["f1_dev"] = os.path.join(out_dir, "f1_dev.png")
        if y_test.size:
            _plot_roc(y_test, prob_test, "TEST ROC", os.path.join(out_dir, "roc_test.png")); images["roc_test"] = os.path.join(out_dir, "roc_test.png")
            _plot_pr(y_test,  prob_test, "TEST PR",  os.path.join(out_dir, "pr_test.png"));  images["pr_test"]  = os.path.join(out_dir, "pr_test.png")

        _write_html_report(out_dir, dev_metrics, test_metrics, images, thr)
        _display_inline(images, args.show_plots)

        # сохраним схему для predict (если есть)
        if schema_joints is not None:
            with open(os.path.join(out_dir, "schema_joints.json"), "w", encoding="utf-8") as f:
                json.dump(schema_joints, f, ensure_ascii=False, indent=2)

    # ===== Сохранение отчётов (текст/JSON) =====
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
    print(f"Report: {os.path.join(out_dir, 'report.html')}")
