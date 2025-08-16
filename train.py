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
    classification_report, confusion_matrix, precision_recall_curve
)
from sklearn.utils.class_weight import compute_class_weight
import joblib

from sklearn.inspection import permutation_importance

# Имена фич для классики: 3 оси * 6 статистик на каждый сустав
_STAT_NAMES = ["mean","std","min","max","dmean","dstd"]
_AXIS = ["x","y","z"]

def classical_feature_names(schema_joints):
    names = []
    for j in schema_joints:
        for ax in _AXIS:
            for st in _STAT_NAMES:
                names.append(f"{j}:{ax}:{st}")
    return names  # порядок совпадает с features_from_sequence()

def aggregate_importance_by_joint(feat_names, importances):
    by_joint = {}
    by_joint_axis = {}
    for nm, imp in zip(feat_names, importances):
        j, ax, st = nm.split(":")
        by_joint[j] = by_joint.get(j, 0.0) + float(imp)
        key = (j, ax)
        by_joint_axis[key] = by_joint_axis.get(key, 0.0) + float(imp)
    return by_joint, by_joint_axis

def save_importance_tables(out_dir, all_rows, by_joint, by_joint_axis, prefix):
    os.makedirs(out_dir, exist_ok=True)
    # детально по фичам
    df_all = pd.DataFrame(all_rows).sort_values("importance", ascending=False)
    df_all.to_csv(os.path.join(out_dir, f"{prefix}_feature_importance.csv"), index=False)
    # по суставам
    df_joint = pd.DataFrame(
        [{"joint": j, "importance": v} for j, v in by_joint.items()]
    ).sort_values("importance", ascending=False)
    df_joint.to_csv(os.path.join(out_dir, f"{prefix}_joint_importance.csv"), index=False)
    # по (сустав,ось)
    df_ja = pd.DataFrame(
        [{"joint": j, "axis": ax, "importance": v} for (j, ax), v in by_joint_axis.items()]
    ).sort_values("importance", ascending=False)
    df_ja.to_csv(os.path.join(out_dir, f"{prefix}_joint_axis_importance.csv"), index=False)


# ===================== Метки =====================

def label_to_int(v):
    """Ровно две метки: 'Injury' -> 1, 'No Injury' -> 0."""
    if v is None:
        return None
    s = str(v).strip().lower()
    if s == "injury":
        return 1
    if s == "no injury":
        return 0
    return None


# ===================== Метрики/утилиты =====================

def compute_metrics(y_true, y_pred, y_prob):
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else float("nan"),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "report": classification_report(y_true, y_pred, digits=3)
    }
    return out

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def best_threshold_by_f1(y_true, proba):
    """Подбор порога по максимуму F1 на dev."""
    pr, rc, th = precision_recall_curve(y_true, proba)  # n+1, n+1, n
    f1_vals = 2 * pr[:-1] * rc[:-1] / np.clip(pr[:-1] + rc[:-1], 1e-12, None)
    if len(f1_vals) == 0 or np.all(np.isnan(f1_vals)):
        return 0.5
    best_idx = int(np.nanargmax(f1_vals))
    return float(th[max(0, min(best_idx, len(th) - 1))])


# ===================== Низкоуровневые загрузчики =====================

def _safe_json_load(path):
    """Быстрый json.load: пробуем orjson (если установлен)."""
    try:
        import orjson
        with open(path, "rb") as f:
            return orjson.loads(f.read())
    except Exception:
        with open(path, "r") as f:
            return json.load(f)

def _stack_motion_frames_with_schema(md, schema_joints):
    """
    Из dict суставов -> (T, 3*|schema|) с заполнением отсутствующих суставов NaN.
    T = min длина по присутствующим суставам; если ни одного — None.
    """
    present = [j for j in schema_joints if j in md]
    if not present:
        return None
    T = min(len(md[j]) for j in present)
    if T <= 1:
        return None
    cols = []
    for j in schema_joints:
        if j in md:
            arr = np.asarray(md[j], dtype=np.float32)[:T]   # (T,3)
        else:
            arr = np.full((T, 3), np.nan, dtype=np.float32) # отсутствующий сустав
        cols.append(arr)
    X = np.concatenate(cols, axis=1)  # (T, 3*|schema|)
    return X


# ===================== Автодетект схемы суставов =====================

def discover_joint_schema(csv_path, data_dir, motion_key,
                          filename_col="filename", label_col="No inj/ inj",
                          sample_max=500, freq_threshold=0.9, use_joints=None):
    """
    Возвращает упорядоченный список суставов (schema_joints).
    - если задан use_joints -> возвращаем его как есть;
    - иначе сканируем до sample_max файлов и берём суставы, встречающиеся в >= freq_threshold доле файлов.
    """
    if use_joints:
        return list(use_joints)

    meta = pd.read_csv(csv_path, usecols=[filename_col, label_col])
    counts = {}
    seen = 0
    for _, row in meta.iterrows():
        if seen >= sample_max:
            break
        fn = str(row[filename_col]).strip() if pd.notnull(row.get(filename_col)) else ""
        y = label_to_int(row.get(label_col))
        if not fn or y is None:
            continue
        p = os.path.join(data_dir, fn)
        if not os.path.exists(p) and not fn.endswith(".json"):
            p2 = p + ".json"
            if os.path.exists(p2):
                p = p2
        if not os.path.exists(p):
            continue
        try:
            data = _safe_json_load(p)
            if motion_key not in data or not isinstance(data[motion_key], dict):
                continue
            joints = list(data[motion_key].keys())
            for j in joints:
                counts[j] = counts.get(j, 0) + 1
            seen += 1
        except Exception:
            continue

    if seen == 0:
        raise RuntimeError("discover_joint_schema: не удалось прочитать ни одного файла.")

    freq = {j: c / seen for j, c in counts.items()}
    schema = [j for j, f in freq.items() if f >= freq_threshold]
    if not schema:
        schema = list(counts.keys())
    schema = sorted(schema)
    return schema


# ===================== Стриминг фич для классики =====================

def _feats_from_file(args):
    json_path, y, motion_key, schema_joints = args
    try:
        data = _safe_json_load(json_path)
        if motion_key not in data or not isinstance(data[motion_key], dict):
            return None
        md = data[motion_key]
        seq = _stack_motion_frames_with_schema(md, schema_joints)
        if seq is None or seq.shape[0] < 2:
            return None
        seq = seq.astype(np.float32, copy=False)
        dif = np.diff(seq, axis=0)
        stat = np.concatenate([
            np.nanmean(seq, axis=0), np.nanstd(seq, axis=0), np.nanmin(seq, axis=0), np.nanmax(seq, axis=0),
            np.nanmean(dif, axis=0), np.nanstd(dif, axis=0)
        ]).astype(np.float32, copy=False)
        stat = np.nan_to_num(stat, nan=0.0, posinf=0.0, neginf=0.0)
        return (stat, int(y))
    except Exception:
        return None

def load_features_streaming(csv_path, data_dir, motion_key="running",
                            filename_col="filename", label_col="No inj/ inj",
                            use_joints=None, workers=0):
    """
    Возвращает X (n, d), y (n,) и schema_joints — фичи для классических моделей.
    """
    schema_joints = discover_joint_schema(
        csv_path, data_dir, motion_key, filename_col, label_col,
        sample_max=500, freq_threshold=0.9, use_joints=use_joints
    )
    print(f"Feature schema joints ({len(schema_joints)}): "
          f"{', '.join(schema_joints[:8])}{' ...' if len(schema_joints) > 8 else ''}")

    meta = pd.read_csv(csv_path, usecols=[filename_col, label_col])
    tasks = []
    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Indexing", dynamic_ncols=True, mininterval=0.2):
    
        fn = str(row[filename_col]).strip() if pd.notnull(row.get(filename_col)) else ""
        y = label_to_int(row.get(label_col))
        if not fn or y is None:
            continue
        p = os.path.join(data_dir, fn)
        if not os.path.exists(p) and not fn.endswith(".json"):
            p2 = p + ".json"
            if os.path.exists(p2):
                p = p2
        if os.path.exists(p):
            tasks.append((p, y, motion_key, schema_joints))

    results = []
    if workers and workers > 0:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for r in tqdm(ex.map(_feats_from_file, tasks, chunksize=16),
                        total=len(tasks), desc="Feats", dynamic_ncols=True, mininterval=0.2):
                if r is not None:
                    results.append(r)

    else:
        for t in tqdm(tasks, desc="Feats", dynamic_ncols=True, mininterval=0.2):
            r = _feats_from_file(t)
            if r is not None:
                results.append(r)


    if not results:
        raise RuntimeError("Не удалось получить ни одной строки фич.")
    X = np.stack([r[0] for r in results]).astype(np.float32, copy=False)
    y = np.array([r[1] for r in results], dtype=np.int32)
    return X, y, schema_joints


# ===================== Ленивая инициализация TensorFlow + модели =====================

def _lazy_tf():
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import (
            LSTM, Dense, Conv1D, BatchNormalization, Dropout,
            GlobalAveragePooling1D, Masking
        )
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from tensorflow.keras.callbacks import EarlyStopping
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
            "pad_sequences": pad_sequences,
            "EarlyStopping": EarlyStopping,
        }
    except Exception as e:
        raise RuntimeError(
            "TensorFlow недоступен или несовместим с текущей средой. "
            "Для 'lstm'/'tcn' установите совместимый TF или используйте классические модели. "
            f"Ошибка: {e}"
        )

def build_lstm(input_shape):
    k = _lazy_tf()
    model = k["Sequential"]([
        k["Masking"](mask_value=0.0, input_shape=input_shape),
        k["LSTM"](128, return_sequences=True),
        k,
        k["Dense"](64, activation='relu'),
        k["Dropout"](0.3),
        k["Dense"](1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_tcn(input_shape, dilations=(1, 2, 4, 8)):
    k = _lazy_tf()
    from tensorflow.keras.layers import Conv1D  # локальный алиас

    model = k["Sequential"]()
    for i, d in enumerate(dilations):
        if i == 0:
            model.add(Conv1D(64, kernel_size=3, padding='causal',
                             dilation_rate=d, activation='relu',
                             input_shape=input_shape))
        else:
            model.add(Conv1D(64, kernel_size=3, padding='causal',
                             dilation_rate=d, activation='relu'))
        model.add(k["BatchNormalization"]())
        model.add(k["Dropout"](0.2))
    model.add(k["GlobalAveragePooling1D"]())
    model.add(k["Dense"](64, activation='relu'))
    model.add(k["Dropout"](0.3))
    model.add(k["Dense"](1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# ===================== Генератор батчей для LSTM/TCN (со схемой суставов) =====================

def _build_index(csv_path, data_dir, filename_col, label_col):
    meta = pd.read_csv(csv_path, usecols=[filename_col, label_col])
    items = []
    for _, row in meta.iterrows():
        fn = str(row[filename_col]).strip() if pd.notnull(row.get(filename_col)) else ""
        y = label_to_int(row.get(label_col))
        if not fn or y is None:
            continue
        p = os.path.join(data_dir, fn)
        if not os.path.exists(p) and not fn.endswith(".json"):
            p2 = p + ".json"
            if os.path.exists(p2):
                p = p2
        if os.path.exists(p):
            items.append((p, int(y)))
    return items

def _probe_max_len(file_label_pairs, motion_key, schema_joints, pctl=95, sample_max=None):
    """Быстрый проход для оценки длины по train: считаем минимальную длину среди присутствующих суставов."""
    L = []
    it = file_label_pairs if sample_max is None else file_label_pairs[:sample_max]
    for p, _ in tqdm(it, desc="Probe lengths", dynamic_ncols=True, mininterval=0.2):
        try:
            data = _safe_json_load(p)
            if motion_key not in data or not isinstance(data[motion_key], dict):
                continue
            md = data[motion_key]
            present = [j for j in schema_joints if j in md]
            if not present:
                continue
            lengths = [len(md[j]) for j in present]
            if lengths:
                L.append(min(lengths))
        except Exception:
            continue
    if not L:
        return None
    return int(np.percentile(L, pctl))

def make_sequence(files_labels, motion_key, schema_joints, max_len, batch_size, shuffle=True):
    k = _lazy_tf()
    tf = k["tf"]
    pad_sequences = k["pad_sequences"]

    class JsonSeq(tf.keras.utils.Sequence):
        def __init__(self, items):
            self.items = list(items)
            self.bs = batch_size
            self.indices = np.arange(len(self.items))
            self.shuffle = shuffle
            self.on_epoch_end()
        def __len__(self):
            return (len(self.items) + self.bs - 1) // self.bs
        def on_epoch_end(self):
            if self.shuffle:
                np.random.shuffle(self.indices)
        def __getitem__(self, idx):
            sl = self.indices[idx*self.bs:(idx+1)*self.bs]
            X_batch, y_batch = [], []
            for i in sl:
                p, y = self.items[i]
                try:
                    data = _safe_json_load(p)
                    if motion_key in data and isinstance(data[motion_key], dict):
                        md = data[motion_key]
                        seq = _stack_motion_frames_with_schema(md, schema_joints)
                        if seq is None:
                            continue
                        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
                    else:
                        continue
                    X_batch.append(seq.astype(np.float32, copy=False))
                    y_batch.append(y)
                except Exception:
                    continue
            if not X_batch:
                return np.zeros((0, max_len, 1), dtype=np.float32), np.zeros((0,), dtype=np.int32)
            Xp = pad_sequences(X_batch, maxlen=max_len, dtype='float32', padding='post', truncating='post')
            return Xp, np.array(y_batch, dtype=np.int32)

    return JsonSeq(files_labels)

# ==== TRANSFORMER ENCODER ====
def build_transformer(input_shape, d_model=128, num_heads=4, ff_dim=256, num_layers=2, dropout=0.1):
    k = _lazy_tf()
    from tensorflow.keras import layers, models
    inp = layers.Input(shape=input_shape)               # (T,F)
    x = layers.Masking(mask_value=0.0)(inp)
    x = layers.Dense(d_model)(x)                        # линейная проекция

    # позиционное кодирование (синус/косинус)
    def add_positional_encoding(x):
        import tensorflow as tf
        T = tf.shape(x)[1]; d = int(x.shape[-1])
        pos = tf.cast(tf.range(T)[:, None], tf.float32)
        i   = tf.cast(tf.range(d)[None, :], tf.float32)
        angle = pos / tf.pow(10000.0, (2*(i//2))/d)
        pe = tf.where(tf.equal(tf.cast(i % 2, tf.int32), 0), tf.sin(angle), tf.cos(angle))
        pe = tf.expand_dims(pe, 0)
        return x + pe

    x = layers.Lambda(add_positional_encoding)(x)
    for _ in range(num_layers):
        a = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads, dropout=dropout)(x, x)
        x = layers.LayerNormalization(epsilon=1e-6)(x + a)
        f = layers.Dense(ff_dim, activation="relu")(x)
        f = layers.Dropout(dropout)(f)
        f = layers.Dense(d_model)(f)
        x = layers.LayerNormalization(epsilon=1e-6)(x + f)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inp, out)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# ==== CNN-LSTM ====
def build_cnn_lstm(input_shape):
    k = _lazy_tf()
    from tensorflow.keras import layers, models
    inp = layers.Input(shape=input_shape)
    x = layers.Masking(mask_value=0.0)(inp)
    x = layers.Conv1D(128, 5, padding="same", activation="relu")(x); x = layers.BatchNormalization()(x); x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 3, padding="same", activation="relu")(x); x = layers.BatchNormalization()(x); x = layers.MaxPooling1D(2)(x)
    x = layers.LSTM(128)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inp, out)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# ==== GRU ====
def build_gru(input_shape):
    k = _lazy_tf()
    from tensorflow.keras import layers, models
    inp = layers.Input(shape=input_shape)
    x = layers.Masking(mask_value=0.0)(inp)
    x = layers.GRU(128, return_sequences=True)(x)
    x = layers.GRU(64)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inp, out)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# ==== TimesNet (лайт, мультишкальные свёртки) ====
def build_timesnet(input_shape):
    k = _lazy_tf()
    from tensorflow.keras import layers, models
    inp = layers.Input(shape=input_shape)
    x = layers.Masking(mask_value=0.0)(inp)
    b1 = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    b2 = layers.Conv1D(64, 5, padding="same", activation="relu")(x)
    b3 = layers.Conv1D(64, 7, padding="same", activation="relu")(x)
    x = layers.Concatenate()([b1, b2, b3])
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inp, out)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# ==== PatchTST (лайт) ====
def build_patchtst(input_shape, patch_len=16, stride=8, d_model=128, num_layers=2, num_heads=4, ff_dim=256):
    k = _lazy_tf()
    from tensorflow.keras import layers, models
    inp = layers.Input(shape=input_shape)
    x = layers.Masking(mask_value=0.0)(inp)
    # патч-эмбеддинг по времени
    x = layers.Conv1D(filters=d_model, kernel_size=patch_len, strides=stride, padding="valid")(x)
    for _ in range(num_layers):
        a = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads)(x, x)
        x = layers.LayerNormalization(epsilon=1e-6)(x + a)
        f = layers.Dense(ff_dim, activation="relu")(x)
        f = layers.Dense(d_model)(f)
        x = layers.LayerNormalization(epsilon=1e-6)(x + f)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inp, out)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# ==== Informer (лайт, encoder-only с downsample) ====
def build_informer(input_shape, d_model=128, num_heads=4, ff_dim=256, num_layers=2, dropout=0.1):
    k = _lazy_tf()
    from tensorflow.keras import layers, models
    inp = layers.Input(shape=input_shape)
    x = layers.Masking(mask_value=0.0)(inp)
    x = layers.Dense(d_model)(x)
    for l in range(num_layers):
        if l > 0:  # distilling: уменьшаем длину
            x = layers.Conv1D(d_model, 3, strides=2, padding="same", activation="relu")(x)
        a = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads, dropout=dropout)(x, x)
        x = layers.LayerNormalization(epsilon=1e-6)(x + a)
        f = layers.Dense(ff_dim, activation="relu")(x)
        f = layers.Dropout(dropout)(f)
        f = layers.Dense(d_model)(f)
        x = layers.LayerNormalization(epsilon=1e-6)(x + f)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inp, out)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


# ===================== Обучение моделей =====================

def train_classical(X_train, X_dev, X_test, y_train, y_dev, y_test, model_type, out_dir):
    if model_type == "rf":
        model = RandomForestClassifier(n_estimators=400, n_jobs=-1, class_weight="balanced", random_state=42)
    elif model_type == "svm":
        model = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)
    elif model_type == "xgb":
        pos = int(np.sum(y_train == 1))
        neg = int(np.sum(y_train == 0))
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

    # DEV
    prob_dev = model.predict_proba(X_dev)[:, 1]
    thr = best_threshold_by_f1(y_dev, prob_dev)
    pred_dev = (prob_dev >= thr).astype(int)
    dev_metrics = compute_metrics(y_dev, pred_dev, prob_dev)

    # TEST
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
    model.fit(
        train_seq,
        validation_data=dev_seq,
        epochs=epochs,
        class_weight=class_weight,
        callbacks=cb,
        verbose=1
    )

    # DEV
    prob_dev, y_dev = [], []
    for Xb, yb in dev_seq:
        if len(Xb) == 0:
            continue
        pb = model.predict(Xb, verbose=0).flatten()
        prob_dev.append(pb); y_dev.append(yb)
    prob_dev = np.concatenate(prob_dev); y_dev = np.concatenate(y_dev)
    thr = best_threshold_by_f1(y_dev, prob_dev)
    pred_dev = (prob_dev >= thr).astype(int)
    dev_metrics = compute_metrics(y_dev, pred_dev, prob_dev)

    # TEST
    prob_test, y_test = [], []
    for Xb, yb in test_seq:
        if len(Xb) == 0:
            continue
        pb = model.predict(Xb, verbose=0).flatten()
        prob_test.append(pb); y_test.append(yb)
    prob_test = np.concatenate(prob_test); y_test = np.concatenate(y_test)
    pred_test = (prob_test >= thr).astype(int)
    test_metrics = compute_metrics(y_test, pred_test, prob_test)

    model.save(os.path.join(out_dir, "model.h5"))
    with open(os.path.join(out_dir, "threshold.txt"), "w") as f:
        f.write(str(thr))
    # Сохраняем max_len в отдельном npz (в генераторе нормировки не делаем)
    np.savez(os.path.join(out_dir, "norm_stats.npz"), max_len=input_shape[0])

    return dev_metrics, test_metrics, thr, model


# ===================== Main =====================

def print_split_stats(name, y):
    n = len(y)
    n1 = int(np.sum(y == 1))
    n0 = n - n1
    p1 = (n1 / n * 100) if n else 0.0
    p0 = (n0 / n * 100) if n else 0.0
    print(f"[{name}] total={n} | Injury=1: {n1} ({p1:.1f}%) | No Injury=0: {n0} ({p0:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Binary injury classifier from 3D joint JSON sequences (70/10/20 split)")
    parser.add_argument(
        "--model", type=str,
        choices=["rf","svm","xgb","lstm","tcn","gru","cnn_lstm","transformer","timesnet","patchtst","informer"],
        required=True
    )
    parser.add_argument("--csv", type=str, required=True, help="Путь к CSV с метаданными")
    parser.add_argument("--data_dir", type=str, required=True, help="Папка с JSON файлами")
    parser.add_argument("--motion_key", type=str, default="running", help="running | walking")
    parser.add_argument("--filename_col", type=str, default="filename")
    parser.add_argument("--label_col", type=str, default="No inj/ inj")
    parser.add_argument("--use_joints", type=str, default="", help="через запятую: pelvis_1,hip_r,... (пусто = авто-схема)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--out_dir", type=str, default="outputs", help="куда сохранять модель/метрики")
    parser.add_argument("--max_len", type=str, default="auto", help="'auto' (95-перцентиль по train) или число для LSTM/TCN")
    parser.add_argument("--loader_workers", type=int, default=0, help="кол-во процессов при извлечении фич (классика)")
   

    args = parser.parse_args()

    use_joints = None
    if args.use_joints.strip():
        use_joints = [s.strip() for s in args.use_joints.split(",") if s.strip()]

    out_dir = os.path.join(args.out_dir, args.model)
    ensure_dir(out_dir)

    if args.model in ["rf", "svm", "xgb"]:
        # ===== КЛАССИКА: стриминг фич + сохранение схемы суставов =====
        X_all, y_all, schema_joints = load_features_streaming(
            args.csv, args.data_dir,
            motion_key=args.motion_key,
            filename_col=args.filename_col,
            label_col=args.label_col,
            use_joints=use_joints,
            workers=args.loader_workers
        )
        # Сохраним схему для predict.py
        with open(os.path.join(out_dir, "schema_joints.json"), "w", encoding="utf-8") as f:
            json.dump(schema_joints, f, ensure_ascii=False, indent=2)


        # Сплит 70/20/10
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X_all, y_all, test_size=0.20, random_state=42, stratify=y_all
        )
        X_train_feat, X_dev_feat, y_train, y_dev = train_test_split(
            X_train_full, y_train_full, test_size=0.125, random_state=42, stratify=y_train_full
        )

        # >>> NEW: сводка по классам
        print("\n=== Split stats (classical) ===")
        print_split_stats("TRAIN (≈70%)", y_train)
        print_split_stats("DEV   (≈10%)", y_dev)
        print_split_stats("TEST  (≈20%)", y_test)

        # Баланс классов (для информации)
        classes = np.array([0, 1])
        w = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
        class_weight = {0: float(w[0]), 1: float(w[1])}
        print("Class weights (train):", class_weight)

        scaler = StandardScaler()
        X_train_feat = scaler.fit_transform(X_train_feat)
        X_dev_feat   = scaler.transform(X_dev_feat)
        X_test_feat  = scaler.transform(X_test)

        dev_metrics, test_metrics, thr, model = train_classical(
            X_train_feat, X_dev_feat, X_test_feat, y_train, y_dev, y_test, args.model, out_dir
        )

        joblib.dump({"model": model, "scaler": scaler}, os.path.join(out_dir, "model.joblib"))

    else:
        # ===== НЕЙРОСЕТИ: схема суставов + генераторы =====
        items = _build_index(args.csv, args.data_dir, args.filename_col, args.label_col)
        assert items, "Нет валидных файлов/меток."

        # Схема суставов (общая для всего датасета) и сохранение
        schema_joints = discover_joint_schema(
            args.csv, args.data_dir, args.motion_key,
            filename_col=args.filename_col, label_col=args.label_col,
            sample_max=500, freq_threshold=0.9, use_joints=use_joints
        )
        with open(os.path.join(out_dir, "schema_joints.json"), "w", encoding="utf-8") as f:
            json.dump(schema_joints, f, ensure_ascii=False, indent=2)
        print(f"Schema joints saved ({len(schema_joints)}): "
              f"{', '.join(schema_joints[:8])}{' ...' if len(schema_joints) > 8 else ''}")

        files = np.array([p for p, _ in items], dtype=object)
        labels = np.array([y for _, y in items], dtype=np.int32)

        idx_train_full, idx_test = train_test_split(
            np.arange(len(items)), test_size=0.20, random_state=42, stratify=labels
        )
        idx_train, idx_dev = train_test_split(
            idx_train_full, test_size=0.125, random_state=42, stratify=labels[idx_train_full]
        )

        # >>> NEW: сводка по классам (через labels и индексы)
        print("\n=== Split stats (deep) ===")
        print_split_stats("TRAIN (≈70%)", labels[idx_train])
        print_split_stats("DEV   (≈10%)", labels[idx_dev])
        print_split_stats("TEST  (≈20%)", labels[idx_test])


        # Оценка max_len по train
        if args.max_len.strip().lower() == "auto":
            max_len = _probe_max_len([items[i] for i in idx_train], args.motion_key, schema_joints, pctl=95, sample_max=None)
            if not max_len or max_len <= 1:
                raise ValueError("Не удалось оценить max_len. Укажи --max_len вручную.")
        else:
            max_len = int(args.max_len)
        print("max_len (train) =", max_len)

        train_seq = make_sequence([items[i] for i in idx_train], args.motion_key, schema_joints, max_len, batch_size=args.batch_size, shuffle=True)
        dev_seq   = make_sequence([items[i] for i in idx_dev],   args.motion_key, schema_joints, max_len, batch_size=args.batch_size, shuffle=False)
        test_seq  = make_sequence([items[i] for i in idx_test],  args.motion_key, schema_joints, max_len, batch_size=args.batch_size, shuffle=False)

        # Получим input_shape
        X_sample, _ = train_seq[0]
        if X_sample.shape[0] == 0:
            raise RuntimeError("Пустой первый батч. Проверь данные.")
        input_shape = (X_sample.shape[1], X_sample.shape[2])

        # Баланс классов по train
        y_train_tmp = np.concatenate([yb for _, yb in train_seq]) if len(train_seq) > 0 else None
        if y_train_tmp is None or len(y_train_tmp) == 0:
            raise RuntimeError("Не удалось собрать метки для train.")
        classes = np.array([0, 1])
        w = compute_class_weight(class_weight="balanced", classes=classes, y=y_train_tmp)
        class_weight = {0: float(w[0]), 1: float(w[1])}
        print("Class weights (train):", class_weight)

        dev_metrics, test_metrics, thr, model = train_deep(
            args.model, train_seq, dev_seq, test_seq, class_weight, args.epochs, out_dir, input_shape
        )

    # ===== Метрики и сохранение =====
    print("\n=== DEV METRICS (threshold tuned here) ===")
    for k in ["accuracy", "f1", "roc_auc", "confusion_matrix"]:
        print(f"{k}: {dev_metrics[k]}")
    print("\nDev classification report:\n", dev_metrics["report"])

    print("\n=== TEST METRICS (using dev-tuned threshold) ===")
    for k in ["accuracy", "f1", "roc_auc", "confusion_matrix"]:
        print(f"{k}: {test_metrics[k]}")
    print("\nTest classification report:\n", test_metrics["report"])

    # JSON-отчёты
    with open(os.path.join(out_dir, "metrics_dev.json"), "w", encoding="utf-8") as f:
        json.dump(dev_metrics, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "metrics_test.json"), "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to: {out_dir}")
