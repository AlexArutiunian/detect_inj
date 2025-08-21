#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import List, Tuple, Dict
from tqdm import tqdm
import os, json, argparse, random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# ---------------------- графики ----------------------
def save_plots(history, out_dir, y_test, prob_test):
    os.makedirs(out_dir, exist_ok=True)

    for metric in ["loss", "accuracy"]:
        train = history.history.get(metric, [])
        val   = history.history.get(f"val_{metric}", [])
        if not train:
            continue
        plt.figure()
        plt.plot(range(1, len(train)+1), train, label=f"train_{metric}")
        if val: plt.plot(range(1, len(val)+1), val, label=f"val_{metric}")
        plt.xlabel("Epoch"); plt.ylabel(metric); plt.legend()
        plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{metric}.png"), dpi=150)
        plt.close()

    if y_test is not None and prob_test is not None and len(np.unique(y_test)) == 2:
        fpr, tpr, _ = roc_curve(y_test, prob_test)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
        plt.plot([0,1], [0,1], linestyle="--")
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend()
        plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "roc_test.png"), dpi=150)
        plt.close()

# ---------------------- утилиты ----------------------
def ensure_dir(p: str): os.makedirs(p, exist_ok=True)

def label_to_int(v):
    if v is None: return None
    s = str(v).strip().lower()
    if s in ("injury", "1"): return 1
    if s in ("no injury", "0"): return 0
    return None

def sanitize_seq(a: np.ndarray) -> np.ndarray:
    """
    Приводит массив к (T, F):
    - заменяет NaN/Inf на 0
    - переносит самую длинную ось как время (ось 0)
    - остальные оси сплющивает в признаки
    """
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    if a.ndim == 1:
        a = a[:, None]  # (T,) -> (T,1)
    elif a.ndim >= 2:
        t_axis = int(np.argmax(a.shape))      # предполагаем: самая длинная ось — время
        if t_axis != 0:
            a = np.moveaxis(a, t_axis, 0)     # время -> ось 0
        if a.ndim > 2:
            a = a.reshape(a.shape[0], -1)     # (T, ...)->(T,F)
    return a.astype(np.float32, copy=False)

def best_threshold(y_true, p, mode="bal_acc", fixed=None, target_spec=None):
    """Устойчивый выбор порога, чтобы не залипать на одном классе."""
    from sklearn.metrics import precision_recall_curve, roc_curve, f1_score, balanced_accuracy_score

    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 1e-7, 1 - 1e-7)

    if fixed is not None:
        return float(fixed)

    if mode == "f1_pos":
        pr, rc, th = precision_recall_curve(y_true, p)
        if len(th) == 0: return 0.5
        f1 = 2*pr[:-1]*rc[:-1]/np.clip(pr[:-1]+rc[:-1], 1e-12, None)
        return float(th[int(np.nanargmax(f1))])

    if mode == "macro_f1":
        thr = np.linspace(0.01, 0.99, 199)
        scores = [f1_score(y_true, p >= t, average="macro") for t in thr]
        return float(thr[int(np.argmax(scores))])

    if mode == "bal_acc":
        thr = np.linspace(0.01, 0.99, 199)
        scores = [balanced_accuracy_score(y_true, p >= t) for t in thr]
        best_i = int(np.argmax(scores))
        print(f"[THR] bal_acc={scores[best_i]:.3f} @ thr={thr[best_i]:.3f}")
        return float(thr[best_i])

    if mode in ("roc_j", "youden", "spec"):
        fpr, tpr, th = roc_curve(y_true, p)
        m = ~np.isinf(th)  # убрать inf
        fpr, tpr, th = fpr[m], tpr[m], th[m]
        spec = 1.0 - fpr
        if mode != "spec":
            j = tpr - fpr
            return float(th[int(np.argmax(j))])
        if target_spec is None: target_spec = 0.5
        idx = np.where(spec >= float(target_spec))[0]
        return float(th[idx[-1]] if len(idx) else th[-1])

    return 0.5

def compute_metrics(y_true, y_pred, y_prob):
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else float("nan"),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "report": classification_report(y_true, y_pred, digits=3),
    }

# ---------------------- маппинг путей ----------------------
def possible_npy_paths(data_dir: str, rel: str) -> List[str]:
    """Генерирует все разумные варианты путей (учёт .json, .npy, .json.npy, базовое имя)."""
    rel = (rel or "").strip().replace("\\", "/").lstrip("/")
    cands = []
    def push(x):
        if x not in cands:
            cands.append(x)

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

def pick_existing_path(cands: List[str]) -> str | None:
    for p in cands:
        if os.path.exists(p):
            return p
    return None

# ---------------------- чтение индекса ----------------------
def build_items(csv_path: str, data_dir: str,
                filename_col="filename", label_col="No inj/ inj",
                debug_index: bool=False) -> Tuple[List[str], List[int], Dict[str,int], List[Tuple[str,str]]]:
    df = pd.read_csv(csv_path)
    cols = {c.lower().strip(): c for c in df.columns}
    fn_col = cols.get(filename_col.lower(), filename_col)
    if fn_col not in df.columns:
        for c in df.columns:
            if "file" in c.lower() or "name" in c.lower():
                fn_col = c; break
    lb_col = cols.get(label_col.lower(), label_col)
    if lb_col not in df.columns:
        for c in df.columns:
            if "inj" in c.lower() or "label" in c.lower() or "target" in c.lower():
                lb_col = c; break

    print(f"Using columns: filename_col='{fn_col}', label_col='{lb_col}'")

    items_x, items_y = [], []
    stats = {"ok":0, "no_file":0, "bad_label":0, "too_short":0, "error":0}
    skipped_examples = []

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Indexing"):
        rel = str(row.get(fn_col, "")).strip()
        lab_raw = row.get(lb_col, None)
        lab = label_to_int(lab_raw)

        status = ""
        resolved = None
        shape_txt = ""

        if not rel:
            stats["no_file"] += 1
            status = "empty-filename"
            if len(skipped_examples)<10: skipped_examples.append((status, str(row.to_dict())))
        elif lab is None:
            stats["bad_label"] += 1
            status = f"bad-label:{lab_raw}"
            if len(skipped_examples)<10: skipped_examples.append((status, rel))
        else:
            path = pick_existing_path(possible_npy_paths(data_dir, rel))
            if not path:
                stats["no_file"] += 1
                status = "not-found"
                if len(skipped_examples)<10: skipped_examples.append((status, rel))
            else:
                try:
                    arr = np.load(path, allow_pickle=False, mmap_mode="r")
                    x = sanitize_seq(arr)  # <-- универсально к (T,F)
                    if x.ndim != 2 or x.shape[0] < 2:
                        stats["too_short"] += 1
                        status = "too-short"
                        if len(skipped_examples)<10: skipped_examples.append((status, os.path.basename(path)))
                    else:
                        items_x.append(path)
                        items_y.append(int(lab))
                        stats["ok"] += 1
                        status = "OK"
                        resolved = path
                        shape_txt = f"shape={tuple(x.shape)}"
                except Exception as e:
                    stats["error"] += 1
                    status = f"np-load:{type(e).__name__}"
                    if len(skipped_examples)<10: skipped_examples.append((status, os.path.basename(path) if path else rel))

        if debug_index:
            print(f"[{i:05d}] csv='{rel}' | label_raw='{lab_raw}' -> {status}"
                  + (f" | path='{resolved}' {shape_txt}" if resolved else ""))

    return items_x, items_y, stats, skipped_examples

def probe_stats(items: List[Tuple[str,int]], downsample: int = 1, pctl: int = 95):
    lengths = []
    feat = None
    for p, _ in items:
        arr = np.load(p, allow_pickle=False, mmap_mode="r")
        x = sanitize_seq(arr)  # <-- сначала нормализуем
        L = int(x.shape[0] // max(1, downsample))
        if x.ndim != 2 or L < 1:
            continue
        if feat is None: feat = int(x.shape[1])
        if L > 0: lengths.append(L)
    max_len = int(np.percentile(lengths, pctl)) if lengths else 256
    return max_len, feat or 1

def compute_norm_stats(items: List[Tuple[str,int]], feat_dim: int, downsample: int, max_len_cap: int, sample_items: int = 512):
    rng = random.Random(42)
    pool = items if len(items) <= sample_items else rng.sample(items, sample_items)
    count = 0
    mean = np.zeros(feat_dim, dtype=np.float64)
    m2   = np.zeros(feat_dim, dtype=np.float64)
    for p, _ in pool:
        arr = np.load(p, allow_pickle=False, mmap_mode="r")
        x = sanitize_seq(arr)              # <-- нормализуем
        x = x[::max(1, downsample)]        # <-- потом downsample по времени
        if x.shape[0] > max_len_cap: x = x[:max_len_cap]
        for t in range(x.shape[0]):
            count += 1
            delta = x[t] - mean
            mean += delta / count
            m2   += delta * (x[t] - mean)
    var = m2 / max(1, (count - 1))
    std = np.sqrt(np.clip(var, 1e-12, None)).astype(np.float32)
    return mean.astype(np.float32), std

# ---------------------- TF часть ----------------------
def lazy_tf():
    import tensorflow as tf
    from tensorflow.keras import layers, models
    return tf, layers, models

def make_datasets(items, labels, max_len, feat_dim, bs, downsample, mean, std, replicas):
    import tensorflow as tf
    AUTOTUNE = tf.data.AUTOTUNE

    def gen(indices):
        def _g():
            for i in indices:
                p = items[i]
                y = labels[i]
                arr = np.load(p, allow_pickle=False, mmap_mode="r")
                x = sanitize_seq(arr)           # <-- нормализуем
                x = x[::max(1, downsample)]     # <-- потом downsample
                if x.shape[0] < 2:
                    continue
                if x.shape[0] > max_len:
                    x = x[:max_len]
                x = (x - mean) / std
                yield x, np.int32(y)
        return _g

    sig = (
        tf.TensorSpec(shape=(None, feat_dim), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )

    def pad_map(x, y):
        T = tf.shape(x)[0]
        pad_t = tf.maximum(0, max_len - T)
        x = tf.pad(x, [[0, pad_t], [0, 0]])
        x = x[:max_len]
        x = tf.where(tf.math.is_finite(x), x, tf.zeros_like(x))  # защита от NaN/Inf
        x.set_shape([max_len, feat_dim])
        return x, y

    def make(indices, shuffle=False, drop_remainder=False):
        ds = tf.data.Dataset.from_generator(gen(indices), output_signature=sig)
        if shuffle:
            ds = ds.shuffle(4096, reshuffle_each_iteration=True)
        ds = ds.map(pad_map, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(bs, drop_remainder=drop_remainder)
        ds = ds.prefetch(AUTOTUNE)
        return ds

    return make

# ---------------------- Модель TCN ----------------------
def build_tcn_model(
    input_shape,
    learning_rate=1e-3,
    mixed_precision=False,
    tcn_filters=64,
    tcn_kernel=5,
    tcn_dilations=(1, 2, 4, 8),
    tcn_block_dropout=0.2,
    tcn_pool="max",
    dense_units=64,
    dense_dropout=0.3,
    norm_type="layer",  # "layer" | "batch" | "none"
):
    tf, layers, models = lazy_tf()

    def _norm(x):
        if norm_type == "layer":
            return layers.LayerNormalization(epsilon=1e-5)(x)
        elif norm_type == "batch":
            return layers.BatchNormalization(epsilon=1e-3)(x)
        else:
            return x

    def tcn_block(x, filters, kernel_size, dilation_rate, dropout):
        y = layers.Conv1D(filters, kernel_size, padding="causal",
                          dilation_rate=dilation_rate,
                          kernel_initializer="he_normal", use_bias=True)(x)
        y = _norm(y)
        y = layers.Activation("relu")(y)
        if dropout and dropout > 0:
            y = layers.Dropout(dropout)(y)

        y = layers.Conv1D(filters, kernel_size, padding="causal",
                          dilation_rate=dilation_rate,
                          kernel_initializer="he_normal", use_bias=True)(y)
        y = _norm(y)

        res = x if x.shape[-1] == filters else layers.Conv1D(filters, 1, padding="same")(x)
        y = layers.Add()([res, y])
        y = layers.Activation("relu")(y)
        return y

    inp = layers.Input(shape=input_shape)  # [T, F]
    x = inp
    for d in tcn_dilations:
        x = tcn_block(x, filters=tcn_filters, kernel_size=tcn_kernel,
                      dilation_rate=int(d), dropout=tcn_block_dropout)

    if tcn_pool.lower() == "avg":
        x = layers.GlobalAveragePooling1D()(x)
    else:
        x = layers.GlobalMaxPooling1D()(x)

    x = layers.Dropout(dense_dropout)(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid", dtype="float32")(x)

    model = models.Model(inp, out)
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-7,
                                   amsgrad=True, clipnorm=1.0)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
    return model

# ---------------------- helpers для CLI ----------------------
def _parse_int_list(s: str, default):
    if s is None or str(s).strip() == "":
        return tuple(default)
    parts = str(s).replace(";", ",").replace(" ", ",").split(",")
    vals = [int(p) for p in parts if p.strip() != ""]
    return tuple(vals) if vals else tuple(default)

# ---------------------- main ----------------------
def main():
    ap = argparse.ArgumentParser("TCN binary classifier over NPY sequences")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--filename_col", default="filename")
    ap.add_argument("--label_col", default="No inj/ inj")
    ap.add_argument("--out_dir", default="output_run_tcn")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--downsample", type=int, default=1)
    ap.add_argument("--max_len", default="auto")
    ap.add_argument("--gpus", default="all")
    ap.add_argument("--mixed_precision", action="store_true")
    ap.add_argument("--no_xla", action="store_true", help="Disable XLA JIT")
    ap.add_argument("--lr", type=float, default=1e-3, help="initial learning rate")

    # ---- TCN hyperparams ----
    ap.add_argument("--tcn_filters", type=int, default=64)
    ap.add_argument("--tcn_kernel", type=int, default=5)
    ap.add_argument("--tcn_dilations", type=str, default="1,2,4,8")
    ap.add_argument("--tcn_block_dropout", type=float, default=0.2)
    ap.add_argument("--tcn_pool", type=str, choices=["max","avg"], default="max")
    ap.add_argument("--tcn_dense_units", type=int, default=64)
    ap.add_argument("--tcn_dense_dropout", type=float, default=0.3)
    ap.add_argument("--norm", choices=["layer","batch","none"], default="layer",
                    help="нормализация внутри TCN-блоков")

    # прочее
    ap.add_argument("--peek", type=int, default=0,
                    help="Показать N успешно сопоставленных путей (форма массива)")

    # выбор порога и веса классов
    ap.add_argument("--threshold_mode", default="bal_acc",
                    choices=["bal_acc", "f1_pos", "macro_f1", "roc_j", "spec", "fixed"])
    ap.add_argument("--threshold_fixed", type=float, default=None)
    ap.add_argument("--target_specificity", type=float, default=None)
    ap.add_argument("--cw_scale0", type=float, default=1.0,
                    help="множитель веса класса 0 (No injury)")
    ap.add_argument("--cw_scale1", type=float, default=1.0,
                    help="множитель веса класса 1 (Injury)")

    args = ap.parse_args()

    # Видимость GPU до импорта TF
    if args.gpus.lower() == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif args.gpus.lower() != "all":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    if args.no_xla:
        os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"

    # Индекс
    items_x, items_y, stats, skipped = build_items(
        args.csv, args.data_dir,
        filename_col=args.filename_col,
        label_col=args.label_col,
        debug_index=False
    )
    assert items_x and items_y, "Не найдено валидных .npy и меток"
    items = list(zip(items_x, items_y))
    paths = items_x
    labels_all = np.array(items_y, dtype=np.int32)

    if args.peek > 0:
        print(f"\n=== Peek first {min(args.peek, len(items))} matched items ===")
        for (pth, lab) in items[:args.peek]:
            try:
                arr = np.load(pth, allow_pickle=False, mmap_mode="r")
                x = sanitize_seq(arr)
                print(f"OK  label={lab} | {pth} | shape={x.shape}")
            except Exception as e:
                print(f"FAIL to load for peek: {pth} -> {type(e).__name__}")

    # Статы по длине/размеру признака
    if str(args.max_len).strip().lower() == "auto":
        max_len, feat_dim = probe_stats(items, downsample=args.downsample, pctl=95)
        max_len = int(max(8, min(max_len, 20000)))
    else:
        max_len = int(args.max_len)
        aa = np.load(paths[0], allow_pickle=False, mmap_mode="r")
        aa = sanitize_seq(aa)  # <-- важно
        feat_dim = int(aa.shape[1])
    print(f"max_len={max_len} | feat_dim={feat_dim}")

    # Сплит 70/10/20
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(paths))
    idx_train_full, idx_test = train_test_split(idx, test_size=0.20, random_state=42, stratify=labels_all)
    idx_train, idx_dev = train_test_split(idx_train_full, test_size=0.125, random_state=42, stratify=labels_all[idx_train_full])

    # Норм-статы по train
    ensure_dir(args.out_dir)
    mean, std = compute_norm_stats([items[i] for i in idx_train], feat_dim, args.downsample, max_len)
    np.savez_compressed(os.path.join(args.out_dir, "norm_stats.npz"), mean=mean, std=std, max_len=max_len)

    # TF / стратегия
    tf, layers, models = lazy_tf()
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        try: tf.config.experimental.set_memory_growth(g, True)
        except Exception: pass

    if args.mixed_precision:
        from tensorflow.keras import mixed_precision as mp
        mp.set_global_policy("mixed_float16")  # выход Dense уже float32

    strategy = tf.distribute.MirroredStrategy() if len(gpus) > 1 else None
    replicas = strategy.num_replicas_in_sync if strategy else 1
    global_bs = args.batch_size * replicas
    print(f"GPUs: {len(gpus)} | replicas: {replicas} | global_batch: {global_bs}")

    # Датасеты
    make_ds = make_datasets(paths, labels_all, max_len, feat_dim, global_bs,
                            args.downsample, mean, std, replicas)
    train_ds = make_ds(idx_train, shuffle=True,  drop_remainder=(replicas > 1))
    dev_ds   = make_ds(idx_dev,   shuffle=False, drop_remainder=False)
    test_ds  = make_ds(idx_test,  shuffle=False, drop_remainder=False)

    # Веса классов
    from sklearn.utils.class_weight import compute_class_weight
    cls = np.array([0, 1], dtype=np.int32)
    w = compute_class_weight("balanced", classes=cls, y=labels_all[idx_train])
    class_weight = {0: float(w[0] * args.cw_scale0), 1: float(w[1] * args.cw_scale1)}
    print("class_weight:", class_weight)

    # ---- распарсим и выведем TCN-параметры ----
    dilations = _parse_int_list(args.tcn_dilations, default=(1,2,4,8))
    print(f"TCN config: filters={args.tcn_filters}, kernel={args.tcn_kernel}, "
          f"dilations={dilations}, block_dropout={args.tcn_block_dropout}, "
          f"pool={args.tcn_pool}, dense_units={args.tcn_dense_units}, "
          f"dense_dropout={args.tcn_dense_dropout}, norm={args.norm}")

    # Строим модель (TCN)
    ctx = strategy.scope() if strategy else contextlib_null()
    with ctx:
        model = build_tcn_model(
            (max_len, feat_dim),
            learning_rate=args.lr,
            mixed_precision=args.mixed_precision,
            tcn_filters=args.tcn_filters,
            tcn_kernel=args.tcn_kernel,
            tcn_dilations=dilations,
            tcn_block_dropout=args.tcn_block_dropout,
            tcn_pool=args.tcn_pool,
            dense_units=args.tcn_dense_units,
            dense_dropout=args.tcn_dense_dropout,
            norm_type=args.norm,
        )

    # Коллбеки
    cb = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    ]

    # Обучение
    hist = model.fit(train_ds,
                     validation_data=dev_ds,
                     epochs=args.epochs,
                     class_weight=class_weight,
                     callbacks=cb,
                     verbose=1)

    # Предсказания и метрики
    prob_dev  = model.predict(dev_ds,  verbose=0).reshape(-1).astype(np.float32)
    y_dev     = labels_all[idx_dev]
    thr = best_threshold(
        y_dev, prob_dev,
        mode=args.threshold_mode,
        fixed=args.threshold_fixed,
        target_spec=args.target_specificity
    )
    print(f"Chosen threshold: {thr:.3f}")

    prob_test = model.predict(test_ds, verbose=0).reshape(-1).astype(np.float32)
    y_test    = labels_all[idx_test]
    pred_test = (prob_test >= thr).astype(np.int32)

    dev_pred  = (prob_dev >= thr).astype(np.int32)
    dev_metrics  = compute_metrics(y_dev, dev_pred, prob_dev)
    test_metrics = compute_metrics(y_test, pred_test, prob_test)
    save_plots(hist, args.out_dir, y_test, prob_test)

    # Сохранения
    model.save(os.path.join(args.out_dir, "model.keras"))
    with open(os.path.join(args.out_dir, "threshold.txt"), "w") as f: f.write(str(thr))
    with open(os.path.join(args.out_dir, "metrics_dev.json"), "w", encoding="utf-8") as f:
        json.dump(dev_metrics, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out_dir, "metrics_test.json"), "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)

    print("\n=== DEV (threshold tuned) ===")
    print(dev_metrics["report"])
    print("\n=== TEST (using DEV threshold) ===")
    print(test_metrics["report"])
    print("Saved to:", args.out_dir)

# заглушка контекста, если нет стратегии
class contextlib_null:
    def __enter__(self): return None
    def __exit__(self, *exc): return False

if __name__ == "__main__":
    main()
