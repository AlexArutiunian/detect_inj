#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ST-GCN бинарный классификатор для последовательностей из .npy
(замена GRU на Spatio-Temporal Graph Convolutional Network)

Ожидается, что вход — это массив формы (T, F), где F = V*C.
V — число узлов графа (например, суставов), C — число каналов на узел (например, x,y,(z)).
Если V и C не заданы, они выводятся автоматически (3->2->1).

Граф (аджacency) можно задать:
  * --edges_json путь к JSON-списку рёбер [[i,j], ...] (0- или 1-индексация)
  * --stgcn_adjacency {identity,ring,full} — предустановки (по умолчанию ring)

Прочее: логика загрузки данных, нормализация, разбиение, сохранение метрик/графиков —
оставлены из исходного скрипта.
"""
from typing import List, Tuple, Dict
from tqdm import tqdm
import os, json, argparse, math, random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # сохраняем графики без GUI
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
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    return a.astype(np.float32, copy=False)

def best_threshold_by_f1(y_true, p):
    from sklearn.metrics import precision_recall_curve
    pr, rc, th = precision_recall_curve(y_true, p)
    if len(th) == 0: return 0.5
    f1 = 2*pr[:-1]*rc[:-1]/np.clip(pr[:-1]+rc[:-1], 1e-12, None)
    return float(th[int(np.nanargmax(f1))])

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

    # как есть и с .npy
    push(os.path.join(data_dir, rel))
    if not rel.endswith(".npy"):
        push(os.path.join(data_dir, rel + ".npy"))

    # варианты с .json → .npy и .json.npy
    if rel.endswith(".json"):
        base_nojson = rel[:-5]
        push(os.path.join(data_dir, base_nojson + ".npy"))
        push(os.path.join(data_dir, rel + ".npy"))          # .json.npy

    # если уже .json.npy — попробуем без .json
    if rel.endswith(".json.npy"):
        push(os.path.join(data_dir, rel.replace(".json.npy", ".npy")))

    # только базовое имя
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
                    if arr.ndim != 2 or arr.shape[0] < 2:
                        stats["too_short"] += 1
                        status = "too-short"
                        if len(skipped_examples)<10: skipped_examples.append((status, os.path.basename(path)))
                    else:
                        items_x.append(path)
                        items_y.append(int(lab))
                        stats["ok"] += 1
                        status = "OK"
                        resolved = path
                        shape_txt = f"shape={tuple(arr.shape)}"
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
        if arr.ndim != 2 or arr.shape[0] < 2: continue
        if feat is None: feat = int(arr.shape[1])
        L = int(arr.shape[0] // max(1, downsample))
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
        x = sanitize_seq(arr[::max(1, downsample)])
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

# ======== ST-GCN utils ========
def infer_nodes_channels(feat_dim: int, channels: int|None, nodes: int|None) -> Tuple[int,int]:
    if nodes and channels and nodes*channels == feat_dim:
        return int(nodes), int(channels)
    if nodes and feat_dim % nodes == 0:
        return int(nodes), int(feat_dim // nodes)
    if channels and feat_dim % channels == 0:
        return int(feat_dim // channels), int(channels)
    if feat_dim % 3 == 0:
        return int(feat_dim // 3), 3
    if feat_dim % 2 == 0:
        return int(feat_dim // 2), 2
    return feat_dim, 1

def load_edges_json(path: str, V: int) -> List[Tuple[int,int]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    edges = []
    for e in raw:
        if isinstance(e, dict) and "src" in e and "dst" in e:
            a, b = int(e["src"]), int(e["dst"])
        else:
            a, b = int(e[0]), int(e[1])
        # поддержка 1-индексации
        if a >= 1 and b >= 1 and (a > V or b > V):
            # если явно больше V, оставляем как есть (возможно V тоже 1-индексировано)
            pass
        # нормируем в 0..V-1, если нужно
        if a >= 1 and b >= 1 and (V == max(a,b)):
            a -= 1; b -= 1
        edges.append((max(0, min(V-1, a)), max(0, min(V-1, b))))
    return edges

def make_adjacency(V: int, preset: str = "ring", edges: List[Tuple[int,int]]|None = None, self_loops: bool = True) -> np.ndarray:
    A = np.zeros((V,V), dtype=np.float32)
    if edges:
        for i,j in edges:
            if 0 <= i < V and 0 <= j < V:
                A[i,j] = 1.0
                A[j,i] = 1.0
    elif preset == "identity":
        pass
    elif preset == "full":
        A[:] = 1.0
    else:  # ring
        for i in range(V-1):
            A[i, i+1] = 1.0
            A[i+1, i] = 1.0
    if self_loops:
        A += np.eye(V, dtype=np.float32)
    # нормализация D^{-1/2} A D^{-1/2}
    deg = np.clip(A.sum(axis=1), 1e-6, None)
    D_inv_sqrt = 1.0 / np.sqrt(deg)
    A_norm = (A * D_inv_sqrt[:, None]) * D_inv_sqrt[None, :]
    return A_norm.astype(np.float32)

# ======== TF datasets ========
def make_datasets(items, labels, max_len, feat_dim, V, C, bs, downsample, mean, std, replicas):
    import tensorflow as tf
    AUTOTUNE = tf.data.AUTOTUNE

    def gen(indices):
        def _g():
            for i in indices:
                p = items[i]
                y = labels[i]
                arr = np.load(p, allow_pickle=False, mmap_mode="r")
                x = sanitize_seq(arr[::max(1, downsample)])
                if x.shape[0] < 2:
                    continue
                if x.shape[0] > max_len:
                    x = x[:max_len]
                # нормализация покомпонентно (по F), затем reshape в (T,V,C)
                x = (x - mean) / std
                T = x.shape[0]
                # fallback, если F не делится как ожидалось — добиваем нулями
                F = x.shape[1]
                if F != V*C:
                    pad_f = (V*C) - (F % (V*C)) if F % (V*C) != 0 else 0
                    if pad_f > 0:
                        x = np.pad(x, [[0,0],[0,pad_f]], mode='constant')
                x = x.reshape(T, V, C)
                yield x, np.int32(y)
        return _g

    sig = (
        tf.TensorSpec(shape=(None, V, C), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )

    def pad_map(x, y):
        T = tf.shape(x)[0]
        pad_t = tf.maximum(0, max_len - T)
        x = tf.pad(x, [[0, pad_t], [0, 0], [0, 0]])
        x = x[:max_len]
        x.set_shape([max_len, V, C])
        return x, y

    def make(indices, shuffle=False, drop_remainder=False):
        ds = tf.data.Dataset.from_generator(gen(indices), output_signature=sig)
        if shuffle:
            ds = ds.shuffle(64, reshuffle_each_iteration=True)
        ds = ds.map(pad_map, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(bs, drop_remainder=drop_remainder)
        ds = ds.prefetch(AUTOTUNE)
        return ds

    return make

# ======== ST-GCN модель ========
def build_stgcn_model(T_max: int, V: int, C: int, A_norm: np.ndarray,
                      learning_rate: float = 1e-3, mixed_precision: bool=False,
                      temporal_kernel: int = 9,
                      channels_list: List[int] = [128, 64],
                      temporal_strides: List[int] | None = None,
                      dropout: float = 0.1):
    tf, layers, models = lazy_tf()

    A_tf = tf.constant(A_norm, dtype=tf.float32)

    def spatial_gcn(x, out_ch):
        # x: (B, T, V, C_in)
        x1 = layers.TimeDistributed(layers.Dense(out_ch))(x)  # (B,T,V,out_ch)
        x2 = layers.Lambda(lambda z: tf.einsum('btuc,vu->btvc', z, A_tf))(x1)  # умножение на A по узлам
        x2 = layers.BatchNormalization()(x2)
        x2 = layers.Activation('relu')(x2)
        return x2

    def stgcn_block(x, out_ch, stride_t: int = 1):
        s = spatial_gcn(x, out_ch)
        t = layers.Conv2D(out_ch, kernel_size=(temporal_kernel, 1), strides=(stride_t, 1), padding='same')(s)  # свёртка по времени с шагом
        t = layers.BatchNormalization()(t)
        t = layers.Activation('relu')(t)
        if dropout and dropout > 0:
            t = layers.Dropout(dropout)(t)
        # residual
        if x.shape[-1] != out_ch or stride_t != 1:
            r = layers.Conv2D(out_ch, kernel_size=(1,1), strides=(stride_t,1), padding='same')(x)
            r = layers.BatchNormalization()(r)
        else:
            r = x
        out = layers.Add()([t, r])
        out = layers.Activation('relu')(out)
        return out

    if temporal_strides is None:
        temporal_strides = [1] * len(channels_list)

    inp = layers.Input(shape=(T_max, V, C))
    x = inp
    for ch, st in zip(channels_list, temporal_strides):
        x = stgcn_block(x, ch, stride_t=int(st))

    x = layers.GlobalAveragePooling2D()(x)  # усреднение по (T,V)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(1, activation='sigmoid', dtype='float32')(x)

    model = models.Model(inp, out)
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ---------------------- main ----------------------
class contextlib_null:
    def __enter__(self): return None
    def __exit__(self, *exc): return False

def main():
    ap = argparse.ArgumentParser("ST-GCN binary classifier over NPY sequences")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--filename_col", default="filename")
    ap.add_argument("--label_col", default="No inj/ inj")
    ap.add_argument("--out_dir", default="output_run_stgcn")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=16)  # локальный BS (умножается на #GPU)
    ap.add_argument("--downsample", type=int, default=1)
    ap.add_argument("--max_len", default="auto")           # "auto" -> 95 перцентиль
    ap.add_argument("--gpus", default="all")               # "all" | "cpu" | "0,1"
    ap.add_argument("--mixed_precision", action="store_true")
    ap.add_argument("--print_csv_preview", action="store_true",
                    help="Показать первые 5 строк CSV и частоты меток")
    ap.add_argument("--debug_index", action="store_true",
                    help="Печатать статус каждой строки при индексации")
    ap.add_argument("--peek", type=int, default=0,
                    help="Показать N успешно сопоставленных путей (форма массива)")
    # ST-GCN специфичные
    ap.add_argument("--nodes", type=int, default=None, help="Число узлов графа V (если известно)")
    ap.add_argument("--channels", type=int, default=None, help="Число каналов C на узел (если известно)")
    ap.add_argument("--edges_json", type=str, default=None, help="JSON со списком рёбер [[i,j], ...] или объектами {src,dst}")
    ap.add_argument("--stgcn_adjacency", choices=["identity","ring","full"], default="ring",
                    help="Предустановленная структура графа, если не указан edges_json")
    ap.add_argument("--temporal_kernel", type=int, default=9, help="Размер ядра по времени в ST-GCN")
    ap.add_argument("--stgcn_channels", type=str, default="128,64", help="Список каналов блоков, через запятую")
    ap.add_argument("--temporal_strides", type=str, default="2,2", help="Шаг по времени в каждом блоке, через запятую (downsample по времени)")
    ap.add_argument("--sample_for_norm", type=int, default=256, help="Сколько клипов брать для оценки mean/std (ускоряет старт)")

    args = ap.parse_args()

    # Видимость GPU до импорта TF
    if args.gpus.lower() == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif args.gpus.lower() != "all":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    # Индекс
    items_x, items_y, stats, skipped = build_items(
        args.csv, args.data_dir,
        filename_col=args.filename_col,
        label_col=args.label_col,
        debug_index=args.debug_index
    )
    assert items_x and items_y, "Не найдено валидных .npy и меток"
    items = list(zip(items_x, items_y))
    paths = items_x
    labels_all = np.array(items_y, dtype=np.int32)

    if args.print_csv_preview:
        print("\n=== CSV preview (first 5 rows) ===")
        df_preview = pd.read_csv(args.csv)
        print(df_preview.head(5).to_string(index=False))
        if "No inj/ inj" in df_preview.columns:
            print("\nLabel value counts in 'No inj/ inj':")
            print(df_preview["No inj/ inj"].value_counts(dropna=False))

    if args.peek > 0:
        print(f"\n=== Peek first {min(args.peek, len(items))} matched items ===")
        for (pth, lab) in items[:args.peek]:
            try:
                arr = np.load(pth, allow_pickle=False, mmap_mode="r")
                print(f"OK  label={lab} | {pth} | shape={arr.shape}")
            except Exception as e:
                print(f"FAIL to load for peek: {pth} -> {type(e).__name__}")

    # Статы по длине/размеру признака
    if str(args.max_len).strip().lower() == "auto":
        max_len, feat_dim = probe_stats(items, downsample=args.downsample, pctl=95)
        max_len = int(max(8, min(max_len, 20000)))
    else:
        max_len = int(args.max_len)
        aa = np.load(paths[0], allow_pickle=False, mmap_mode="r")
        feat_dim = int(aa.shape[1])
    print(f"max_len={max_len} | feat_dim={feat_dim}")

    # Выводим V, C
    V, C = infer_nodes_channels(feat_dim, channels=args.channels, nodes=args.nodes)
    print(f"Inferred graph: V={V} nodes, C={C} channels (V*C={V*C})")

    # Строим матрицу смежности
    if args.edges_json:
        edges = load_edges_json(args.edges_json, V)
        A = make_adjacency(V, preset="identity", edges=edges, self_loops=True)
        print(f"Adjacency: loaded from {args.edges_json}, edges={len(edges)}")
    else:
        A = make_adjacency(V, preset=args.stgcn_adjacency, edges=None, self_loops=True)
        print(f"Adjacency preset: {args.stgcn_adjacency}")

    # Сплит 70/10/20
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(paths))
    idx_train_full, idx_test = train_test_split(idx, test_size=0.20, random_state=42, stratify=labels_all)
    idx_train, idx_dev = train_test_split(idx_train_full, test_size=0.125, random_state=42, stratify=labels_all[idx_train_full])

    # Норм-статы по train
    ensure_dir(args.out_dir)
    mean, std = compute_norm_stats([items[i] for i in idx_train], feat_dim, args.downsample, max_len, sample_items=int(args.sample_for_norm))
    np.savez_compressed(os.path.join(args.out_dir, "norm_stats.npz"), mean=mean, std=std, max_len=max_len, V=V, C=C)

    # TF / стратегия
    tf, layers, models = lazy_tf()
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        try: tf.config.experimental.set_memory_growth(g, True)
        except Exception: pass

    if args.mixed_precision:
        from tensorflow.keras import mixed_precision as mp
        mp.set_global_policy("mixed_float16")  # выходной Dense уже float32

    strategy = tf.distribute.MirroredStrategy() if len(gpus) > 1 else None
    replicas = strategy.num_replicas_in_sync if strategy else 1
    global_bs = args.batch_size * replicas
    print(f"GPUs: {len(gpus)} | replicas: {replicas} | global_batch: {global_bs}")

    # Датасеты
    make_ds = make_datasets(paths, labels_all, max_len, feat_dim, V, C, global_bs,
                            args.downsample, mean, std, replicas)
    train_ds = make_ds(idx_train, shuffle=True,  drop_remainder=(replicas > 1))
    dev_ds   = make_ds(idx_dev,   shuffle=False, drop_remainder=False)
    test_ds  = make_ds(idx_test,  shuffle=False, drop_remainder=False)

    # Веса классов
    from sklearn.utils.class_weight import compute_class_weight
    cls = np.array([0, 1], dtype=np.int32)
    w = compute_class_weight("balanced", classes=cls, y=labels_all[idx_train])
    class_weight = {0: float(w[0]), 1: float(w[1])}
    print("class_weight:", class_weight)

    # Параметры ST-GCN
    channels_list = [int(x) for x in str(args.stgcn_channels).split(',') if str(x).strip()]
    # подготовим страйды по времени (если их меньше, добьём единицами)
    temporal_strides = [int(x) for x in str(args.temporal_strides).split(',') if str(x).strip()]
    if len(temporal_strides) < len(channels_list):
        temporal_strides += [1] * (len(channels_list) - len(temporal_strides))
    else:
        temporal_strides = temporal_strides[:len(channels_list)]

    # Строим модель
    ctx = strategy.scope() if strategy else contextlib_null()
    with ctx:
        model = build_stgcn_model(max_len, V, C, A_norm=A,
                                  learning_rate=1e-3,
                                  mixed_precision=args.mixed_precision,
                                  temporal_kernel=int(args.temporal_kernel),
                                  channels_list=channels_list,
                                  temporal_strides=temporal_strides,
                                  dropout=0.1)

    # короткая прогонка одной пачки — убедиться, что формы корректны
    for bx, by in train_ds.take(1):
        print("warmup batch:", bx.shape, by.shape)
        break

    # Коллбеки
    cb = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)
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
    thr = best_threshold_by_f1(y_dev, prob_dev)

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

if __name__ == "__main__":
    main()
