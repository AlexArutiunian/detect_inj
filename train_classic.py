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

# ===================== Фиксированная схема суставов =====================
DEFAULT_SCHEMA_JOINTS = [
  "L_foot_1","L_foot_2","L_foot_3","L_foot_4",
  "L_shank_1","L_shank_2","L_shank_3","L_shank_4",
  "L_thigh_1","L_thigh_2","L_thigh_3","L_thigh_4",
  "R_foot_1","R_foot_2","R_foot_3","R_foot_4",
  "R_shank_1","R_shank_2","R_shank_3","R_shank_4",
  "R_thigh_1","R_thigh_2","R_thigh_3","R_thigh_4",
  "pelvis_1","pelvis_2","pelvis_3","pelvis_4"
]
_JOINTS_DIM = len(DEFAULT_SCHEMA_JOINTS)  # 28
_AXIS_DIM = 3

# Имена фич для классики: 3 оси * 6 статистик на каждый сустав
_STAT_NAMES = ["mean","std","min","max","dmean","dstd"]
_AXIS = ["x","y","z"]

def classical_feature_names(schema_joints):
    names = []
    for j in schema_joints:
        for ax in _AXIS:
            for st in _STAT_NAMES:
                names.append(f"{j}:{ax}:{st}")
    return names

# ===================== Метки/утилиты =====================

def label_to_int(v):
    """Ровно две метки: 'Injury' -> 1, 'No Injury' -> 0."""
    if v is None:
        return None
    s = str(v).strip().lower()
    if s == "injury": return 1
    if s == "no injury": return 0
    return None

def compute_metrics(y_true, y_pred, y_prob):
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else float("nan"),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "report": classification_report(y_true, y_pred, digits=3)
    }
    return out

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def best_threshold_by_f1(y_true, proba):
    pr, rc, th = precision_recall_curve(y_true, proba)  # n+1, n+1, n
    f1_vals = 2 * pr[:-1] * rc[:-1] / np.clip(pr[:-1] + rc[:-1], 1e-12, None)
    if len(f1_vals) == 0 or np.all(np.isnan(f1_vals)): return 0.5
    best_idx = int(np.nanargmax(f1_vals))
    return float(th[max(0, min(best_idx, len(th) - 1))])

# ===================== Загрузка NPY (без JSON) =====================

def _resolve_npy_path(data_dir, fn):
    """Ищем .npy: как есть, иначе добавляем .npy."""
    p = os.path.join(data_dir, fn)
    if os.path.exists(p) and p.endswith(".npy"):
        return p
    if not p.endswith(".npy"):
        p2 = p + ".npy"
        if os.path.exists(p2):
            return p2
    return p if os.path.exists(p) else None

def _load_npy(path):
    """Грузит .npy. Поддерживает dict (object) или ndarray."""
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        try:
            arr = arr.item()
        except Exception:
            pass
    return arr

def _stack_motion_frames_with_schema(md, schema_joints):
    """
    Поддерживает:
      1) md: dict[joint] -> (T,3)
      2) md: ndarray (T, J, 3) или (T, 3*J) с J == len(schema_joints)
    Возвращает (T, 3*|schema|).
    """
    # dict: ожидаем ключи из schema_joints
    if isinstance(md, dict):
        present = [j for j in schema_joints if j in md]
        if not present: return None
        T = min(len(md[j]) for j in present)
        if T <= 1: return None
        cols = []
        for j in schema_joints:
            if j in md:
                arr = np.asarray(md[j], dtype=np.float32)[:T]
                if arr.ndim != 2 or arr.shape[1] != _AXIS_DIM:
                    raise ValueError(f"{j}: ожидалось (T,3), получено {arr.shape}")
            else:
                arr = np.full((T, _AXIS_DIM), np.nan, dtype=np.float32)
            cols.append(arr)
        return np.concatenate(cols, axis=1)

    # ndarray
    arr = np.asarray(md, dtype=np.float32)
    if arr.ndim == 3:
        T, J, C = arr.shape
        if C != _AXIS_DIM:
            raise ValueError(f"Ожидалось C=3, получено {C} в массиве {arr.shape}")
        if J != len(schema_joints):
            raise ValueError(f"Число суставов J={J} не совпадает со схемой ({len(schema_joints)}).")
        return arr.reshape(T, J * _AXIS_DIM)
    elif arr.ndim == 2:
        if arr.shape[1] % _AXIS_DIM != 0:
            raise ValueError(f"Вторая размерность должна делиться на 3, получено {arr.shape[1]}")
        J = arr.shape[1] // _AXIS_DIM
        if J != len(schema_joints):
            raise ValueError(f"Число суставов J={J} не совпадает со схемой ({len(schema_joints)}).")
        return arr
    else:
        return None

# ===================== Стриминг фич для классики (NPY only) =====================

def _feats_from_file(args):
    path, y, motion_key, schema_joints = args
    try:
        obj = _load_npy(path)
        # извлекаем "md" — либо dict с motion_key, либо сразу массив
        if isinstance(obj, dict):
            if motion_key not in obj: return None
            md = obj[motion_key]
        else:
            md = obj

        seq = _stack_motion_frames_with_schema(md, schema_joints)
        if seq is None or seq.shape[0] < 2:
            return None
        seq = seq.astype(np.float32, copy=False)
        dif = np.diff(seq, axis=0)
        stat = np.concatenate([
            np.nanmean(seq, axis=0), np.nanstd(seq, axis=0),
            np.nanmin(seq, axis=0), np.nanmax(seq, axis=0),
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
    Используется фиксированная схема (или переданная через --use_joints).
    """
    schema_joints = list(use_joints) if use_joints else list(DEFAULT_SCHEMA_JOINTS)
    print(f"Feature schema joints ({len(schema_joints)}): "
          f"{', '.join(schema_joints[:8])}{' ...' if len(schema_joints) > 8 else ''}")

    meta = pd.read_csv(csv_path, usecols=[filename_col, label_col])
    tasks = []
    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Indexing", dynamic_ncols=True, mininterval=0.2):
        fn = str(row[filename_col]).strip() if pd.notnull(row.get(filename_col)) else ""
        y = label_to_int(row.get(label_col))
        if not fn or y is None:
            continue
        p = _resolve_npy_path(data_dir, fn)
        if p and os.path.exists(p):
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
        raise RuntimeError("Не удалось получить ни одной строки фич (проверь .npy файлы и motion_key).")
    X = np.stack([r[0] for r in results]).astype(np.float32, copy=False)
    y = np.array([r[1] for r in results], dtype=np.int32)
    return X, y, schema_joints

# ===================== Обучение моделей (классика) =====================

def train_classical(X_train, X_dev, X_test, y_train, y_dev, y_test, model_type, out_dir):
    if model_type == "rf":
        model = RandomForestClassifier(n_estimators=400, n_jobs=-1, class_weight="balanced", random_state=42)
    elif model_type == "svm":
        model = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)
    elif model_type == "xgb":
        pos = int(np.sum(y_train == 1)); neg = int(np.sum(y_train == 0))
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

# ===================== Main =====================

def print_split_stats(name, y):
    n = len(y); n1 = int(np.sum(y == 1)); n0 = n - n1
    p1 = (n1 / n * 100) if n else 0.0; p0 = (n0 / n * 100) if n else 0.0
    print(f"[{name}] total={n} | Injury=1: {n1} ({p1:.1f}%) | No Injury=0: {n0} ({p0:.1f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Binary injury classifier from NPY 3D joint sequences (70/10/20 split)")
    parser.add_argument("--model", type=str, choices=["rf","svm","xgb"], required=True)
    parser.add_argument("--csv", type=str, required=True, help="Путь к CSV с метаданными")
    parser.add_argument("--data_dir", type=str, required=True, help="Папка с .npy файлами")
    parser.add_argument("--motion_key", type=str, default="running", help="ключ для dict внутри .npy, если используется")
    parser.add_argument("--filename_col", type=str, default="filename")
    parser.add_argument("--label_col", type=str, default="No inj/ inj")
    parser.add_argument("--use_joints", type=str, default="", help="кастомная схема: имя1,имя2,... (по умолчанию фиксированная)")
    parser.add_argument("--out_dir", type=str, default="outputs", help="куда сохранять модель/метрики")
    parser.add_argument("--loader_workers", type=int, default=0, help="кол-во процессов при извлечении фич")

    args = parser.parse_args()

    use_joints = None
    if args.use_joints.strip():
        use_joints = [s.strip() for s in args.use_joints.split(",") if s.strip()]

    out_dir = os.path.join(args.out_dir, args.model)
    ensure_dir(out_dir)

    # ===== Стриминг фич из .npy + сохранение схемы суставов =====
    X_all, y_all, schema_joints = load_features_streaming(
        args.csv, args.data_dir,
        motion_key=args.motion_key,
        filename_col=args.filename_col,
        label_col=args.label_col,
        use_joints=use_joints,
        workers=args.loader_workers
    )
    with open(os.path.join(out_dir, "schema_joints.json"), "w", encoding="utf-8") as f:
        json.dump(schema_joints, f, ensure_ascii=False, indent=2)

    # Сплит 70/20/10
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_all, y_all, test_size=0.20, random_state=42, stratify=y_all
    )
    X_train_feat, X_dev_feat, y_train, y_dev = train_test_split(
        X_train_full, y_train_full, test_size=0.125, random_state=42, stratify=y_train_full
    )

    # Сводка по классам
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

    # ===== Метрики и сохранение =====
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
