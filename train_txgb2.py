#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# train_xgb_gpu_feats.py — XGBoost для бинарной классификации injury/no-injury
# с ускорённым (опционально) расчётом агрегатных фич на GPU (CuPy/Torch).

import os, json, argparse
import numpy as np, pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             confusion_matrix, classification_report,
                             roc_curve, precision_recall_curve,
                             average_precision_score)
from xgboost import XGBClassifier
try:
    from xgboost import callback as xgb_callback
except Exception:
    xgb_callback = None

import joblib

# Matplotlib без GUI
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

def print_split_stats(name, y):
    n = len(y); n1 = int(np.sum(y==1)); n0 = n-n1
    p1 = (n1/n*100) if n else 0.0; p0 = (n0/n*100) if n else 0.0
    print(f"[{name}] total={n} | Injury=1: {n1} ({p1:.1f}%) | No Injury=0: {n0} ({p0:.1f}%)")

# ---- выбор бэкенда для фичей (CPU/ CuPy / Torch) ----
def _get_xpu(gpu_feats: str):
    gpu = (gpu_feats or "off").lower()
    if gpu == "cupy":
        import cupy as cp
        def to_numpy(x): return cp.asnumpy(x)
        return cp, to_numpy, "cupy"
    if gpu == "torch":
        import torch
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        class TorchXP:
            ndarray = torch.Tensor
            float32 = torch.float32
            def asarray(a, dtype=None):
                t = torch.as_tensor(a)
                if dtype is not None: t = t.to(dtype)
                return t.to(dev)
            def diff(a, axis=0):
                if axis != 0: raise ValueError("torch shim supports axis=0 only")
                return a[1:] - a[:-1]
            def nanmean(a, axis=0): return torch.nanmean(a, dim=axis)
            def nanstd(a, axis=0):  return torch.nanstd(a, dim=axis, unbiased=False)
            def nanmin(a, axis=0):  return torch.nanmin(a, dim=axis).values
            def nanmax(a, axis=0):  return torch.nanmax(a, dim=axis).values
            def concatenate(lst, axis=0): return torch.cat(lst, dim=axis)
            def isnan(a): return torch.isnan(a)
            def zeros(shape, dtype=None):
                t = torch.zeros(*shape, device=dev)
                return t.to(dtype) if dtype is not None else t
            def clip(a, a_min, a_max): return torch.clamp(a, min=a_min, max=a_max)
            def where(cond, x, y): return torch.where(cond, x, y)
            def astype(a, dtype): return a.to(dtype)
        def to_numpy(x): return x.detach().cpu().numpy()
        return TorchXP(), to_numpy, f"torch({dev})"
    import numpy as np
    def to_numpy(x): return x
    return np, to_numpy, "numpy(cpu)"

# ===================== ФИЧИ ИЗ NPY =====================
_STAT_NAMES = ["mean","std","min","max","dmean","dstd"]
_AXIS = ["x","y","z"]

def classical_feature_names(schema_joints, d_dim):
    if schema_joints:
        names=[]
        for j in schema_joints:
            for ax in _AXIS:
                for st in _STAT_NAMES:
                    names.append(f"{j}:{ax}:{st}")
        return names
    return [f"f_{i}" for i in range(d_dim)]

def _map_to_npy_path(npy_dir, rel_path):
    base = rel_path[:-5] if str(rel_path).endswith(".json") else str(rel_path)
    if not base.endswith(".npy"):
        base = base + ".npy"
    return os.path.join(npy_dir, base)

def _features_from_seq_cpu(seq_np: np.ndarray) -> np.ndarray:
    seq = np.nan_to_num(seq_np, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    if seq.ndim != 2 or seq.shape[0] < 2:
        return None
    dif = np.diff(seq, axis=0)
    stat = np.concatenate([
        np.nanmean(seq, axis=0), np.nanstd(seq, axis=0),
        np.nanmin(seq, axis=0),  np.nanmax(seq, axis=0),
        np.nanmean(dif, axis=0), np.nanstd(dif, axis=0)
    ], axis=0).astype(np.float32, copy=False)
    return stat

def _features_from_seq_backend(seq_np: np.ndarray, xp, to_numpy):
    a = xp.asarray(seq_np, dtype=xp.float32)
    # nan/inf -> 0
    if xp is np:
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        a = xp.where(xp.isnan(a), xp.zeros(a.shape, dtype=a.dtype), a)
        a = xp.clip(a, -1e30, 1e30)
    if getattr(a, "ndim", 2) != 2 or a.shape[0] < 2:
        return None
    dif = xp.diff(a, axis=0)
    stat = xp.concatenate([
        xp.nanmean(a, axis=0), xp.nanstd(a, axis=0),
        xp.nanmin(a, axis=0),  xp.nanmax(a, axis=0),
        xp.nanmean(dif, axis=0), xp.nanstd(dif, axis=0)
    ], axis=0)
    return to_numpy(stat).astype(np.float32, copy=False)

def _feats_from_npy_task(args):
    npy_path, y, downsample, gpu_feats = args
    xp, to_numpy, _ = _get_xpu(gpu_feats)
    try:
        arr = np.load(npy_path, allow_pickle=False, mmap_mode="r")  # чтение с CPU
        if arr.ndim != 2 or arr.shape[0] < 2:
            return None
        seq = arr[::downsample] if downsample > 1 else arr
        if gpu_feats == "off":
            feat = _features_from_seq_cpu(seq)
        else:
            feat = _features_from_seq_backend(seq, xp, to_numpy)
        if feat is None: return None
        return (feat, int(y))
    except Exception:
        return None

def load_features_from_npy(csv_path, npy_dir, filename_col="filename",
                           label_col="No inj/ inj", downsample=1, workers=0,
                           gpu_feats="off"):
    meta = pd.read_csv(csv_path, usecols=[filename_col, label_col])
    tasks=[]
    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Indexing(NPY)", dynamic_ncols=True, mininterval=0.2):
        fn = str(row[filename_col]).strip() if pd.notnull(row.get(filename_col)) else ""
        y = label_to_int(row.get(label_col))
        if not fn or y is None: continue
        p = _map_to_npy_path(npy_dir, fn)
        if os.path.exists(p): tasks.append((p, y, int(max(1, downsample)), gpu_feats))

    results=[]
    # GPU-вариант считаем в одном процессе (быстрее, чем делить один GPU на много процессов)
    if workers and workers>0 and gpu_feats == "off":
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

# ===================== ВИЗУАЛИЗАЦИИ / ОТЧЁТ =====================
CLASS_NAMES = ["No Injury (0)", "Injury (1)"]

def _plot_confusion(cm, title, path):
    cm = np.asarray(cm, dtype=np.int32)
    cmn = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(4.5,4))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_title(title); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(CLASS_NAMES, rotation=15, ha="right"); ax.set_yticklabels(CLASS_NAMES)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]}\n{cmn[i,j]:.2f}", ha="center", va="center", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)

def _plot_roc(y_true, y_prob, title, path):
    if len(np.unique(y_true))<2: return None
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5,4))
    ax.plot(fpr, tpr, label=f"AUC = {auc_val:.3f}"); ax.plot([0,1],[0,1],"--")
    ax.set_xlim([0,1]); ax.set_ylim([0,1])
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title(title); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
    return auc_val

def _plot_pr(y_true, y_prob, title, path):
    p, r, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5,4))
    ax.plot(r, p, label=f"AP = {ap:.3f}")
    ax.set_xlim([0,1]); ax.set_ylim([0,1])
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title(title); ax.legend(loc="lower left")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
    return ap

def _plot_f1_threshold(y_true, y_prob, thr, title, path):
    p, r, t = precision_recall_curve(y_true, y_prob)
    f1 = 2*p[:-1]*r[:-1]/np.clip(p[:-1]+r[:-1],1e-12,None)
    t = np.asarray(t)
    fig, ax = plt.subplots(figsize=(5.5,4))
    ax.plot(t, f1); ax.axvline(thr, linestyle='--', label=f"best thr={thr:.3f}")
    ax.set_xlabel("Threshold"); ax.set_ylabel("F1"); ax.set_title(title); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)

def _plot_feature_importance(model, feature_names, topn, title, path):
    if not hasattr(model, "feature_importances_"): return False
    imp = np.asarray(model.feature_importances_, dtype=float)
    idx = np.argsort(-imp)[:topn]
    names = [feature_names[i] for i in idx]; vals = imp[idx]
    fig, ax = plt.subplots(figsize=(7, max(3, 0.25*len(idx)+1)))
    ax.barh(range(len(idx)), vals)
    ax.set_yticks(range(len(idx))); ax.set_yticklabels(names); ax.invert_yaxis()
    ax.set_xlabel("Importance"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig); return True

def _write_html_report(out_dir, dev_metrics, test_metrics, images, thr):
    html=[]
    html.append("<html><head><meta charset='utf-8'><title>Training Report</title></head><body>")
    html.append("<h1>XGBoost report</h1>")
    html.append(f"<p><b>Best threshold (from DEV)</b>: {thr:.6f}</p>")
    for name, m in [("DEV metrics",dev_metrics), ("TEST metrics",test_metrics)]:
        html.append(f"<h2>{name}</h2><ul>")
        html.append(f"<li>accuracy: {m['accuracy']:.4f}</li>")
        html.append(f"<li>f1: {m['f1']:.4f}</li>")
        html.append(f"<li>roc_auc: {m['roc_auc']:.4f}</li></ul>")
    def img(title, key):
        if key in images:
            html.append(f"<h3>{title}</h3><img src='{os.path.basename(images[key])}' style='max-width:720px'>")
    img("DEV: Confusion Matrix","cm_dev"); img("DEV: ROC","roc_dev"); img("DEV: PR","pr_dev"); img("DEV: F1 vs thr","f1_dev")
    img("TEST: Confusion Matrix","cm_test"); img("TEST: ROC","roc_test"); img("TEST: PR","pr_test")
    img("Feature importance (top)","fi")
    html.append("</body></html>")
    with open(os.path.join(out_dir,"report.html"),"w",encoding="utf-8") as f:
        f.write("\n".join(html))

# ===================== ПОДБОР ПОРОГА =====================
def thr_max_tnr_at_min_recall(y_true, proba, min_recall1=0.8):
    ths = np.r_[0.0, np.unique(proba), 1.0]
    best_thr = 0.5; best_tnr = -1.0
    y_true = np.asarray(y_true)
    for t in ths:
        pred = (proba >= t).astype(int)
        TP = np.sum((y_true == 1) & (pred == 1))
        FN = np.sum((y_true == 1) & (pred == 0))
        TN = np.sum((y_true == 0) & (pred == 0))
        FP = np.sum((y_true == 0) & (pred == 1))
        rec1 = TP / max(TP + FN, 1)
        tnr  = TN / max(TN + FP, 1)
        if rec1 >= min_recall1 and tnr > best_tnr:
            best_tnr, best_thr = float(tnr), float(t)
    return best_thr, best_tnr

# ===================== ОБУЧЕНИЕ =====================
def train_xgb(
    X_train, X_dev, X_test, y_train, y_dev, y_test, out_dir,
    n_estimators=2000, learning_rate=0.05, max_depth=6,
    subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
    early_stopping_rounds=50, random_state=42,
    # веса классов через sample_weight (усиливаем класс 0)
    w0=2.0, w1=1.0,
    # ограничение при подборе порога
    min_recall1=0.93
):
    pos, neg = int(np.sum(y_train == 1)), int(np.sum(y_train == 0))
    spw = (neg / max(pos, 1)) if pos > 0 else 1.0

    model = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        scale_pos_weight=spw,
        random_state=random_state,
        n_jobs=-1
    )

    sw_train = np.where(y_train == 0, w0, w1).astype(np.float32)
    sw_dev   = np.where(y_dev   == 0, w0, w1).astype(np.float32)

    use_best = False
    try:
        if xgb_callback is not None:
            es = xgb_callback.EarlyStopping(rounds=early_stopping_rounds, save_best=True, maximize=True)
            try:
                model.fit(X_train, y_train,
                          sample_weight=sw_train,
                          eval_set=[(X_dev, y_dev)],
                          eval_sample_weight=[sw_dev],
                          callbacks=[es], verbose=False)
            except TypeError:
                model.fit(X_train, y_train,
                          sample_weight=sw_train,
                          eval_set=[(X_dev, y_dev)],
                          callbacks=[es], verbose=False)
            use_best = True
        else:
            raise TypeError("callbacks not available")
    except TypeError:
        try:
            try:
                model.fit(X_train, y_train,
                          sample_weight=sw_train,
                          eval_set=[(X_dev, y_dev)],
                          eval_sample_weight=[sw_dev],
                          early_stopping_rounds=early_stopping_rounds, verbose=False)
            except TypeError:
                model.fit(X_train, y_train,
                          sample_weight=sw_train,
                          eval_set=[(X_dev, y_dev)],
                          early_stopping_rounds=early_stopping_rounds, verbose=False)
            use_best = True
        except TypeError:
            print("[warn] early stopping недоступен — обучаю без него.")
            model.fit(X_train, y_train, sample_weight=sw_train, verbose=False)

    # предсказания с учётом best_iteration
    kw={}
    if use_best:
        bi = getattr(model, "best_iteration", None)
        if bi is None: bi = getattr(model, "best_iteration_", None)
        if bi is not None:
            try: kw={"iteration_range": (0, int(bi)+1)}
            except TypeError: kw={}
        if not kw:
            bntl = getattr(model, "best_ntree_limit", None)
            if bntl is not None: kw={"ntree_limit": int(bntl)}

    prob_dev  = model.predict_proba(X_dev,  **kw)[:, 1]
    thr, _    = thr_max_tnr_at_min_recall(y_dev, prob_dev, min_recall1=min_recall1)
    pred_dev  = (prob_dev >= thr).astype(int)
    dev_metrics = compute_metrics(y_dev, pred_dev, prob_dev)

    prob_test = model.predict_proba(X_test, **kw)[:, 1]
    pred_test = (prob_test >= thr).astype(int)
    test_metrics = compute_metrics(y_test, pred_test, prob_test)

    with open(os.path.join(out_dir, "threshold.txt"), "w") as f:
        f.write(str(thr))

    return dev_metrics, test_metrics, thr, model, prob_dev, prob_test

# ===================== MAIN =====================
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="XGBoost injury/no-injury (NPY -> aggregated features, GPU feats optional)")
    # данные
    p.add_argument("--csv", required=True, type=str)
    p.add_argument("--data_dir", required=True, type=str, help="папка с .npy")
    p.add_argument("--filename_col", default="filename", type=str)
    p.add_argument("--label_col", default="No inj/ inj", type=str)
    p.add_argument("--schema_json", default="", type=str, help="список суставов (для имён фич, опционально)")
    p.add_argument("--out_dir", default="outputs/xgb", type=str)
    p.add_argument("--downsample", default=1, type=int)
    p.add_argument("--loader_workers", default=0, type=int)
    p.add_argument("--gpu_feats", type=str, default="off", choices=["off","cupy","torch"],
                   help="считать фичи на GPU (cupy/torch) или на CPU (off)")

    # XGB гиперпараметры
    p.add_argument("--n_estimators", type=int, default=2000)
    p.add_argument("--early_stopping_rounds", type=int, default=50)
    p.add_argument("--learning_rate", type=float, default=0.05)
    p.add_argument("--max_depth", type=int, default=6)
    p.add_argument("--subsample", type=float, default=0.9)
    p.add_argument("--colsample_bytree", type=float, default=0.9)
    p.add_argument("--reg_lambda", type=float, default=1.0)

    # веса классов в sample_weight
    p.add_argument("--w0", type=float, default=2.0, help="вес класса 0 (No injury)")
    p.add_argument("--w1", type=float, default=1.0, help="вес класса 1 (Injury)")

    # подбор порога
    p.add_argument("--min_recall1", type=float, default=0.93, help="ограничение на recall класса 1 при выборе порога")

    # визуализации
    p.add_argument("--top_features", type=int, default=30)
    p.add_argument("--show_plots", action="store_true")

    args = p.parse_args()
    ensure_dir(args.out_dir)

    # (опц) имена фич по суставам
    schema = None
    if args.schema_json and os.path.exists(args.schema_json):
        with open(args.schema_json, "r", encoding="utf-8") as f:
            schema = json.load(f)

    print(f"[info] gpu_feats backend: { _get_xpu(args.gpu_feats)[2] }")

    # 1) Фичи
    X_all, y_all = load_features_from_npy(
        args.csv, args.data_dir,
        filename_col=args.filename_col, label_col=args.label_col,
        downsample=args.downsample, workers=args.loader_workers,
        gpu_feats=args.gpu_feats
    )

    # 2) Сплиты 70/10/20
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_all, y_all, test_size=0.20, random_state=42, stratify=y_all
    )
    X_train, X_dev, y_train, y_dev = train_test_split(
        X_train_full, y_train_full, test_size=0.125, random_state=42, stratify=y_train_full
    )

    print("\n=== Split stats ===")
    print_split_stats("TRAIN (≈70%)", y_train)
    print_split_stats("DEV   (≈10%)", y_dev)
    print_split_stats("TEST  (≈20%)", y_test)

    # >>> сохраняем CSV для теста <<<
    meta = pd.read_csv(args.csv, usecols=[args.filename_col, args.label_col])
    # Переведём метки в int (0/1)
    meta["label_int"] = meta[args.label_col].map(label_to_int)
    # Чтобы правильно выбрать test, надо опираться на индексы
    # train_test_split возвращает массивы, сохраним индексы явно
    _, test_idx = train_test_split(
        np.arange(len(meta)), test_size=0.20, random_state=42, stratify=meta["label_int"]
    )
    test_meta = meta.iloc[test_idx].copy()
    test_meta.to_csv(os.path.join(args.out_dir, "test_manifest.csv"), index=False)
    print(f"[ok] Test manifest saved: {os.path.join(args.out_dir, 'test_manifest.csv')}")


    # 3) Нормализация
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_dev_s   = scaler.transform(X_dev)
    X_test_s  = scaler.transform(X_test)

    # 4) Обучение XGB
    dev_metrics, test_metrics, thr, model, prob_dev, prob_test = train_xgb(
        X_train_s, X_dev_s, X_test_s, y_train, y_dev, y_test,
        out_dir=args.out_dir,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        early_stopping_rounds=args.early_stopping_rounds,
        random_state=42,
        w0=args.w0, w1=args.w1,
        min_recall1=args.min_recall1
    )

    # 5) Сохранения
    joblib.dump({"model": model, "scaler": scaler}, os.path.join(args.out_dir, "model.joblib"))
    with open(os.path.join(args.out_dir,"metrics_dev.json"),"w",encoding="utf-8") as f:
        json.dump(dev_metrics, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out_dir,"metrics_test.json"),"w",encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)

    # 6) Визуализации
    images={}
    _plot_confusion(np.array(dev_metrics["confusion_matrix"]), "DEV Confusion Matrix",  os.path.join(args.out_dir,"cm_dev.png"));  images["cm_dev"]=os.path.join(args.out_dir,"cm_dev.png")
    _plot_confusion(np.array(test_metrics["confusion_matrix"]), "TEST Confusion Matrix", os.path.join(args.out_dir,"cm_test.png")); images["cm_test"]=os.path.join(args.out_dir,"cm_test.png")
    _plot_roc(y_dev, prob_dev, "DEV ROC", os.path.join(args.out_dir,"roc_dev.png"));     images["roc_dev"]=os.path.join(args.out_dir,"roc_dev.png")
    _plot_pr(y_dev,  prob_dev, "DEV PR",  os.path.join(args.out_dir,"pr_dev.png"));      images["pr_dev"]=os.path.join(args.out_dir,"pr_dev.png")
    _plot_f1_threshold(y_dev, prob_dev, thr, "DEV F1 vs threshold", os.path.join(args.out_dir,"f1_dev.png")); images["f1_dev"]=os.path.join(args.out_dir,"f1_dev.png")
    _plot_roc(y_test, prob_test, "TEST ROC", os.path.join(args.out_dir,"roc_test.png")); images["roc_test"]=os.path.join(args.out_dir,"roc_test.png")
    _plot_pr(y_test,  prob_test,"TEST PR",  os.path.join(args.out_dir,"pr_test.png"));   images["pr_test"]=os.path.join(args.out_dir,"pr_test.png")

    feat_names = classical_feature_names(schema, X_train_s.shape[1])
    if _plot_feature_importance(model, feat_names, args.top_features,
                                "Feature importance (top)",
                                os.path.join(args.out_dir,"feature_importance_top.png")):
        images["fi"]=os.path.join(args.out_dir,"feature_importance_top.png")

    _write_html_report(args.out_dir, dev_metrics, test_metrics, images, thr)

    # 7) Лог
    print("\n=== DEV METRICS (threshold tuned here) ===")
    for k in ["accuracy","f1","roc_auc","confusion_matrix"]:
        print(f"{k}: {dev_metrics[k]}")
    print("\nDev classification report:\n", dev_metrics["report"])

    print("\n=== TEST METRICS (using dev-tuned threshold) ===")
    for k in ["accuracy","f1","roc_auc","confusion_matrix"]:
        print(f"{k}: {test_metrics[k]}")
    print("\nTest classification report:\n", test_metrics["report"])

    print(f"\nSaved to: {args.out_dir}")
    print(f"Report: {os.path.join(args.out_dir, 'report.html')}")
