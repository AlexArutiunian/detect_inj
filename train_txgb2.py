#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# XGBoost binary (injury / no-injury) over NPY sequences — rich features + constrained thresholding

import os, json, argparse, math, warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             confusion_matrix, classification_report,
                             roc_curve, precision_recall_curve,
                             average_precision_score)

from xgboost import XGBClassifier
try:
    from xgboost import callback as xgb_callback  # может отсутствовать в старых версиях
except Exception:
    xgb_callback = None

import joblib

# ---- Matplotlib (без GUI) ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============== УТИЛИТЫ ===============
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def label_to_int(v):
    if v is None: return None
    s = str(v).strip().lower()
    if s in ("injury","1"): return 1
    if s in ("no injury","0"): return 0
    return None

def print_split_stats(name, y):
    n = len(y); n1 = int(np.sum(y==1)); n0 = n-n1
    p1 = (n1/n*100) if n else 0.0; p0 = (n0/n*100) if n else 0.0
    print(f"[{name}] total={n} | Injury=1: {n1} ({p1:.1f}%) | No Injury=0: {n0} ({p0:.1f}%)")

def compute_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else float("nan"),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "report": classification_report(y_true, y_pred, digits=3),
    }

def choose_threshold_constrained(
    y_true, proba,
    min_recall1=0.90,
    min_precision1=None,        # если None — не ограничиваем precision
    prefer="tnr",               # "tnr" | "f1" | "balacc"
):
    """
    Возвращает (thr, info_dict): порог, дающий max prefer среди порогов,
    где recall(1) >= min_recall1 и (при наличии) precision(1) >= min_precision1.
    Если таких нет — берём лучший компромисс (штрафуем недобор recall/precision).
    """
    y_true = np.asarray(y_true).astype(int)
    proba  = np.asarray(proba, dtype=float)
    ths = np.r_[0.0, np.unique(proba), 1.0]

    def stats_at(t):
        pred = (proba >= t).astype(int)
        TP = np.sum((y_true == 1) & (pred == 1))
        FN = np.sum((y_true == 1) & (pred == 0))
        TN = np.sum((y_true == 0) & (pred == 0))
        FP = np.sum((y_true == 0) & (pred == 1))
        rec1 = TP / max(TP + FN, 1)
        prec1 = TP / max(TP + FP, 1)
        tnr = TN / max(TN + FP, 1)
        f1 = (2*prec1*rec1) / max(prec1 + rec1, 1e-12)
        balacc = 0.5 * (tnr + rec1)
        return dict(thr=float(t), TP=int(TP), FN=int(FN), TN=int(TN), FP=int(FP),
                    recall1=float(rec1), precision1=float(prec1),
                    tnr=float(tnr), f1=float(f1), balacc=float(balacc))

    stats = [stats_at(t) for t in ths]

    feasible = []
    for s in stats:
        ok_rec = (s["recall1"] >= min_recall1)
        ok_pre = (True if (min_precision1 is None) else (s["precision1"] >= min_precision1))
        if ok_rec and ok_pre:
            feasible.append(s)

    keymap = {"tnr":"tnr","f1":"f1","balacc":"balacc"}
    key = keymap.get(prefer, "tnr")

    if feasible:
        best = max(feasible, key=lambda s: s[key])
        best["feasible"] = True
        return best["thr"], best

    # --- компромисс: штрафуем недобор recall/precision, вознаграждаем tnr/recall
    w_rec_short, w_prec_short, w_tnr_reward, w_rec_reward = 10.0, 5.0, 1.0, 0.2
    def compromise_score(s):
        v_rec  = max(0.0, min_recall1   - s["recall1"])
        v_prec = max(0.0, (min_precision1 if min_precision1 is not None else 0.0) - s["precision1"])
        return -(w_rec_short*v_rec + w_prec_short*v_prec) + (w_tnr_reward*s["tnr"] + w_rec_reward*s["recall1"])
    best = max(stats, key=compromise_score)
    best["feasible"] = False
    return best["thr"], best

# =============== ФИЧИ ИЗ NPY (расширенные) ===============
# Базовые статистики по каналу
def _basic_stats(x):
    # x: (T,)
    x = np.asarray(x, dtype=np.float32)
    dx = np.diff(x) if x.size > 1 else np.array([0.0], dtype=np.float32)

    def safe_std(v): return float(np.nanstd(v)) if v.size else 0.0
    def safe_mean(v): return float(np.nanmean(v)) if v.size else 0.0
    def safe_min(v): return float(np.nanmin(v)) if v.size else 0.0
    def safe_max(v): return float(np.nanmax(v)) if v.size else 0.0

    # квантили
    def q(v, q): return float(np.nanpercentile(v, q)) if v.size else 0.0

    # форма распределения
    def safe_skew(v):
        v = v[np.isfinite(v)]
        if v.size < 3: return 0.0
        m = v.mean(); s = v.std()
        return float(np.mean(((v - m) / (s + 1e-12))**3))
    def safe_kurt(v):
        v = v[np.isfinite(v)]
        if v.size < 4: return 0.0
        m = v.mean(); s = v.std()
        return float(np.mean(((v - m) / (s + 1e-12))**4) - 3.0)

    # автокорреляция lag k
    def acf(v, k=1):
        v = v[np.isfinite(v)]
        n = v.size
        if n <= k or n < 2: return 0.0
        m = v.mean()
        v0 = v - m
        num = np.sum(v0[:-k] * v0[k:])
        den = np.sum(v0 * v0) + 1e-12
        return float(num / den)

    # оконные куски
    def window_stats(v):
        n = v.size
        if n < 3:
            return [safe_mean(v), safe_std(v), safe_min(v), safe_max(v)]*3
        a = v[:n//3]; b = v[n//3:2*n//3]; c = v[2*n//3:]
        out=[]
        for w in (a,b,c):
            out.extend([safe_mean(w), safe_std(w), safe_min(w), safe_max(w)])
        return out

    out = [
        safe_mean(x), safe_std(x), safe_min(x), safe_max(x),
        q(x,25), q(x,50), q(x,75),
        safe_skew(x), safe_kurt(x),
        safe_mean(dx), safe_std(dx), safe_min(dx), safe_max(dx),
        acf(x,1), acf(x,2),
    ]
    out.extend(window_stats(x))  # 12 дополнительных
    return np.array(out, dtype=np.float32)  # итого 15 + 12 = 27

def _spectral_stats(x, sr=1.0):
    # простые спектральные признаки: энергия в 3 диапазонах + спектральная энтропия
    x = np.asarray(x, dtype=np.float32)
    x = x - np.nanmean(x)
    n = x.size
    if n < 4: return np.zeros(4, dtype=np.float32)
    # односторонний спектр
    fft = np.fft.rfft(x * np.hanning(n))
    pxx = (fft.real**2 + fft.imag**2)
    pxx = np.clip(pxx, 1e-12, None)
    # три диапазона равной ширины
    k = pxx.size
    b = k // 3
    e1 = float(pxx[:b].sum())
    e2 = float(pxx[b:2*b].sum())
    e3 = float(pxx[2*b:].sum())
    p = pxx / pxx.sum()
    sent = float(-(p * np.log(p)).sum())
    return np.array([e1,e2,e3,sent], dtype=np.float32)

def _extract_features(seq_2d):
    """
    seq_2d: (T, F)
    На каждый канал → 27 (basic+windows) + 4 (spectral) = 31 фича.
    """
    T, F = seq_2d.shape
    feats = []
    for j in range(F):
        x = np.asarray(seq_2d[:, j], dtype=np.float32)
        feats.append(_basic_stats(x))
        feats.append(_spectral_stats(x))
    return np.concatenate(feats, axis=0).astype(np.float32, copy=False)  # (F*31,)

def _map_to_npy_path(npy_dir, rel_path):
    # filename может быть *.json — ищем одноимённый .npy
    base = str(rel_path)
    if base.endswith(".json"):
        base = base[:-5]
    if not base.endswith(".npy"):
        base = base + ".npy"
    return os.path.join(npy_dir, base)

def _sanitize_seq(a: np.ndarray, downsample: int) -> np.ndarray:
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    if a.ndim == 1:
        a = a[:, None]
    elif a.ndim >= 2:
        t_axis = int(np.argmax(a.shape))
        if t_axis != 0:
            a = np.moveaxis(a, t_axis, 0)
        if a.ndim > 2:
            a = a.reshape(a.shape[0], -1)
    if downsample > 1:
        a = a[::downsample]
    return a.astype(np.float32, copy=False)

def _feats_from_path_task(args):
    npy_path, y, downsample = args
    try:
        arr = np.load(npy_path, allow_pickle=False, mmap_mode="r")
        x = _sanitize_seq(arr, downsample)
        if x.ndim != 2 or x.shape[0] < 2:
            return None
        f = _extract_features(x)
        return (f, int(y))
    except Exception:
        return None

def load_features_from_npy(csv_path, npy_dir, filename_col="filename",
                           label_col="No inj/ inj", downsample=1, workers=0):
    meta = pd.read_csv(csv_path, usecols=[filename_col, label_col])
    tasks=[]
    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Indexing(NPY)", dynamic_ncols=True, mininterval=0.2):
        fn = str(row[filename_col]).strip() if pd.notnull(row.get(filename_col)) else ""
        y = label_to_int(row.get(label_col))
        if not fn or y is None: continue
        p = _map_to_npy_path(npy_dir, fn)
        if os.path.exists(p): tasks.append((p, y, int(max(1, downsample))))
    results=[]
    if workers and workers>0:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for r in tqdm(ex.map(_feats_from_path_task, tasks, chunksize=16),
                          total=len(tasks), desc="Feats(NPY)", dynamic_ncols=True, mininterval=0.2):
                if r is not None: results.append(r)
    else:
        for t in tqdm(tasks, desc="Feats(NPY)", dynamic_ncols=True, mininterval=0.2):
            r = _feats_from_path_task(t)
            if r is not None: results.append(r)

    if not results:
        raise RuntimeError("Не удалось получить ни одной строки фич из NPY.")

    X = np.stack([r[0] for r in results]).astype(np.float32, copy=False)
    y = np.array([r[1] for r in results], dtype=np.int32)
    return X, y

# =============== ВИЗУАЛИЗАЦИИ / ОТЧЁТ ===============
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

# =============== ОБУЧЕНИЕ ===============
def train_xgb(
    X_train, X_dev, X_test, y_train, y_dev, y_test, out_dir,
    # бустинг (адекватные дефолты)
    n_estimators=2000, learning_rate=0.03, max_depth=6,
    subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, reg_alpha=0.0,
    min_child_weight=5.0, gamma=0.5, max_delta_step=1.0,
    # ES/seed
    early_stopping_rounds=50, random_state=42,
    # веса классов
    w0=2.0, w1=1.0, use_scale_pos_weight=True,
    # порог
    thr_mode="tnr_constrained", min_recall1=0.90, min_precision1=None, fixed_thr=None,
    thr_prefer="tnr",
    # калибровка вероятностей
    calibrate="none"  # 'none' | 'isotonic' | 'platt'
):
    # scale_pos_weight из дисбаланса train
    spw = 1.0
    if use_scale_pos_weight:
        pos, neg = int(np.sum(y_train == 1)), int(np.sum(y_train == 0))
        spw = (neg / max(pos, 1)) if pos > 0 else 1.0

    model = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        min_child_weight=min_child_weight,
        gamma=gamma,
        max_delta_step=max_delta_step,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        scale_pos_weight=spw,
        random_state=random_state,
        n_jobs=-1
    )

    # sample_weight усиливает класс 0 (No Injury)
    sw_train = np.where(y_train == 0, w0, w1).astype(np.float32)
    sw_dev   = np.where(y_dev   == 0, w0, w1).astype(np.float32)

    # ---- fit с поддержкой и без callbacks / eval_sample_weight
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

    # ---- получить предсказания c учётом best_iteration
    kw = {}
    if use_best:
        bi = getattr(model, "best_iteration", None)
        if bi is None:
            bi = getattr(model, "best_iteration_", None)
        if bi is not None:
            try:
                kw = {"iteration_range": (0, int(bi)+1)}
            except TypeError:
                kw = {}
        if not kw:
            bntl = getattr(model, "best_ntree_limit", None)
            if bntl is not None:
                kw = {"ntree_limit": int(bntl)}

    # ---- калибровка вероятностей по DEV (по желанию)
    calibrator = None
    def _apply_cal(p): return calibrator.transform(p) if calibrator is not None else p

    prob_dev_raw  = model.predict_proba(X_dev,  **kw)[:, 1]
    if calibrate.lower() == "isotonic":
        try:
            from sklearn.isotonic import IsotonicRegression
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(prob_dev_raw, y_dev)
        except Exception as e:
            print(f"[warn] isotonic calibration disabled: {e}")
    elif calibrate.lower() == "platt":
        try:
            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression(max_iter=1000)
            lr.fit(prob_dev_raw.reshape(-1,1), y_dev)
            class _LRWrap:
                def __init__(self, lr): self.lr = lr
                def transform(self, p):
                    return self.lr.predict_proba(p.reshape(-1,1))[:,1]
            calibrator = _LRWrap(lr)
        except Exception as e:
            print(f"[warn] platt calibration disabled: {e}")

    prob_dev = _apply_cal(prob_dev_raw)

    # ---- подбор порога
    if fixed_thr is not None:
        thr = float(fixed_thr)
        thr_info = {"thr": thr, "feasible": True}
    else:
        prefer = thr_prefer if thr_mode == "tnr_constrained" else "f1"
        thr, thr_info = choose_threshold_constrained(
            y_dev, prob_dev,
            min_recall1=min_recall1,
            min_precision1=(min_precision1 if thr_mode == "tnr_constrained" else None),
            prefer=prefer,
        )

    print(f"[thr] feasible={thr_info.get('feasible')} thr={thr:.4f} | "
          f"TNR={thr_info.get('tnr', float('nan')):.3f} | "
          f"R1={thr_info.get('recall1', float('nan')):.3f} | "
          f"P1={thr_info.get('precision1', float('nan')):.3f}")

    with open(os.path.join(out_dir, "threshold.txt"), "w") as f:
        f.write(str(thr))
    with open(os.path.join(out_dir, "threshold_info.json"), "w", encoding="utf-8") as f:
        json.dump(thr_info, f, ensure_ascii=False, indent=2)
    if calibrator is not None:
        joblib.dump(calibrator, os.path.join(out_dir, "calibrator.joblib"))

    # ---- метрики DEV/TEST
    pred_dev = (prob_dev >= thr).astype(int)
    dev_metrics = compute_metrics(y_dev, pred_dev, prob_dev)

    prob_test_raw = model.predict_proba(X_test, **kw)[:, 1]
    prob_test = _apply_cal(prob_test_raw)
    pred_test = (prob_test >= thr).astype(int)
    test_metrics = compute_metrics(y_test, pred_test, prob_test)

    return dev_metrics, test_metrics, thr, model, prob_dev, prob_test

# =============== MAIN ===============
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="XGBoost injury/no-injury (NPY -> rich aggregated features)")

    # пути/данные
    p.add_argument("--csv", required=True, type=str)
    p.add_argument("--data_dir", required=True, type=str, help="папка с .npy")
    p.add_argument("--filename_col", default="filename", type=str)
    p.add_argument("--label_col", default="No inj/ inj", type=str)
    p.add_argument("--out_dir", default="outputs/xgb_plus", type=str)
    p.add_argument("--downsample", default=1, type=int)
    p.add_argument("--loader_workers", default=0, type=int)

    # сплиты
    p.add_argument("--test_size", type=float, default=0.20)
    p.add_argument("--dev_size_from_train", type=float, default=0.125)
    p.add_argument("--seed", type=int, default=42)

    # XGB гиперпараметры
    p.add_argument("--n_estimators", type=int, default=2000)
    p.add_argument("--early_stopping_rounds", type=int, default=50)
    p.add_argument("--learning_rate", type=float, default=0.03)
    p.add_argument("--max_depth", type=int, default=6)
    p.add_argument("--subsample", type=float, default=0.9)
    p.add_argument("--colsample_bytree", type=float, default=0.9)
    p.add_argument("--reg_lambda", type=float, default=1.0)
    p.add_argument("--reg_alpha", type=float, default=0.0)
    p.add_argument("--min_child_weight", type=float, default=5.0)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--max_delta_step", type=float, default=1.0)
    p.add_argument("--use_scale_pos_weight", action="store_true")

    # веса классов в sample_weight
    p.add_argument("--w0", type=float, default=2.0, help="вес класса 0 (No injury)")
    p.add_argument("--w1", type=float, default=1.0, help="вес класса 1 (Injury)")

    # стратегия порога
    p.add_argument("--thr_mode", type=str, default="tnr_constrained", choices=["f1","tnr_constrained"])
    p.add_argument("--min_recall1", type=float, default=0.90, help="ограничение на recall класса 1")
    p.add_argument("--min_precision1", type=float, default=None, help="минимальный precision класса 1 (используется при tnr_constrained, опционально)")
    p.add_argument("--fixed_thr", type=float, default=None, help="если задано — использовать фиксированный порог")
    p.add_argument("--thr_prefer", type=str, default="tnr", choices=["tnr","f1","balacc"])

    # визуализации
    p.add_argument("--top_features", type=int, default=30)
    p.add_argument("--show_plots", action="store_true")

    args = p.parse_args()
    ensure_dir(args.out_dir)

    # 1) Фичи
    X_all, y_all = load_features_from_npy(
        args.csv, args.data_dir,
        filename_col=args.filename_col, label_col=args.label_col,
        downsample=args.downsample, workers=args.loader_workers
    )

    # 2) Сплиты 70/10/20
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_all, y_all, test_size=args.test_size, random_state=args.seed, stratify=y_all
    )
    X_train, X_dev, y_train, y_dev = train_test_split(
        X_train_full, y_train_full, test_size=args.dev_size_from_train, random_state=args.seed, stratify=y_train_full
    )

    print("\n=== Split stats ===")
    print_split_stats("TRAIN (≈70%)", y_train)
    print_split_stats("DEV   (≈10%)", y_dev)
    print_split_stats("TEST  (≈20%)", y_test)

    # 3) Нормализация
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_dev_s   = scaler.transform(X_dev)
    X_test_s  = scaler.transform(X_test)

    # 4) Обучение XGB + порог
    dev_metrics, test_metrics, thr, model, prob_dev, prob_test = train_xgb(
        X_train_s, X_dev_s, X_test_s, y_train, y_dev, y_test,
        out_dir=args.out_dir,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        reg_alpha=args.reg_alpha,
        min_child_weight=args.min_child_weight,
        gamma=args.gamma,
        max_delta_step=args.max_delta_step,
        early_stopping_rounds=args.early_stopping_rounds,
        random_state=args.seed,
        w0=args.w0, w1=args.w1, use_scale_pos_weight=args.use_scale_pos_weight,
        thr_mode=args.thr_mode, min_recall1=args.min_recall1, min_precision1=args.min_precision1,
        fixed_thr=args.fixed_thr, thr_prefer=args.thr_prefer,
        calibrate="none"
    )

    # 5) Сохранения
    joblib.dump({"model": model, "scaler": scaler}, os.path.join(args.out_dir, "model.joblib"))
    with open(os.path.join(args.out_dir,"metrics_dev.json"),"w",encoding="utf-8") as f:
        json.dump(dev_metrics, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out_dir,"metrics_test.json"),"w",encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)

    # 6) Визуализации
    kw={}
    if hasattr(model,"best_iteration_") or hasattr(model,"best_iteration"):
        bi = getattr(model,"best_iteration_", None)
        if bi is None: bi = getattr(model,"best_iteration", None)
        if bi is not None:
            try: kw={"iteration_range":(0, int(bi)+1)}
            except TypeError: kw={}
    prob_dev  = model.predict_proba(X_dev_s,  **kw)[:,1]
    prob_test = model.predict_proba(X_test_s, **kw)[:,1]

    images={}
    _plot_confusion(np.array(dev_metrics["confusion_matrix"]), "DEV Confusion Matrix",  os.path.join(args.out_dir,"cm_dev.png"));  images["cm_dev"]=os.path.join(args.out_dir,"cm_dev.png")
    _plot_confusion(np.array(test_metrics["confusion_matrix"]), "TEST Confusion Matrix", os.path.join(args.out_dir,"cm_test.png")); images["cm_test"]=os.path.join(args.out_dir,"cm_test.png")
    _plot_roc(y_dev, prob_dev, "DEV ROC", os.path.join(args.out_dir,"roc_dev.png"));     images["roc_dev"]=os.path.join(args.out_dir,"roc_dev.png")
    _plot_pr(y_dev,  prob_dev, "DEV PR",  os.path.join(args.out_dir,"pr_dev.png"));      images["pr_dev"]=os.path.join(args.out_dir,"pr_dev.png")
    _plot_f1_threshold(y_dev, prob_dev, thr, "DEV F1 vs threshold", os.path.join(args.out_dir,"f1_dev.png")); images["f1_dev"]=os.path.join(args.out_dir,"f1_dev.png")
    _plot_roc(y_test, prob_test, "TEST ROC", os.path.join(args.out_dir,"roc_test.png")); images["roc_test"]=os.path.join(args.out_dir,"roc_test.png")
    _plot_pr(y_test,  prob_test,"TEST PR",  os.path.join(args.out_dir,"pr_test.png"));   images["pr_test"]=os.path.join(args.out_dir,"pr_test.png")

    

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
