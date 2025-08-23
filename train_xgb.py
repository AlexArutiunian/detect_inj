#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, argparse, random, re, warnings, itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore", category=UserWarning)

# =============== GLOBAL (для форков) ===============
schema_path_global = None

# =============== Utils ===============
def seed_all(s=42):
    random.seed(s); np.random.seed(s)

def label_to_int(v):
    if v is None: return None
    s = str(v).strip().lower()
    if s.startswith("inj"): return 1
    if s.startswith("no"):  return 0
    if s in ("1","0"): return int(s)
    return None

def guess_subject_id(filename: str) -> str:
    b = Path(filename).stem
    m = re.match(r"(\d{8})", b)   # напр. 20120717...
    return m.group(1) if m else b.split("T")[0][:8]

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

# =============== Schema + I/O ===============
@dataclass
class Schema:
    names: List[str]
    groups: Dict[str, List[int]]

def load_schema(schema_json: str) -> Schema:
    names = json.load(open(schema_json, "r", encoding="utf-8"))
    def idxs(prefix):
        return [i for i, n in enumerate(names) if n.lower().startswith(prefix)]
    groups = {
        "pelvis":  idxs("pelvis"),
        "L_foot":  idxs("l_foot"),
        "L_shank": idxs("l_shank"),
        "L_thigh": idxs("l_thigh"),
        "R_foot":  idxs("r_foot"),
        "R_shank": idxs("r_shank"),
        "R_thigh": idxs("r_thigh"),
    }
    for k, v in groups.items():
        if len(v)==0:
            print(f"[warn] schema: group '{k}' пуст")
    return Schema(names=names, groups=groups)

def load_npy(path: str, schema: Schema) -> np.ndarray:
    arr = np.load(path, allow_pickle=False)
    if arr.ndim == 2 and arr.shape[1] == 3*len(schema.names):
        arr = arr.reshape(arr.shape[0], len(schema.names), 3)
    elif arr.ndim == 3 and arr.shape[2] == 3 and arr.shape[1] == len(schema.names):
        pass
    else:
        raise ValueError(f"Unexpected array shape {arr.shape} for N={len(schema.names)}")
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    return arr  # (T, N, 3)

# =============== Biomech helpers ===============
def centroid_series(x: np.ndarray) -> np.ndarray:
    """(T, K, 3) -> (T,3)"""
    return x.mean(axis=1)

def center_and_scale(all_xyz: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    pelvis = all_xyz.get("pelvis")
    if pelvis is not None:
        c = centroid_series(pelvis)
        for k in all_xyz:
            all_xyz[k] = all_xyz[k] - c[:, None, :]
    scale_list = []
    for side in ("L", "R"):
        th = all_xyz.get(f"{side}_thigh"); sh = all_xyz.get(f"{side}_shank")
        if th is not None and sh is not None:
            th_c = centroid_series(th); sh_c = centroid_series(sh)
            d = np.linalg.norm(th_c - sh_c, axis=1)
            scale_list.append(np.median(d[d>0]))
    scale = float(np.median(scale_list)) if scale_list else 1.0
    scale = max(scale, 1e-6)
    for k in all_xyz:
        all_xyz[k] = all_xyz[k] / scale
    return all_xyz

def angle_between(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    dot = (u*v).sum(axis=1)
    nu = np.linalg.norm(u, axis=1); nv = np.linalg.norm(v, axis=1)
    c = np.clip(dot/(np.maximum(nu*nv, 1e-8)), -1.0, 1.0)
    return np.arccos(c)

# быстрая оценка осей сегментов (без покадрового SVD)
def fast_segment_axes(cogs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    axes = {}
    for side in ("L", "R"):
        th = cogs.get(f"{side}_thigh")
        sh = cogs.get(f"{side}_shank")
        ft = cogs.get(f"{side}_foot")
        if th is not None and sh is not None:
            v = sh - th
            axes[f"{side}_thigh"] = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
        if sh is not None and ft is not None:
            v = ft - sh
            axes[f"{side}_shank"] = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
        if ft is not None:
            dv = np.vstack([ft[1:] - ft[:-1], ft[[-1]] - ft[[-2]]])
            axes[f"{side}_foot"] = dv / (np.linalg.norm(dv, axis=1, keepdims=True) + 1e-8)
    pel = cogs.get("pelvis")
    thL = cogs.get("L_thigh"); thR = cogs.get("R_thigh")
    if pel is not None and (thL is not None or thR is not None):
        tgt = pel.copy()
        if thL is not None and thR is not None:
            tgt = (thL + thR)/2.0
        elif thL is not None:
            tgt = thL
        else:
            tgt = thR
        v = tgt - pel
        axes["pelvis"] = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
    return axes

# детектор событий шага (по стопе)
def detect_gait_events(foot_centroid: np.ndarray, fps: int = 30):
    y = foot_centroid[:, 1]
    vy = np.gradient(y)
    from scipy.signal import find_peaks
    min_dist = max(3, int(0.20 * fps))
    inv = -y
    mins, _ = find_peaks(inv, distance=min_dist)
    maxs, _ = find_peaks(y,   distance=min_dist)
    fs = [i for i in mins if i > 1 and vy[i-1] < 0 <= vy[i]]
    fo = [i for i in maxs if i > 1 and vy[i-1] > 0 >= vy[i]]
    return np.asarray(fs, dtype=int), np.asarray(fo, dtype=int)

def iqr(a: np.ndarray):
    q75, q25 = np.nanpercentile(a, [75, 25])
    return float(q75 - q25)

# =============== Feature Extraction ===============
def extract_features(path: str, schema: Schema, fps: int = 30, stride: int = 1, fast_axes_flag: bool = True) -> pd.DataFrame:
    """
    Возвращает одну строку числовых признаков.
    Ускорения:
      - stride: берём каждый s-й кадр
      - fast_axes_flag: быстрые оси сегментов (centroid-to-centroid)
    """
    A = load_npy(path, schema)
    if stride > 1:
        A = A[::stride]
        fps = max(1, int(round(fps/stride)))
    T = A.shape[0]
    # сегменты
    seg = {k: A[:, idx, :] for k, idx in schema.groups.items() if len(idx) > 0}
    if not seg:
        raise ValueError("Schema produced no segments")
    seg = center_and_scale(seg)
    # центроиды и оси
    cogs = {k: centroid_series(v) for k, v in seg.items()}
    axes = fast_segment_axes(cogs) if fast_axes_flag else {}  # для ROM, если пусто — некоторые ROM будут нулевые

    rows = []
    # дополнительные последовательные фичи (для smoothness)
    pelvis = cogs.get("pelvis")
    if pelvis is None: raise ValueError("No pelvis in schema")

    # вспом: ширина шага (средняя ML-дистанция между стопами)
    step_width_ml = None
    if "L_foot" in cogs and "R_foot" in cogs:
        step_width_ml = float(np.nanmedian(np.abs(cogs["L_foot"][:,2] - cogs["R_foot"][:,2])))

    for side in ("L", "R"):
        foot = cogs.get(f"{side}_foot"); sh = axes.get(f"{side}_shank")
        th  = axes.get(f"{side}_thigh"); pel = axes.get("pelvis")
        if foot is None or pel is None:
            continue
        fs, fo = detect_gait_events(foot, fps=fps)
        if len(fs) < 2:
            fs = np.array([0, T-1], dtype=int)

        knee = angle_between(th, sh) if (th is not None and sh is not None) else np.zeros(T)
        hip  = angle_between(pel, th) if (pel is not None and th is not None) else np.zeros(T)
        foot_axis = axes.get(f"{side}_foot")
        ankle = angle_between(sh, foot_axis) if (sh is not None and foot_axis is not None) else np.zeros(T)

        for i in range(len(fs)-1):
            a, b = fs[i], fs[i+1]
            if b <= a+1: 
                continue
            segslice = slice(a, b)
            step_t = (b-a)/fps
            pelvis_y = pelvis[:,1]
            pelvis_z = pelvis[:,2]  # ML
            vo = float(np.max(pelvis_y[segslice]) - np.min(pelvis_y[segslice]))
            ml = float(np.max(pelvis_z[segslice]) - np.min(pelvis_z[segslice]))
            p0 = pelvis[a, [0,2]]; p1 = pelvis[b, [0,2]]
            step_len = float(np.linalg.norm(p1 - p0))
            fo_in = fo[(fo > a) & (fo < b)]
            stance = float((fo_in[0]-a)/fps) if len(fo_in) else np.nan
            speed = step_len/step_t if step_t>1e-6 else 0.0

            rows.append({
                "side": 0 if side=="L" else 1,
                "step_time": step_t,
                "cadence": 60.0/step_t if step_t>1e-6 else np.nan,
                "step_len": step_len,
                "step_speed": speed,
                "pelvis_vert_osc": vo,
                "pelvis_ml_osc": ml,
                "rom_knee": float(np.max(knee[segslice]) - np.min(knee[segslice])),
                "rom_hip":  float(np.max(hip[segslice])  - np.min(hip[segslice])),
                "rom_ankle": float(np.max(ankle[segslice]) - np.min(ankle[segslice])),
                "stance_time": stance,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        # fallback минимум
        return pd.DataFrame([{
            "step_time_median": np.nan, "step_time_mean": np.nan, "step_time_std": 0.0,
            "cadence_median": np.nan, "cadence_mean": np.nan, "cadence_std": 0.0,
            "step_len_median": 0.0, "step_len_mean": 0.0, "step_len_std": 0.0,
            "step_speed_median": 0.0, "step_speed_mean": 0.0, "step_speed_std": 0.0,
            "pelvis_vert_osc_median": 0.0, "pelvis_vert_osc_mean": 0.0, "pelvis_vert_osc_std": 0.0,
            "pelvis_ml_osc_median": 0.0, "pelvis_ml_osc_mean": 0.0, "pelvis_ml_osc_std": 0.0,
            "rom_knee_median": 0.0, "rom_knee_mean": 0.0, "rom_knee_std": 0.0,
            "rom_hip_median": 0.0,  "rom_hip_mean": 0.0,  "rom_hip_std": 0.0,
            "rom_ankle_median": 0.0, "rom_ankle_mean": 0.0, "rom_ankle_std": 0.0,
            "stance_time_median": np.nan, "stance_time_mean": np.nan, "stance_time_std": 0.0,
            "cv_step_time": np.nan, "cv_step_len": np.nan, "cv_cadence": np.nan, "cv_stance_time": np.nan,
            "iqr_step_time": np.nan, "iqr_step_len": np.nan, "iqr_cadence": np.nan, "iqr_stance_time": np.nan,
            "asym_step_time": np.nan, "asym_step_len": np.nan, "asym_rom_knee": np.nan,
            "asym_rom_hip": np.nan, "asym_rom_ankle": np.nan, "asym_stance_time": np.nan,
            "asym_cadence": np.nan, "asym_pelvis_vert_osc": np.nan, "asym_pelvis_ml_osc": np.nan,
            "pelvis_jerk_rms_y": 0.0, "pelvis_jerk_rms_z": 0.0,
            "step_width_ml": float(step_width_ml) if step_width_ml is not None else np.nan,
            "cycles_count": 0,
            "path_straightness": 0.0
        }])

    # агрегаты по шагам
    num = df.select_dtypes(include=[np.number]).copy()
    side_col = num.pop("side")
    agg_tbl = num.agg(["median","mean","std"]).T
    flat = {f"{feat}_{stat}": float(agg_tbl.loc[feat, stat])
            for feat in agg_tbl.index for stat in agg_tbl.columns}
    # CV/IQR — для «здоровых» важно низкое разброс/вариабельность
    for col in ["step_time","step_len","cadence","stance_time","pelvis_vert_osc","pelvis_ml_osc"]:
        v = num[col].values
        if np.isfinite(v).sum() >= 2:
            m = np.nanmean(v); s = np.nanstd(v)
            flat[f"cv_{col}"] = float(s/(abs(m)+1e-6))
            flat[f"iqr_{col}"] = iqr(v)
        else:
            flat[f"cv_{col}"] = np.nan; flat[f"iqr_{col}"] = np.nan

    # асимметрии по медианам L/R
    if (side_col==0).any() and (side_col==1).any():
        def med(side, col):
            vv = num.loc[side_col==side, col]
            return float(np.median(vv)) if len(vv) else np.nan
        for col in ["step_time","step_len","rom_knee","rom_hip","rom_ankle","stance_time","cadence","pelvis_vert_osc","pelvis_ml_osc"]:
            L = med(0,col); R = med(1,col)
            denom = (abs(L)+abs(R))/2 + 1e-6
            flat[f"asym_{col}"] = abs(L-R)/denom

    # плавность (jerk RMS) таза по вертикали/ML
    def jerk_rms(sig):
        v  = np.gradient(sig)         # vel
        a  = np.gradient(v)           # acc
        j  = np.gradient(a)           # jerk
        return float(np.sqrt(np.nanmean(j*j)))
    flat["pelvis_jerk_rms_y"] = jerk_rms(pelvis[:,1])
    flat["pelvis_jerk_rms_z"] = jerk_rms(pelvis[:,2])

    # средняя ширина шага (ML между стопами)
    flat["step_width_ml"] = float(step_width_ml) if step_width_ml is not None else np.nan

    # straightness пути таза: фин.смещение / сумма приращений
    horiz = pelvis[:,[0,2]]
    net = np.linalg.norm(horiz[-1]-horiz[0])
    seglen = np.linalg.norm(np.diff(horiz, axis=0), axis=1)
    flat["path_straightness"] = float(net / (seglen.sum()+1e-6))

    flat["cycles_count"] = int(len(num))
    return pd.DataFrame([flat])

# =============== Parallel dataset build ===============
def _extract_one(args):
    p, fn, y_val, fps, stride, fast_axes = args
    schema = load_schema(schema_path_global)
    feats = extract_features(p, schema, fps=fps, stride=stride, fast_axes_flag=fast_axes)
    d = feats.iloc[0].to_dict()
    return d, int(y_val), guess_subject_id(fn), p

def build_table(manifest_csv: str, data_dir: str, schema: Schema, fps: int,
                workers: int = 1, stride: int = 1, fast_axes: bool = True
               ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    df = pd.read_csv(manifest_csv)
    assert "filename" in df.columns, "CSV must have 'filename'"
    y_col = None
    for c in df.columns:
        if "inj" in c.lower():
            y_col = c; break
    assert y_col is not None, "CSV must have an injury label column (e.g., 'No inj/ inj')"

    f_idx = df.columns.get_loc("filename")
    y_idx = df.columns.get_loc(y_col)

    tasks = []
    for row in df.itertuples(index=False, name=None):
        fn = str(row[f_idx])
        p = os.path.join(data_dir, fn)
        if not p.endswith(".npy"): p += ".npy"
        if not os.path.exists(p): 
            continue
        y_val = label_to_int(row[y_idx])
        tasks.append((p, fn, y_val, fps, stride, fast_axes))
    if not tasks:
        raise RuntimeError("No valid .npy paths")

    print(f"[info] Extracting features: {len(tasks)} files | workers={workers} | stride={stride} | fast_axes={fast_axes}")
    X_rows = []; y_list = []; groups = []; files = []
    skipped = 0

    if workers <= 1:
        for t in tqdm(tasks, desc="Extract features"):
            try:
                d, yv, grp, fp = _extract_one(t)
                X_rows.append(d); y_list.append(yv); groups.append(grp); files.append(fp)
            except Exception as e:
                skipped += 1
                tqdm.write(f"[skip] {t[0]} -> {type(e).__name__} {e}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_extract_one, t) for t in tasks]
            for f in tqdm(as_completed(futs), total=len(futs), desc="Extract features (parallel)"):
                try:
                    d, yv, grp, fp = f.result()
                    X_rows.append(d); y_list.append(yv); groups.append(grp); files.append(fp)
                except Exception as e:
                    skipped += 1
                    tqdm.write(f"[skip] -> {type(e).__name__} {e}")

    X = pd.DataFrame(X_rows).fillna(0.0)
    y = np.asarray(y_list, dtype=np.int64)
    groups = np.asarray(groups)
    files = np.asarray(files)

    ok = np.isin(y, [0,1])
    bad = (~ok).sum()
    if bad:
        print(f"[warn] dropped {bad} rows due to bad labels")
    X = X.loc[ok].reset_index(drop=True)
    y = y[ok]; groups = groups[ok]; files = files[ok]

    X = X.select_dtypes(include=[np.number]).copy()

    print(f"[info] Ready: {len(X)} samples (skipped {skipped}); positives={int((y==1).sum())}, negatives={int((y==0).sum())}, features={X.shape[1]}")
    return X, y, groups, files

# =============== Confusion matrix plotting ===============
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm: np.ndarray, class_names, normalize: bool, save_path: str, title: str = None):
    cm = cm.astype(np.float64)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm_norm = cm / row_sums
    else:
        cm_norm = cm
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True",
        xlabel="Predicted",
    )
    ax.set_ylim(len(class_names)-0.5, -0.5)
    if title is None:
        title = "Confusion Matrix (normalized)" if normalize else "Confusion Matrix (counts)"
    ax.set_title(title, pad=14)
    thresh = cm_norm.max()/2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        text = f"{cm_norm[i,j]:.2f}\n({int(cm[i,j])})" if normalize else f"{int(cm[i,j])}"
        ax.text(j, i, text,
                ha="center", va="center",
                color="white" if cm_norm[i,j] > thresh else "black",
                fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close(fig)

# =============== Threshold search with constraints ===============
from sklearn.metrics import recall_score, precision_recall_curve, roc_auc_score, average_precision_score, confusion_matrix, classification_report

def find_threshold_balanced(y_true, prob, target_r1=0.95, target_r0=0.90):
    thr_grid = np.linspace(0.0, 1.0, 500)
    best = {"thr": 0.5, "r1": 0.0, "r0": 0.0, "dist": 1e9}
    for thr in thr_grid:
        pred = (prob >= thr).astype(int)
        r1 = recall_score(y_true, pred, pos_label=1)
        r0 = recall_score(y_true, 1-pred, pos_label=1)
        # расстояние от цели
        dist = abs(r1-target_r1) + abs(r0-target_r0)
        if dist < best["dist"]:
            best.update({"thr": thr, "r1": r1, "r0": r0, "dist": dist})
    return best


# =============== Train XGBoost ===============
def train_xgb(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, files: np.ndarray,
              out_dir: str, use_gpu: bool, min_recall_pos: float, min_recall_neg: float,
              seed: int = 42):
    ensure_dir(out_dir)
    from sklearn.model_selection import GroupShuffleSplit
    from xgboost import XGBClassifier, plot_importance

    # subject-wise split 60/20/20
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    tr_idx, te_idx = next(gss1.split(X, y, groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    tr2_idx, dv_idx = next(gss2.split(X.iloc[tr_idx], y[tr_idx], groups[tr_idx]))
    tr_idx = tr_idx[tr2_idx]

    def stat(idx):
        return {"n": len(idx), "pos": int((y[idx]==1).sum()), "neg": int((y[idx]==0).sum()),
                "subjects": int(len(np.unique(groups[idx])))}
    print("[split] subject-wise 60/20/20")
    print("  train:", stat(tr_idx))
    print("  dev:  ", stat(dv_idx))
    print("  test: ", stat(te_idx))

    split_df = pd.DataFrame({"file": files, "group": groups, "y": y, "split": "train"})
    split_df.loc[dv_idx, "split"] = "dev"
    split_df.loc[te_idx, "split"] = "test"
    split_df.to_csv(os.path.join(out_dir, "split_subjectwise.csv"), index=False)

    # модель
    scale_pos_weight = float((y[tr_idx]==0).sum() / max(1,(y[tr_idx]==1).sum()))
    params = dict(
        n_estimators=1600,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=2,
        subsample=0.85,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        tree_method="hist" if not use_gpu else "gpu_hist",
        eval_metric="logloss",
        random_state=seed,
        scale_pos_weight=scale_pos_weight,
        n_jobs=0
    )
    model = XGBClassifier(**params)

    X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
    X_dv, y_dv = X.iloc[dv_idx], y[dv_idx]
    X_te, y_te = X.iloc[te_idx], y[te_idx]

    print("[train] XGBoost fitting...")
    model.fit(X_tr, y_tr,
              eval_set=[(X_tr, y_tr), (X_dv, y_dv)],
              verbose=50, early_stopping_rounds=100)

    # вероятности
    prob_dv = model.predict_proba(X_dv)[:,1]
    prob_te = model.predict_proba(X_te)[:,1]

    # PR/ROC
    auprc_dv = average_precision_score(y_dv, prob_dv)
    auroc_dv = roc_auc_score(y_dv, prob_dv)
    auprc_te = average_precision_score(y_te, prob_te)
    auroc_te = roc_auc_score(y_te, prob_te)

    # подбор порога под цели
    best = find_threshold_balanced(y_dv, prob_dv,
                                           min_recall_pos=min_recall_pos,
                                           min_recall_neg=min_recall_neg)
    thr = best["thr"]
    print(f"[threshold] chosen={thr:.4f} | dev recall(1)={best['r1']:.3f} recall(0)={best['r0']:.3f} ok={best['ok']}")

    # тест с этим порогом
    pred_te = (prob_te >= thr).astype(int)
    cm = confusion_matrix(y_te, pred_te)
    rep = classification_report(y_te, pred_te, digits=3, output_dict=False)
    print("\n=== TEST REPORT ===\n", rep)

    # сохраняем метрики
    metrics = {
        "dev_auprc": float(auprc_dv), "dev_auroc": float(auroc_dv),
        "test_auprc": float(auprc_te), "test_auroc": float(auroc_te),
        "threshold": float(thr),
        "threshold_search_ok": bool(best["ok"]),
        "dev_recall_injury": float(best["r1"]), "dev_recall_noinj": float(best["r0"]),
        "test_cm": cm.tolist(),
        "test_report": rep
    }
    json.dump(metrics, open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    open(os.path.join(out_dir, "threshold.txt"), "w").write(str(thr))

    # матрицы путаницы (картинки)
    class_names = ["No Injury", "Injury"]
    plot_confusion_matrix(cm, class_names, normalize=False,
                          save_path=os.path.join(out_dir, "confusion_matrix_counts.png"),
                          title="Confusion Matrix (counts)")
    plot_confusion_matrix(cm, class_names, normalize=True,
                          save_path=os.path.join(out_dir, "confusion_matrix_normalized.png"),
                          title="Confusion Matrix (row-normalized)")

    # важности признаков (top-25)
    try:
        from xgboost import plot_importance
        fig, ax = plt.subplots(figsize=(8, 10))
        plot_importance(model, max_num_features=25, importance_type="gain", ax=ax)
        plt.title("XGBoost Feature Importance (top-25)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "feature_importance_top25.png"), dpi=160)
        plt.close(fig)
    except Exception as e:
        print("[warn] feature importance plot failed:", e)

    # сохраним модель
    model.save_model(os.path.join(out_dir, "xgb.json"))
    json.dump(list(X.columns), open(os.path.join(out_dir, "features_cols.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("Saved to:", out_dir)
    return model, thr, (X_te, y_te, prob_te)

# =============== Cache helpers ===============
def extract_and_save(manifest_csv: str, data_dir: str, schema_path: str, fps: int, out_dir: str,
                     workers: int = 1, stride: int = 1, fast_axes: bool = True):
    ensure_dir(out_dir)
    global schema_path_global
    schema_path_global = schema_path
    schema = load_schema(schema_path)
    X, y, groups, files = build_table(manifest_csv, data_dir, schema, fps=fps,
                                      workers=workers, stride=stride, fast_axes=fast_axes)
    X.to_parquet(os.path.join(out_dir, "features.parquet"))
    np.save(os.path.join(out_dir, "labels.npy"), y)
    np.save(os.path.join(out_dir, "groups.npy"), groups)
    with open(os.path.join(out_dir, "files.txt"), "w", encoding="utf-8") as f:
        for p in files: f.write(str(p)+"\n")
    meta = {
        "manifest_csv": manifest_csv, "data_dir": data_dir,
        "fps": fps, "stride": stride, "fast_axes": fast_axes,
        "n_samples": int(len(X)), "n_features": int(X.shape[1]),
        "positives": int((y==1).sum()), "negatives": int((y==0).sum()),
        "schema": os.path.abspath(schema_path),
    }
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[extract] saved cache to: {out_dir}")
    return out_dir

def load_features_cache(features_path: str, labels_path: str, groups_path: str, files_path: str):
    X = pd.read_parquet(features_path)
    y = np.load(labels_path)
    groups = np.load(groups_path, allow_pickle=True)
    with open(files_path, "r", encoding="utf-8") as f:
        files = np.array([ln.strip() for ln in f if ln.strip()])
    X = X.select_dtypes(include=[np.number]).copy()
    return X, y, groups, files

# =============== Predict ===============
def predict_one(npy_path: str, schema_path: str, out_dir: str, fps: int = 30, stride: int = 1, fast_axes: bool = True):
    from xgboost import XGBClassifier
    schema = load_schema(schema_path)
    feats = extract_features(npy_path, schema, fps=fps, stride=stride, fast_axes_flag=fast_axes)
    cols = json.load(open(os.path.join(out_dir, "features_cols.json"), "r", encoding="utf-8"))
    X = feats.reindex(columns=cols, fill_value=0.0)
    model = XGBClassifier()
    model.load_model(os.path.join(out_dir, "xgb.json"))
    thr = float(open(os.path.join(out_dir, "threshold.txt")).read().strip())
    prob = float(model.predict_proba(X)[:,1][0])
    pred = int(prob >= thr)
    return {"file": npy_path, "prob_injury": prob, "pred": pred, "threshold": thr}

# =============== CLI ===============
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["extract","train","train_features","predict"], default="train")
    ap.add_argument("--csv", help="manifest.csv (для extract/train)")
    ap.add_argument("--data_dir", help="папка с .npy (для extract/train)")
    ap.add_argument("--schema", required=True, help="schema_joints.json")
    ap.add_argument("--out_dir", default="out_xgb")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--stride", type=int, default=2, help="темп прореживания кадров")
    ap.add_argument("--fast_axes", action="store_true", help="использовать быстрые оси сегментов (без SVD)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)-1), help="число процессов при extract/build")

    ap.add_argument("--use_gpu", action="store_true", help="XGBoost GPU (gpu_hist)")
    ap.add_argument("--min_recall_injury", type=float, default=0.95)
    ap.add_argument("--min_recall_noinj",  type=float, default=0.90)

    # cache paths для train_features
    ap.add_argument("--features", help="path to features.parquet")
    ap.add_argument("--labels", help="path to labels.npy")
    ap.add_argument("--groups", help="path to groups.npy")
    ap.add_argument("--files_list", help="path to files.txt")

    # predict
    ap.add_argument("--npy", help="путь к одному .npy (для predict)")

    args = ap.parse_args()
    seed_all(42)

    global schema_path_global
    schema_path_global = args.schema

    if args.mode == "extract":
        assert args.csv and args.data_dir, "--csv и --data_dir обязательны"
        extract_and_save(args.csv, args.data_dir, args.schema, fps=args.fps,
                         out_dir=args.out_dir, workers=args.workers, stride=args.stride, fast_axes=args.fast_axes)
        return

    if args.mode == "train":
        assert args.csv and args.data_dir, "--csv и --data_dir обязательны"
        schema = load_schema(args.schema)
        X, y, groups, files = build_table(args.csv, args.data_dir, schema, fps=args.fps,
                                          workers=args.workers, stride=args.stride, fast_axes=args.fast_axes)
        train_xgb(X, y, groups, files, args.out_dir, use_gpu=args.use_gpu,
                  min_recall_pos=args.min_recall_injury, min_recall_neg=args.min_recall_noinj)
        return

    if args.mode == "train_features":
        features = args.features or os.path.join(args.out_dir, "features.parquet")
        labels   = args.labels   or os.path.join(args.out_dir, "labels.npy")
        groups   = args.groups   or os.path.join(args.out_dir, "groups.npy")
        files_l  = args.files_list or os.path.join(args.out_dir, "files.txt")
        X, y, groups, files = load_features_cache(features, labels, groups, files_l)
        train_xgb(X, y, groups, files, args.out_dir, use_gpu=args.use_gpu,
                  min_recall_pos=args.min_recall_injury, min_recall_neg=args.min_recall_noinj)
        return

    # predict
    assert args.npy, "--npy обязателен в режиме predict"
    res = predict_one(args.npy, args.schema, args.out_dir, fps=args.fps, stride=args.stride, fast_axes=args.fast_axes)
    print(json.dumps(res, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
