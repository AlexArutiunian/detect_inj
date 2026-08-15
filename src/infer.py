#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
infer_xgb_rich.py — Инференс классической XGBoost-модели (injury / no-injury)
с теми же RICH-фичами, что и в train_xgb_plus:
на канал 27 (basic+windows) + 4 (spectral) = 31 признаков.

Поддерживает входы: .npy или .json (нужна схема суставов).
Умеет брать список файлов из CSV (столбец --filename_col) и/или из сканирования папки.
Результат: CSV с prob/pred + “confidence buckets” и PNG-график распределений.
"""

from __future__ import annotations
import os, sys, json, argparse
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib

# --- Matplotlib без GUI ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# ================== ВСПОМОГАТЕЛЬНОЕ ==================
def entropy_binary(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-8, 1 - 1e-8)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def label_to_int(v):
    if v is None: return None
    s = str(v).strip().lower()
    if s in ("injury","1"): return 1
    if s in ("no injury","0"): return 0
    return None


# ================== ПРОЧТЕНИЕ ВХОДОВ ==================
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
        if b.endswith(".json"):    push(os.path.join(data_dir, b[:-5] + ".npy"))
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

def choose_paths_scan_then_csv(data_dir: str,
                               csv_path: Optional[str],
                               filename_col: str,
                               recursive: bool,
                               input_format: str) -> List[str]:
    ext = ".npy" if input_format == "npy" else ".json"
    disk_files = list_files_with_ext(data_dir, ext, recursive=recursive)

    if not csv_path or not os.path.exists(csv_path):
        if csv_path and not os.path.exists(csv_path):
            print(f"[warn] CSV '{csv_path}' не найден; используем только файлы на диске.", file=sys.stderr)
        return disk_files

    df = pd.read_csv(csv_path)
    cols = {c.lower().strip(): c for c in df.columns}
    fn_col = cols.get(filename_col.lower(), filename_col)
    if fn_col not in df.columns:
        for c in df.columns:
            if "file" in c.lower() or "name" in c.lower():
                fn_col = c; break

    by_base: dict[str, str] = {}
    for p in disk_files:
        b = os.path.basename(p)
        if b not in by_base:
            by_base[b] = p

    chosen: list[str] = []
    missing = 0
    for raw in df[fn_col].astype(str).tolist():
        hit = None
        for c in possible_paths(data_dir, raw, input_format):
            nc = os.path.normpath(c)
            if os.path.exists(nc):
                hit = nc; break
            b = os.path.basename(nc)
            if b in by_base:
                hit = by_base[b]; break
        if hit: chosen.append(hit)
        else:   missing += 1

    if missing:
        print(f"[warn] В CSV указано файлов, которых нет на диске: {missing}", file=sys.stderr)

    # dedup
    seen, out = set(), []
    for p in chosen:
        if p not in seen:
            seen.add(p); out.append(p)
    return out


# ================== JSON/NPY helpers ==================
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


# ================== Приведение к (T,F) ==================
def as_TxF(a: np.ndarray) -> Optional[np.ndarray]:
    """Поддерживает (T,F) и (T,V,C) → (T, F)"""
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


# ================== RICH-фичи (31 на канал) ==================
def _basic_stats(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    dx = np.diff(x) if x.size > 1 else np.array([0.0], dtype=np.float32)

    def safe_std(v):  return float(np.nanstd(v)) if v.size else 0.0
    def safe_mean(v): return float(np.nanmean(v)) if v.size else 0.0
    def safe_min(v):  return float(np.nanmin(v)) if v.size else 0.0
    def safe_max(v):  return float(np.nanmax(v)) if v.size else 0.0
    def q(v, qq):     return float(np.nanpercentile(v, qq)) if v.size else 0.0

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

    def acf(v, k=1):
        v = v[np.isfinite(v)]
        n = v.size
        if n <= k or n < 2: return 0.0
        m = v.mean()
        v0 = v - m
        num = np.sum(v0[:-k] * v0[k:])
        den = np.sum(v0 * v0) + 1e-12
        return float(num / den)

    def window_stats(v):
        n = v.size
        if n < 3:
            return [safe_mean(v), safe_std(v), safe_min(v), safe_max(v)] * 3
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
    out.extend(window_stats(x))  # +12
    return np.array(out, dtype=np.float32)  # 27 на канал

def _spectral_stats(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = x - np.nanmean(x)
    n = x.size
    if n < 4: return np.zeros(4, dtype=np.float32)
    fft = np.fft.rfft(x * np.hanning(n))
    pxx = (fft.real**2 + fft.imag**2)
    pxx = np.clip(pxx, 1e-12, None)
    k = pxx.size
    b = k // 3
    e1 = float(pxx[:b].sum())
    e2 = float(pxx[b:2*b].sum())
    e3 = float(pxx[2*b:].sum())
    p = pxx / pxx.sum()
    sent = float(-(p * np.log(p)).sum())
    return np.array([e1, e2, e3, sent], dtype=np.float32)



def _extract_features_rich(seq_2d: np.ndarray) -> np.ndarray:
    """seq_2d: (T, F) → (F*31,)"""
    T, F = seq_2d.shape
    feats = []
    for j in range(F):
        x = np.asarray(seq_2d[:, j], dtype=np.float32)
        feats.append(_basic_stats(x))
        feats.append(_spectral_stats(x))
    return np.concatenate(feats, axis=0).astype(np.float32, copy=False)
# ==== два экстрактора: CLASSIC (6*F) и RICH (31*F) ====

def _extract_features_classic(seq_2d: np.ndarray) -> np.ndarray:
    """(T,F) -> (F*6,) : mean, std, min, max, dmean, dstd"""
    seq = np.asarray(seq_2d, dtype=np.float32)
    dif = np.diff(seq, axis=0) if seq.shape[0] > 1 else np.zeros_like(seq)
    stat = np.concatenate([
        np.nanmean(seq, axis=0), np.nanstd(seq, axis=0),
        np.nanmin(seq, axis=0),  np.nanmax(seq, axis=0),
        np.nanmean(dif, axis=0), np.nanstd(dif, axis=0)
    ], axis=0).astype(np.float32, copy=False)
    return np.nan_to_num(stat, nan=0.0, posinf=0.0, neginf=0.0)

# (RICH уже есть) _extract_features_rich(seq_2d) -> (F*31,)

# ==== универсальный билдер с выбором режима ====
def build_features_for_paths(paths: List[str],
                             input_format: str,
                             downsample: int,
                             schema_joints: Optional[List[str]],
                             motion_key: str="running",
                             feat_mode: str="auto",
                             exp_dim: Optional[int]=None) -> Tuple[np.ndarray, List[str]]:
    """
    feat_mode: 'auto' | 'rich' | 'classic'
    exp_dim:   ожидаемая размерность признаков (например, scaler.n_features_in_)
    """
    # helper для одного массива
    def _one_feats(seq_2d: np.ndarray, mode: str) -> np.ndarray:
        if mode == "rich":
            return _extract_features_rich(seq_2d)
        elif mode == "classic":
            return _extract_features_classic(seq_2d)
        else:
            raise ValueError("mode must be 'rich' or 'classic'")

    # если auto — определим на первом валидном примере
    detected_mode = None

    X_list, keep_paths = [], []
    for p in tqdm(paths, desc="Feats(AUTO)" if feat_mode=="auto" else f"Feats({feat_mode.upper()})",
                  dynamic_ncols=True, mininterval=0.2):
        try:
            # загрузка последовательности -> (T,F)
            if input_format == "npy":
                arr = np.load(p, allow_pickle=False, mmap_mode="r")
                arr = as_TxF(arr)
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
                if not isinstance(md, dict): 
                    continue
                if not schema_joints: 
                    continue
                seq = _stack_motion_frames_with_schema(md, schema_joints)
                if seq is None or seq.shape[0] < 2: 
                    continue
                if downsample > 1: 
                    seq = seq[::downsample]

            seq = np.asarray(seq, dtype=np.float32)

            # автоопределение
            if feat_mode == "auto" and detected_mode is None and exp_dim is not None:
                F = seq.shape[1]
                cand = []
                if F*31 == exp_dim: cand.append("rich")
                if F*6  == exp_dim: cand.append("classic")
                # если обе подходят — предпочтем RICH (т.к. она “старше” в твоём пайплайне)
                if cand:
                    detected_mode = "rich" if "rich" in cand else cand[0]
                else:
                    # пробуем вычислить и сравнить длину напрямую
                    r = _extract_features_rich(seq).shape[0]
                    c = _extract_features_classic(seq).shape[0]
                    if r == exp_dim: detected_mode = "rich"
                    elif c == exp_dim: detected_mode = "classic"
                    else:
                        raise ValueError(f"Cannot auto-detect features: exp={exp_dim}, got rich={r}, classic={c}")

            mode_to_use = detected_mode if feat_mode == "auto" else feat_mode
            X_list.append(_one_feats(seq, mode_to_use))
            keep_paths.append(p)
        except Exception:
            continue

    if not X_list:
        raise RuntimeError("Не удалось собрать ни одной строки фич.")
    X = np.stack(X_list).astype(np.float32, copy=False)
    return X, keep_paths


# ================== МОДЕЛЬ/КАЛИБРАТОР ==================
def load_classical_bundle(path: str):
    obj = joblib.load(path)
    if isinstance(obj, dict):
        return obj.get("model", obj), obj.get("scaler", None)
    return obj, None

def maybe_load_calibrator(model_dir: str):
    path = os.path.join(model_dir, "calibrator.joblib")
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception:
            pass
    return None


# ================== ГРАФИК ГРУПП ==================
def plot_group_bars(df: pd.DataFrame, out_path: str, title_prefix: str = "", dpi: int = 180):
    conf_labels = ["conf <50%", "conf 50–60%", "conf 60–80%", "conf >80%"]
    inj_labels  = ["inj <50%", "inj 50–60%", "inj 60–80%", "inj >80%"]
    noi_labels  = ["no-injury <50%", "no-injury 50–60%", "no-injury 60–80%", "no-injury >80%"]

    conf_counts = [int((df["confidence_group"] == g).sum()) for g in conf_labels]
    inj_counts  = [int((df["inj_group"] == g).sum()) for g in inj_labels]
    noi_counts  = [int((df["noinj_group"] == g).sum()) for g in noi_labels]

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
        ax.set_xlim(0, max(1, int(maxc * 1.18)))

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


# ================== MAIN ==================
def main():
    ap = argparse.ArgumentParser("Inference (XGB classic) with RICH features for NPY/JSON")
    # формат
    ap.add_argument("--input_format", choices=["npy","json"], default="npy")
    ap.add_argument("--motion_key", default="running", help="для JSON: running|walking")
    ap.add_argument("--schema_json", default="", help="для JSON: путь к schema_joints.json")
    ap.add_argument("--use_joints",  default="", help="для JSON: список через запятую (приоритет над schema_json)")

    # модель/порог
    ap.add_argument("--model", required=True, help="path to model.joblib (из train_xgb_plus)")
    ap.add_argument("--threshold",  default=None, help="число или путь к threshold.txt (если не задано — 0.5)")
    ap.add_argument("--downsample", type=int, default=1)

    # входы
    ap.add_argument("--csv",         default=None, help="CSV с колонкой filename")
    ap.add_argument("--data_dir",    default=".",  help="корневая папка данных")
    ap.add_argument("--filename_col", default="filename")
    ap.add_argument("--recursive",  action="store_true", help="глубокий скан при --scan_first")
    ap.add_argument("--scan_first", action="store_true", help="сначала скан папки, затем пересечение с CSV")
    ap.add_argument("--out_csv",    required=True)
    ap.add_argument("inputs", nargs="*", help="необязательный явный список файлов/имен (сопоставится через --data_dir)")
    
    ap.add_argument("--feat_mode", choices=["auto","rich","classic"], default="auto",
                help="Тип фич: auto подбирает по модели/скейлеру")

    # графики
    ap.add_argument("--plot_summary", default="", help="PNG для групп; по умолчанию <out_csv>_groups.png")
    ap.add_argument("--plot_title",   default="", help="Префикс к заголовкам")
    ap.add_argument("--plot_dpi",     type=int, default=180)

    args = ap.parse_args()

    # входные пути
    if args.scan_first and args.data_dir:
        paths = choose_paths_scan_then_csv(args.data_dir, args.csv, args.filename_col, args.recursive, args.input_format)
    elif args.csv:
        paths = resolve_inputs_from_csv(args.csv, args.data_dir or ".", args.filename_col, args.input_format)
    else:
        # явные inputs → сопоставим с data_dir при необходимости
        paths = []
        for p in (args.inputs or []):
            if os.path.exists(p): paths.append(p)
            else:
                r = pick_existing_path(possible_paths(args.data_dir or ".", p, args.input_format))
                if r: paths.append(r)

    if not paths:
        print("Нет входных файлов. Укажи --scan_first или --csv, или список путей.", file=sys.stderr)
        sys.exit(2)

    # схема суставов (для JSON)
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
            raise RuntimeError("Для JSON не удалось определить схему суставов. Передай --schema_json или --use_joints.")

    # порог
    thr = 0.5
    if args.threshold:
        if os.path.exists(args.threshold):
            try:
                with open(args.threshold, "r", encoding="utf-8") as f:
                    thr = float(f.read().strip())
            except Exception:
                print("[warn] не удалось прочитать threshold, используем 0.5", file=sys.stderr)
        else:
            try: thr = float(args.threshold)
            except Exception: pass

    # загрузка модели/скейлера и (возможно) калибратора
    model, scaler = load_classical_bundle(args.model)
    calibrator = None
    model_dir = os.path.dirname(os.path.abspath(args.model))
    cal = maybe_load_calibrator(model_dir)
    if cal is not None:
        calibrator = cal
        print("[info] calibrator loaded (isotonic/Platt)")

    exp_dim = None
    if scaler is not None and hasattr(scaler, "n_features_in_"):
        exp_dim = int(scaler.n_features_in_)
    elif hasattr(model, "n_features_in_"):
        exp_dim = int(model.n_features_in_)

    # считаем фичи (авто/ручной режим)
    X, keep_paths = build_features_for_paths(
        paths, args.input_format, max(1, int(args.downsample)),
        schema_joints, motion_key=args.motion_key,
        feat_mode=args.feat_mode, exp_dim=exp_dim
    )

    # строгая проверка на совпадение размерности, если известно exp_dim
    if exp_dim is not None and X.shape[1] != exp_dim:
        raise ValueError(
            f"Feature mismatch: got {X.shape[1]}, expected {exp_dim}. "
            f"Проверь --feat_mode/раскладку/downsample."
    )
    # проверка совместимости с scaler
    if scaler is not None and hasattr(scaler, "n_features_in_"):
        exp = int(scaler.n_features_in_)
        got = int(X.shape[1])
        if got != exp:
            raise ValueError(
                f"Feature mismatch: got {got}, expected {exp}. "
                f"Убедись, что инференс использует RICH-фичи (31*F), тот же downsample/раскладку, что при обучении."
            )

    Xs = scaler.transform(X) if scaler is not None else X

    # вероятности
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(Xs)[:, 1]
    elif hasattr(model, "decision_function"):
        d = model.decision_function(Xs)
        d = (d - d.min()) / (d.max() - d.min() + 1e-9)
        prob = d
    else:
        prob = model.predict(Xs).astype(np.float32)

    probs = np.asarray(prob, dtype=np.float32)
    if calibrator is not None:
        try:
            probs = calibrator.transform(probs)  # isotonic
        except Exception:
            try:
                probs = calibrator.predict_proba(probs.reshape(-1,1))[:,1]  # если вдруг классическая LR
            except Exception:
                pass

    # бинаризация и вспомогательные величины
    pred = (probs >= thr).astype(np.int32)
    p_clip = np.clip(probs, 1e-8, 1 - 1e-8)
    logit = np.log(p_clip / (1.0 - p_clip)).astype(np.float32)
    confidence = np.where(pred == 1, p_clip, 1.0 - p_clip).astype(np.float32)
    entr = entropy_binary(p_clip).astype(np.float32)

    # группировки
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

    # CSV
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
    print(f"[OK] saved: {args.out_csv}")

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

if __name__ == "__main__":
    main()
