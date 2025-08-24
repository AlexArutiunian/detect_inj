#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Injury vs No-Injury (binary) on NPY sequences — enhanced.
Добавлены фичи по походке и улучшения обучения, чтобы уменьшить FP и поднять recall класса No-Injury.

Новое:
- Улучшенный поиск порога с приоритетом на recall(0) (здоровые).
- Опциональная калибровка вероятностей (isotonic/sigmoid/none) по DEV.
- Асимметричное масштабирование веса здоровых (w0_scale).
- Улучшенный детектор шагов с учётом оценённой частоты каденса (f0).
- Доп. фичи:
  * peak width/ prominence вокруг f0 для вертикали таза
  * spectral flatness и spectral centroid/spread для вертикали/ML
  * ACF-regularity около периода шага
  * RMSSD по step_time/step_len/cadence и дрейф каденса (rolling-median std)
  * variability step_width (std/CV)
  * foot clearance (swing)
  * lag-consistency L/R по AP/ML (std лага по циклам)
  * turn-mask: «прямые» шаги выделяются по кривизне пути; рассчитываются *_straight агрегаты

Совместим с исходным форматом CSV/NPY/Schema.
"""

import os, json, argparse, random, re, warnings, itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore", category=UserWarning)

# -------------------- GLOBAL --------------------
schema_path_global = None

# -------------------- Utils --------------------
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
    m = re.match(r"(\d{8})", b)
    return m.group(1) if m else b.split("T")[0][:8]

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

# -------------------- Schema + I/O --------------------
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

# -------------------- Biomech helpers --------------------
def centroid_series(x: np.ndarray) -> np.ndarray:
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

# -------------------- Events + helpers --------------------
from scipy.signal import find_peaks, peak_widths, peak_prominences

def detect_gait_events(foot_centroid: np.ndarray, fps: int = 30, f0: Optional[float] = None):
    """Возвращает индексы FS/FO с адаптивной минимальной дистанцией между событиями.
    f0 — оценка частоты шага (Hz)."""
    y = foot_centroid[:, 1]
    vy = np.gradient(y)
    # базовый минимум между событиями
    min_dist = max(3, int(0.20 * fps))
    if f0 is not None and np.isfinite(f0) and f0 > 0.3:
        # ориентир: ~0.35 периода шага
        est = int(max(3, 0.35 * fps / max(f0, 1e-3)))
        min_dist = max(min_dist, est)
    inv = -y
    mins, _ = find_peaks(inv, distance=min_dist)
    maxs, _ = find_peaks(y,   distance=min_dist)
    fs = [i for i in mins if i > 1 and vy[i-1] < 0 <= vy[i]]  # foot strike
    fo = [i for i in maxs if i > 1 and vy[i-1] > 0 >= vy[i]]  # foot off
    return np.asarray(fs, dtype=int), np.asarray(fo, dtype=int)

# -------------------- Stats/Spectral --------------------
def iqr(a: np.ndarray):
    q75, q25 = np.nanpercentile(a, [75, 25])
    return float(q75 - q25)

def rmssd(a: np.ndarray) -> float:
    a = np.asarray(a, float)
    if a.size < 3: return np.nan
    d = np.diff(a)
    return float(np.sqrt(np.nanmean(d*d)))

# базовые и спектральные фичи по каналу

def _basic_stats(x):
    x = np.asarray(x, dtype=np.float32)
    dx = np.diff(x) if x.size>1 else np.array([0.0], np.float32)
    def q(v,a): return float(np.nanpercentile(v,a)) if v.size else 0.0
    def smean(v): return float(np.nanmean(v)) if v.size else 0.0
    def sstd(v):  return float(np.nanstd(v))  if v.size else 0.0
    def smin(v):  return float(np.nanmin(v))  if v.size else 0.0
    def smax(v):  return float(np.nanmax(v))  if v.size else 0.0
    def skew(v):
        v=v[np.isfinite(v)]
        if v.size<3: return 0.0
        m=v.mean(); s=v.std()
        return float(np.mean(((v-m)/(s+1e-12))**3))
    def kurt(v):
        v=v[np.isfinite(v)]
        if v.size<4: return 0.0
        m=v.mean(); s=v.std()
        return float(np.mean(((v-m)/(s+1e-12))**4)-3.0)
    def acf(v,k=1):
        v=v[np.isfinite(v)]; n=v.size
        if n<=k or n<2: return 0.0
        m=v.mean(); v0=v-m
        return float(np.sum(v0[:-k]*v0[k:])/(np.sum(v0*v0)+1e-12))
    def win(v):
        n=v.size
        if n<3: return [smean(v), sstd(v), smin(v), smax(v)]*3
        a=v[:n//3]; b=v[n//3:2*n//3]; c=v[2*n//3:]
        out=[]
        for w in (a,b,c): out += [smean(w), sstd(w), smin(w), smax(w)]
        return out
    out=[smean(x), sstd(x), smin(x), smax(x), q(x,25), q(x,50), q(x,75),
         skew(x), kurt(x), smean(dx), sstd(dx), smin(dx), smax(dx),
         acf(x,1), acf(x,2)]
    out += win(x)
    return np.array(out, np.float32)


def _spectral_stats(x):
    x=np.asarray(x, np.float32); x=x-np.nanmean(x); n=x.size
    if n<4: return np.zeros(4, np.float32)
    fft=np.fft.rfft(x*np.hanning(n)); p=(fft.real**2+fft.imag**2); p=np.clip(p,1e-12,None)
    k=p.size; b=k//3; e1=float(p[:b].sum()); e2=float(p[b:2*b].sum()); e3=float(p[2*b:].sum())
    pp=p/p.sum(); sent=float(-(pp*np.log(pp)).sum())
    return np.array([e1,e2,e3,sent], np.float32)


def _extract_generic_feats(seq_2d):
    T, F = seq_2d.shape
    feats = []
    feat_names = []
    for j in range(F):
        x = np.asarray(seq_2d[:, j], np.float32)
        prefix = f"ch{j:03d}"
        basic = _basic_stats(x)
        spec = _spectral_stats(x)
        feats.append(basic)
        feats.append(spec)
        for i in range(len(basic)):
            feat_names.append(f"{prefix}_stat_{i}")
        for i in range(len(spec)):
            feat_names.append(f"{prefix}_spec_{i}")
    feats = np.concatenate(feats, axis=0).astype(np.float32, copy=False)
    return feats, feat_names


# -------------------- Feature Extraction --------------------

def _harmonic_ratio(sig, fps, cadence_hz=None):
    x = np.asarray(sig, np.float32); x = x - np.nanmean(x); n = x.size
    if n < 16: return np.nan, np.nan
    win = np.hanning(n); fft = np.fft.rfft(x*win); p = (fft.real**2+fft.imag**2)
    freqs = np.fft.rfftfreq(n, d=1.0/max(fps,1))
    # estimate cadence peak if not given: in [0.5..3] Hz
    if cadence_hz is None:
        mask = (freqs>=0.5)&(freqs<=3.0)
        if not mask.any(): return np.nan, np.nan
        i0 = np.argmax(p[mask]); f0 = float(freqs[mask][i0])
    else:
        f0 = float(cadence_hz)
    # pick nearest bins
    i1 = int(np.argmin(np.abs(freqs - f0)))
    i2 = int(np.argmin(np.abs(freqs - 2.0*f0)))
    p1 = float(p[i1]) if i1 < len(p) else np.nan
    p2 = float(p[i2]) if i2 < len(p) else np.nan
    hr = (p2/(p1+1e-12)) if np.isfinite(p1) and p1>0 else np.nan
    return hr, f0


def _crosscorr_max(x, y, max_lag):
    x = np.asarray(x, np.float32); y = np.asarray(y, np.float32)
    n = min(len(x), len(y))
    if n < 4: return np.nan, np.nan
    x = (x - np.mean(x)); y = (y - np.mean(y))
    max_lag = int(max(1, max_lag))
    lags = range(-max_lag, max_lag+1)
    best = (0.0, 0)
    denom = (np.linalg.norm(x)*np.linalg.norm(y) + 1e-12)
    for L in lags:
        if L<0:
            a=x[-L:]; b=y[:n+L]
        elif L>0:
            a=x[:n-L]; b=y[L:]
        else:
            a=x; b=y
        if len(a)<3: continue
        c = float(np.sum(a*b)/denom)
        if abs(c) > abs(best[0]): best = (c, L)
    return float(best[0]), int(best[1])


def spectral_extras(sig: np.ndarray, fps: int, f0: Optional[float]):
    x = np.asarray(sig, np.float32); x = x - np.nanmean(x); n = len(x)
    out = {
        "spec_flatness": np.nan,
        "spec_centroid": np.nan,
        "spec_spread": np.nan,
        "peak_width_hz": np.nan,
        "peak_prominence": np.nan,
    }
    if n < 16: return out
    w = np.hanning(n)
    fft = np.fft.rfft(x*w)
    p = (fft.real**2+fft.imag**2)
    p = np.clip(p, 1e-12, None)
    freqs = np.fft.rfftfreq(n, d=1.0/max(fps,1))
    # flatness
    gm = np.exp(np.mean(np.log(p)))
    am = np.mean(p)
    out["spec_flatness"] = float(gm/am)
    # centroid + spread
    centroid = float(np.sum(freqs*p)/np.sum(p))
    spread = float(np.sqrt(np.sum(p*(freqs-centroid)**2)/np.sum(p)))
    out["spec_centroid"] = centroid
    out["spec_spread"] = spread
    # peak around f0
    if f0 is None or not np.isfinite(f0):
        mask = (freqs>=0.5)&(freqs<=3.0)
        if not mask.any():
            return out
        i0 = int(np.argmax(p[mask]))
        # index in full array
        idx = np.where(mask)[0][i0]
    else:
        idx = int(np.argmin(np.abs(freqs - f0)))
    peaks, _ = find_peaks(p)
    if len(peaks)==0:
        return out
    # choose peak closest to idx
    k = peaks[np.argmin(np.abs(peaks - idx))]
    try:
        widths = peak_widths(p, [k], rel_height=0.5)
        out["peak_width_hz"] = float((widths[0][0]) * (freqs[1]-freqs[0]))
        prom = peak_prominences(p, [k])
        out["peak_prominence"] = float(prom[0][0])
    except Exception:
        pass
    return out


def acf_regularities(sig: np.ndarray, fps: int, f0: Optional[float]):
    x = np.asarray(sig, np.float32)
    x = x - np.nanmean(x)
    n = len(x)
    out = {"acf_at_step": np.nan, "acf_at_2step": np.nan, "acf_ratio_2_1": np.nan}
    if n < 16 or f0 is None or not np.isfinite(f0) or f0 <= 0:
        return out
    step = int(round(fps / max(f0, 1e-6)))
    if step < 1 or step >= n: return out
    def acf(v, k):
        if k >= len(v): return np.nan
        v0 = v - np.mean(v)
        num = np.sum(v0[:-k]*v0[k:])
        den = np.sum(v0*v0) + 1e-12
        return float(num/den)
    a1 = acf(x, step)
    a2 = acf(x, 2*step) if 2*step < n else np.nan
    out["acf_at_step"] = a1
    out["acf_at_2step"] = a2
    out["acf_ratio_2_1"] = (a2/(abs(a1)+1e-6)) if np.isfinite(a1) else np.nan
    return out


# -------------------- Extract --------------------

def extract_features(path: str, schema: Schema, fps: int = 30, stride: int = 1,
                     fast_axes_flag: bool = True, turn_curv_thr: float = 0.15) -> pd.DataFrame:
    """Возвращает одну строку фич: биомех по шагам + новые + общестат/спектр."""
    A = load_npy(path, schema)
    if stride > 1:
        A = A[::stride]
        fps = max(1, int(round(fps/stride)))
    T = A.shape[0]
    seg = {k: A[:, idx, :] for k, idx in schema.groups.items() if len(idx) > 0}
    if not seg:
        raise ValueError("Schema produced no segments")
    seg = center_and_scale(seg)
    cogs = {k: centroid_series(v) for k, v in seg.items()}
    axes = fast_segment_axes(cogs) if fast_axes_flag else {}

    pelvis = cogs.get("pelvis")
    if pelvis is None: raise ValueError("No pelvis in schema")

    # curvature & straightness per frame
    horiz = pelvis[:,[0,2]]
    net = np.linalg.norm(horiz[-1]-horiz[0])
    seglen = np.linalg.norm(np.diff(horiz, axis=0), axis=1)
    path_straightness = float(net / (seglen.sum()+1e-6))
    if len(horiz) >= 3:
        v = np.gradient(horiz, axis=0)
        a = np.gradient(v, axis=0)
        numc = np.abs(v[:,0]*a[:,1]-v[:,1]*a[:,0])
        denc = (np.linalg.norm(v,axis=1)**3 + 1e-12)
        kappa = numc/denc
        path_curv_mean = float(np.nanmean(kappa))
        path_curv_rms  = float(np.sqrt(np.nanmean(kappa*kappa)))
    else:
        kappa = np.zeros(len(horiz))
        path_curv_mean = 0.0; path_curv_rms = 0.0

    # step width median
    step_width_ml = None
    if "L_foot" in cogs and "R_foot" in cogs:
        step_width_ml = float(np.nanmedian(np.abs(cogs["L_foot"][:,2] - cogs["R_foot"][:,2])))

    # Harmonic ratio + cadence
    hr_vert, f0 = _harmonic_ratio(pelvis[:,1], fps, cadence_hz=None)
    hr_ml,   _  = _harmonic_ratio(pelvis[:,2], fps, cadence_hz=f0 if np.isfinite(f0) else None)

    # spectral extras
    specY = spectral_extras(pelvis[:,1], fps, f0 if np.isfinite(f0) else None)
    specZ = spectral_extras(pelvis[:,2], fps, f0 if np.isfinite(f0) else None)
    acfY  = acf_regularities(pelvis[:,1], fps, f0 if np.isfinite(f0) else None)
    acfZ  = acf_regularities(pelvis[:,2], fps, f0 if np.isfinite(f0) else None)

    # collect step cycles
    rows = []
    cadence_list = []
    foot_clearances = []
    cycle_kappa_median = []  # для turn-mask

    for side in ("L", "R"):
        foot = cogs.get(f"{side}_foot"); sh = axes.get(f"{side}_shank")
        th  = axes.get(f"{side}_thigh"); pel_ax = axes.get("pelvis")
        if foot is None or pel_ax is None:
            continue
        fs, fo = detect_gait_events(foot, fps=fps, f0=f0 if np.isfinite(f0) else None)
        if len(fs) < 2:  # fallback одна «ступень»
            fs = np.array([0, T-1], dtype=int)

        knee = angle_between(th, sh) if (th is not None and sh is not None) else np.zeros(T)
        hip  = angle_between(pel_ax, th) if (pel_ax is not None and th is not None) else np.zeros(T)
        foot_axis = axes.get(f"{side}_foot")
        ankle = angle_between(sh, foot_axis) if (sh is not None and foot_axis is not None) else np.zeros(T)

        for i in range(len(fs)-1):
            a, b = fs[i], fs[i+1]
            if b <= a+1:
                continue
            segslice = slice(a, b)
            step_t = (b-a)/fps
            cadence = 60.0/step_t if step_t>1e-6 else np.nan
            cadence_list.append(cadence)

            # turn-ness этого шага
            med_k = float(np.nanmedian(kappa[segslice]))
            cycle_kappa_median.append(med_k)

            pel_y = pelvis[:,1]
            pel_z = pelvis[:,2]  # ML
            pel_x = pelvis[:,0]  # AP
            vo = float(np.max(pel_y[segslice]) - np.min(pel_y[segslice]))
            ml = float(np.max(pel_z[segslice]) - np.min(pel_z[segslice]))
            ap = float(np.max(pel_x[segslice]) - np.min(pel_x[segslice]))
            # step length via pelvis horiz displacement
            p0 = pelvis[a, [0,2]]; p1 = pelvis[b, [0,2]]
            step_len = float(np.linalg.norm(p1 - p0))
            # duty factor (stance_time / step_time) using FO inside
            fo_in = fo[(fo > a) & (fo < b)]
            stance = float((fo_in[0]-a)/fps) if len(fo_in) else np.nan
            duty = (stance/step_t) if (np.isfinite(stance) and step_t>0) else np.nan
            speed = step_len/step_t if step_t>1e-6 else 0.0

            # foot clearance: swing peak (FO -> next FS)
            clearance = np.nan
            if len(fo_in):
                st = int(fo_in[0]); en = b
                yy = foot[st:en,1]
                if yy.size>2:
                    base = np.nanmedian(foot[max(a-3,0):min(a+3,T),1])
                    clearance = float(np.nanmax(yy) - base)
            foot_clearances.append(clearance)

            rows.append(dict(
                side=0 if side=="L" else 1,
                step_time=step_t, cadence=cadence, step_len=step_len, step_speed=speed,
                pelvis_vert_osc=vo, pelvis_ml_osc=ml, pelvis_ap_osc=ap,
                rom_knee=float(np.max(knee[segslice]) - np.min(knee[segslice])),
                rom_hip=float(np.max(hip[segslice])  - np.min(hip[segslice])),
                rom_ankle=float(np.max(ankle[segslice]) - np.min(ankle[segslice])),
                stance_time=stance, duty_factor=duty,
                kappa_med=med_k,
                foot_clearance=clearance,
            ))

    df = pd.DataFrame(rows)

    def _empty_row():
        return dict(
            step_time_median=np.nan, step_time_mean=np.nan, step_time_std=0.0,
            cadence_median=np.nan, cadence_mean=np.nan, cadence_std=0.0,
            step_len_median=0.0, step_len_mean=0.0, step_len_std=0.0,
            step_speed_median=0.0, step_speed_mean=0.0, step_speed_std=0.0,
            pelvis_vert_osc_median=0.0, pelvis_vert_osc_mean=0.0, pelvis_vert_osc_std=0.0,
            pelvis_ml_osc_median=0.0, pelvis_ml_osc_mean=0.0, pelvis_ml_osc_std=0.0,
            pelvis_ap_osc_median=0.0, pelvis_ap_osc_mean=0.0, pelvis_ap_osc_std=0.0,
            rom_knee_median=0.0, rom_knee_mean=0.0, rom_knee_std=0.0,
            rom_hip_median=0.0,  rom_hip_mean=0.0,  rom_hip_std=0.0,
            rom_ankle_median=0.0, rom_ankle_mean=0.0, rom_ankle_std=0.0,
            stance_time_median=np.nan, stance_time_mean=np.nan, stance_time_std=0.0,
            duty_factor_median=np.nan, duty_factor_mean=np.nan, duty_factor_std=0.0,
            cv_step_time=np.nan, cv_step_len=np.nan, cv_cadence=np.nan, cv_stance_time=np.nan,
            iqr_step_time=np.nan, iqr_step_len=np.nan, iqr_cadence=np.nan, iqr_stance_time=np.nan,
            rmssd_step_time=np.nan, rmssd_step_len=np.nan, rmssd_cadence=np.nan,
            drift_cadence_std=np.nan,
            asym_step_time=np.nan, asym_step_len=np.nan, asym_rom_knee=np.nan,
            asym_rom_hip=np.nan, asym_rom_ankle=np.nan, asym_stance_time=np.nan,
            asym_cadence=np.nan, asym_pelvis_vert_osc=np.nan, asym_pelvis_ml_osc=np.nan, asym_pelvis_ap_osc=np.nan,
            si_step_len=np.nan, si_step_time=np.nan, si_cadence=np.nan,
            step_width_ml=float(step_width_ml) if step_width_ml is not None else np.nan,
            step_width_std=np.nan, step_width_cv=np.nan,
            foot_clearance_median=np.nan, foot_clearance_std=np.nan,
            cycles_count=0,
            path_straightness=path_straightness,
            path_curv_mean=path_curv_mean, path_curv_rms=path_curv_rms,
            hr_vert=np.nan, hr_ml=np.nan, cadence_hz_est=np.nan,
            specY_flatness=np.nan, specY_centroid=np.nan, specY_spread=np.nan,
            specY_peak_width_hz=np.nan, specY_peak_prominence=np.nan,
            specZ_flatness=np.nan, specZ_centroid=np.nan, specZ_spread=np.nan,
            specZ_peak_width_hz=np.nan, specZ_peak_prominence=np.nan,
            acfY_step=np.nan, acfY_2step=np.nan, acfY_ratio=np.nan,
            acfZ_step=np.nan, acfZ_2step=np.nan, acfZ_ratio=np.nan,
            ds_ratio=np.nan,
            cc_AP=np.nan, cc_AP_lag=np.nan, cc_ML=np.nan, cc_ML_lag=np.nan,
            lag_ap_std=np.nan, lag_ml_std=np.nan,
            freq_peak_power=np.nan,
            outlier_rate=0.0,
            # straight duplicates
            straight_frac=np.nan,
            step_len_mean_straight=np.nan, step_time_mean_straight=np.nan, cadence_mean_straight=np.nan,
        )

    if df.empty:
        out = _empty_row()
        A2 = A.reshape(A.shape[0], -1).astype(np.float32, copy=False)
        raw_vals, raw_names = _extract_generic_feats(A2)
        for k, name in enumerate(raw_names):
            out[name] = float(raw_vals[k])

        return pd.DataFrame([out])

    # ---- Aggregates over cycles (all)
    num = df.select_dtypes(include=[np.number]).copy()
    side_col = num.pop("side")
    agg_tbl = num.agg(["median","mean","std"]).T
    flat = {f"{feat}_{stat}": float(agg_tbl.loc[feat, stat])
            for feat in agg_tbl.index for stat in agg_tbl.columns}

    # CV/IQR/RMSSD/Drift
    def _roll_median_std(v: np.ndarray, w: int = 5) -> float:
        v = np.asarray(v, float)
        if v.size < w+2: return np.nan
        med = pd.Series(v).rolling(window=w, center=True, min_periods=w//2).median()
        return float(np.nanstd(med.values))

    for col in ["step_time","step_len","cadence","stance_time","pelvis_vert_osc","pelvis_ml_osc","pelvis_ap_osc","duty_factor","foot_clearance"]:
        v = num[col].values if col in num else np.array([])
        if np.isfinite(v).sum() >= 2:
            m = np.nanmean(v); s = np.nanstd(v)
            flat[f"cv_{col}"] = float(s/(abs(m)+1e-6))
            flat[f"iqr_{col}"] = iqr(v)
            if col in ("step_time","step_len","cadence"):
                flat[f"rmssd_{col}"] = rmssd(v)
                if col=="cadence":
                    flat["drift_cadence_std"] = _roll_median_std(v, w=5)
        else:
            flat[f"cv_{col}"] = np.nan; flat[f"iqr_{col}"] = np.nan
            if col in ("step_time","step_len","cadence"):
                flat[f"rmssd_{col}"] = np.nan
                if col=="cadence": flat["drift_cadence_std"] = np.nan

    # step width variability
    if "L_foot" in cogs and "R_foot" in cogs:
        sw = np.abs(cogs["L_foot"][:,2] - cogs["R_foot"][:,2])
        flat["step_width_ml"] = float(np.nanmedian(sw))
        flat["step_width_std"] = float(np.nanstd(sw))
        flat["step_width_cv"]  = float(np.nanstd(sw)/(np.nanmean(sw)+1e-6))
    else:
        flat["step_width_ml"] = float(step_width_ml) if step_width_ml is not None else np.nan
        flat["step_width_std"] = np.nan
        flat["step_width_cv"]  = np.nan

    # asymmetry L/R + SI
    if (side_col==0).any() and (side_col==1).any():
        def med(side, col):
            vv = num.loc[side_col==side, col]
            return float(np.median(vv)) if len(vv) else np.nan
        for col in ["step_time","step_len","rom_knee","rom_hip","rom_ankle","stance_time","cadence","pelvis_vert_osc","pelvis_ml_osc","pelvis_ap_osc","foot_clearance"]:
            L = med(0,col); R = med(1,col)
            denom = (abs(L)+abs(R))/2 + 1e-6
            flat[f"asym_{col}"] = abs(L-R)/denom
            if col in ("step_len","step_time","cadence"):
                flat[f"si_{col}"] = (L-R)/((L+R)/2 + 1e-6)
    else:
        for col in ("step_len","step_time","cadence"):
            flat[f"si_{col}"] = np.nan

    # jerk RMS of pelvis axes и нормированный jerk
    def jerk_rms(sig):
        v  = np.gradient(sig)
        a  = np.gradient(v)
        j  = np.gradient(a)
        return float(np.sqrt(np.nanmean(j*j)))
    flat["pelvis_jerk_rms_y"] = jerk_rms(pelvis[:,1])
    flat["pelvis_jerk_rms_z"] = jerk_rms(pelvis[:,2])
    flat["pelvis_jerk_rms_x"] = jerk_rms(pelvis[:,0])
    mean_step_t = np.nanmean(num["step_time"]) if "step_time" in num else np.nan
    mean_step_l = np.nanmean(num["step_len"]) if "step_len" in num else np.nan
    denom_norm = (abs(mean_step_t)*abs(mean_step_l) + 1e-6)
    flat["pelvis_jerk_norm"] = (flat["pelvis_jerk_rms_y"]+flat["pelvis_jerk_rms_z"])/denom_norm

    # path metrics
    flat["path_straightness"] = path_straightness
    flat["path_curv_mean"] = path_curv_mean
    flat["path_curv_rms"]  = path_curv_rms

    # HR + cadence estimate
    flat["hr_vert"] = float(hr_vert) if np.isfinite(hr_vert) else np.nan
    flat["hr_ml"]   = float(hr_ml) if np.isfinite(hr_ml) else np.nan
    flat["cadence_hz_est"] = float(f0) if np.isfinite(f0) else np.nan

    # spectral extras записываем
    flat.update({
        "specY_flatness": specY["spec_flatness"],
        "specY_centroid": specY["spec_centroid"],
        "specY_spread":   specY["spec_spread"],
        "specY_peak_width_hz": specY["peak_width_hz"],
        "specY_peak_prominence": specY["peak_prominence"],
        "specZ_flatness": specZ["spec_flatness"],
        "specZ_centroid": specZ["spec_centroid"],
        "specZ_spread":   specZ["spec_spread"],
        "specZ_peak_width_hz": specZ["peak_width_hz"],
        "specZ_peak_prominence": specZ["peak_prominence"],
        "acfY_step":  acfY["acf_at_step"],
        "acfY_2step": acfY["acf_at_2step"],
        "acfY_ratio": acfY["acf_ratio_2_1"],
        "acfZ_step":  acfZ["acf_at_step"],
        "acfZ_2step": acfZ["acf_at_2step"],
        "acfZ_ratio": acfZ["acf_ratio_2_1"],
    })

    # Frequency peak power near cadence (vertical pelvis)
    if np.isfinite(f0):
        x = pelvis[:,1]-np.nanmean(pelvis[:,1]); n=len(x)
        if n>=16:
            fft=np.fft.rfft(x*np.hanning(n)); p=(fft.real**2+fft.imag**2); freqs=np.fft.rfftfreq(n, d=1.0/max(fps,1))
            mask=(freqs>=max(0.3,f0*0.7))&(freqs<=min(5.0,f0*1.3))
            flat["freq_peak_power"] = float(np.max(p[mask])) if mask.any() else np.nan
        else:
            flat["freq_peak_power"]=np.nan
    else:
        flat["freq_peak_power"]=np.nan

    # Cross-correlation L/R (AP=x, ML=z) и lag-consistency по шагам
    lag_ap_list=[]; lag_ml_list=[]; cc_ap_list=[]; cc_ml_list=[]
    if "L_foot" in cogs and "R_foot" in cogs:
        maxlag = int(0.5*fps)
        cc_ap, lag_ap = _crosscorr_max(cogs["L_foot"][:,0], cogs["R_foot"][:,0], maxlag)
        cc_ml, lag_ml = _crosscorr_max(cogs["L_foot"][:,2], cogs["R_foot"][:,2], maxlag)
        flat["cc_AP"] = float(cc_ap); flat["cc_AP_lag"]=float(lag_ap/fps)
        flat["cc_ML"] = float(cc_ml); flat["cc_ML_lag"]=float(lag_ml/fps)
        # по циклам: лаг по сегментам FS_i..FS_{i+1} (по AP и ML)
        # возьмём интервалы по левой ноге как опору, если есть
        Lfs,_ = detect_gait_events(cogs["L_foot"], fps=fps, f0=f0 if np.isfinite(f0) else None)
        for i in range(len(Lfs)-1):
            a,b = Lfs[i], Lfs[i+1]
            if b<=a+3: continue
            ca, la = _crosscorr_max(cogs["L_foot"][a:b,0], cogs["R_foot"][a:b,0], min(maxlag, (b-a)//2))
            cm, lm = _crosscorr_max(cogs["L_foot"][a:b,2], cogs["R_foot"][a:b,2], min(maxlag, (b-a)//2))
            if np.isfinite(la): lag_ap_list.append(la/fps)
            if np.isfinite(lm): lag_ml_list.append(lm/fps)
            if np.isfinite(ca): cc_ap_list.append(ca)
            if np.isfinite(cm): cc_ml_list.append(cm)
    else:
        flat["cc_AP"]=np.nan; flat["cc_AP_lag"]=np.nan; flat["cc_ML"]=np.nan; flat["cc_ML_lag"]=np.nan
    flat["lag_ap_std"] = float(np.nanstd(lag_ap_list)) if len(lag_ap_list) else np.nan
    flat["lag_ml_std"] = float(np.nanstd(lag_ml_list)) if len(lag_ml_list) else np.nan

    # Double-support ratio (approx)
    ds_vals=[]
    if "L_foot" in cogs and "R_foot" in cogs:
        fsL, foL = detect_gait_events(cogs["L_foot"], fps=fps, f0=f0 if np.isfinite(f0) else None)
        fsR, foR = detect_gait_events(cogs["R_foot"], fps=fps, f0=f0 if np.isfinite(f0) else None)
        def make_intervals(fs,fo):
            out=[]
            for s in fs:
                f_after = fo[fo>s]
                if len(f_after):
                    out.append((s, int(f_after[0])))
            return out
        intL = make_intervals(fsL, foL); intR = make_intervals(fsR, foR)
        mean_step = np.nanmean(num["step_time"]) if "step_time" in num else np.nan
        if np.isfinite(mean_step) and mean_step>0:
            i,j=0,0
            while i<len(intL) and j<len(intR):
                a0,a1=intL[i]; b0,b1=intR[j]
                ov = max(0, min(a1,b1)-max(a0,b0))
                if ov>0: ds_vals.append(ov/(mean_step*fps))
                if a1<b1: i+=1
                else: j+=1
    flat["ds_ratio"] = float(np.nanmean(ds_vals)) if ds_vals else np.nan

    # Outlier rate across per-step key features
    key = ["step_len","step_time","cadence","stance_time","pelvis_vert_osc","pelvis_ml_osc"]
    count_out = 0; count_all = 0
    for k in key:
        if k not in num: continue
        v = num[k].values; m = np.isfinite(v); vv = v[m]
        if vv.size < 3: continue
        med = np.median(vv)
        i = np.percentile(vv, 75) - np.percentile(vv, 25)
        lo = med - 1.5 * i; hi = med + 1.5 * i
        out = np.sum((vv < lo) | (vv > hi))
        count_out += int(out); count_all += int(vv.size)
    flat["outlier_rate"] = float(count_out / max(1, count_all))

    # cycles_count
    flat["cycles_count"] = int(len(num))

    # straight-only агрегаты (turn-mask)
    straight_mask = (df["kappa_med"].values <= float(turn_curv_thr)) if "kappa_med" in df.columns else np.zeros(len(df), dtype=bool)
    straight_frac = float(np.mean(straight_mask)) if len(df) else np.nan
    flat["straight_frac"] = straight_frac
    if straight_mask.any():
        ssub = df.loc[straight_mask]
        flat["step_len_mean_straight"]  = float(np.nanmean(ssub["step_len"]))
        flat["step_time_mean_straight"] = float(np.nanmean(ssub["step_time"]))
        flat["cadence_mean_straight"]   = float(np.nanmean(ssub["cadence"]))
    else:
        flat["step_len_mean_straight"]=np.nan
        flat["step_time_mean_straight"]=np.nan
        flat["cadence_mean_straight"]=np.nan

    # foot clearance agg
    fc = df["foot_clearance"].values if "foot_clearance" in df else np.array([])
    flat["foot_clearance_median"] = float(np.nanmedian(fc)) if np.isfinite(fc).any() else np.nan
    flat["foot_clearance_std"]    = float(np.nanstd(fc))    if np.isfinite(fc).any() else np.nan

    # ---- Add generic per-channel stats/spectrum 31*F from raw array
    A2 = A.reshape(A.shape[0], -1).astype(np.float32, copy=False)
    raw_vals, raw_names = _extract_generic_feats(A2)
    for k, name in enumerate(raw_names):
        out[name] = float(raw_vals[k])

    return pd.DataFrame([flat])

# -------------------- Parallel dataset build --------------------

def _extract_one(args):
    p, fn, y_val, fps, stride, fast_axes, turn_thr = args
    schema = load_schema(schema_path_global)
    feats = extract_features(p, schema, fps=fps, stride=stride, fast_axes_flag=fast_axes, turn_curv_thr=turn_thr)
    d = feats.iloc[0].to_dict()
    return d, int(y_val), guess_subject_id(fn), p


def build_table(manifest_csv: str, data_dir: str, schema: Schema, fps: int,
                workers: int = 1, stride: int = 1, fast_axes: bool = True, turn_thr: float = 0.15
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
        tasks.append((p, fn, y_val, fps, stride, fast_axes, turn_thr))
    if not tasks:
        raise RuntimeError("No valid .npy paths")

    print(f"[info] Extracting features: {len(tasks)} files | workers={workers} | stride={stride} | fast_axes={fast_axes} | turn_thr={turn_thr}")
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

# -------------------- Plots --------------------
import matplotlib
matplotlib.use("Agg")
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

# -------------------- Threshold search / metrics --------------------
from sklearn.metrics import (
    recall_score, precision_recall_curve, roc_auc_score,
    average_precision_score, confusion_matrix, classification_report,
    roc_curve, auc
)

def find_threshold_balanced(y_true, probs, target_r1=0.90, target_r0=0.96):
    best = {"thr": 0.5, "r1": 0.0, "r0": 0.0, "ok": False, "bal": -1.0}
    for thr in np.linspace(0.0, 1.0, 2001):
        pred = (probs >= thr).astype(int)
        r1 = recall_score(y_true, pred, pos_label=1)
        r0 = recall_score(y_true, pred, pos_label=0)
        bal = 0.5*(r1 + r0)
        if (r1 >= target_r1) and (r0 >= target_r0):
            if not best["ok"] or bal > best["bal"]:
                best = {"thr": float(thr), "r1": float(r1), "r0": float(r0), "ok": True, "bal": float(bal)}
        if not best["ok"] and bal > best["bal"]:
            best = {"thr": float(thr), "r1": float(r1), "r0": float(r0), "ok": False, "bal": float(bal)}
    return best


def plot_roc_pr_curves(y_true: np.ndarray, prob: np.ndarray, save_path: str, title_prefix: str):
    y_true = np.asarray(y_true).astype(int)
    prob   = np.asarray(prob).astype(float)
    if len(np.unique(y_true)) < 2:
        print(f"[warn] {title_prefix}: один класс -> ROC/PR не рисуем")
        return
    fpr, tpr, _ = roc_curve(y_true, prob); roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y_true, prob); auprc = average_precision_score(y_true, prob)
    fig = plt.figure(figsize=(9,4))
    ax1 = plt.subplot(1,2,1)
    ax1.plot(fpr, tpr, lw=2, label=f"AUC={roc_auc:.3f}"); ax1.plot([0,1],[0,1],"--", lw=1)
    ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR"); ax1.set_title(f"{title_prefix} ROC"); ax1.legend(loc="lower right")
    ax2 = plt.subplot(1,2,2)
    ax2.plot(rec, prec, lw=2, label=f"AP={auprc:.3f}")
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision"); ax2.set_title(f"{title_prefix} PR"); ax2.legend(loc="lower left")
    plt.tight_layout(); plt.savefig(save_path, dpi=160); plt.close(fig)
    base,_=os.path.splitext(save_path)
    pd.DataFrame({"fpr":fpr,"tpr":tpr}).to_csv(base+"_roc_points.csv", index=False)
    pd.DataFrame({"recall":rec,"precision":prec}).to_csv(base+"_pr_points.csv", index=False)

# -------------------- Calibration --------------------
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import joblib

class Calibrator:
    def __init__(self, method: str = "none"):
        self.method = method
        self.iso: Optional[IsotonicRegression] = None
        self.lr:  Optional[LogisticRegression] = None
    def fit(self, prob: np.ndarray, y: np.ndarray):
        if self.method == "isotonic":
            self.iso = IsotonicRegression(out_of_bounds='clip')
            self.iso.fit(prob, y)
        elif self.method == "sigmoid":
            x = prob.reshape(-1,1)
            self.lr = LogisticRegression(max_iter=1000)
            self.lr.fit(x, y)
        else:
            pass
    def transform(self, prob: np.ndarray) -> np.ndarray:
        if self.method == "isotonic" and self.iso is not None:
            return self.iso.predict(prob)
        if self.method == "sigmoid" and self.lr is not None:
            x = prob.reshape(-1,1)
            return self.lr.predict_proba(x)[:,1]
        return prob

# -------------------- Train XGBoost --------------------
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb


def train_xgb(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, files: np.ndarray,
              out_dir: str, use_gpu: bool,
              min_recall_pos: float, min_recall_neg: float,
              w0_scale: float = 1.0,
              calibrate: str = "isotonic",
              seed: int = 42):

    ensure_dir(out_dir)

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

    X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
    X_dv, y_dv = X.iloc[dv_idx], y[dv_idx]
    X_te, y_te = X.iloc[te_idx], y[te_idx]

    # balanced class weights + усиление веса здоровых
    classes = np.array([0, 1], dtype=np.int32)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_tr)
    w0, w1 = float(cw[0])*float(w0_scale), float(cw[1])
    print(f"[train] class_weight(balanced)*scales: w0(NoInjury)={w0:.3f}, w1(Injury)={w1:.3f}")
    sw_tr = np.where(y_tr == 0, w0, w1).astype(np.float32)
    sw_dv = np.where(y_dv == 0, w0, w1).astype(np.float32)

    # DMatrix
    dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=sw_tr)
    dvalid = xgb.DMatrix(X_dv, label=y_dv, weight=sw_dv)
    dtest  = xgb.DMatrix(X_te, label=y_te)

    params = dict(
        objective="binary:logistic",
        learning_rate=0.03,
        max_depth=5,
        min_child_weight=6,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        reg_alpha=0.0,
        tree_method="gpu_hist" if use_gpu else "hist",
        eval_metric="logloss",
        seed=seed,
    )
    num_boost_round = 2000

    print("[train] XGBoost (xgb.train) fitting...")
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=100,
        verbose_eval=50,
    )

    # Probabilities
    prob_dv = bst.predict(dvalid, iteration_range=(0, bst.best_iteration + 1))
    prob_te = bst.predict(dtest,  iteration_range=(0, bst.best_iteration + 1))

    # Calibration
    cal = Calibrator(method=calibrate)
    cal_fitted = False
    if calibrate.lower() != "none" and len(np.unique(y_dv))==2:
        cal.fit(prob_dv, y_dv)
        prob_dv = cal.transform(prob_dv)
        prob_te = cal.transform(prob_te)
        joblib.dump(cal, os.path.join(out_dir, "calibrator.joblib"))
        cal_fitted = True
        print(f"[calibration] fitted {calibrate} calibrator on DEV")
    else:
        print("[calibration] skipped (none or single-class DEV)")

    # ROC/PR numbers + plots
    auprc_dv = average_precision_score(y_dv, prob_dv)
    auroc_dv = roc_auc_score(y_dv, prob_dv)
    auprc_te = average_precision_score(y_te, prob_te)
    auroc_te = roc_auc_score(y_te, prob_te)
    plot_roc_pr_curves(y_dv, prob_dv, os.path.join(out_dir, "roc_pr_dev.png"),  "DEV")
    plot_roc_pr_curves(y_te, prob_te, os.path.join(out_dir, "roc_pr_test.png"), "TEST")

    # Threshold tuned on DEV (приоритет здоровых)
    best = find_threshold_balanced(y_dv, prob_dv,
                                   target_r1=min_recall_pos,
                                   target_r0=min_recall_neg)
    thr = best["thr"]
    print(f"[threshold] chosen={thr:.4f} | dev recall(1)={best['r1']:.3f} recall(0)={best['r0']:.3f} ok={best['ok']}")

    # Test report
    pred_te = (prob_te >= thr).astype(int)
    cm = confusion_matrix(y_te, pred_te)
    rep = classification_report(y_te, pred_te, digits=3, output_dict=False)
    print("\n=== TEST REPORT ===\n", rep)

    # Save metrics
    metrics = {
        "dev_auprc": float(auprc_dv), "dev_auroc": float(auroc_dv),
        "test_auprc": float(auprc_te), "test_auroc": float(auroc_te),
        "threshold": float(thr),
        "threshold_search_ok": bool(best["ok"]),
        "dev_recall_injury": float(best["r1"]), "dev_recall_noinj": float(best["r0"]),
        "test_cm": cm.tolist(),
        "test_report": rep,
        "calibration": calibrate if cal_fitted else "none",
        "w0_scale": w0_scale,
    }
    json.dump(metrics, open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    open(os.path.join(out_dir, "threshold.txt"), "w").write(str(thr))

    # Confusion matrices
    class_names = ["No Injury", "Injury"]
    plot_confusion_matrix(cm, class_names, normalize=False,
                          save_path=os.path.join(out_dir, "confusion_matrix_counts.png"),
                          title="Confusion Matrix (counts)")
    plot_confusion_matrix(cm, class_names, normalize=True,
                          save_path=os.path.join(out_dir, "confusion_matrix_normalized.png"),
                          title="Confusion Matrix (row-normalized)")

    # Feature importance (top-25)
    try:
        score = bst.get_score(importance_type="gain")
        if len(score) > 0:
            items = sorted(score.items(), key=lambda x: x[1], reverse=True)[:25]
            names, vals = zip(*items)
            plt.figure(figsize=(8,10))
            y_pos = np.arange(len(names))
            plt.barh(y_pos, vals)
            plt.yticks(y_pos, names)
            plt.gca().invert_yaxis()
            plt.xlabel("Gain")
            plt.title("XGBoost Feature Importance (top-25)")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "feature_importance_top25.png"), dpi=160)
            plt.close()
    except Exception as e:
        print("[warn] feature importance plot failed:", e)

    # Save model + feature columns
    bst.save_model(os.path.join(out_dir, "xgb.json"))
    json.dump(list(X.columns), open(os.path.join(out_dir, "features_cols.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("Saved to:", out_dir)
    return bst, thr, (X_te, y_te, prob_te)

# -------------------- Cache helpers --------------------

def extract_and_save(manifest_csv: str, data_dir: str, schema_path: str, fps: int, out_dir: str,
                     workers: int = 1, stride: int = 1, fast_axes: bool = True, turn_thr: float = 0.15):
    ensure_dir(out_dir)
    global schema_path_global
    schema_path_global = schema_path
    schema = load_schema(schema_path)
    X, y, groups, files = build_table(manifest_csv, data_dir, schema, fps=fps,
                                      workers=workers, stride=stride, fast_axes=fast_axes, turn_thr=turn_thr)
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
        "turn_curv_thr": float(turn_thr),
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

# -------------------- Predict --------------------

def predict_one(npy_path: str, schema_path: str, out_dir: str, fps: int = 30, stride: int = 1, fast_axes: bool = True, turn_thr: float = 0.15):
    schema = load_schema(schema_path)
    feats = extract_features(npy_path, schema, fps=fps, stride=stride, fast_axes_flag=fast_axes, turn_curv_thr=turn_thr)
    cols = json.load(open(os.path.join(out_dir, "features_cols.json"), "r", encoding="utf-8"))
    X = feats.reindex(columns=cols, fill_value=0.0)
    import xgboost as xgb
    bst = xgb.Booster()
    bst.load_model(os.path.join(out_dir, "xgb.json"))
    thr = float(open(os.path.join(out_dir, "threshold.txt")).read().strip())
    prob = float(bst.predict(xgb.DMatrix(X))[0])
    # apply calibrator if exists
    cal_path = os.path.join(out_dir, "calibrator.joblib")
    if os.path.exists(cal_path):
        cal: Calibrator = joblib.load(cal_path)
        prob = float(cal.transform(np.array([prob]))[0])
    pred = int(prob >= thr)
    return {"file": npy_path, "prob_injury": prob, "pred": pred, "threshold": thr}

# -------------------- CLI --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["extract","train","train_features","predict"], default="train")
    ap.add_argument("--csv", help="manifest.csv (для extract/train)")
    ap.add_argument("--data_dir", help="папка с .npy (для extract/train)")
    ap.add_argument("--schema", required=True, help="schema_joints.json")
    ap.add_argument("--out_dir", default="out_xgb_all")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--stride", type=int, default=2, help="темп прореживания кадров")
    ap.add_argument("--fast_axes", action="store_true", help="быстрые оси сегментов (без SVD)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)-1), help="число процессов при extract/build")
    ap.add_argument("--turn_curv_thr", type=float, default=0.15, help="порог кривизны для straight шагов")

    ap.add_argument("--use_gpu", action="store_true", help="XGBoost GPU (gpu_hist)")
    ap.add_argument("--min_recall_injury", type=float, default=0.90, help="цель по recall класса 1")
    ap.add_argument("--min_recall_noinj",  type=float, default=0.96, help="цель по recall класса 0")
    ap.add_argument("--w0_scale", type=float, default=1.0, help="доп. масштаб веса для No-Injury в трейне")
    ap.add_argument("--calibrate", choices=["isotonic","sigmoid","none"], default="isotonic", help="калибровка вероятностей по DEV")

    # cache paths
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
                         out_dir=args.out_dir, workers=args.workers, stride=args.stride,
                         fast_axes=args.fast_axes, turn_thr=args.turn_curv_thr)
        return

    if args.mode == "train":
        assert args.csv and args.data_dir, "--csv и --data_dir обязательны"
        schema = load_schema(args.schema)
        X, y, groups, files = build_table(args.csv, args.data_dir, schema, fps=args.fps,
                                          workers=args.workers, stride=args.stride, fast_axes=args.fast_axes, turn_thr=args.turn_curv_thr)
        train_xgb(X, y, groups, files, args.out_dir, use_gpu=args.use_gpu,
                  min_recall_pos=args.min_recall_injury, min_recall_neg=args.min_recall_noinj,
                  w0_scale=args.w0_scale, calibrate=args.calibrate)
        return

    if args.mode == "train_features":
        features = args.features or os.path.join(args.out_dir, "features.parquet")
        labels   = args.labels   or os.path.join(args.out_dir, "labels.npy")
        groups   = args.groups   or os.path.join(args.out_dir, "groups.npy")
        files_l  = args.files_list or os.path.join(args.out_dir, "files.txt")
        X, y, groups, files = load_features_cache(features, labels, groups, files_l)
        train_xgb(X, y, groups, files, args.out_dir, use_gpu=args.use_gpu,
                  min_recall_pos=args.min_recall_injury, min_recall_neg=args.min_recall_noinj,
                  w0_scale=args.w0_scale, calibrate=args.calibrate)
        return

    # predict
    assert args.npy, "--npy обязателен в режиме predict"
    res = predict_one(args.npy, args.schema, args.out_dir, fps=args.fps, stride=args.stride, fast_axes=args.fast_axes, turn_thr=args.turn_curv_thr)
    print(json.dumps(res, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()