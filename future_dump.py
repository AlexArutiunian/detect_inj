#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NPY → gait features (features-only).

- Рекурсивно обходит папку с .npy
- Загрузка схемы суставов (schema_joints.json)
- Нормализация: центрирование по тазу, масштаб по бедро↔голень
- Детектор шагов по вертикали стопы (FS/FO)
- Перешаговые признаки и агрегаты
- Спектр/HR/ACF и метрики «прямолинейности»
- Сохранение: общий features.csv (+ опц. parquet) и per-file CSV (по флагу)

Требования: numpy, pandas, scipy, tqdm
"""

import os
import re
import json
import argparse
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import find_peaks, peak_widths, peak_prominences


# -------------------- Утилиты --------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def guess_subject_id(filename: str) -> str:
    b = Path(filename).stem
    m = re.match(r"(\d{8})", b)
    return m.group(1) if m else b.split("T")[0][:8]


# -------------------- Схема / ввод --------------------

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
    return Schema(names=names, groups=groups)

def load_npy(path: str, schema: Schema) -> np.ndarray:
    arr = np.load(path, allow_pickle=False)
    if arr.ndim == 2 and arr.shape[1] == 3 * len(schema.names):
        arr = arr.reshape(arr.shape[0], len(schema.names), 3)
    elif arr.ndim == 3 and arr.shape[2] == 3 and arr.shape[1] == len(schema.names):
        pass
    else:
        raise ValueError(f"Unexpected array shape {arr.shape} for N={len(schema.names)}")
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    return arr  # (T, N, 3)


# -------------------- Биомех-хелперы --------------------

def centroid_series(x: np.ndarray) -> np.ndarray:
    # x: (T, K, 3) -> (T, 3)
    return x.mean(axis=1)

def center_and_scale(all_xyz: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    pelvis = all_xyz.get("pelvis")
    if pelvis is not None:
        c = centroid_series(pelvis)
        for k in all_xyz:
            all_xyz[k] = all_xyz[k] - c[:, None, :]
    scale_list = []
    for side in ("L", "R"):
        th = all_xyz.get(f"{side}_thigh")
        sh = all_xyz.get(f"{side}_shank")
        if th is not None and sh is not None:
            th_c = centroid_series(th); sh_c = centroid_series(sh)
            d = np.linalg.norm(th_c - sh_c, axis=1)
            pos = d[d > 0]
            if pos.size:
                scale_list.append(np.median(pos))
    scale = float(np.median(scale_list)) if scale_list else 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    for k in all_xyz:
        all_xyz[k] = all_xyz[k] / scale
    return all_xyz

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
        if thL is not None and thR is not None: tgt = (thL + thR) / 2.0
        elif thL is not None: tgt = thL
        else: tgt = thR
        v = tgt - pel
        axes["pelvis"] = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
    return axes


# -------------------- События шага и спектр --------------------

def detect_gait_events(foot_centroid: np.ndarray, fps: int = 30, f0: Optional[float] = None):
    """Возвращает индексы FS/FO (по вертикали стопы)."""
    y = foot_centroid[:, 1]
    if len(y) >= 5:
        y = pd.Series(y).rolling(window=5, center=True, min_periods=1).median().values
    if f0 is None or not np.isfinite(f0):
        min_dist = max(5, int(0.3 * fps))
    else:
        min_dist = max(5, int((0.5 / max(f0, 1e-3)) * fps))
    peaks, _ = find_peaks(y, distance=min_dist)  # mid-swing
    if len(peaks) < 2:
        step = max(8, int(0.5 * fps))
        peaks = np.arange(step, len(y) - 1, step)
    fs = []; fo = []
    vy = np.gradient(y) if len(y) > 2 else np.zeros_like(y)
    for k in range(len(peaks) - 1):
        a, b = peaks[k], peaks[k + 1]
        seg = y[a:b+1]
        if seg.size < 3: continue
        i_min = int(np.argmin(seg)) + a
        fs.append(i_min)
        if k == 0:
            pre = y[:peaks[k]+1]
            if pre.size >= 3:
                fo.append(int(np.argmin(pre)))
        else:
            pre = y[peaks[k-1]:peaks[k]+1]
            if pre.size >= 3:
                fo.append(int(np.argmin(pre)) + peaks[k-1])
    return np.array(fs, int), np.array(fo, int)

def _harmonic_ratio(x: np.ndarray, fps: int, cadence_hz: Optional[float] = None) -> Tuple[float, float]:
    x = np.asarray(x, np.float32)
    x = x - np.nanmean(x)
    n = len(x)
    if n < 16:
        return np.nan, np.nan
    fft = np.fft.rfft(x * np.hanning(n))
    p = (fft.real**2 + fft.imag**2)
    freqs = np.fft.rfftfreq(n, d=1.0/max(fps,1))
    if cadence_hz is None or not np.isfinite(cadence_hz):
        mask = (freqs >= 0.5) & (freqs <= 3.0)
        if not mask.any(): return np.nan, np.nan
        k = np.argmax(p[mask])
        f0 = float(freqs[mask][k])
    else:
        f0 = float(cadence_hz)
    def band_power(mult):
        f = mult * f0
        idx = np.argmin(np.abs(freqs - f))
        return float(p[idx])
    even = sum(band_power(2*m) for m in range(1, 4))
    odd  = sum(band_power(2*m-1) for m in range(1, 4))
    hr = even / (odd + 1e-8)
    return float(hr), f0

def spectral_extras(x: np.ndarray, fps: int, f0: Optional[float]) -> Dict[str, float]:
    x = np.asarray(x, np.float32)
    x = x - np.nanmean(x)
    n = len(x)
    out = dict(spec_flatness=np.nan, spec_centroid=np.nan, spec_spread=np.nan,
               peak_width_hz=np.nan, peak_prominence=np.nan)
    if n < 16:
        return out
    fft = np.fft.rfft(x * np.hanning(n))
    p = (fft.real**2 + fft.imag**2)
    p = np.clip(p, 1e-12, None)
    freqs = np.fft.rfftfreq(n, d=1.0/max(fps,1))
    gm = float(np.exp(np.mean(np.log(p)))); am = float(np.mean(p))
    out["spec_flatness"] = gm / am
    centroid = float(np.sum(freqs * p) / np.sum(p))
    spread = float(np.sqrt(np.sum(p * (freqs - centroid)**2) / np.sum(p)))
    out["spec_centroid"] = centroid
    out["spec_spread"] = spread
    if f0 is None or not np.isfinite(f0):
        mask = (freqs >= 0.5) & (freqs <= 3.0)
        if not mask.any(): return out
        idx = int(np.argmax(p[mask])); idx = np.where(mask)[0][idx]
    else:
        idx = int(np.argmin(np.abs(freqs - f0)))
    peaks, _ = find_peaks(p)
    if len(peaks) == 0: return out
    k = peaks[np.argmin(np.abs(peaks - idx))]
    try:
        widths = peak_widths(p, [k], rel_height=0.5)
        out["peak_width_hz"] = float(widths[0][0] * (freqs[1] - freqs[0]))
        prom = peak_prominences(p, [k])
        out["peak_prominence"] = float(prom[0][0])
    except Exception:
        pass
    return out

def acf_regularities(x: np.ndarray, fps: int, f0: Optional[float]) -> Dict[str, float]:
    x = np.asarray(x, np.float32) - np.nanmean(x)
    n = len(x)
    if n < 8:
        return dict(acf_at_step=np.nan, acf_at_2step=np.nan, acf_ratio_2_1=np.nan)
    if f0 is None or not np.isfinite(f0):
        L = max(4, int(0.6 * fps))
    else:
        L = max(4, int((1.0 / max(f0, 1e-3)) * fps))
    def acf_at(k):
        if k >= len(x): return np.nan
        v0 = x[:-k]; v1 = x[k:]
        return float(np.sum(v0 * v1) / (np.sum(x * x) + 1e-12))
    a1 = acf_at(L); a2 = acf_at(2 * L)
    return dict(acf_at_step=a1, acf_at_2step=a2, acf_ratio_2_1=(a2 / (abs(a1) + 1e-6)))


# -------------------- Извлечение фич (ядро) --------------------

def extract_features(path: str, schema: Schema, fps: int = 30, stride: int = 1,
                     fast_axes_flag: bool = True, turn_curv_thr: float = 0.15) -> pd.DataFrame:
    """
    Возвращает одну строку фич по .npy:
      - агрегаты по шагам L/R
      - HR/спектральные/ACF
      - метрики «прямолинейности» траектории таза
    """
    A = load_npy(path, schema)
    if stride > 1:
        A = A[::stride]
        fps = max(1, int(round(fps / stride)))

    seg = {k: A[:, idx, :] for k, idx in schema.groups.items() if len(idx) > 0}
    if not seg:
        raise ValueError("Schema produced no segments")

    seg = center_and_scale(seg)
    cogs = {k: centroid_series(v) for k, v in seg.items()}
    axes = fast_segment_axes(cogs) if fast_axes_flag else {}

    pelvis = cogs.get("pelvis")
    if pelvis is None:
        raise ValueError("pelvis segment is required")

    # Путь таза в горизонте: прямолинейность / кривизна
    d = pelvis[:, [0, 2]]
    net = float(np.linalg.norm(d[-1] - d[0])) if len(d) else 0.0
    seglen = np.linalg.norm(np.diff(d, axis=0), axis=1) if len(d) > 1 else np.array([0.0])
    path_straightness = float(net / (seglen.sum() + 1e-6))
    if len(d) >= 3:
        v = np.gradient(d, axis=0)
        a = np.gradient(v, axis=0)
        num = np.abs(v[:, 0] * a[:, 1] - v[:, 1] * a[:, 0])
        den = (np.linalg.norm(v, axis=1) ** 3 + 1e-12)
        kappa = num / den
        path_curv_mean = float(np.nanmean(kappa))
        path_curv_rms = float(np.sqrt(np.nanmean(kappa * kappa)))
    else:
        kappa = np.zeros(len(d))
        path_curv_mean = 0.0
        path_curv_rms = 0.0

    # Каденс / HR / спектр/ACF по тазу
    hr_vert, f0 = _harmonic_ratio(pelvis[:, 1], fps, cadence_hz=None)
    hr_ml,   _  = _harmonic_ratio(pelvis[:, 2], fps, cadence_hz=f0 if np.isfinite(f0) else None)
    specY = spectral_extras(pelvis[:, 1], fps, f0 if np.isfinite(f0) else None)
    specZ = spectral_extras(pelvis[:, 2], fps, f0 if np.isfinite(f0) else None)
    acfY  = acf_regularities(pelvis[:, 1], fps, f0 if np.isfinite(f0) else None)
    acfZ  = acf_regularities(pelvis[:, 2], fps, f0 if np.isfinite(f0) else None)

    # Перешаговые признаки и агрегаты
    rows = []
    for side in ("L", "R"):
        foot = cogs.get(f"{side}_foot")
        pel_ax = axes.get("pelvis")
        sh = axes.get(f"{side}_shank")
        th = axes.get(f"{side}_thigh")
        if foot is None or pel_ax is None:
            continue
        fs_idx, fo_idx = detect_gait_events(foot, fps=fps, f0=f0 if np.isfinite(f0) else None)
        if len(fs_idx) < 2:
            # один «псевдо-цикл» на весь трек
            a = 0; b = len(pelvis) - 1
            if b <= a: continue
            step_time = (b - a) / max(1, fps)
            step_len = float(np.linalg.norm(pelvis[b, [0, 2]] - pelvis[a, [0, 2]]))
            rows.append(dict(
                side=side, step_time=step_time, cadence=1.0/max(step_time,1e-6),
                step_len=step_len, step_speed=step_len/max(step_time,1e-6),
                pelvis_vert_osc=np.ptp(pelvis[a:b+1, 1]),
                pelvis_ml_osc=np.ptp(pelvis[a:b+1, 2]),
                pelvis_ap_osc=np.ptp(pelvis[a:b+1, 0]),
                kappa_med=np.nan, foot_clearance=np.nan
            ))
        else:
            for k in range(len(fs_idx) - 1):
                a, b = int(fs_idx[k]), int(fs_idx[k + 1])
                if b - a < max(4, int(0.2 * fps)): continue
                segslice = slice(a, b + 1)
                step_time = (b - a) / max(1, fps)
                p0 = pelvis[a, [0, 2]]; p1 = pelvis[b, [0, 2]]
                step_len = float(np.linalg.norm(p1 - p0))
                vo = float(np.ptp(pelvis[segslice, 1]))
                ml = float(np.ptp(pelvis[segslice, 2]))
                ap = float(np.ptp(pelvis[segslice, 0]))
                clearance = np.nan
                fy = foot[:, 1]
                if fy.size:
                    clearance = float(np.nanmedian(fy[segslice]) - np.nanmin(fy[segslice]))
                rows.append(dict(
                    side=side, step_time=step_time, cadence=1.0/max(step_time,1e-6),
                    step_len=step_len, step_speed=step_len/max(step_time,1e-6),
                    pelvis_vert_osc=vo, pelvis_ml_osc=ml, pelvis_ap_osc=ap,
                    kappa_med=float(np.nanmedian(kappa[a:b])) if np.isfinite(kappa[a:b]).any() else np.nan,
                    foot_clearance=clearance
                ))

    df = pd.DataFrame(rows)

    # Базовые агрегаты (median/mean/std) + счётчик циклов
    base = dict(
        step_time_median=np.nan, step_time_mean=np.nan, step_time_std=0.0,
        cadence_median=np.nan, cadence_mean=np.nan, cadence_std=0.0,
        step_len_median=0.0, step_len_mean=0.0, step_len_std=0.0,
        step_speed_median=0.0, step_speed_mean=0.0, step_speed_std=0.0,
        pelvis_vert_osc_median=0.0, pelvis_vert_osc_mean=0.0, pelvis_vert_osc_std=0.0,
        pelvis_ml_osc_median=0.0, pelvis_ml_osc_mean=0.0, pelvis_ml_osc_std=0.0,
        pelvis_ap_osc_median=0.0, pelvis_ap_osc_mean=0.0, pelvis_ap_osc_std=0.0,
        foot_clearance_median=np.nan, foot_clearance_mean=np.nan, foot_clearance_std=np.nan,
        cycles_count=0,
    )
    if df.empty:
        out = base.copy()
    else:
        num = df.select_dtypes(include=[np.number]).copy()
        agg = num.agg(["median", "mean", "std"]).T
        out = {f"{feat}_{stat}": float(agg.loc[feat, stat]) for feat in agg.index for stat in agg.columns}
        out["cycles_count"] = int(len(df))

    # Прямолинейность / HR / спектр / ACF
    out.update({
        "path_straightness": path_straightness,
        "path_curv_mean": path_curv_mean,
        "path_curv_rms": path_curv_rms,
        "hr_vert": float(hr_vert) if np.isfinite(hr_vert) else np.nan,
        "hr_ml":   float(hr_ml)   if np.isfinite(hr_ml)   else np.nan,
        "cadence_hz_est": float(f0) if np.isfinite(f0) else np.nan,
        "specY_flatness": specY["spec_flatness"], "specY_centroid": specY["spec_centroid"],
        "specY_spread":   specY["spec_spread"],   "specY_peak_width_hz": specY["peak_width_hz"],
        "specY_peak_prominence": specY["peak_prominence"],
        "specZ_flatness": specZ["spec_flatness"], "specZ_centroid": specZ["spec_centroid"],
        "specZ_spread":   specZ["spec_spread"],   "specZ_peak_width_hz": specZ["peak_width_hz"],
        "specZ_peak_prominence": specZ["peak_prominence"],
        "acfY_step": acfY["acf_at_step"], "acfY_2step": acfY["acf_at_2step"], "acfY_ratio": acfY["acf_ratio_2_1"],
        "acfZ_step": acfZ["acf_at_step"], "acfZ_2step": acfZ["acf_at_2step"], "acfZ_ratio": acfZ["acf_ratio_2_1"],
    })

    # Доля «прямых» шагов (mask по медианной кривизне шага)
    if not df.empty and "kappa_med" in df.columns:
        straight_mask = (df["kappa_med"].values <= float(turn_curv_thr))
        out["straight_frac"] = float(np.mean(straight_mask))
    else:
        out["straight_frac"] = np.nan

    return pd.DataFrame([out])


# -------------------- Пакетная обвязка (без меток) --------------------

def process_one(npy_path: Path, schema_path: Path, fps: int, stride: int,
                fast_axes: bool, turn_thr: float, per_file_dir: Optional[Path]):
    schema = load_schema(str(schema_path))
    feats_df = extract_features(str(npy_path), schema, fps=fps, stride=stride,
                                fast_axes_flag=fast_axes, turn_curv_thr=turn_thr)
    meta = {
        "file": str(npy_path.resolve()),
        "basename": npy_path.name,
        "stem": npy_path.stem,
        "subject": guess_subject_id(npy_path.name),
    }
    # per-file CSV по желанию
    if per_file_dir is not None:
        ensure_dir(per_file_dir)
        per_file = feats_df.copy()
        for k, v in list(meta.items())[::-1]:
            per_file.insert(0, k, v)
        (per_file_dir / f"{npy_path.stem}.csv").write_text(per_file.to_csv(index=False), encoding="utf-8")
    return meta, feats_df.iloc[0]


def main():
    ap = argparse.ArgumentParser(description="Features-only extractor from .npy")
    ap.add_argument("--data_dir", required=True, help="Папка с .npy (обход рекурсивный)")
    ap.add_argument("--schema", required=True, help="schema_joints.json")
    ap.add_argument("--out_dir", default="out_features", help="Куда сохранять общий features.csv")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--fast_axes", action="store_true")
    ap.add_argument("--turn_curv_thr", type=float, default=0.15)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--per_file", action="store_true", help="Сохранять per-file CSV")
    ap.add_argument("--parquet", action="store_true", help="Сохранить features.parquet")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    schema_path = Path(args.schema)
    out_dir = Path(args.out_dir)
    per_file_dir = (out_dir / "per_file") if args.per_file else None
    ensure_dir(out_dir)

    files = sorted(data_dir.rglob("*.npy"))
    if not files:
        raise SystemExit(f"[err] В {data_dir} не найдено .npy")

    print(f"[info] Found {len(files)} .npy; workers={args.workers}; stride={args.stride}; fast_axes={args.fast_axes}")

    tasks = [(p, schema_path, args.fps, args.stride, args.fast_axes, args.turn_curv_thr, per_file_dir) for p in files]
    rows_meta, rows_feats, skipped = [], [], 0

    if args.workers <= 1:
        for t in tqdm(tasks, desc="Extract"):
            try:
                meta, s = process_one(*t)
                rows_meta.append(meta); rows_feats.append(s)
            except Exception as e:
                skipped += 1
                tqdm.write(f"[skip] {t[0]} -> {type(e).__name__}: {e}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(process_one, *t) for t in tasks]
            for f in tqdm(as_completed(futs), total=len(futs), desc="Extract (parallel)"):
                try:
                    meta, s = f.result()
                    rows_meta.append(meta); rows_feats.append(s)
                except Exception as e:
                    skipped += 1
                    tqdm.write(f"[skip] -> {type(e).__name__}: {e}")

    if not rows_feats:
        raise SystemExit("[err] Не удалось извлечь ни одной строки фичей.")

    meta_df  = pd.DataFrame(rows_meta).reset_index(drop=True)
    feats_df = pd.DataFrame(rows_feats).reset_index(drop=True)

    feats_num = feats_df.select_dtypes(include=[np.number]).copy()
    feats_num = feats_num.loc[:, ~feats_num.columns.duplicated(keep="first")]

    # на всякий случай выравниваем длины
    n = min(len(meta_df), len(feats_num))
    if len(meta_df) != len(feats_num):
        print(f"[warn] Length mismatch: meta={len(meta_df)} feats={len(feats_num)} -> trunc to {n}")
    meta_df = meta_df.iloc[:n].reset_index(drop=True)
    feats_num = feats_num.iloc[:n].reset_index(drop=True)

    final = pd.concat([meta_df, feats_num], axis=1)
    final_csv = out_dir / "features.csv"
    final.to_csv(final_csv, index=False)

    if args.parquet:
        final_parq = out_dir / "features.parquet"
        final.to_parquet(final_parq)
        print(f"[done] Saved: {final_csv} and {final_parq}  |  skipped files: {skipped}")
    else:
        print(f"[done] Saved: {final_csv}  |  skipped files: {skipped}")


if __name__ == "__main__":
    main()
