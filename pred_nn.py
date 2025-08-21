#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, argparse, json
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

def load_norm_stats(path: str) -> Tuple[np.ndarray, np.ndarray, int]:
    d = np.load(path, allow_pickle=False)
    mean = d["mean"].astype(np.float32)
    std  = d["std"].astype(np.float32)
    max_len = int(d["max_len"])
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std, max_len

def as_TxF(a: np.ndarray) -> Optional[np.ndarray]:
    a = np.asarray(a)
    if a.ndim == 2: return a
    if a.ndim == 3:
        T,V,C = a.shape
        return a.reshape(T, V*C)
    if a.ndim == 1:
        return a.reshape(-1,1)
    return None

def list_files_from_csv_or_dir(data_dir: str, csv_path: Optional[str], filename_col: str) -> List[str]:
    def _possible(rel: str) -> List[str]:
        rel = (rel or "").replace("\\","/").lstrip("/")
        cands = [
            os.path.join(data_dir, rel),
            os.path.join(data_dir, rel + ".npy"),
            os.path.join(data_dir, os.path.basename(rel)),
            os.path.join(data_dir, os.path.basename(rel) + ".npy"),
        ]
        return list(dict.fromkeys(cands))
    out = []
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        cols = {c.lower().strip(): c for c in df.columns}
        fn_col = cols.get(filename_col.lower(), filename_col)
        if fn_col not in df.columns:
            for c in df.columns:
                if "file" in c.lower() or "name" in c.lower():
                    fn_col = c; break
        missing = 0
        for r in df[fn_col].astype(str).tolist():
            hit = None
            for c in _possible(r):
                if os.path.exists(c):
                    hit = c; break
            if hit: out.append(hit)
            else: missing += 1
        if missing:
            print(f"[warn] В CSV указано файлов, которых нет на диске: {missing}", file=sys.stderr)
        return out
    # иначе — просто все .npy в папке (без рекурсии)
    for f in sorted(os.listdir(data_dir)):
        p = os.path.join(data_dir, f)
        if os.path.isfile(p) and f.lower().endswith(".npy"):
            out.append(p)
    return out

# ---- надёжная загрузка keras-модели с Lambda-патчем ----
def _register_custom_objects():
    import tensorflow as tf
    import keras

    @keras.saving.register_keras_serializable(package="custom")
    def masked_mean(inputs):
        if isinstance(inputs, (list, tuple)):
            x = inputs[0]; mask = inputs[1] if len(inputs)>1 else None
        else:
            x = inputs; mask = None
        x = tf.convert_to_tensor(x)
        if mask is not None:
            mask = tf.cast(mask, x.dtype)
            mask = tf.expand_dims(mask, -1)           # (B,T,1)
            num = tf.reduce_sum(x*mask, axis=1)       # (B,D)
            den = tf.reduce_sum(mask, axis=1) + 1e-8  # (B,1)
            y = num/den
        else:
            y = tf.reduce_mean(x, axis=1)             # (B,D)
        if x.shape.rank is not None and x.shape.rank>=2:
            y.set_shape([None, x.shape[-1]])
        return y

    @keras.saving.register_keras_serializable(package="custom")
    class SafeLambda(keras.layers.Lambda):
        def compute_output_shape(self, input_shape):
            s = input_shape
            if isinstance(s,(list,tuple)) and s and isinstance(s[0],(list,tuple)):
                s = s[0]
            try:
                b = s[0] if isinstance(s,(list,tuple)) else None
                d = s[-1] if isinstance(s,(list,tuple)) else None
                if d is not None: return (b, d)
            except Exception:
                pass
            return super().compute_output_shape(input_shape)

    # Патч двух путей до загрузки
    keras.layers.Lambda = SafeLambda
    try:
        from keras.src.layers.core import lambda_layer as _ll
        _ll.Lambda = SafeLambda
    except Exception:
        pass

    try:
        co = keras.saving.get_custom_objects()
        co["Lambda"] = SafeLambda
        co["masked_mean"] = masked_mean
    except Exception:
        pass

    return {"Lambda": SafeLambda, "masked_mean": masked_mean}

def load_model_any(path: str, unsafe_deser: bool=False):
    import keras, tensorflow as tf
    custom = _register_custom_objects()
    # Keras часто блокирует кастомы в safe_mode — снимем
    safe_mode = False
    with keras.utils.custom_object_scope(custom):
        # сначала tf.keras
        try:
            return tf.keras.models.load_model(path, custom_objects=custom, compile=False, safe_mode=safe_mode)
        except TypeError:
            return tf.keras.models.load_model(path, custom_objects=custom, compile=False)

def make_ds_npy(paths: List[str], mean: np.ndarray, std: np.ndarray, max_len: int, batch: int):
    import tensorflow as tf
    F = int(mean.shape[0])
    def gen():
        for p in paths:
            try:
                a = np.load(p, allow_pickle=False, mmap_mode="r")
                a = as_TxF(a)
                if a is None or a.shape[0] < 1: continue
                x = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                if x.shape[1] != F:
                    if x.shape[1] > F: x = x[:, :F]
                    else: x = np.pad(x, [[0,0],[0, F-x.shape[1]]], mode="constant")
                if x.shape[0] > max_len: x = x[:max_len]
                x = (x - mean) / std
                yield x, p
            except Exception as e:
                print(f"[warn] skip '{p}': {type(e).__name__}: {e}", file=sys.stderr)
    sig = (tf.TensorSpec(shape=(None,F), dtype=tf.float32),
           tf.TensorSpec(shape=(), dtype=tf.string))
    def pad_map(x,path):
        import tensorflow as tf
        T = tf.shape(x)[0]
        pad = tf.maximum(0, max_len-T)
        x = tf.pad(x, [[0,pad],[0,0]])[:max_len]
        x.set_shape([max_len, F])
        return x, path
    ds = tf.data.Dataset.from_generator(gen, output_signature=sig)
    ds = ds.map(pad_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds

def entropy_binary(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-8, 1-1e-8)
    return -(p*np.log(p) + (1-p)*np.log(1-p))

def main():
    ap = argparse.ArgumentParser("Simple NN predictor for NPY with summary")
    ap.add_argument("--model", required=True)
    ap.add_argument("--norm_stats", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--filename_col", default="filename")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--threshold", default=None, help="float or path to txt")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--mc_passes", type=int, default=1)
    ap.add_argument("--gpus", default="all")
    args = ap.parse_args()

    # GPU env
    if args.gpus.lower() == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif args.gpus.lower() != "all":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    # входные файлы
    paths = list_files_from_csv_or_dir(args.data_dir, args.csv, args.filename_col)
    if not paths:
        print("Нет входных файлов.", file=sys.stderr)
        sys.exit(2)

    # нормировки
    mean, std, max_len = load_norm_stats(args.norm_stats)

    # порог
    thr = 0.5
    if args.threshold:
        if os.path.exists(args.threshold):
            try:
                with open(args.threshold, "r", encoding="utf-8") as f:
                    thr = float(f.read().strip())
            except Exception:
                print("[warn] threshold-файл не прочитан, используем 0.5", file=sys.stderr)
        else:
            try:
                thr = float(args.threshold)
            except Exception:
                pass

    # загрузка модели
    model = load_model_any(args.model)

    # датасет и предсказания
    ds = make_ds_npy(paths, mean, std, max_len, args.batch_size)
    import tensorflow as tf
    all_probs, all_paths = [], []
    if args.mc_passes <= 1:
        for xb, pb in ds:
            p = model(xb, training=False)
            p = tf.cast(p, tf.float32).numpy().reshape(-1)
            all_probs.append(p)
            all_paths.extend([s.decode("utf-8") for s in pb.numpy().tolist()])
        probs = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0,), np.float32)
        prob_std = None
    else:
        mc = []
        first_paths = None
        for t in range(args.mc_passes):
            probs_t, paths_t = [], []
            for xb, pb in ds:
                p = model(xb, training=True)  # MC Dropout
                probs_t.append(np.asarray(p).reshape(-1))
                if t == 0:
                    paths_t.extend([s.decode("utf-8") for s in pb.numpy().tolist()])
            mc.append(np.concatenate(probs_t, axis=0) if probs_t else np.zeros((0,), np.float32))
            if t == 0: first_paths = paths_t
        mc_arr = np.stack(mc, axis=0)
        probs = mc_arr.mean(axis=0)
        prob_std = mc_arr.std(axis=0)
        all_paths = first_paths

    # пост-обработка и сводка
    probs = probs.astype(np.float32)
    p_clip = np.clip(probs, 1e-8, 1-1e-8)
    pred = (p_clip >= thr).astype(np.int32)
    confidence = np.where(pred==1, p_clip, 1.0 - p_clip).astype(np.float32)
    logit = np.log(p_clip/(1.0-p_clip)).astype(np.float32)
    entr = entropy_binary(p_clip).astype(np.float32)

    df = pd.DataFrame({
        "path": all_paths,
        "prob": probs,
        "pred": pred,
        "confidence": confidence,
        "logit": logit,
        "entropy": entr,
    })
    if prob_std is not None:
        df["prob_std"] = prob_std.astype(np.float32)
    df = df.sort_values("confidence", ascending=False).reset_index(drop=True)
    df.to_csv(args.out_csv, index=False, float_format="%.6f")

    # короткая статистика
    n = len(df)
    n_pos = int(df["pred"].sum())
    n_neg = n - n_pos
    c80 = int((df["confidence"] >= 0.8).sum())
    c60_80 = int(((df["confidence"] >= 0.6) & (df["confidence"] < 0.8)).sum())
    c50_60 = int(((df["confidence"] >= 0.5) & (df["confidence"] < 0.6)).sum())
    c_lt50 = n - (c50_60 + c60_80 + c80)
    print(f"Predicted: injury={n_pos} | no-injury={n_neg} | total={n}")
    print(f"Confidence buckets: <50%={c_lt50} | 50–60%={c50_60} | 60–80%={c60_80} | >80%={c80}")
    if "prob_std" in df.columns:
        print(f"MC Dropout: passes={args.mc_passes} | mean std={df['prob_std'].mean():.6f}")

if __name__ == "__main__":
    main()
