import os
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib

# ========= утилиты общие =========

def label_to_str(p):  # 0/1 -> string
    return "Injury" if p == 1 else "No Injury"

def ensure_listdir(dir_path):
    files = []
    for fn in os.listdir(dir_path):
        if fn.lower().endswith(".json"):
            files.append(os.path.join(dir_path, fn))
    return sorted(files)

def read_threshold(path):
    with open(path, "r") as f:
        return float(f.read().strip())

def safe_json_load(path):
    try:
        import orjson
        with open(path, "rb") as f:
            return orjson.loads(f.read())
    except Exception:
        with open(path, "r") as f:
            return json.load(f)

def stack_motion_frames(motion_dict, schema_joints):
    """
    Преобразуем dict {joint: [[x,y,z],...]} в (T, 3*|schema|), заполняя отсутствующие суставы NaN.
    T берём как минимум по присутствующим суставам.
    """
    present = [j for j in schema_joints if j in motion_dict]
    if not present:
        return None
    T = min(len(motion_dict[j]) for j in present)
    if T <= 1:
        return None
    cols = []
    for j in schema_joints:
        if j in motion_dict:
            arr = np.asarray(motion_dict[j], dtype=np.float32)[:T]  # (T,3)
        else:
            arr = np.full((T, 3), np.nan, dtype=np.float32)
        cols.append(arr)
    X = np.concatenate(cols, axis=1)  # (T, 3*|schema|)
    return X

def seq_from_file(json_path, motion_key, schema_joints):
    data = safe_json_load(json_path)
    if motion_key not in data or not isinstance(data[motion_key], dict):
        return None
    return stack_motion_frames(data[motion_key], schema_joints)

# ========= извлечение фич для классики (тот же рецепт, что в train.py) =========

def features_from_sequence(seq):
    seq = seq.astype(np.float32, copy=False)
    dif = np.diff(seq, axis=0)
    stat = np.concatenate([
        np.nanmean(seq, axis=0), np.nanstd(seq, axis=0), np.nanmin(seq, axis=0), np.nanmax(seq, axis=0),
        np.nanmean(dif, axis=0), np.nanstd(dif, axis=0)
    ]).astype(np.float32, copy=False)
    stat = np.nan_to_num(stat, nan=0.0, posinf=0.0, neginf=0.0)
    return stat

# ========= ленивый TF для LSTM/TCN =========

def lazy_tf():
    try:
        import tensorflow as tf
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        return {"tf": tf, "pad_sequences": pad_sequences}
    except Exception as e:
        raise RuntimeError(
            "TensorFlow недоступен/несовместим. Для моделей lstm/tcn установите TF или используйте классические модели. "
            f"Ошибка: {e}"
        )

# ========= инференс =========

def infer_classical(model_dir, json_files, motion_key, schema_joints):
    # загрузка модели/скейлера/порога
    bundle = joblib.load(os.path.join(model_dir, "model.joblib"))
    model = bundle["model"]
    scaler = bundle["scaler"]
    thr = read_threshold(os.path.join(model_dir, "threshold.txt"))

    rows = []
    feats = []
    kept_files = []

    for p in tqdm(json_files, desc="Infer (feats)"):
        seq = seq_from_file(p, motion_key, schema_joints)
        if seq is None or seq.shape[0] < 2:
            continue
        f = features_from_sequence(seq)
        feats.append(f)
        kept_files.append(os.path.basename(p))

    if not feats:
        raise RuntimeError("Нет валидных файлов для инференса (классика).")
    X = np.stack(feats).astype(np.float32, copy=False)
    X = scaler.transform(X)
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= thr).astype(int)

    for fn, pr, pd_ in zip(kept_files, prob, pred):
        rows.append({"filename": fn, "prob_injury": float(pr), "pred_label": label_to_str(int(pd_))})
    return pd.DataFrame(rows)

def infer_deep(model_dir, json_files, motion_key, schema_joints, batch_size):
    # загрузка TF, модели и max_len/threshold
    k = lazy_tf()
    tf = k["tf"]; pad_sequences = k["pad_sequences"]
    from tensorflow.keras.models import load_model

    model = load_model(os.path.join(model_dir, "model.h5"))
    thr = read_threshold(os.path.join(model_dir, "threshold.txt"))

    # max_len из norm_stats.npz (в train.py мы сохраняли только max_len)
    stats = np.load(os.path.join(model_dir, "norm_stats.npz"))
    if "max_len" in stats:
        max_len = int(stats["max_len"])
    else:
        # на всякий случай поддержка варианта с mu/sd/max_len:
        max_len = int(stats["max_len"])

    rows, probs, preds, kept_files = [], [], [], []

    # батчевый проход (читаем и паддим на лету)
    cur_X, cur_names = [], []
    for p in tqdm(json_files, desc="Infer (batches)"):
        seq = seq_from_file(p, motion_key, schema_joints)
        if seq is None:
            continue
        cur_X.append(seq.astype(np.float32, copy=False))
        cur_names.append(os.path.basename(p))
        if len(cur_X) >= batch_size:
            Xp = pad_sequences(cur_X, maxlen=max_len, dtype='float32', padding='post', truncating='post')
            pr = model.predict(Xp, verbose=0).flatten()
            pb = pr
            pd_ = (pb >= thr).astype(int)
            probs.extend(pb.tolist()); preds.extend(pd_.tolist()); kept_files.extend(cur_names)
            cur_X, cur_names = [], []

    # остаток
    if cur_X:
        Xp = pad_sequences(cur_X, maxlen=max_len, dtype='float32', padding='post', truncating='post')
        pr = model.predict(Xp, verbose=0).flatten()
        pb = pr
        pd_ = (pb >= thr).astype(int)
        probs.extend(pb.tolist()); preds.extend(pd_.tolist()); kept_files.extend(cur_names)

    for fn, pr, pd_ in zip(kept_files, probs, preds):
        rows.append({"filename": fn, "prob_injury": float(pr), "pred_label": label_to_str(int(pd_))})
    if not rows:
        raise RuntimeError("Нет валидных файлов для инференса (нейросеть).")
    return pd.DataFrame(rows)

# ========= main =========

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Inference for injury classifier")
    ap.add_argument("--model_dir", required=True, help="Папка с обученной моделью (например outputs/rf или outputs/tcn)")
    ap.add_argument("--motion_key", default="running", help="running|walking")
    ap.add_argument("--input_json", default="", help="Один .json файл для инференса")
    ap.add_argument("--input_dir", default="", help="Папка с .json для инференса")
    ap.add_argument("--use_joints", default="", help="Список суставов через запятую. Если не задан, попробуем schema_joints.json")
    ap.add_argument("--batch_size", type=int, default=64, help="батч для нейросетей")
    ap.add_argument("--out_csv", default="predictions.csv")
    args = ap.parse_args()

    # какие суставы использовать в нужном порядке
    schema_path = os.path.join(args.model_dir, "schema_joints.json")
    schema_joints = None
    if args.use_joints.strip():
        schema_joints = [s.strip() for s in args.use_joints.split(",") if s.strip()]
    elif os.path.exists(schema_path):
        with open(schema_path, "r", encoding="utf-8") as f:
            schema_joints = json.load(f)
    else:
        # последний шанс: возьмём из первого файла, но это может сломать форму между файлами
        print("WARNING: schema_joints.json не найден и --use_joints не задан. "
              "Попробуем взять суставы из первого файла (риск несовпадения формы).")
        # определим позже из первого valid файла

    # какие json прогонять
    files = []
    if args.input_json:
        files = [args.input_json]
    elif args.input_dir:
        files = ensure_listdir(args.input_dir)
    else:
        raise SystemExit("Укажи --input_json ИЛИ --input_dir.")

    if not files:
        raise SystemExit("Нет json файлов для инференса.")

    # определяем тип модели по содержимому папки
    is_classical = os.path.exists(os.path.join(args.model_dir, "model.joblib"))
    is_deep = os.path.exists(os.path.join(args.model_dir, "model.h5"))

    if not (is_classical or is_deep):
        raise SystemExit("Не найдено файлов модели в model_dir. Ожидались model.joblib (классика) или model.h5 (нейросети)")

    # если схема суставов ещё None — попытаемся взять из первого файла
    if schema_joints is None:
        first_ok = None
        for p in files:
            try:
                d = safe_json_load(p)
                if args.motion_key in d and isinstance(d[args.motion_key], dict):
                    schema_joints = sorted(list(d[args.motion_key].keys()))
                    first_ok = p
                    break
            except Exception:
                continue
        if schema_joints is None:
            raise SystemExit("Не удалось определить список суставов. Укажи --use_joints.")

        print(f"Using joints discovered from: {os.path.basename(first_ok)}")
        print(f"Joints ({len(schema_joints)}): {', '.join(schema_joints[:8])}{' ...' if len(schema_joints)>8 else ''}")

    # инференс
    if is_classical:
        df = infer_classical(args.model_dir, files, args.motion_key, schema_joints)
    else:
        df = infer_deep(args.model_dir, files, args.motion_key, schema_joints, batch_size=args.batch_size)

    df.to_csv(args.out_csv, index=False)
    print(f"Saved: {args.out_csv}")
