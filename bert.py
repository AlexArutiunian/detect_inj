#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, argparse, math, random
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, precision_recall_curve

# -------------------- Utils --------------------

def seed_all(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def stem_lower(p: str) -> str:
    b = os.path.basename(str(p))
    return os.path.splitext(b)[0].lower()

def pick_label_col(df: pd.DataFrame, user_col: Optional[str]) -> str:
    if user_col and user_col in df.columns:
        return user_col
    # авто-поиск
    for c in df.columns:
        cl = c.strip().lower()
        if cl == "label" or "inj" in cl:  # "No inj/ inj" тоже покроется
            return c
    raise SystemExit("[err] не нашёл колонку меток — укажите --label_col")

def pick_fname_col(df: pd.DataFrame, user_col: Optional[str]) -> str:
    if user_col and user_col in df.columns:
        return user_col
    for c in ("filename", "file", "path", "basename", "stem"):
        if c in df.columns: return c
    raise SystemExit("[err] не нашёл колонку с именем файла — укажите --fname_col")

def map_label(v):
    s = str(v).strip().lower()
    if s in ("1","injury","inj","yes","y","true","t","1.0"): return 1
    if s in ("0","no injury","no inj","no","n","false","f","0.0"): return 0
    try:
        f = float(s); 
        if f in (0.0, 1.0): return int(f)
    except: pass
    return np.nan

def to_T_N_3(A: np.ndarray) -> np.ndarray:
    if A.ndim == 3 and A.shape[2] == 3:
        return A.astype(np.float32, copy=False)
    if A.ndim == 2 and A.shape[1] % 3 == 0:
        N = A.shape[1] // 3
        return A.reshape(A.shape[0], N, 3).astype(np.float32, copy=False)
    raise ValueError(f"bad shape {A.shape}, expect (T,N,3) or (T,3N)")

def per_sample_zscore(x: np.ndarray, eps=1e-6) -> np.ndarray:
    # x: (T, F)
    m = x.mean(axis=0, keepdims=True)
    s = x.std(axis=0, keepdims=True)
    return (x - m) / (s + eps)

def smart_downsample_len(T: int, max_len: int) -> Tuple[int, np.ndarray]:
    """Вернёт stride и индексы, чтобы получить <= max_len точек равномерно по времени."""
    if T <= max_len:
        idx = np.arange(T, dtype=np.int64)
        return 1, idx
    stride = math.ceil(T / max_len)
    idx = np.arange(0, T, stride, dtype=np.int64)
    if len(idx) > max_len:
        idx = idx[:max_len]
    return stride, idx

# -------------------- Dataset --------------------

class NpySeqDataset(Dataset):
    def __init__(self, rows: List[Tuple[str, int]], max_len: int = 256, center_pelvis: bool = True, augment: bool = False):
        """
        rows: list of (npy_path, label)
        """
        self.rows = rows
        self.max_len = max_len
        self.center_pelvis = center_pelvis
        self.augment = augment

    def __len__(self): return len(self.rows)

    def _load_one(self, path: str) -> np.ndarray:
        A = np.load(path, allow_pickle=False)
        A = to_T_N_3(A)   # (T, N, 3)
        # центрирование по тазу, если есть
        # найдём индекс pelvis эвристически: если есть имя в пути — пропустим; тут просто берём 0-й сегмент как таз не всегда корректно.
        # Для универсальности центрируем по среднему всех суставов — устойчиво.
        if self.center_pelvis:
            center = A.mean(axis=1, keepdims=True)     # (T,1,3)
            A = A - center
        T, N, _ = A.shape
        X = A.reshape(T, N*3)                          # (T, F)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X = per_sample_zscore(X)
        # аугментации (легкие): шум и маленький дропаут кадров
        if self.augment:
            if random.random() < 0.5:
                X = X + np.random.normal(0, 0.02, size=X.shape).astype(np.float32)
            if random.random() < 0.2 and X.shape[0] > 16:
                mask = np.ones(X.shape[0], dtype=bool)
                drop = np.random.choice(X.shape[0], size=min(8, X.shape[0]//10), replace=False)
                mask[drop] = False
                X = X[mask]
        # даунсэмпл/кроп до max_len
        _, idx = smart_downsample_len(X.shape[0], self.max_len)
        X = X[idx]
        return X.astype(np.float32, copy=False)

    def __getitem__(self, i):
        p, y = self.rows[i]
        x = self._load_one(p)
        return x, np.int64(y)

def collate_pad(batch):
    xs, ys = zip(*batch)
    lens = [x.shape[0] for x in xs]
    F = xs[0].shape[1]
    Tm = max(lens)
    pad = np.zeros((len(xs), Tm, F), dtype=np.float32)
    attn = np.zeros((len(xs), Tm), dtype=np.bool_)
    for i, x in enumerate(xs):
        t = x.shape[0]
        pad[i, :t] = x
        attn[i, :t] = True
    return torch.from_numpy(pad), torch.from_numpy(attn), torch.from_numpy(np.asarray(ys, dtype=np.int64))

# -------------------- BERT-like model --------------------

class TimeSeriesBERT(nn.Module):
    def __init__(self, feature_dim: int, d_model: int = 128, nhead: int = 4, num_layers: int = 4,
                 dim_ff: int = 256, dropout: float = 0.1, max_len: int = 1024):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.proj = nn.Linear(feature_dim, d_model)
        self.pos = nn.Parameter(torch.randn(1, max_len + 1, d_model) * 0.02)  # +CLS
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
                                               dropout=dropout, batch_first=True, activation="gelu", norm_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, F), attn_mask: (B, T) bool (True=valid)
        """
        B, T, F = x.shape
        h = self.proj(x)  # (B,T,D)
        cls = self.cls.expand(B, 1, -1)  # (B,1,D)
        h = torch.cat([cls, h], dim=1)   # (B,T+1,D)
        pos = self.pos[:, :T+1, :]       # (1,T+1,D)
        h = h + pos
        # attention mask: 1 for keep, 0 for pad -> transformer expects True for keep via key_padding_mask=True means positions with True will be ignored
        # So we build key_padding_mask: (B, T+1), True means PAD
        pad_mask = ~attn_mask  # (B,T)
        pad_mask = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=pad_mask.device), pad_mask], dim=1)
        # TransformerEncoder: key_padding_mask=True means those positions are ignored
        h = self.enc(h, src_key_padding_mask=pad_mask)
        cls_h = self.norm(h[:, 0, :])    # (B,D)
        logit = self.head(cls_h).squeeze(-1)  # (B,)
        return logit

# -------------------- Train/Eval --------------------

@torch.no_grad()
def evaluate(model, loader, device) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    probs = []; ys = []
    for xb, mb, yb in loader:
        xb = xb.to(device); mb = mb.to(device)
        logits = model(xb, mb)
        p = torch.sigmoid(logits).cpu().numpy()
        probs.append(p); ys.append(yb.numpy())
    probs = np.concatenate(probs)
    ys = np.concatenate(ys)
    try:
        auc = roc_auc_score(ys, probs)
    except:
        auc = float("nan")
    return auc, probs, ys

def find_balanced_threshold(y_true: np.ndarray, prob: np.ndarray) -> Tuple[float, Dict[str, float]]:
    best = (0.5, -1.0, 0.0, 0.0)  # thr, bal, r0, r1
    for thr in np.linspace(0, 1, 1001):
        pred = (prob >= thr).astype(int)
        tp = ((pred==1)&(y_true==1)).sum(); fn = ((pred==0)&(y_true==1)).sum()
        tn = ((pred==0)&(y_true==0)).sum(); fp = ((pred==1)&(y_true==0)).sum()
        r1 = tp / (tp+fn+1e-12)
        r0 = tn / (tn+fp+1e-12)
        bal = 0.5*(r0+r1)
        if bal > best[1]:
            best = (thr, bal, r0, r1)
    thr, bal, r0, r1 = best
    return float(thr), {"balanced": float(bal), "recall0": float(r0), "recall1": float(r1)}

def train_loop(model, train_loader, dev_loader, device, epochs=25, lr=2e-4, weight_decay=1e-4, pos_weight: Optional[float]=None, patience=7):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    else:
        criterion = nn.BCEWithLogitsLoss()
    best_dev = -1e9
    best_state = None
    wait = 0

    for ep in range(1, epochs+1):
        model.train()
        losses = []
        for xb, mb, yb in train_loader:
            xb = xb.to(device); mb = mb.to(device); yb = yb.float().to(device)
            logits = model(xb, mb)
            loss = criterion(logits, yb)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            losses.append(loss.item())
        # dev
        auc_dev, prob_dev, y_dev = evaluate(model, dev_loader, device)
        thr_dev, info = find_balanced_threshold(y_dev, prob_dev)
        score = 0.5*info["recall0"] + 0.5*info["recall1"]  # то же, что bal
        print(f"[ep {ep:02d}] train_loss={np.mean(losses):.4f}  dev_auc={auc_dev:.3f}  thr={thr_dev:.3f}  r0={info['recall0']:.3f} r1={info['recall1']:.3f}")
        if score > best_dev:
            best_dev = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("[early] patience reached")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

# -------------------- Main --------------------

def main():
    ap = argparse.ArgumentParser(description="BERT-like transformer on .npy time series")
    ap.add_argument("--data_dir", required=True, help="Папка с .npy")
    ap.add_argument("--labels_csv", required=True, help="CSV с filename и label (0/1 или 'No inj/ inj')")
    ap.add_argument("--label_col", default=None, help="Имя колонки метки (авто, если не указано)")
    ap.add_argument("--fname_col",  default=None, help="Имя колонки с путём/именем файла (авто)")
    ap.add_argument("--max_len", type=int, default=256, help="макс длина последовательности")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--ff", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--center_pelvis", action="store_true", help="центрировать по среднему всех суставов")
    ap.add_argument("--augment", action="store_true", help="лёгкие аугментации на трейне")
    ap.add_argument("--out_dir", default="out_bert_ts")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_all(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[device]", device)

    # load labels
    L = pd.read_csv(args.labels_csv)
    label_col = pick_label_col(L, args.label_col)
    fname_col  = pick_fname_col(L, args.fname_col)

    L["_key"] = L[fname_col].astype(str).map(stem_lower)
    L["label"] = L[label_col].map(map_label)
    L = L[L["label"].isin([0,1])].copy()
    if len(L) == 0:
        raise SystemExit("[err] нет валидных меток 0/1")

    data_dir = Path(args.data_dir)
    # сматчим реальные пути к npy
    rows = []
    miss = 0
    for stem, raw, lab in L[["_key", fname_col, "label"]].itertuples(index=False, name=None):
        p = Path(str(raw))
        print(p)
        if not p.is_file():
            p1 = data_dir / (str(stem) + ".npy")
            p2 = data_dir / str(stem)
            if p1.is_file():
                p = p1
            elif p2.is_file():
                p = p2
            else:
                miss += 1
                continue
        rows.append((str(p), int(lab)))

    if miss:
        print(f"[warn] пропущено {miss} строк (файлы не найдены)")
    if not rows:
        raise SystemExit("[err] не найдено ни одного npy по CSV")

    # быстрый «прогрев» чтобы выяснить размер признаков F
    tmpA = np.load(rows[0][0], allow_pickle=False)
    tmpA = to_T_N_3(tmpA)
    Fdim = tmpA.shape[1] * 3
    print(f"[info] samples={len(rows)}  feature_dim={Fdim}")

    # split: train/dev/test = 60/20/20, стратифицировано по y
    paths = [r[0] for r in rows]
    ys    = np.array([r[1] for r in rows], dtype=np.int64)
    p_tr, p_te, y_tr, y_te = train_test_split(paths, ys, test_size=0.20, random_state=args.seed, stratify=ys)
    p_tr, p_dv, y_tr, y_dv = train_test_split(p_tr, y_tr, test_size=0.25, random_state=args.seed, stratify=y_tr)
    print(f"[split] train={len(p_tr)}  dev={len(p_dv)}  test={len(p_te)}")

    # dataloaders
    ds_tr = NpySeqDataset(list(zip(p_tr, y_tr)), max_len=args.max_len, center_pelvis=args.center_pelvis, augment=args.augment)
    ds_dv = NpySeqDataset(list(zip(p_dv, y_dv)), max_len=args.max_len, center_pelvis=args.center_pelvis, augment=False)
    ds_te = NpySeqDataset(list(zip(p_te, y_te)), max_len=args.max_len, center_pelvis=args.center_pelvis, augment=False)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=collate_pad, pin_memory=True)
    dl_dv = DataLoader(ds_dv, batch_size=args.batch_size*2, shuffle=False, num_workers=2, collate_fn=collate_pad, pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size*2, shuffle=False, num_workers=2, collate_fn=collate_pad, pin_memory=True)

    # class imbalance -> pos_weight (для BCEWithLogitsLoss) = N_neg / N_pos
    pos = (y_tr == 1).sum(); neg = (y_tr == 0).sum()
    pos_weight = None
    if pos > 0:
        pos_weight = max(1.0, float(neg) / float(pos))
        print(f"[train] pos_weight={pos_weight:.3f} (neg/pos)")

    # model
    model = TimeSeriesBERT(
        feature_dim=Fdim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.layers,
        dim_ff=args.ff,
        dropout=args.dropout,
        max_len=args.max_len+1,
    )

    # train
    model = train_loop(model, dl_tr, dl_dv, device, epochs=args.epochs, lr=args.lr,
                       weight_decay=args.weight_decay, pos_weight=pos_weight, patience=7)

    # choose threshold on DEV, evaluate on TEST
    auc_dev, prob_dev, y_dev = evaluate(model, dl_dv, device)
    thr, info = find_balanced_threshold(y_dev, prob_dev)
    print(f"[thr] DEV chosen: {thr:.3f} | r0={info['recall0']:.3f} r1={info['recall1']:.3f} bal={info['balanced']:.3f} | AUC={auc_dev:.3f}")

    auc_te, prob_te, y_te = evaluate(model, dl_te, device)
    pred_te = (prob_te >= thr).astype(int)
    cm = confusion_matrix(y_te, pred_te)
    print(f"\n[TEST] AUC: {auc_te:.3f}")
    print("[TEST] Confusion matrix:\n", cm)
    print("\n[TEST] Report:\n", classification_report(y_te, pred_te, digits=3))

    # save
    torch.save({
        "model_state": model.state_dict(),
        "config": {
            "feature_dim": Fdim, "d_model": args.d_model, "nhead": args.nhead,
            "layers": args.layers, "dim_ff": args.ff, "dropout": args.dropout,
            "max_len": args.max_len+1
        },
        "threshold": float(thr)
    }, os.path.join(args.out_dir, "bert_timeseries.pt"))
    json.dump({"thr": float(thr), "dev_auc": float(auc_dev), "test_auc": float(auc_te),
               "dev_balanced": info}, open(os.path.join(args.out_dir, "metrics.json"), "w"), ensure_ascii=False, indent=2)
    print("\n[done] saved to", args.out_dir)

if __name__ == "__main__":
    main()
