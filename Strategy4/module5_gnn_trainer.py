# module_gnn_trainer.py （训练/推理，按“日期样本”迭代）
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple

from module5_gnn_model import GNNRetVol

def _date_range(start: str, end: str) -> set:
    dr = pd.date_range(pd.to_datetime(start), pd.to_datetime(end), freq="D")
    return set(pd.DatetimeIndex(dr).normalize().tolist())

def fit_scaler(samples: List[Dict], date_set: set) -> Tuple[np.ndarray, np.ndarray]:
    Xs = [s["X"] for s in samples if s["date"] in date_set]
    X = np.concatenate(Xs, axis=0)
    mu = X.mean(axis=0)
    std = X.std(axis=0) + 1e-6
    return mu, std

def train_gnn(
    samples: List[Dict],
    train_start: str,
    train_end: str,
    val_end: str,
    hidden: int = 128,
    layers: int = 2,
    dropout: float = 0.1,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 30,
    device: str = "cpu",
) -> Tuple[GNNRetVol, Dict]:

    train_set = _date_range(train_start, train_end)
    val_set = _date_range(pd.to_datetime(train_end) + pd.Timedelta(days=1), val_end)

    mu, std = fit_scaler(samples, train_set)

    in_dim = samples[0]["X"].shape[1]
    model = GNNRetVol(in_dim, hidden=hidden, layers=layers, dropout=dropout).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_ret_fn = nn.MSELoss(reduction="none")
    loss_vol_fn = nn.MSELoss(reduction="none")

    best_val = float("inf")
    best_state = None

    for ep in range(epochs):
        model.train()
        tr = []

        for s in samples:
            if s["date"] not in train_set:
                continue

            X = ((s["X"] - mu) / std).astype(np.float32)
            A = torch.from_numpy(s["A"]).to(device)
            Xt = torch.from_numpy(X).to(device)
            y_ret = torch.from_numpy(s["y_ret"]).to(device)
            y_vol = torch.from_numpy(s["y_vol"]).to(device)
            mask = torch.from_numpy(s["mask"]).to(device)

            pred_ret, pred_vol = model(A, Xt)

            lret = loss_ret_fn(pred_ret, y_ret) * mask
            lvol = loss_vol_fn(pred_vol, y_vol) * mask
            loss = (lret.sum()/(mask.sum()+1e-6)) + 0.5*(lvol.sum()/(mask.sum()+1e-6))

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tr.append(loss.item())

        model.eval()
        va = []
        with torch.no_grad():
            for s in samples:
                if s["date"] not in val_set:
                    continue
                X = ((s["X"] - mu) / std).astype(np.float32)
                A = torch.from_numpy(s["A"]).to(device)
                Xt = torch.from_numpy(X).to(device)
                y_ret = torch.from_numpy(s["y_ret"]).to(device)
                y_vol = torch.from_numpy(s["y_vol"]).to(device)
                mask = torch.from_numpy(s["mask"]).to(device)

                pr, pv = model(A, Xt)
                lret = loss_ret_fn(pr, y_ret) * mask
                lvol = loss_vol_fn(pv, y_vol) * mask
                loss = (lret.sum()/(mask.sum()+1e-6)) + 0.5*(lvol.sum()/(mask.sum()+1e-6))
                va.append(loss.item())

        tr_m = float(np.mean(tr)) if tr else np.nan
        va_m = float(np.mean(va)) if va else np.nan
        print(f"[Epoch {ep+1:03d}] train={tr_m:.6f} val={va_m:.6f}")

        if va_m < best_val:
            best_val = va_m
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    scaler = {"mu": mu, "std": std}
    return model, scaler


def predict_gnn(model: GNNRetVol, samples: List[Dict], scaler: Dict, device: str = "cpu") -> pd.DataFrame:
    mu, std = scaler["mu"], scaler["std"]
    rows = []
    model.eval()
    with torch.no_grad():
        for s in samples:
            X = ((s["X"] - mu) / std).astype(np.float32)
            A = torch.from_numpy(s["A"]).to(device)
            Xt = torch.from_numpy(X).to(device)

            pr, pv = model(A, Xt)
            pr = pr.detach().cpu().numpy()
            pv = pv.detach().cpu().numpy()

            for code, r, v in zip(s["codes"], pr, pv):
                rows.append({"date": s["date"], "code": code, "pred_ret": float(r), "pred_vol": float(v)})

    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["date"])
    return out
