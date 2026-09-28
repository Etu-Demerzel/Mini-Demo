from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class MLP(torch.nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.15),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.10),
            torch.nn.Linear(64, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit_predict(xtr, ytr, xte, seed, dev):
    torch.manual_seed(seed)
    if dev.type == "cpu":
        np.random.seed(seed)
    scaler = StandardScaler().fit(xtr)
    xtr_s = scaler.transform(xtr).astype(np.float32)
    xte_s = scaler.transform(xte).astype(np.float32)
    model = MLP(xtr_s.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-4)
    loss_fn = torch.nn.SmoothL1Loss()
    X = torch.from_numpy(xtr_s).to(dev)
    Y = torch.from_numpy(ytr.astype(np.float32)).to(dev)
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    for epoch in range(1, 401):
        model.train()
        pred = model(X)
        loss = loss_fn(pred, Y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        val = float(loss.detach().cpu())
        if val < best_loss - 1e-5:
            best_loss = val
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(xte_s).to(dev)).detach().cpu().numpy()
    return pred


def metrics(y, p):
    rmse = float(np.sqrt(np.mean((y-p)**2)))
    mae = float(np.mean(np.abs(y-p)))
    return {
        "rmse": rmse,
        "mae": mae,
        "pearson_r": float(pearsonr(y, p).statistic),
        "spearman_rho": float(spearmanr(y, p).statistic),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rows = pd.read_json(args.features)
    ligand = np.array(rows["ligand"].tolist(), dtype=np.float32)
    pocket = np.array(rows["pocket"].tolist(), dtype=np.float32)
    y = rows["label"].to_numpy(dtype=np.float32)
    Xs = {"ligand_only": ligand, "ligand_plus_pocket": np.concatenate([ligand, pocket], axis=1)}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    pred_rows, summary = [], []
    dev = device()
    print("device:", dev)
    for name, X in Xs.items():
        fold_stats = []
        oof = np.zeros(len(y), dtype=np.float32)
        for fold, (tr, te) in enumerate(kf.split(X), start=1):
            p = fit_predict(X[tr], y[tr], X[te], seed=100+fold, dev=dev)
            oof[te] = p
            for idx, pred in zip(te, p):
                pred_rows.append({"model": name, "fold": fold, "pdb_id": rows.iloc[idx]["pdb_id"], "y_true": float(y[idx]), "y_pred": float(pred)})
            fold_stats.append(metrics(y[te], p))
        overall = metrics(y, oof)
        summary.append({"model": name, **overall,
                        "fold_rmse_mean": float(np.mean([m["rmse"] for m in fold_stats])),
                        "fold_rmse_std": float(np.std([m["rmse"] for m in fold_stats]))})
    pd.DataFrame(pred_rows).to_csv(out / "cv_predictions.csv", index=False)
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
