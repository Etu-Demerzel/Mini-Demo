from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SmallMLP(torch.nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def metric_dict(y, p):
    return {
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "mae": float(np.mean(np.abs(y - p))),
        "pearson_r": float(pearsonr(y, p).statistic),
        "spearman_rho": float(spearmanr(y, p).statistic),
    }


def fit_mlp_predict(xtr, ytr, xte, seed, dev):
    # Feature scaling is fit ONLY on the outer training fold.
    x_scaler = StandardScaler().fit(xtr)
    xtr_s = x_scaler.transform(xtr).astype(np.float32)
    xte_s = x_scaler.transform(xte).astype(np.float32)

    # Target scaling is also fit ONLY on the outer training fold.
    y_mean = float(np.mean(ytr))
    y_std = float(np.std(ytr))
    if y_std < 1e-8:
        y_std = 1.0
    ytr_s = ((ytr - y_mean) / y_std).astype(np.float32)

    # Inner validation split for early stopping.
    idx = np.arange(len(ytr_s))
    tr_idx, va_idx = train_test_split(
        idx, test_size=0.20, random_state=seed
    )

    torch.manual_seed(seed)
    if dev.type == "cpu":
        np.random.seed(seed)

    model = SmallMLP(xtr_s.shape[1]).to(dev)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=2e-3,
        weight_decay=1e-3,
    )
    loss_fn = torch.nn.MSELoss()

    X = torch.from_numpy(xtr_s).to(dev)
    Y = torch.from_numpy(ytr_s).to(dev)
    Xtr = X[tr_idx]
    Ytr = Y[tr_idx]
    Xva = X[va_idx]
    Yva = Y[va_idx]

    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience = 80
    bad_epochs = 0

    for epoch in range(1, 1001):
        model.train()
        pred = model(Xtr)
        loss = loss_fn(pred, Ytr)

        opt.zero_grad()
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(Xva), Yva).detach().cpu())

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        p_scaled = model(
            torch.from_numpy(xte_s).to(dev)
        ).detach().cpu().numpy()

    # Return to original affinity scale.
    return p_scaled * y_std + y_mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = pd.read_json(args.features)
    ligand = np.array(rows["ligand"].tolist(), dtype=np.float32)
    pocket = np.array(rows["pocket"].tolist(), dtype=np.float32)
    y = rows["label"].to_numpy(dtype=np.float32)

    Xs = {
        "ligand_only": ligand,
        "ligand_plus_pocket": np.concatenate([ligand, pocket], axis=1),
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    dev = get_device()
    print("device:", dev)
    print("N:", len(y))
    print("target mean/std:", float(y.mean()), float(y.std()))

    pred_rows = []
    summary = []

    for feature_name, X in Xs.items():
        model_outputs = {}
        fold_stats = {"mean": [], "ridge": [], "mlp": []}

        for fold, (tr, te) in enumerate(kf.split(X), start=1):
            # Proper CV mean baseline: training-fold mean only.
            mean_pred = np.full(len(te), y[tr].mean(), dtype=np.float32)

            # Linear ridge baseline.
            ridge = make_pipeline(
                StandardScaler(),
                Ridge(alpha=10.0)
            )
            ridge.fit(X[tr], y[tr])
            ridge_pred = ridge.predict(X[te]).astype(np.float32)

            # Small neural network with target scaling.
            mlp_pred = fit_mlp_predict(
                X[tr], y[tr], X[te],
                seed=100 + fold,
                dev=dev,
            ).astype(np.float32)

            model_outputs["mean"] = model_outputs.get("mean", []) + [(
                te, mean_pred
            )]
            model_outputs["ridge"] = model_outputs.get("ridge", []) + [(
                te, ridge_pred
            )]
            model_outputs["mlp"] = model_outputs.get("mlp", []) + [(
                te, mlp_pred
            )]

            fold_stats["mean"].append(metric_dict(y[te], mean_pred))
            fold_stats["ridge"].append(metric_dict(y[te], ridge_pred))
            fold_stats["mlp"].append(metric_dict(y[te], mlp_pred))

        for model_name in ("mean", "ridge", "mlp"):
            oof = np.zeros(len(y), dtype=np.float32)
            for te, pred in model_outputs[model_name]:
                oof[te] = pred

            overall = metric_dict(y, oof)
            summary.append({
                "feature_set": feature_name,
                "model": model_name,
                **overall,
                "fold_rmse_mean": float(
                    np.mean([m["rmse"] for m in fold_stats[model_name]])
                ),
                "fold_rmse_std": float(
                    np.std([m["rmse"] for m in fold_stats[model_name]])
                ),
                "pred_mean": float(oof.mean()),
                "true_mean": float(y.mean()),
            })

            for i, pred in enumerate(oof):
                pred_rows.append({
                    "feature_set": feature_name,
                    "model": model_name,
                    "fold": int(
                        next(
                            fold for fold, (tr, te) in enumerate(
                                kf.split(X), start=1
                            ) if i in te
                        )
                    ),
                    "pdb_id": rows.iloc[i]["pdb_id"],
                    "y_true": float(y[i]),
                    "y_pred": float(pred),
                })

    summary_df = pd.DataFrame(summary)
    pred_df = pd.DataFrame(pred_rows)

    summary_df.to_csv(out / "summary_v2.csv", index=False)
    pred_df.to_csv(out / "cv_predictions_v2.csv", index=False)
    (out / "summary_v2.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    print("\n=== SUMMARY V2 ===")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
