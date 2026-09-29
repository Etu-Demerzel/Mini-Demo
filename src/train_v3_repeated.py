from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class SmallMLP(torch.nn.Module):
    def __init__(self, d_in: int):
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


def get_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def metric_dict(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    valid = np.std(y_true) > 1e-12 and np.std(y_pred) > 1e-12
    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "pearson_r": float(pearsonr(y_true, y_pred).statistic) if valid else np.nan,
        "spearman_rho": float(spearmanr(y_true, y_pred).statistic) if valid else np.nan,
    }


def fit_mlp_predict(x_train, y_train, x_test, seed, device):
    x_scaler = StandardScaler()
    x_train = x_scaler.fit_transform(x_train).astype(np.float32)
    x_test = x_scaler.transform(x_test).astype(np.float32)

    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train)) or 1.0
    y_scaled = ((y_train - y_mean) / y_std).astype(np.float32)

    idx = np.arange(len(y_train))
    train_idx, val_idx = train_test_split(
        idx, test_size=0.20, random_state=seed, shuffle=True
    )

    np.random.seed(seed)
    torch.manual_seed(seed)

    model = SmallMLP(x_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    loss_fn = torch.nn.MSELoss()

    X = torch.from_numpy(x_train).to(device)
    Y = torch.from_numpy(y_scaled).to(device)

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    bad_epochs = 0

    for _ in range(1000):
        model.train()
        pred = model(X[train_idx])
        loss = loss_fn(pred, Y[train_idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(X[val_idx]), Y[val_idx]).cpu())

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= 80:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.from_numpy(x_test).to(device)).cpu().numpy()
    return (pred_scaled * y_std + y_mean).astype(np.float32)


def run_fold(X, y, train_idx, test_idx, seed, device):
    xtr, xte = X[train_idx], X[test_idx]
    ytr = y[train_idx]

    mean_pred = np.full(len(test_idx), ytr.mean(), dtype=np.float32)

    ridge = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    ridge.fit(xtr, ytr)
    ridge_pred = ridge.predict(xte).astype(np.float32)

    mlp_pred = fit_mlp_predict(xtr, ytr, xte, seed, device)
    return {"mean": mean_pred, "ridge": ridge_pred, "mlp": mlp_pred}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seeds", default="42,123,456,789,2026")
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_json(args.features)
    ligand = np.asarray(df["ligand"].tolist(), dtype=np.float32)
    pocket = np.asarray(df["pocket"].tolist(), dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.float32)
    pdb_ids = df["pdb_id"].astype(str).to_numpy()

    feature_sets = {
        "ligand_only": ligand,
        "ligand_plus_pocket": np.concatenate([ligand, pocket], axis=1),
    }

    device = get_device()
    print(f"device={device} | N={len(y)} | ligand={ligand.shape[1]} | pocket={pocket.shape[1]}")

    pred_rows, fold_rows, seed_rows = [], [], []

    for cv_seed in seeds:
        print(f"\n=== CV seed {cv_seed} ===")
        splits = list(KFold(5, shuffle=True, random_state=cv_seed).split(np.arange(len(y))))

        for feature_name, X in feature_sets.items():
            oof = {m: np.zeros(len(y), dtype=np.float32) for m in ["mean", "ridge", "mlp"]}

            for fold, (train_idx, test_idx) in enumerate(splits, start=1):
                model_seed = cv_seed * 100 + fold
                preds = run_fold(X, y, train_idx, test_idx, model_seed, device)

                for model_name, y_pred in preds.items():
                    oof[model_name][test_idx] = y_pred
                    fold_rows.append({
                        "cv_seed": cv_seed,
                        "fold": fold,
                        "feature_set": feature_name,
                        "model": model_name,
                        "n_test": len(test_idx),
                        **metric_dict(y[test_idx], y_pred),
                    })

                    pred_rows.extend({
                        "cv_seed": cv_seed,
                        "fold": fold,
                        "feature_set": feature_name,
                        "model": model_name,
                        "pdb_id": pdb_ids[g],
                        "y_true": float(y[g]),
                        "y_pred": float(y_pred[j]),
                    } for j, g in enumerate(test_idx))

            for model_name, y_pred in oof.items():
                seed_rows.append({
                    "cv_seed": cv_seed,
                    "feature_set": feature_name,
                    "model": model_name,
                    **metric_dict(y, y_pred),
                    "pred_mean": float(y_pred.mean()),
                    "true_mean": float(y.mean()),
                })

    predictions = pd.DataFrame(pred_rows)
    fold_metrics = pd.DataFrame(fold_rows)
    seed_summary = pd.DataFrame(seed_rows)

    predictions.to_csv(out_dir / "cv_predictions_repeated.csv", index=False)
    fold_metrics.to_csv(out_dir / "fold_metrics_repeated.csv", index=False)
    seed_summary.to_csv(out_dir / "seed_summary_repeated.csv", index=False)

    summary = seed_summary.groupby(["feature_set", "model"], as_index=False).agg(
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        mae_mean=("mae", "mean"), mae_std=("mae", "std"),
        pearson_mean=("pearson_r", "mean"), pearson_std=("pearson_r", "std"),
        spearman_mean=("spearman_rho", "mean"), spearman_std=("spearman_rho", "std"),
        n_seeds=("cv_seed", "nunique"),
    )
    summary.to_csv(out_dir / "summary_repeated.csv", index=False)

    paired = fold_metrics.pivot_table(
        index=["cv_seed", "fold", "model"],
        columns="feature_set",
        values=["rmse", "mae", "pearson_r", "spearman_rho"],
    )
    paired.columns = [f"{m}__{f}" for m, f in paired.columns]
    paired = paired.reset_index()

    for metric in ["rmse", "mae", "pearson_r", "spearman_rho"]:
        paired[f"delta_{metric}"] = (
            paired[f"{metric}__ligand_plus_pocket"] - paired[f"{metric}__ligand_only"]
        )

    paired.to_csv(out_dir / "pocket_effect_repeated.csv", index=False)

    pocket_summary = paired[paired["model"].isin(["ridge", "mlp"])].groupby("model", as_index=False).agg(
        n_fold_comparisons=("delta_rmse", "size"),
        mean_delta_rmse=("delta_rmse", "mean"),
        std_delta_rmse=("delta_rmse", "std"),
        median_delta_rmse=("delta_rmse", "median"),
        pocket_rmse_improved_count=("delta_rmse", lambda s: int((s < 0).sum())),
        pocket_rmse_worsened_count=("delta_rmse", lambda s: int((s > 0).sum())),
        mean_delta_mae=("delta_mae", "mean"),
        mean_delta_pearson=("delta_pearson_r", "mean"),
        mean_delta_spearman=("delta_spearman_rho", "mean"),
    )
    pocket_summary.to_csv(out_dir / "pocket_effect_summary.csv", index=False)

    print("\n=== summary_repeated.csv ===")
    print(summary.to_string(index=False))
    print("\n=== pocket_effect_summary.csv ===")
    print(pocket_summary.to_string(index=False))


if __name__ == "__main__":
    main()
