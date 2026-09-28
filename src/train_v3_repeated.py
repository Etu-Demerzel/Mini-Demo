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


# ============================================================
# Repeated CV seeds
# ============================================================

CV_SEEDS = [
    42,
    123,
    456,
    789,
    2026,
]


# ============================================================
# Device
# ============================================================

def get_device():

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# ============================================================
# Small MLP
#
# Same philosophy as train_v2.py:
#
# input
#  ↓
# 64
#  ↓
# ReLU
#  ↓
# 32
#  ↓
# ReLU
#  ↓
# 1
# ============================================================

class SmallMLP(torch.nn.Module):

    def __init__(self, d_in):

        super().__init__()

        self.net = torch.nn.Sequential(

            torch.nn.Linear(
                d_in,
                64,
            ),

            torch.nn.ReLU(),

            torch.nn.Linear(
                64,
                32,
            ),

            torch.nn.ReLU(),

            torch.nn.Linear(
                32,
                1,
            ),
        )

    def forward(self, x):

        return self.net(x).squeeze(-1)


# ============================================================
# Metrics
# ============================================================

def safe_pearson(y, p):

    if len(y) < 2:
        return np.nan

    if np.std(y) < 1e-12:
        return np.nan

    if np.std(p) < 1e-12:
        return np.nan

    try:
        return float(
            pearsonr(y, p).statistic
        )

    except Exception:
        return np.nan


def safe_spearman(y, p):

    if len(y) < 2:
        return np.nan

    if np.std(y) < 1e-12:
        return np.nan

    if np.std(p) < 1e-12:
        return np.nan

    try:
        return float(
            spearmanr(y, p).statistic
        )

    except Exception:
        return np.nan


def metric_dict(y, p):

    y = np.asarray(
        y,
        dtype=np.float64,
    )

    p = np.asarray(
        p,
        dtype=np.float64,
    )

    return {

        "rmse": float(
            np.sqrt(
                np.mean(
                    (y - p) ** 2
                )
            )
        ),

        "mae": float(
            np.mean(
                np.abs(
                    y - p
                )
            )
        ),

        "pearson_r":
            safe_pearson(
                y,
                p,
            ),

        "spearman_rho":
            safe_spearman(
                y,
                p,
            ),
    }


# ============================================================
# MLP training
#
# Important:
#
# 1. X scaler is fitted only on outer training fold
# 2. y scaling is fitted only on outer training fold
# 3. outer training fold is split again into:
#       inner train
#       inner validation
# 4. early stopping uses validation loss
# 5. outer test fold is never used for training
# ============================================================

def fit_mlp_predict(
    xtr,
    ytr,
    xte,
    seed,
    device,
):

    # --------------------------------------------------------
    # Feature scaling
    # --------------------------------------------------------

    x_scaler = StandardScaler()

    x_scaler.fit(
        xtr
    )

    xtr_scaled = (
        x_scaler
        .transform(xtr)
        .astype(np.float32)
    )

    xte_scaled = (
        x_scaler
        .transform(xte)
        .astype(np.float32)
    )

    # --------------------------------------------------------
    # Target scaling
    # --------------------------------------------------------

    y_mean = float(
        np.mean(ytr)
    )

    y_std = float(
        np.std(ytr)
    )

    if y_std < 1e-8:
        y_std = 1.0

    ytr_scaled = (
        (
            ytr
            - y_mean
        )
        / y_std
    ).astype(np.float32)

    # --------------------------------------------------------
    # Inner validation split
    # --------------------------------------------------------

    all_idx = np.arange(
        len(ytr_scaled)
    )

    train_idx, val_idx = train_test_split(

        all_idx,

        test_size=0.20,

        random_state=seed,

        shuffle=True,
    )

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    np.random.seed(seed)

    torch.manual_seed(seed)

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = SmallMLP(
        xtr_scaled.shape[1]
    ).to(device)

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=2e-3,

        weight_decay=1e-3,
    )

    loss_fn = torch.nn.MSELoss()

    # --------------------------------------------------------
    # Tensor conversion
    # --------------------------------------------------------

    X_all = torch.from_numpy(
        xtr_scaled
    ).to(device)

    Y_all = torch.from_numpy(
        ytr_scaled
    ).to(device)

    X_train = X_all[
        train_idx
    ]

    Y_train = Y_all[
        train_idx
    ]

    X_val = X_all[
        val_idx
    ]

    Y_val = Y_all[
        val_idx
    ]

    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    best_val_loss = float(
        "inf"
    )

    best_state = copy.deepcopy(
        model.state_dict()
    )

    patience = 80

    bad_epochs = 0

    max_epochs = 1000

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    for epoch in range(
        1,
        max_epochs + 1,
    ):

        model.train()

        pred_train = model(
            X_train
        )

        train_loss = loss_fn(
            pred_train,
            Y_train,
        )

        optimizer.zero_grad()

        train_loss.backward()

        optimizer.step()

        # validation
        model.eval()

        with torch.no_grad():

            pred_val = model(
                X_val
            )

            val_loss = float(

                loss_fn(
                    pred_val,
                    Y_val,
                )
                .detach()
                .cpu()
            )

        # early stopping
        if (
            val_loss
            < best_val_loss - 1e-5
        ):

            best_val_loss = (
                val_loss
            )

            best_state = (
                copy.deepcopy(
                    model.state_dict()
                )
            )

            bad_epochs = 0

        else:

            bad_epochs += 1

        if bad_epochs >= patience:
            break

    # --------------------------------------------------------
    # Restore best model
    # --------------------------------------------------------

    model.load_state_dict(
        best_state
    )

    model.eval()

    # --------------------------------------------------------
    # Test prediction
    # --------------------------------------------------------

    X_test = torch.from_numpy(
        xte_scaled
    ).to(device)

    with torch.no_grad():

        pred_scaled = (
            model(X_test)
            .detach()
            .cpu()
            .numpy()
        )

    # --------------------------------------------------------
    # Return to original affinity scale
    # --------------------------------------------------------

    pred_original = (

        pred_scaled
        * y_std
        + y_mean
    )

    return pred_original.astype(
        np.float32
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--features",
        required=True,
        help=(
            "Example: "
            "outputs/strict182/features.json"
        ),
    )

    parser.add_argument(
        "--out_dir",
        required=True,
        help=(
            "Example: "
            "outputs/repeated_cv"
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    out_dir = Path(
        args.out_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load features
    # --------------------------------------------------------

    df = pd.read_json(
        args.features
    )

    ligand = np.asarray(

        df[
            "ligand"
        ].tolist(),

        dtype=np.float32,
    )

    pocket = np.asarray(

        df[
            "pocket"
        ].tolist(),

        dtype=np.float32,
    )

    y = df[
        "label"
    ].to_numpy(
        dtype=np.float32
    )

    pdb_ids = df[
        "pdb_id"
    ].astype(
        str
    ).to_numpy()

    # --------------------------------------------------------
    # Feature sets
    # --------------------------------------------------------

    feature_sets = {

        "ligand_only":
            ligand,

        "ligand_plus_pocket":
            np.concatenate(
                [
                    ligand,
                    pocket,
                ],
                axis=1,
            ),
    }

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = get_device()

    print(
        "device:",
        device,
    )

    print(
        "N:",
        len(y),
    )

    print(
        "target mean/std:",
        float(y.mean()),
        float(y.std()),
    )

    print(
        "ligand dim:",
        ligand.shape[1],
    )

    print(
        "pocket dim:",
        pocket.shape[1],
    )

    print(
        "CV seeds:",
        CV_SEEDS,
    )

    print()

    # ========================================================
    # Storage
    # ========================================================

    prediction_rows = []

    fold_metric_rows = []

    seed_summary_rows = []

    # ========================================================
    # Repeated CV
    # ========================================================

    for cv_seed in CV_SEEDS:

        print(
            "=" * 70
        )

        print(
            f"CV SEED = {cv_seed}"
        )

        print(
            "=" * 70
        )

        kf = KFold(

            n_splits=5,

            shuffle=True,

            random_state=cv_seed,
        )

        # ----------------------------------------------------
        # Same fold splits are used for both feature sets.
        # This is important for paired comparison.
        # ----------------------------------------------------

        splits = list(
            kf.split(
                np.arange(
                    len(y)
                )
            )
        )

        for feature_name, X in (
            feature_sets.items()
        ):

            print()
            print(
                "Feature set:",
                feature_name,
            )

            # -----------------------------------------------
            # OOF predictions for this specific CV seed
            # -----------------------------------------------

            seed_oof = {

                "mean":
                    np.zeros(
                        len(y),
                        dtype=np.float32,
                    ),

                "ridge":
                    np.zeros(
                        len(y),
                        dtype=np.float32,
                    ),

                "mlp":
                    np.zeros(
                        len(y),
                        dtype=np.float32,
                    ),
            }

            # -----------------------------------------------
            # 5 folds
            # -----------------------------------------------

            for fold_number, (
                train_idx,
                test_idx,
            ) in enumerate(
                splits,
                start=1,
            ):

                X_train = X[
                    train_idx
                ]

                X_test = X[
                    test_idx
                ]

                y_train = y[
                    train_idx
                ]

                y_test = y[
                    test_idx
                ]

                # ============================================
                # Mean baseline
                # ============================================

                mean_pred = np.full(

                    len(test_idx),

                    y_train.mean(),

                    dtype=np.float32,
                )

                # ============================================
                # Ridge
                # ============================================

                ridge = make_pipeline(

                    StandardScaler(),

                    Ridge(
                        alpha=10.0
                    ),
                )

                ridge.fit(
                    X_train,
                    y_train,
                )

                ridge_pred = (
                    ridge
                    .predict(X_test)
                    .astype(np.float32)
                )

                # ============================================
                # MLP
                # ============================================

                # Different inner-training seed
                # for each outer CV seed and fold.
                #
                # This stays deterministic but avoids using
                # exactly the same initialization everywhere.
                mlp_seed = (
                    cv_seed * 100
                    + fold_number
                )

                mlp_pred = fit_mlp_predict(

                    X_train,
                    y_train,
                    X_test,

                    seed=mlp_seed,

                    device=device,
                )

                model_predictions = {

                    "mean":
                        mean_pred,

                    "ridge":
                        ridge_pred,

                    "mlp":
                        mlp_pred,
                }

                # ============================================
                # Save each fold
                # ============================================

                for (
                    model_name,
                    predictions,
                ) in model_predictions.items():

                    seed_oof[
                        model_name
                    ][
                        test_idx
                    ] = predictions

                    metrics = metric_dict(
                        y_test,
                        predictions,
                    )

                    fold_metric_rows.append(
                        {
                            "cv_seed":
                                cv_seed,

                            "fold":
                                fold_number,

                            "feature_set":
                                feature_name,

                            "model":
                                model_name,

                            "n_test":
                                len(test_idx),

                            **metrics,
                        }
                    )

                    # individual sample predictions
                    for local_i, global_i in enumerate(
                        test_idx
                    ):

                        prediction_rows.append(
                            {
                                "cv_seed":
                                    cv_seed,

                                "fold":
                                    fold_number,

                                "feature_set":
                                    feature_name,

                                "model":
                                    model_name,

                                "pdb_id":
                                    pdb_ids[
                                        global_i
                                    ],

                                "y_true":
                                    float(
                                        y[
                                            global_i
                                        ]
                                    ),

                                "y_pred":
                                    float(
                                        predictions[
                                            local_i
                                        ]
                                    ),
                            }
                        )

                print(
                    f"  fold {fold_number} done"
                )

            # -----------------------------------------------
            # Metrics across the full OOF set for this seed
            # -----------------------------------------------

            for model_name in [

                "mean",
                "ridge",
                "mlp",

            ]:

                overall_metrics = (
                    metric_dict(
                        y,
                        seed_oof[
                            model_name
                        ],
                    )
                )

                seed_summary_rows.append(
                    {
                        "cv_seed":
                            cv_seed,

                        "feature_set":
                            feature_name,

                        "model":
                            model_name,

                        **overall_metrics,

                        "pred_mean":
                            float(
                                seed_oof[
                                    model_name
                                ].mean()
                            ),

                        "true_mean":
                            float(
                                y.mean()
                            ),
                    }
                )

    # ========================================================
    # Save raw results
    # ========================================================

    predictions_df = pd.DataFrame(
        prediction_rows
    )

    fold_metrics_df = pd.DataFrame(
        fold_metric_rows
    )

    seed_summary_df = pd.DataFrame(
        seed_summary_rows
    )

    predictions_df.to_csv(

        out_dir
        / "cv_predictions_repeated.csv",

        index=False,
    )

    fold_metrics_df.to_csv(

        out_dir
        / "fold_metrics_repeated.csv",

        index=False,
    )

    seed_summary_df.to_csv(

        out_dir
        / "seed_summary_repeated.csv",

        index=False,
    )

    # ========================================================
    # Aggregate summary across seeds
    # ========================================================

    aggregate_rows = []

    for (
        feature_set,
        model_name,
    ), group in seed_summary_df.groupby(

        [
            "feature_set",
            "model",
        ]
    ):

        aggregate_rows.append(
            {
                "feature_set":
                    feature_set,

                "model":
                    model_name,

                "rmse_mean":
                    group[
                        "rmse"
                    ].mean(),

                "rmse_std":
                    group[
                        "rmse"
                    ].std(),

                "mae_mean":
                    group[
                        "mae"
                    ].mean(),

                "mae_std":
                    group[
                        "mae"
                    ].std(),

                "pearson_mean":
                    group[
                        "pearson_r"
                    ].mean(),

                "pearson_std":
                    group[
                        "pearson_r"
                    ].std(),

                "spearman_mean":
                    group[
                        "spearman_rho"
                    ].mean(),

                "spearman_std":
                    group[
                        "spearman_rho"
                    ].std(),

                "n_seeds":
                    len(group),
            }
        )

    aggregate_df = pd.DataFrame(
        aggregate_rows
    )

    aggregate_df.to_csv(

        out_dir
        / "summary_repeated.csv",

        index=False,
    )

    # ========================================================
    # Paired pocket-vs-ligand comparison
    #
    # Same seed + same fold + same model
    # ========================================================

    paired_rows = []

    for cv_seed in CV_SEEDS:

        for fold_number in range(
            1,
            6,
        ):

            for model_name in [
                "ridge",
                "mlp",
            ]:

                ligand_row = (
                    fold_metrics_df[
                        (
                            fold_metrics_df[
                                "cv_seed"
                            ]
                            == cv_seed
                        )
                        &
                        (
                            fold_metrics_df[
                                "fold"
                            ]
                            == fold_number
                        )
                        &
                        (
                            fold_metrics_df[
                                "feature_set"
                            ]
                            == "ligand_only"
                        )
                        &
                        (
                            fold_metrics_df[
                                "model"
                            ]
                            == model_name
                        )
                    ]
                    .iloc[0]
                )

                pocket_row = (
                    fold_metrics_df[
                        (
                            fold_metrics_df[
                                "cv_seed"
                            ]
                            == cv_seed
                        )
                        &
                        (
                            fold_metrics_df[
                                "fold"
                            ]
                            == fold_number
                        )
                        &
                        (
                            fold_metrics_df[
                                "feature_set"
                            ]
                            == "ligand_plus_pocket"
                        )
                        &
                        (
                            fold_metrics_df[
                                "model"
                            ]
                            == model_name
                        )
                    ]
                    .iloc[0]
                )

                paired_rows.append(
                    {
                        "cv_seed":
                            cv_seed,

                        "fold":
                            fold_number,

                        "model":
                            model_name,

                        # negative = pocket improves error
                        "delta_rmse":
                            pocket_row[
                                "rmse"
                            ]
                            -
                            ligand_row[
                                "rmse"
                            ],

                        "delta_mae":
                            pocket_row[
                                "mae"
                            ]
                            -
                            ligand_row[
                                "mae"
                            ],

                        # positive = pocket improves correlation
                        "delta_pearson":
                            pocket_row[
                                "pearson_r"
                            ]
                            -
                            ligand_row[
                                "pearson_r"
                            ],

                        "delta_spearman":
                            pocket_row[
                                "spearman_rho"
                            ]
                            -
                            ligand_row[
                                "spearman_rho"
                            ],
                    }
                )

    paired_df = pd.DataFrame(
        paired_rows
    )

    paired_df.to_csv(

        out_dir
        / "pocket_effect_repeated.csv",

        index=False,
    )

    # ========================================================
    # Pocket effect summary
    # ========================================================

    effect_summary_rows = []

    for model_name, group in (
        paired_df.groupby(
            "model"
        )
    ):

        effect_summary_rows.append(
            {
                "model":
                    model_name,

                "n_fold_comparisons":
                    len(group),

                "mean_delta_rmse":
                    group[
                        "delta_rmse"
                    ].mean(),

                "std_delta_rmse":
                    group[
                        "delta_rmse"
                    ].std(),

                "median_delta_rmse":
                    group[
                        "delta_rmse"
                    ].median(),

                "pocket_rmse_improved_count":
                    int(
                        (
                            group[
                                "delta_rmse"
                            ]
                            < 0
                        ).sum()
                    ),

                "pocket_rmse_worsened_count":
                    int(
                        (
                            group[
                                "delta_rmse"
                            ]
                            > 0
                        ).sum()
                    ),

                "mean_delta_mae":
                    group[
                        "delta_mae"
                    ].mean(),

                "mean_delta_pearson":
                    group[
                        "delta_pearson"
                    ].mean(),

                "mean_delta_spearman":
                    group[
                        "delta_spearman"
                    ].mean(),
            }
        )

    effect_summary_df = pd.DataFrame(
        effect_summary_rows
    )

    effect_summary_df.to_csv(

        out_dir
        / "pocket_effect_summary.csv",

        index=False,
    )

    # ========================================================
    # Console output
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "REPEATED CV SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        aggregate_df.to_string(
            index=False
        )
    )

    print()
    print(
        "=" * 70
    )

    print(
        "POCKET EFFECT SUMMARY"
    )

    print(
        "Negative delta RMSE = pocket improves"
    )

    print(
        "=" * 70
    )

    print(
        effect_summary_df.to_string(
            index=False
        )
    )

    print()
    print(
        "Saved:"
    )

    print(
        out_dir
        / "summary_repeated.csv"
    )

    print(
        out_dir
        / "seed_summary_repeated.csv"
    )

    print(
        out_dir
        / "fold_metrics_repeated.csv"
    )

    print(
        out_dir
        / "pocket_effect_repeated.csv"
    )

    print(
        out_dir
        / "pocket_effect_summary.csv"
    )

    print(
        out_dir
        / "cv_predictions_repeated.csv"
    )


if __name__ == "__main__":

    main()