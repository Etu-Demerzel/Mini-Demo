from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# General plotting settings
# ============================================================

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ============================================================
# Utility
# ============================================================

def savefig(path: Path):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"[saved] {path}"
    )


# ============================================================
# Figure 1
# Dataset curation / quality control
# ============================================================

def plot_dataset_qc(
    index_csv: Path,
    features_json: Path,
    out_dir: Path,
):

    index_df = pd.read_csv(
        index_csv
    )

    features_df = pd.read_json(
        features_json
    )

    total_n = len(index_df)

    included_n = len(features_df)

    excluded_n = (
        total_n - included_n
    )

    values = [
        total_n,
        included_n,
        excluded_n,
    ]

    labels = [
        "Original\nCore set",
        "Strictly\nsanitized",
        "Excluded",
    ]

    fig, ax = plt.subplots(
        figsize=(7.2, 4.6)
    )

    bars = ax.bar(
        labels,
        values,
        width=0.60,
    )

    ax.set_ylabel(
        "Number of complexes"
    )

    ax.set_title(
        "Dataset curation"
    )

    ax.set_ylim(
        0,
        max(values) * 1.18,
    )

    for bar, value in zip(
        bars,
        values,
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,

            bar.get_height() + 3,

            str(value),

            ha="center",
            va="bottom",
            fontsize=12,
        )

    ax.text(
        0.98,
        0.95,
        (
            f"Usable fraction: "
            f"{included_n / total_n * 100:.1f}%"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
    )

    savefig(
        out_dir
        / "01_dataset_qc.png"
    )


# ============================================================
# Figure 2
# Target distribution of strict 182 subset
# ============================================================

def plot_target_distribution(
    features_json: Path,
    out_dir: Path,
):

    df = pd.read_json(
        features_json
    )

    y = df[
        "label"
    ].to_numpy()

    fig, ax = plt.subplots(
        figsize=(7.2, 4.6)
    )

    ax.hist(
        y,
        bins=15,
        edgecolor="black",
        linewidth=0.7,
    )

    ax.axvline(
        np.mean(y),
        linestyle="--",
        linewidth=1.5,
        label=(
            f"Mean = "
            f"{np.mean(y):.2f}"
        ),
    )

    ax.set_xlabel(
        "Binding affinity label (-logKd/Ki)"
    )

    ax.set_ylabel(
        "Number of complexes"
    )

    ax.set_title(
        "Binding-affinity distribution"
    )

    ax.legend(
        frameon=False
    )

    ax.text(
        0.98,
        0.95,
        (
            f"N = {len(y)}\n"
            f"SD = {np.std(y):.2f}\n"
            f"Range = "
            f"{np.min(y):.2f}–"
            f"{np.max(y):.2f}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
    )

    savefig(
        out_dir
        / "02_target_distribution.png"
    )


# ============================================================
# Figure 3
# Baseline model comparison
#
# Reads actual train_v2 summary
# ============================================================

def plot_v2_model_comparison(
    summary_csv: Path,
    out_dir: Path,
):

    df = pd.read_csv(
        summary_csv
    )

    # We only need RMSE here.
    pivot = df.pivot(
        index="model",
        columns="feature_set",
        values="rmse",
    )

    # Keep intuitive model ordering.
    desired_order = [
        "mean",
        "ridge",
        "mlp",
    ]

    models = [
        x
        for x in desired_order
        if x in pivot.index
    ]

    pivot = pivot.loc[
        models
    ]

    x = np.arange(
        len(models)
    )

    width = 0.34

    fig, ax = plt.subplots(
        figsize=(8.2, 4.8)
    )

    if (
        "ligand_only"
        in pivot.columns
    ):

        values = pivot[
            "ligand_only"
        ].to_numpy()

        bars1 = ax.bar(
            x - width / 2,
            values,
            width,
            label="Ligand only",
        )

        for bar, value in zip(
            bars1,
            values,
        ):
            ax.text(
                bar.get_x()
                + bar.get_width()/2,

                value + 0.025,

                f"{value:.3f}",

                ha="center",
                va="bottom",
                fontsize=9,
            )

    if (
        "ligand_plus_pocket"
        in pivot.columns
    ):

        values = pivot[
            "ligand_plus_pocket"
        ].to_numpy()

        bars2 = ax.bar(
            x + width / 2,
            values,
            width,
            label="Ligand + coarse pocket",
        )

        for bar, value in zip(
            bars2,
            values,
        ):
            ax.text(
                bar.get_x()
                + bar.get_width()/2,

                value + 0.025,

                f"{value:.3f}",

                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        [
            model.upper()
            if model != "mean"
            else "Mean"
            for model in models
        ]
    )

    ax.set_ylabel(
        "OOF RMSE"
    )

    ax.set_title(
        "Baseline performance on the strict 182-complex subset"
    )

    ax.legend(
        frameon=False
    )

    savefig(
        out_dir
        / "03_baseline_rmse.png"
    )


# ============================================================
# Figure 4
# MLP parity plots
#
# Automatically handles either common column layout:
#
# feature_set/model
#
# or older format in which 'model'
# may directly contain ligand_only etc.
# ============================================================

def plot_parity(
    predictions_csv: Path,
    out_dir: Path,
):

    df = pd.read_csv(
        predictions_csv
    )

    # New v2 format
    if (
        "feature_set" in df.columns
        and "model" in df.columns
    ):

        subset = df[
            df["model"] == "mlp"
        ].copy()

        feature_column = (
            "feature_set"
        )

    else:

        # Fallback for older prediction output
        subset = df.copy()

        feature_column = (
            "model"
        )

    desired_sets = [
        "ligand_only",
        "ligand_plus_pocket",
    ]

    for feature_set in desired_sets:

        current = subset[
            subset[
                feature_column
            ]
            == feature_set
        ]

        if len(current) == 0:
            continue

        y = current[
            "y_true"
        ].to_numpy()

        p = current[
            "y_pred"
        ].to_numpy()

        rmse = np.sqrt(
            np.mean(
                (y - p) ** 2
            )
        )

        corr = np.corrcoef(
            y,
            p,
        )[0, 1]

        lo = min(
            np.min(y),
            np.min(p),
        )

        hi = max(
            np.max(y),
            np.max(p),
        )

        fig, ax = plt.subplots(
            figsize=(5.5, 5.2)
        )

        ax.scatter(
            y,
            p,
            alpha=0.72,
            s=34,
        )

        ax.plot(
            [lo, hi],
            [lo, hi],
            linestyle="--",
            linewidth=1.2,
        )

        ax.set_xlim(
            lo - 0.3,
            hi + 0.3,
        )

        ax.set_ylim(
            lo - 0.3,
            hi + 0.3,
        )

        ax.set_xlabel(
            "Experimental affinity"
        )

        ax.set_ylabel(
            "Predicted affinity"
        )

        pretty_name = (
            "Ligand only"
            if feature_set
            == "ligand_only"
            else
            "Ligand + coarse pocket"
        )

        ax.set_title(
            f"MLP parity plot — {pretty_name}"
        )

        ax.text(
            0.04,
            0.96,
            (
                f"RMSE = {rmse:.3f}\n"
                f"Pearson r = {corr:.3f}\n"
                f"N = {len(y)}"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
        )

        filename = (
            "04a_parity_ligand.png"
            if feature_set
            == "ligand_only"
            else
            "04b_parity_ligand_pocket.png"
        )

        savefig(
            out_dir
            / filename
        )


# ============================================================
# Figure 5
# Prediction residuals
# ============================================================

def plot_residuals(
    predictions_csv: Path,
    out_dir: Path,
):

    df = pd.read_csv(
        predictions_csv
    )

    if (
        "feature_set" in df.columns
        and "model" in df.columns
    ):

        df = df[
            df["model"] == "mlp"
        ]

        feature_col = (
            "feature_set"
        )

    else:

        feature_col = (
            "model"
        )

    fig, ax = plt.subplots(
        figsize=(7.3, 4.7)
    )

    for feature_set, label in [

        (
            "ligand_only",
            "Ligand only",
        ),

        (
            "ligand_plus_pocket",
            "Ligand + coarse pocket",
        ),

    ]:

        temp = df[
            df[
                feature_col
            ]
            == feature_set
        ]

        if len(temp) == 0:
            continue

        residual = (
            temp[
                "y_pred"
            ].to_numpy()
            -
            temp[
                "y_true"
            ].to_numpy()
        )

        ax.hist(
            residual,
            bins=18,
            alpha=0.55,
            label=label,
        )

    ax.axvline(
        0,
        linestyle="--",
        linewidth=1.2,
    )

    ax.set_xlabel(
        "Residual (prediction − experiment)"
    )

    ax.set_ylabel(
        "Count"
    )

    ax.set_title(
        "MLP residual distribution"
    )

    ax.legend(
        frameon=False
    )

    savefig(
        out_dir
        / "05_mlp_residuals.png"
    )


# ============================================================
# Figure 6
# Repeated CV summary
#
# Shows mean RMSE ± SD across the five CV seeds.
# ============================================================

def plot_repeated_cv_rmse(
    summary_csv: Path,
    out_dir: Path,
):

    df = pd.read_csv(
        summary_csv
    )

    df = df[
        df["model"] == "mlp"
    ].copy()

    order = [
        "ligand_only",
        "ligand_plus_pocket",
    ]

    df[
        "feature_set"
    ] = pd.Categorical(
        df["feature_set"],
        categories=order,
        ordered=True,
    )

    df = (
        df
        .sort_values(
            "feature_set"
        )
    )

    labels = [
        "Ligand only",
        "Ligand +\ncoarse pocket",
    ]

    means = df[
        "rmse_mean"
    ].to_numpy()

    stds = df[
        "rmse_std"
    ].to_numpy()

    x = np.arange(
        len(means)
    )

    fig, ax = plt.subplots(
        figsize=(6.8, 4.7)
    )

    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=6,
        width=0.60,
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        labels
    )

    ax.set_ylabel(
        "RMSE"
    )

    ax.set_title(
        "Repeated 5-fold CV: MLP performance"
    )

    for bar, mean, std in zip(
        bars,
        means,
        stds,
    ):

        ax.text(
            bar.get_x()
            + bar.get_width()/2,

            mean + std + 0.025,

            f"{mean:.3f} ± {std:.3f}",

            ha="center",
            va="bottom",
            fontsize=10,
        )

    savefig(
        out_dir
        / "06_repeated_cv_rmse.png"
    )


# ============================================================
# Figure 7
# Pocket effect:
# improved vs worsened fold counts
# ============================================================

def plot_pocket_effect_counts(
    effect_summary_csv: Path,
    out_dir: Path,
):

    df = pd.read_csv(
        effect_summary_csv
    )

    desired_order = [
        "mlp",
        "ridge",
    ]

    df[
        "model"
    ] = pd.Categorical(
        df["model"],
        categories=desired_order,
        ordered=True,
    )

    df = (
        df
        .sort_values("model")
    )

    x = np.arange(
        len(df)
    )

    width = 0.34

    improved = df[
        "pocket_rmse_improved_count"
    ].to_numpy()

    worsened = df[
        "pocket_rmse_worsened_count"
    ].to_numpy()

    fig, ax = plt.subplots(
        figsize=(7.0, 4.8)
    )

    bars1 = ax.bar(
        x - width / 2,
        improved,
        width,
        label="Pocket improved RMSE",
    )

    bars2 = ax.bar(
        x + width / 2,
        worsened,
        width,
        label="Pocket worsened RMSE",
    )

    for bars in [
        bars1,
        bars2,
    ]:

        for bar in bars:

            ax.text(
                bar.get_x()
                + bar.get_width()/2,

                bar.get_height()
                + 0.4,

                f"{int(bar.get_height())}",

                ha="center",
                va="bottom",
                fontsize=11,
            )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        [
            str(x).upper()
            for x in df[
                "model"
            ].astype(str)
        ]
    )

    ax.set_ylabel(
        "Number of paired folds"
    )

    ax.set_ylim(
        0,
        max(
            improved.max(),
            worsened.max(),
        ) + 4,
    )

    ax.set_title(
        "Does the coarse pocket representation improve RMSE?"
    )

    ax.legend(
        frameon=False
    )

    ax.text(
        0.98,
        0.04,
        "5 seeds × 5 folds = 25 paired comparisons per model",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )

    savefig(
        out_dir
        / "07_pocket_effect_counts.png"
    )


# ============================================================
# Figure 8
# Distribution of paired delta RMSE
#
# This is scientifically more informative than counts alone.
#
# delta = pocket RMSE - ligand-only RMSE
#
# < 0 : pocket helps
# > 0 : pocket hurts
# ============================================================

def plot_paired_delta_rmse(
    paired_csv: Path,
    out_dir: Path,
):

    df = pd.read_csv(
        paired_csv
    )

    fig, ax = plt.subplots(
        figsize=(7.4, 4.8)
    )

    positions = []

    data = []

    labels = []

    for i, model in enumerate(
        [
            "mlp",
            "ridge",
        ],
        start=1,
    ):

        temp = df[
            df["model"]
            == model
        ]

        if len(temp) == 0:
            continue

        positions.append(
            i
        )

        data.append(
            temp[
                "delta_rmse"
            ].to_numpy()
        )

        labels.append(
            model.upper()
        )

    ax.boxplot(
        data,
        positions=positions,
        widths=0.55,
        showfliers=False,
    )

    # Overlay every actual fold result
    rng = np.random.default_rng(
        42
    )

    for pos, values in zip(
        positions,
        data,
    ):

        jitter = rng.normal(
            0,
            0.035,
            len(values),
        )

        ax.scatter(
            np.full(
                len(values),
                pos,
            )
            + jitter,

            values,

            alpha=0.65,
            s=30,
        )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1.2,
    )

    ax.set_xticks(
        positions
    )

    ax.set_xticklabels(
        labels
    )

    ax.set_ylabel(
        "ΔRMSE = pocket − ligand only"
    )

    ax.set_title(
        "Paired effect of coarse pocket features across CV folds"
    )

    ax.text(
        0.02,
        0.96,
        (
            "ΔRMSE < 0: pocket improves\n"
            "ΔRMSE > 0: pocket worsens"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )

    savefig(
        out_dir
        / "08_paired_delta_rmse.png"
    )


# ============================================================
# Figure 9
# Feature dimensionality / representation diagram
#
# This one uses only your actual dimensions.
# ============================================================

def plot_feature_dimensions(
    features_json: Path,
    out_dir: Path,
):

    df = pd.read_json(
        features_json
    )

    ligand_dim = len(
        df.iloc[0][
            "ligand"
        ]
    )

    pocket_dim = len(
        df.iloc[0][
            "pocket"
        ]
    )

    dims = [
        ligand_dim,
        pocket_dim,
        ligand_dim
        + pocket_dim,
    ]

    labels = [
        "Ligand",
        "Coarse\npocket",
        "Ligand +\npocket",
    ]

    fig, ax = plt.subplots(
        figsize=(7.0, 4.6)
    )

    bars = ax.bar(
        labels,
        dims,
        width=0.60,
    )

    ax.set_ylabel(
        "Feature dimensions"
    )

    ax.set_title(
        "Input representation dimensionality"
    )

    for bar, value in zip(
        bars,
        dims,
    ):

        ax.text(
            bar.get_x()
            + bar.get_width()/2,

            value + 12,

            str(value),

            ha="center",
            va="bottom",
            fontsize=11,
        )

    savefig(
        out_dir
        / "09_feature_dimensions.png"
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        default="data/v2013-core",
    )

    parser.add_argument(
        "--strict_dir",
        default="outputs/strict182",
    )

    parser.add_argument(
        "--repeated_dir",
        default="outputs/repeated_cv",
    )

    parser.add_argument(
        "--out_dir",
        default="figs/final",
    )

    args = parser.parse_args()

    data_dir = Path(
        args.data_dir
    )

    strict_dir = Path(
        args.strict_dir
    )

    repeated_dir = Path(
        args.repeated_dir
    )

    out_dir = Path(
        args.out_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    index_csv = (
        data_dir
        / "pdbbind_v2013_core.csv"
    )

    features_json = (
        strict_dir
        / "features.json"
    )

    summary_v2 = (
        strict_dir
        / "summary_v2.csv"
    )

    predictions_v2 = (
        strict_dir
        / "cv_predictions_v2.csv"
    )

    summary_repeated = (
        repeated_dir
        / "summary_repeated.csv"
    )

    pocket_summary = (
        repeated_dir
        / "pocket_effect_summary.csv"
    )

    pocket_repeated = (
        repeated_dir
        / "pocket_effect_repeated.csv"
    )

    # --------------------------------------------------------
    # Validate expected files
    # --------------------------------------------------------

    required = [

        index_csv,
        features_json,
        summary_v2,
        predictions_v2,
        summary_repeated,
        pocket_summary,
        pocket_repeated,

    ]

    missing = [

        p
        for p in required
        if not p.exists()

    ]

    if missing:

        print(
            "\nMissing required files:"
        )

        for p in missing:

            print(
                "  ",
                p,
            )

        raise FileNotFoundError(
            "Please fix the paths above "
            "before running visualization."
        )

    # --------------------------------------------------------
    # Generate all final figures
    # --------------------------------------------------------

    plot_dataset_qc(
        index_csv,
        features_json,
        out_dir,
    )

    plot_target_distribution(
        features_json,
        out_dir,
    )

    plot_v2_model_comparison(
        summary_v2,
        out_dir,
    )

    plot_parity(
        predictions_v2,
        out_dir,
    )

    plot_residuals(
        predictions_v2,
        out_dir,
    )

    plot_repeated_cv_rmse(
        summary_repeated,
        out_dir,
    )

    plot_pocket_effect_counts(
        pocket_summary,
        out_dir,
    )

    plot_paired_delta_rmse(
        pocket_repeated,
        out_dir,
    )

    plot_feature_dimensions(
        features_json,
        out_dir,
    )

    print()
    print("=" * 70)
    print("FINAL VISUALIZATION COMPLETE")
    print("=" * 70)

    print(
        f"Figures saved to: {out_dir}"
    )


if __name__ == "__main__":
    main()