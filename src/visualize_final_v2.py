from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10,
    "figure.dpi": 160,
    "savefig.dpi": 320,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.18,
})


def finish(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path.name}")


def pretty(name):
    return {"ligand_only": "Ligand only", "ligand_plus_pocket": "Ligand + coarse pocket"}.get(name, name)


def plot_dataset_qc(index_csv, features_json, out_dir):
    total = len(pd.read_csv(index_csv))
    used = len(pd.read_json(features_json))
    values = [total, used, total-used]
    labels = ["Original Core set", "Strictly sanitized", "Excluded"]

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    bars = ax.bar(labels, values, width=0.62)
    ax.set_title("Dataset curation")
    ax.set_ylabel("Number of complexes")
    ax.set_ylim(0, total * 1.15)
    ax.grid(axis="x", visible=False)

    for bar, value in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+4, str(value), ha="center", fontweight="bold")

    ax.text(0.98, 0.93, f"Usable fraction: {used/total*100:.1f}%", transform=ax.transAxes, ha="right")
    finish(fig, out_dir / "01_dataset_curation.png")


def plot_target_distribution(features_json, out_dir):
    y = pd.read_json(features_json)["label"].to_numpy()
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.hist(y, bins=15, edgecolor="white", linewidth=0.8)
    ax.axvline(y.mean(), linestyle="--", linewidth=1.6)
    ax.set_title("Binding-affinity distribution")
    ax.set_xlabel("Experimental affinity (-logKd/Ki)")
    ax.set_ylabel("Number of complexes")
    ax.text(0.98, 0.94, f"N = {len(y)}\nMean = {y.mean():.2f}\nSD = {y.std():.2f}", transform=ax.transAxes, ha="right", va="top")
    finish(fig, out_dir / "02_target_distribution.png")


def plot_repeated_rmse(summary_csv, out_dir):
    df = pd.read_csv(summary_csv)
    models = ["mean", "ridge", "mlp"]
    x = np.arange(len(models))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8.4, 4.9))
    for offset, feature in [(-width/2, "ligand_only"), (width/2, "ligand_plus_pocket")]:
        rows = df[df.feature_set == feature].set_index("model").reindex(models)
        bars = ax.bar(x+offset, rows.rmse_mean, yerr=rows.rmse_std, capsize=5, width=width, label=pretty(feature))
        for bar, value in zip(bars, rows.rmse_mean):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.03, f"{value:.3f}", ha="center", fontsize=9.5)

    ax.set_xticks(x)
    ax.set_xticklabels(["Mean", "Ridge", "MLP"])
    ax.set_ylabel("RMSE across repeated CV")
    ax.set_title("Model performance across repeated 5-fold cross-validation")
    ax.legend(frameon=False)
    ax.grid(axis="x", visible=False)
    finish(fig, out_dir / "03_repeated_cv_rmse.png")


def plot_paired_delta_rmse(paired_csv, out_dir):
    df = pd.read_csv(paired_csv)
    models = ["ridge", "mlp"]
    values = [df.loc[df.model == m, "delta_rmse"].to_numpy() for m in models]

    fig, ax = plt.subplots(figsize=(7.2, 4.9))
    ax.boxplot(values, tick_labels=["Ridge", "MLP"], widths=0.5, showfliers=False, medianprops={"linewidth": 1.8})

    rng = np.random.default_rng(42)
    for i, vals in enumerate(values, 1):
        ax.scatter(np.full(len(vals), i) + rng.normal(0, 0.035, len(vals)), vals, s=34, alpha=0.72)

    ax.axhline(0, linestyle="--", linewidth=1.3)
    ax.set_title("Paired effect of coarse pocket features")
    ax.set_ylabel("ΔRMSE = pocket − ligand only")
    ax.grid(axis="x", visible=False)
    ax.text(0.02, 0.96, "ΔRMSE < 0  →  pocket improves\nΔRMSE > 0  →  pocket worsens", transform=ax.transAxes, va="top", fontsize=9.5)
    finish(fig, out_dir / "04_paired_delta_rmse.png")


def plot_pocket_counts(summary_csv, out_dir):
    df = pd.read_csv(summary_csv)
    df = df[df.model.isin(["ridge", "mlp"])].copy()
    df["model"] = pd.Categorical(df.model, ["ridge", "mlp"], ordered=True)
    df = df.sort_values("model")

    x = np.arange(len(df))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    b1 = ax.bar(x-width/2, df.pocket_rmse_improved_count, width, label="Improved")
    b2 = ax.bar(x+width/2, df.pocket_rmse_worsened_count, width, label="Worsened")

    for bars in [b1, b2]:
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.35, f"{int(bar.get_height())}", ha="center", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(["Ridge", "MLP"])
    ax.set_ylabel("Number of paired folds")
    ax.set_title("How often do coarse pocket features help?")
    ax.legend(frameon=False)
    ax.grid(axis="x", visible=False)
    finish(fig, out_dir / "05_pocket_effect_counts.png")


def plot_seed_stability(seed_csv, out_dir):
    df = pd.read_csv(seed_csv)
    df = df[df.model == "mlp"]

    fig, ax = plt.subplots(figsize=(8.0, 4.7))
    for feature in ["ligand_only", "ligand_plus_pocket"]:
        temp = df[df.feature_set == feature].sort_values("cv_seed")
        ax.plot(temp.cv_seed.astype(str), temp.rmse, marker="o", linewidth=1.8, label=pretty(feature))

    ax.set_title("MLP stability across CV seeds")
    ax.set_xlabel("Outer-CV random seed")
    ax.set_ylabel("OOF RMSE")
    ax.legend(frameon=False)
    finish(fig, out_dir / "06_seed_stability_mlp.png")


def plot_repeated_parity(pred_csv, out_dir):
    df = pd.read_csv(pred_csv)
    df = df[df.model == "mlp"]

    for feature in ["ligand_only", "ligand_plus_pocket"]:
        avg = df[df.feature_set == feature].groupby("pdb_id", as_index=False).agg(y_true=("y_true", "first"), y_pred=("y_pred", "mean"))
        y, p = avg.y_true.to_numpy(), avg.y_pred.to_numpy()
        rmse = np.sqrt(np.mean((y-p)**2))
        r = np.corrcoef(y, p)[0, 1]
        lo, hi = min(y.min(), p.min()), max(y.max(), p.max())

        fig, ax = plt.subplots(figsize=(5.4, 5.1))
        ax.scatter(y, p, s=38, alpha=0.72)
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2)
        ax.set_xlim(lo-0.3, hi+0.3)
        ax.set_ylim(lo-0.3, hi+0.3)
        ax.set_xlabel("Experimental affinity")
        ax.set_ylabel("Mean OOF prediction across CV seeds")
        ax.set_title(pretty(feature))
        ax.text(0.04, 0.96, f"RMSE = {rmse:.3f}\nPearson r = {r:.3f}\nN = {len(avg)}", transform=ax.transAxes, va="top")

        name = "07a_parity_ligand.png" if feature == "ligand_only" else "07b_parity_ligand_pocket.png"
        finish(fig, out_dir / name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/v2013-core")
    ap.add_argument("--features", default="outputs/strict182/features.json")
    ap.add_argument("--repeated_dir", default="outputs/repeated_cv")
    ap.add_argument("--out_dir", default="figs/final_v2")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    features = Path(args.features)
    repeated = Path(args.repeated_dir)
    out = Path(args.out_dir)

    files = {
        "index": data_dir / "pdbbind_v2013_core.csv",
        "summary": repeated / "summary_repeated.csv",
        "seed": repeated / "seed_summary_repeated.csv",
        "paired": repeated / "pocket_effect_repeated.csv",
        "effect": repeated / "pocket_effect_summary.csv",
        "pred": repeated / "cv_predictions_repeated.csv",
    }

    missing = [str(p) for p in [features, *files.values()] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing files:\n" + "\n".join(missing))

    plot_dataset_qc(files["index"], features, out)
    plot_target_distribution(features, out)
    plot_repeated_rmse(files["summary"], out)
    plot_paired_delta_rmse(files["paired"], out)
    plot_pocket_counts(files["effect"], out)
    plot_seed_stability(files["seed"], out)
    plot_repeated_parity(files["pred"], out)
    print(f"\nAll figures saved to: {out}")


if __name__ == "__main__":
    main()
