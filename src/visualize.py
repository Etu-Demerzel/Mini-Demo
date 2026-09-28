from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    preds = pd.read_csv(args.results)

    # Parity plot
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    for model, g in preds.groupby("model"):
        ax.scatter(g.y_true, g.y_pred, alpha=0.65, s=24, label=model)
    lo = min(preds.y_true.min(), preds.y_pred.min())
    hi = max(preds.y_true.max(), preds.y_pred.max())
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.set_xlabel("Experimental -logKd/Ki")
    ax.set_ylabel("Predicted")
    ax.set_title("5-fold out-of-fold prediction")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "01_parity.png", dpi=180); plt.close(fig)

    # Error comparison
    rows = []
    for model, g in preds.groupby("model"):
        e = g.y_pred - g.y_true
        rows.append({"model": model, "RMSE": float(np.sqrt(np.mean(e*e))), "MAE": float(np.mean(np.abs(e)))})
    m = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    x = np.arange(len(m)); w = 0.34
    ax.bar(x-w/2, m.RMSE, width=w, label="RMSE")
    ax.bar(x+w/2, m.MAE, width=w, label="MAE")
    ax.set_xticks(x); ax.set_xticklabels([s.replace("_", "\n") for s in m.model])
    ax.set_ylabel("Error")
    ax.set_title("Model comparison")
    ax.legend()
    fig.tight_layout(); fig.savefig(out / "02_model_comparison.png", dpi=180); plt.close(fig)

    # Residuals
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for model, g in preds.groupby("model"):
        ax.hist(g.y_pred-g.y_true, bins=18, alpha=0.5, label=model)
    ax.axvline(0, linestyle="--")
    ax.set_xlabel("Prediction error")
    ax.set_ylabel("Count")
    ax.set_title("Residual distribution")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "03_residuals.png", dpi=180); plt.close(fig)

    # Contact-feature heatmap for the first complex.
    recs = pd.read_json(args.features)
    if len(recs):
        p = np.array(recs.iloc[0].pocket, dtype=float)
        hist = p[20:140].reshape(20, 6)
        fig, ax = plt.subplots(figsize=(7.0, 5.2))
        im = ax.imshow(hist, aspect="auto")
        ax.set_yticks(range(20)); ax.set_yticklabels(list("ARNDCQEGHILKMFPSTWYV"))
        ax.set_xticks(range(6)); ax.set_xticklabels(["<4", "4-6", "6-8", "8-10", "10-12", ">=12"])
        ax.set_xlabel("Nearest protein Cα to ligand heavy atom distance (Å)")
        ax.set_ylabel("Residue type")
        ax.set_title(f"Coarse pocket interaction fingerprint: {recs.iloc[0].pdb_id}")
        fig.colorbar(im, ax=ax, label="Normalized residue count")
        fig.tight_layout(); fig.savefig(out / "04_pocket_heatmap.png", dpi=180); plt.close(fig)

    print(f"figures saved to {out}")


if __name__ == "__main__":
    main()
