from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from rdkit import Chem

AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

def audit_dataset(data_dir="data/v2013-core",
                  features_path="outputs/features.json",
                  pred_path="outputs/cv_predictions_v2.csv"):
    data_dir = Path(data_dir)
    index_path = data_dir / "pdbbind_v2013_core.csv"

    index = pd.read_csv(index_path)
    feats = pd.read_json(features_path)

    index_ids = set(index["pdb_id"].astype(str))
    feat_ids = set(feats["pdb_id"].astype(str))
    missing_ids = sorted(index_ids - feat_ids)

    print("=" * 70)
    print("DATASET AUDIT")
    print("=" * 70)
    print(f"Index samples                  : {len(index)}")
    print(f"Feature samples                : {len(feats)}")
    print(f"Not represented in features    : {len(missing_ids)}")
    print()

    if missing_ids:
        print("Missing/failed feature IDs")
        print("-" * 70)
        for pdb in missing_ids:
            sdf = data_dir / pdb / f"{pdb}_ligand.sdf"
            pocket = data_dir / pdb / f"{pdb}_pocket.pdb"
            protein = data_dir / pdb / f"{pdb}_protein.pdb"

            reason = []
            if not sdf.exists():
                reason.append("ligand_sdf_missing")
            else:
                try:
                    suppl = Chem.SDMolSupplier(
                        str(sdf), sanitize=True, removeHs=False
                    )
                    mols = [m for m in suppl if m is not None]
                    if not mols:
                        reason.append("sdf_parse_failed")
                    elif all(m.GetNumConformers() == 0 for m in mols):
                        reason.append("no_3d_conformer")
                except Exception as e:
                    reason.append(f"sdf_exception:{type(e).__name__}")

            if pocket.exists():
                pocket_status = "pocket_ok"
            elif protein.exists():
                pocket_status = "pocket_missing_but_protein_exists"
            else:
                pocket_status = "protein_and_pocket_missing"
            reason.append(pocket_status)

            label = float(
                index.loc[index["pdb_id"].astype(str) == pdb, "label"].iloc[0]
            )
            print(f"{pdb:8s} label={label:5.2f}  " + "; ".join(reason))
    else:
        print("No missing IDs.")

    print()
    print("FEATURE VARIANCE")
    print("-" * 70)
    L = np.asarray(feats.iloc[0]["ligand"], dtype=float) if len(feats) else np.array([])
    P = np.asarray(feats.iloc[0]["pocket"], dtype=float) if len(feats) else np.array([])
    if len(feats):
        Lall = np.asarray(feats["ligand"].tolist(), dtype=float)
        Pall = np.asarray(feats["pocket"].tolist(), dtype=float)
        print(f"Ligand dimensions with nonzero variance : "
              f"{(Lall.std(axis=0) > 1e-8).sum()} / {Lall.shape[1]}")
        print(f"Pocket dimensions with nonzero variance : "
              f"{(Pall.std(axis=0) > 1e-8).sum()} / {Pall.shape[1]}")

        # The 42 zero-variance pocket dimensions are expected to be uninformative
        # for this exact benchmark and can be removed safely.
        print(f"Pocket zero-variance dimensions         : "
              f"{(Pall.std(axis=0) <= 1e-8).sum()}")

    if Path(pred_path).exists():
        pred = pd.read_csv(pred_path)
        print()
        print("PER-FOLD PAIRED COMPARISON")
        print("-" * 70)

        rows = []
        for feature_set in pred["feature_set"].unique():
            sub = pred[pred["feature_set"] == feature_set]
            models = sorted(sub["model"].unique())
            for fold in sorted(sub["fold"].unique()):
                fold_sub = sub[sub["fold"] == fold]
                for model in models:
                    g = fold_sub[fold_sub["model"] == model]
                    y = g["y_true"].to_numpy()
                    p = g["y_pred"].to_numpy()
                    rows.append({
                        "feature_set": feature_set,
                        "fold": int(fold),
                        "model": model,
                        "rmse": np.sqrt(np.mean((y-p)**2)),
                        "mae": np.mean(np.abs(y-p)),
                        "pearson": pearsonr(y, p).statistic,
                        "spearman": spearmanr(y, p).statistic,
                    })

        fold_df = pd.DataFrame(rows)
        fold_df.to_csv("outputs/fold_metrics_v2.csv", index=False)
        print(fold_df.to_string(index=False))

        print()
        print("MLP effect of adding pocket (negative = improvement in error)")
        print("-" * 70)
        a = fold_df[
            (fold_df.feature_set == "ligand_only") &
            (fold_df.model == "mlp")
        ].set_index("fold")
        b = fold_df[
            (fold_df.feature_set == "ligand_plus_pocket") &
            (fold_df.model == "mlp")
        ].set_index("fold")

        d_rmse = b["rmse"] - a["rmse"]
        d_mae = b["mae"] - a["mae"]
        d_spear = b["spearman"] - a["spearman"]

        comparison = pd.DataFrame({
            "fold": d_rmse.index,
            "delta_rmse_pocket_minus_ligand": d_rmse.values,
            "delta_mae_pocket_minus_ligand": d_mae.values,
            "delta_spearman_pocket_minus_ligand": d_spear.values,
        })
        comparison.to_csv("outputs/pocket_effect_by_fold_v2.csv", index=False)
        print(comparison.to_string(index=False))
        print()
        print("Mean delta RMSE    :", float(d_rmse.mean()))
        print("Mean delta MAE     :", float(d_mae.mean()))
        print("Mean delta Spearman:", float(d_spear.mean()))

if __name__ == "__main__":
    audit_dataset()
