from pathlib import Path
import numpy as np
import pandas as pd

features = Path("outputs/features.json")
df = pd.read_json(features)

y = df["label"].to_numpy(dtype=float)
L = np.array(df["ligand"].tolist(), dtype=float)
P = np.array(df["pocket"].tolist(), dtype=float)

print("=" * 60)
print("CURRENT DATA DIAGNOSTICS")
print("=" * 60)
print(f"N samples       : {len(df)}")
print(f"y mean/std      : {y.mean():.4f} / {y.std():.4f}")
print(f"y min/max       : {y.min():.4f} / {y.max():.4f}")
print(f"ligand dim      : {L.shape[1]}")
print(f"pocket dim      : {P.shape[1]}")
print(f"ligand zero var : {(L.std(axis=0) < 1e-8).sum()} / {L.shape[1]}")
print(f"pocket zero var : {(P.std(axis=0) < 1e-8).sum()} / {P.shape[1]}")
print(f"ligand NaN      : {np.isnan(L).sum()}")
print(f"pocket NaN      : {np.isnan(P).sum()}")
print()

if Path("outputs/cv_predictions.csv").exists():
    pred = pd.read_csv("outputs/cv_predictions.csv")
    print("PREDICTION DIAGNOSTICS")
    print("-" * 60)
    for model, g in pred.groupby("model"):
        err = g["y_pred"] - g["y_true"]
        print(
            f"{model:24s} "
            f"pred_mean={g.y_pred.mean():.4f}  "
            f"true_mean={g.y_true.mean():.4f}  "
            f"pred_min={g.y_pred.min():.4f}  "
            f"pred_max={g.y_pred.max():.4f}  "
            f"bias={err.mean():+.4f}"
        )

print()
print("DATA-DIRECTORY CHECK")
print("-" * 60)
data_dir = Path("data/v2013-core")
if data_dir.exists():
    pocket_n = len(list(data_dir.glob("*/*_pocket.pdb")))
    protein_n = len(list(data_dir.glob("*/*_protein.pdb")))
    print(f"*_pocket.pdb files  : {pocket_n}")
    print(f"*_protein.pdb files : {protein_n}")
    print("If pocket files are substantially fewer than the extracted samples,")
    print("some samples may have used the full protein as a fallback.")
else:
    print(f"Missing: {data_dir}")
