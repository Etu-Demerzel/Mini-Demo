# Mini-IGN / Hou Lab 48h Project

## Question
Can adding coarse 3D protein-pocket information improve protein–ligand binding-affinity prediction beyond ligand-only molecular fingerprints?

## Data
PDBBind v2013 Core (193 complexes). The dataset contains ligand structures, protein structures/pockets and affinity labels. The label used here is the PDBBind `-logKd/Ki` field.

## Features
- Ligand: RDKit Morgan fingerprint (512 bits) + six simple descriptors.
- Pocket: 20 amino-acid counts + 20x6 residue-type-by-nearest-ligand-distance histogram + three global geometry features.
- Models: two small PyTorch MLPs with the same architecture; one sees ligand features only, the other sees ligand + pocket features.
- Evaluation: 5-fold cross-validation; RMSE, MAE, Pearson r, Spearman rho.

## Important scope note
This is a deliberately small, CPU/MPS-friendly proof-of-concept inspired by the structure-aware idea behind the Hou lab's IGN/RTMScore work. It is not a reproduction of IGN or RTMScore.

## Run
1. Download/extract PDBBind v2013 Core using the command in the main instructions.
2. Ensure the extracted directory is `data/v2013-core`.
3. Run:

```bash
python src/build_features.py --data_dir data/v2013-core --out outputs/features.csv
python src/train.py --features outputs/features.csv --out_dir outputs
python src/visualize.py --features outputs/features.csv --results outputs/cv_predictions.csv --out_dir figs
```

The scripts automatically use MPS when available and otherwise CPU.
