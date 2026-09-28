from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors, AllChem

AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
AAS = list("ARNDCQEGHILKMFPSTWYV")
DIST_BINS = np.array([0.0, 4.0, 6.0, 8.0, 10.0, 12.0, np.inf])


def load_one_mol(path: Path):
    suppl = Chem.SDMolSupplier(str(path), sanitize=True, removeHs=False)
    mol = next((m for m in suppl if m is not None), None)
    if mol is None or mol.GetNumConformers() == 0:
        raise ValueError(f"Could not read 3D ligand: {path}")
    return mol


def ligand_features(mol):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=512)
    fp_arr = np.asarray(fp, dtype=np.float32)
    desc = np.array([
        Descriptors.MolWt(mol),
        Crippen.MolLogP(mol),
        rdMolDescriptors.CalcTPSA(mol),
        Lipinski.NumHDonors(mol),
        Lipinski.NumHAcceptors(mol),
        Lipinski.RingCount(mol),
    ], dtype=np.float32)
    conf = mol.GetConformer()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())], dtype=np.float32)
    heavy = np.array([mol.GetAtomWithIdx(i).GetAtomicNum() != 1 for i in range(mol.GetNumAtoms())])
    return np.concatenate([fp_arr, desc]), coords[heavy]


def pocket_features(pdb_path: Path, ligand_heavy_coords: np.ndarray):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("p", str(pdb_path))
    aa_counts = np.zeros(len(AAS), dtype=np.float32)
    dist_hist = np.zeros((len(AAS), len(DIST_BINS) - 1), dtype=np.float32)
    nearest = []
    for model in structure:
        for chain in model:
            for residue in chain:
                resname = residue.get_resname().strip()
                if resname not in AA3 or "CA" not in residue:
                    continue
                aa = AA3[resname]
                ai = AAS.index(aa)
                ca = np.asarray(residue["CA"].coord, dtype=np.float32)
                d = np.linalg.norm(ligand_heavy_coords - ca[None, :], axis=1)
                dmin = float(np.min(d))
                nearest.append(dmin)
                aa_counts[ai] += 1.0
                bi = np.digitize(dmin, DIST_BINS[1:-1], right=False)
                if bi < dist_hist.shape[1]:
                    dist_hist[ai, bi] += 1.0
        break
    n = max(float(aa_counts.sum()), 1.0)
    aa_counts /= n
    dist_hist /= n
    nearest_arr = np.asarray(nearest, dtype=np.float32)
    global_feats = np.array([
        n,
        float(np.mean(nearest_arr)) if nearest_arr.size else 99.0,
        float(np.min(nearest_arr)) if nearest_arr.size else 99.0,
    ], dtype=np.float32)
    return np.concatenate([aa_counts, dist_hist.reshape(-1), global_feats])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    data_dir = Path(args.data_dir)
    index = pd.read_csv(data_dir / "pdbbind_v2013_core.csv")
    rows = []
    for _, r in index.iterrows():
        pdb = str(r["pdb_id"])
        sdf = data_dir / pdb / f"{pdb}_ligand.sdf"
        pocket = data_dir / pdb / f"{pdb}_pocket.pdb"
        if not sdf.exists():
            print(f"[skip] ligand missing {pdb}")
            continue
        if not pocket.exists():
            protein = data_dir / pdb / f"{pdb}_protein.pdb"
            if protein.exists():
                pocket = protein
            else:
                print(f"[skip] protein/pocket missing {pdb}")
                continue
        try:
            mol = load_one_mol(sdf)
            lf, heavy_coords = ligand_features(mol)
            pf = pocket_features(pocket, heavy_coords)
            rows.append({"pdb_id": pdb, "label": float(r["label"]), "ligand": lf.tolist(), "pocket": pf.tolist()})
        except Exception as e:
            print(f"[skip] {pdb}: {e}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_json(out, orient="records")
    print(f"saved {len(rows)} complexes -> {out}")


if __name__ == "__main__":
    main()
