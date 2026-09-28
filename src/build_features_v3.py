from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from Bio.PDB import PDBParser

from rdkit import DataStructs
from rdkit.Chem import (
    AllChem,
    Crippen,
    Descriptors,
    Lipinski,
    rdMolDescriptors,
)

from molecule_io import load_ligand


# ============================================================
# Amino acid definitions
# ============================================================

AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


AA_ORDER = list(
    "ARNDCQEGHILKMFPSTWYV"
)


# ============================================================
# Distance bins
#
# 0-4
# 4-6
# 6-8
# 8-10
# 10-12
# >=12
#
# 6 bins
#
# 20 AA × 6 = 120 features
# ============================================================

DISTANCE_EDGES = np.array(
    [
        0.0,
        4.0,
        6.0,
        8.0,
        10.0,
        12.0,
        np.inf,
    ],
    dtype=np.float32,
)


# ============================================================
# Ligand features
#
# Morgan fingerprint:
# 512
#
# descriptors:
# 6
#
# total:
# 518
# ============================================================

def extract_ligand_features(mol):

    # --------------------------------------------------------
    # Morgan fingerprint
    # --------------------------------------------------------

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=2,
        nBits=512,
    )

    fp_array = np.zeros(
        (512,),
        dtype=np.float32,
    )

    DataStructs.ConvertToNumpyArray(
        fp,
        fp_array,
    )

    # --------------------------------------------------------
    # Molecular descriptors
    # --------------------------------------------------------

    descriptors = np.array(
        [
            Descriptors.MolWt(mol),
            Crippen.MolLogP(mol),
            rdMolDescriptors.CalcTPSA(mol),
            Lipinski.NumHDonors(mol),
            Lipinski.NumHAcceptors(mol),
            Lipinski.RingCount(mol),
        ],
        dtype=np.float32,
    )

    ligand_feature = np.concatenate(
        [
            fp_array,
            descriptors,
        ]
    )

    # --------------------------------------------------------
    # Extract ligand atom coordinates
    #
    # Pocket feature 只需要 heavy atom coordinates
    # --------------------------------------------------------

    conf = mol.GetConformer()

    all_coords = []

    heavy_mask = []

    for i in range(mol.GetNumAtoms()):

        pos = conf.GetAtomPosition(i)

        all_coords.append(
            [
                pos.x,
                pos.y,
                pos.z,
            ]
        )

        atom = mol.GetAtomWithIdx(i)

        heavy_mask.append(
            atom.GetAtomicNum() != 1
        )

    all_coords = np.asarray(
        all_coords,
        dtype=np.float32,
    )

    heavy_mask = np.asarray(
        heavy_mask,
        dtype=bool,
    )

    heavy_coords = all_coords[
        heavy_mask
    ]

    if len(heavy_coords) == 0:

        raise ValueError(
            "Ligand contains no heavy atoms."
        )

    return ligand_feature, heavy_coords


# ============================================================
# Pocket features
#
# AA composition:
# 20
#
# AA × distance histogram:
# 20 × 6 = 120
#
# global:
# 3
#
# total:
# 143
# ============================================================

def extract_pocket_features(
    pdb_path: Path,
    ligand_heavy_coords: np.ndarray,
):

    parser = PDBParser(
        QUIET=True
    )

    structure = parser.get_structure(
        "protein",
        str(pdb_path),
    )

    # --------------------------------------------------------
    # 20 AA composition
    # --------------------------------------------------------

    aa_counts = np.zeros(
        len(AA_ORDER),
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # 20 × 6 distance histogram
    # --------------------------------------------------------

    distance_hist = np.zeros(
        (
            len(AA_ORDER),
            len(DISTANCE_EDGES) - 1,
        ),
        dtype=np.float32,
    )

    nearest_distances = []

    # --------------------------------------------------------
    # Usually PDB contains one model.
    # We use only the first model.
    # --------------------------------------------------------

    for model in structure:

        for chain in model:

            for residue in chain:

                resname = (
                    residue
                    .get_resname()
                    .strip()
                )

                # only standard amino acids
                if resname not in AA3_TO_1:
                    continue

                # require alpha carbon
                if "CA" not in residue:
                    continue

                aa_one = AA3_TO_1[
                    resname
                ]

                aa_index = AA_ORDER.index(
                    aa_one
                )

                # --------------------------------------------
                # C-alpha coordinate
                # --------------------------------------------

                ca_coord = np.asarray(
                    residue["CA"].coord,
                    dtype=np.float32,
                )

                # --------------------------------------------
                # Distance:
                #
                # one residue C-alpha
                # vs
                # every ligand heavy atom
                # --------------------------------------------

                distances = np.linalg.norm(
                    ligand_heavy_coords
                    - ca_coord[None, :],
                    axis=1,
                )

                nearest = float(
                    np.min(distances)
                )

                nearest_distances.append(
                    nearest
                )

                # --------------------------------------------
                # amino acid count
                # --------------------------------------------

                aa_counts[
                    aa_index
                ] += 1.0

                # --------------------------------------------
                # distance bin
                # --------------------------------------------

                bin_index = np.digitize(
                    nearest,
                    DISTANCE_EDGES[1:-1],
                    right=False,
                )

                if (
                    0
                    <= bin_index
                    < distance_hist.shape[1]
                ):

                    distance_hist[
                        aa_index,
                        bin_index,
                    ] += 1.0

        # first model only
        break

    total_residues = float(
        aa_counts.sum()
    )

    if total_residues <= 0:

        raise ValueError(
            f"No standard residues found in "
            f"pocket/protein: {pdb_path}"
        )

    # --------------------------------------------------------
    # Normalize composition/histogram
    # --------------------------------------------------------

    aa_composition = (
        aa_counts
        / total_residues
    )

    distance_hist_norm = (
        distance_hist
        / total_residues
    )

    nearest_distances = np.asarray(
        nearest_distances,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # 3 global features
    # --------------------------------------------------------

    global_features = np.array(
        [
            total_residues,

            float(
                np.mean(
                    nearest_distances
                )
            ),

            float(
                np.min(
                    nearest_distances
                )
            ),
        ],
        dtype=np.float32,
    )

    pocket_feature = np.concatenate(
        [
            aa_composition,
            distance_hist_norm.reshape(-1),
            global_features,
        ]
    )

    # safety check
    if len(pocket_feature) != 143:

        raise RuntimeError(
            "Pocket feature dimension "
            f"is {len(pocket_feature)}, "
            "expected 143."
        )

    return pocket_feature


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        required=True,
        help=(
            "Example: data/v2013-core"
        ),
    )

    parser.add_argument(
        "--out",
        required=True,
        help=(
            "Output feature JSON path."
        ),
    )

    parser.add_argument(
        "--quality_log",
        default=None,
        help=(
            "Optional CSV for data-quality "
            "and repair metadata."
        ),
    )

    args = parser.parse_args()

    data_dir = Path(
        args.data_dir
    )

    output_path = Path(
        args.out
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Read PDBBind index
    # --------------------------------------------------------

    index_path = (
        data_dir
        / "pdbbind_v2013_core.csv"
    )

    df = pd.read_csv(
        index_path
    )

    rows = []

    quality_rows = []

    # ========================================================
    # Process each complex
    # ========================================================

    for _, row in df.iterrows():

        pdb_id = str(
            row["pdb_id"]
        )

        label = float(
            row["label"]
        )

        complex_dir = (
            data_dir
            / pdb_id
        )

        sdf_path = (
            complex_dir
            / f"{pdb_id}_ligand.sdf"
        )

        pocket_path = (
            complex_dir
            / f"{pdb_id}_pocket.pdb"
        )

        protein_path = (
            complex_dir
            / f"{pdb_id}_protein.pdb"
        )

        # ----------------------------------------------------
        # Determine protein structure source
        # ----------------------------------------------------

        if pocket_path.exists():

            structure_path = (
                pocket_path
            )

            structure_source = (
                "pocket"
            )

        elif protein_path.exists():

            structure_path = (
                protein_path
            )

            structure_source = (
                "protein_fallback"
            )

        else:

            print(
                f"[skip] {pdb_id}: "
                "no pocket/protein PDB"
            )

            quality_rows.append(
                {
                    "pdb_id": pdb_id,
                    "label": label,
                    "status": "failed",
                    "ligand_method": "",
                    "repaired_atoms": "",
                    "structure_source": "",
                    "message": (
                        "No pocket or protein PDB."
                    ),
                }
            )

            continue

        # ----------------------------------------------------
        # Ligand processing
        # ----------------------------------------------------

        try:

            mol, mol_info = load_ligand(
                sdf_path
            )

        except Exception as e:

            print(
                f"[skip] {pdb_id}: "
                f"ligand error: {e}"
            )

            quality_rows.append(
                {
                    "pdb_id": pdb_id,
                    "label": label,
                    "status": "failed",
                    "ligand_method": "failed",
                    "repaired_atoms": "",
                    "structure_source": (
                        structure_source
                    ),
                    "message": str(e),
                }
            )

            continue

        # ----------------------------------------------------
        # Feature extraction
        # ----------------------------------------------------

        try:

            ligand_feature, heavy_coords = (
                extract_ligand_features(
                    mol
                )
            )

            pocket_feature = (
                extract_pocket_features(
                    structure_path,
                    heavy_coords,
                )
            )

        except Exception as e:

            print(
                f"[skip] {pdb_id}: "
                f"feature error: {e}"
            )

            quality_rows.append(
                {
                    "pdb_id": pdb_id,
                    "label": label,
                    "status": "failed",
                    "ligand_method": (
                        mol_info.method
                    ),
                    "repaired_atoms": ",".join(
                        map(
                            str,
                            mol_info.repaired_atoms,
                        )
                    ),
                    "structure_source": (
                        structure_source
                    ),
                    "message": (
                        f"Feature extraction "
                        f"failed: {e}"
                    ),
                }
            )

            continue

        # ----------------------------------------------------
        # Save successful feature row
        # ----------------------------------------------------

        rows.append(
            {
                "pdb_id": pdb_id,
                "label": label,

                "ligand": (
                    ligand_feature.tolist()
                ),

                "pocket": (
                    pocket_feature.tolist()
                ),

                # extra provenance fields
                "ligand_method": (
                    mol_info.method
                ),

                "repaired_atoms": (
                    mol_info.repaired_atoms
                ),
            }
        )

        quality_rows.append(
            {
                "pdb_id": pdb_id,
                "label": label,
                "status": "included",
                "ligand_method": (
                    mol_info.method
                ),
                "repaired_atoms": ",".join(
                    map(
                        str,
                        mol_info.repaired_atoms,
                    )
                ),
                "structure_source": (
                    structure_source
                ),
                "message": (
                    mol_info.message
                ),
            }
        )

        if (
            mol_info.method
            == "strict"
        ):

            print(
                f"[ok] {pdb_id}: "
                "strict"
            )

        else:

            print(
                f"[repair] {pdb_id}: "
                f"{mol_info.method}, "
                f"atoms="
                f"{mol_info.repaired_atoms}"
            )

    # ========================================================
    # Save features
    # ========================================================

    feature_df = pd.DataFrame(
        rows
    )

    feature_df.to_json(
        output_path,
        orient="records",
    )

    # ========================================================
    # Save quality log
    # ========================================================

    if args.quality_log:

        quality_path = Path(
            args.quality_log
        )

    else:

        quality_path = (
            output_path.parent
            / "data_quality.csv"
        )

    quality_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    quality_df = pd.DataFrame(
        quality_rows
    )

    quality_df.to_csv(
        quality_path,
        index=False,
    )

    # ========================================================
    # Final summary
    # ========================================================

    included = int(
        (
            quality_df["status"]
            == "included"
        ).sum()
    )

    failed = int(
        (
            quality_df["status"]
            == "failed"
        ).sum()
    )

    strict_n = int(
        (
            quality_df["ligand_method"]
            == "strict"
        ).sum()
    )

    repaired_n = int(
        (
            quality_df["ligand_method"]
            == "repair_N_charge_plus1"
        ).sum()
    )

    print()
    print("=" * 70)
    print("BUILD FEATURES V3 SUMMARY")
    print("=" * 70)

    print(
        "Index rows      :",
        len(df),
    )

    print(
        "Included        :",
        included,
    )

    print(
        "Failed          :",
        failed,
    )

    print(
        "Strict ligands  :",
        strict_n,
    )

    print(
        "Repaired ligands:",
        repaired_n,
    )

    print(
        "Feature rows    :",
        len(feature_df),
    )

    print(
        "Feature output  :",
        output_path,
    )

    print(
        "Quality log     :",
        quality_path,
    )


if __name__ == "__main__":
    main()