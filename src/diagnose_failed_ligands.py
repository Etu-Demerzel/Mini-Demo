from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from rdkit import Chem


# ============================================================
# Utilities
# ============================================================

def bond_order_value(bond: Chem.Bond) -> float:
    """
    Convert RDKit bond type to an approximate bond-order value.

    Used only for diagnostics.
    """

    bt = bond.GetBondType()

    if bt == Chem.BondType.SINGLE:
        return 1.0

    if bt == Chem.BondType.DOUBLE:
        return 2.0

    if bt == Chem.BondType.TRIPLE:
        return 3.0

    if bt == Chem.BondType.AROMATIC:
        return 1.5

    return 0.0


def describe_atom(
    atom: Chem.Atom,
    mol: Chem.Mol,
):
    """
    Return detailed information about one atom and its local environment.
    """

    idx = atom.GetIdx()

    neighbors = []

    bond_order_sum = 0.0

    for nbr in atom.GetNeighbors():

        bond = mol.GetBondBetweenAtoms(
            idx,
            nbr.GetIdx(),
        )

        bond_value = bond_order_value(
            bond
        )

        bond_order_sum += bond_value

        neighbors.append(
            {
                "neighbor_idx": nbr.GetIdx(),
                "neighbor_element": nbr.GetSymbol(),
                "neighbor_charge": nbr.GetFormalCharge(),
                "bond_type": str(
                    bond.GetBondType()
                ),
                "bond_order": bond_value,
                "bond_aromatic": bond.GetIsAromatic(),
            }
        )

    # Some valence-related properties may fail on
    # incompletely sanitized molecules.
    try:
        explicit_valence = atom.GetExplicitValence()
    except Exception:
        explicit_valence = None

    try:
        implicit_h = atom.GetNumImplicitHs()
    except Exception:
        implicit_h = None

    return {
        "atom_idx": idx,
        "element": atom.GetSymbol(),
        "atomic_num": atom.GetAtomicNum(),
        "formal_charge": atom.GetFormalCharge(),
        "degree": atom.GetDegree(),
        "explicit_valence": explicit_valence,
        "implicit_h": implicit_h,
        "aromatic": atom.GetIsAromatic(),
        "bond_order_sum": bond_order_sum,
        "neighbors": neighbors,
    }


# ============================================================
# RDKit chemistry problem detection
# ============================================================

def detect_rdkit_problems(
    mol: Chem.Mol,
):
    """
    Ask RDKit to identify chemistry problems without requiring
    successful full sanitization.

    Returns a list of dictionaries.
    """

    problems = []

    try:

        detected = Chem.DetectChemistryProblems(
            mol
        )

        for problem in detected:

            item = {
                "problem_type": (
                    problem.GetType()
                    if hasattr(problem, "GetType")
                    else type(problem).__name__
                ),

                "message": (
                    problem.Message()
                    if hasattr(problem, "Message")
                    else str(problem)
                ),

                "atom_idx": None,
            }

            # AtomValenceException etc. often expose atom index.
            if hasattr(
                problem,
                "GetAtomIdx",
            ):
                try:
                    item[
                        "atom_idx"
                    ] = problem.GetAtomIdx()
                except Exception:
                    pass

            problems.append(
                item
            )

    except Exception as e:

        problems.append(
            {
                "problem_type":
                    "DetectChemistryProblems_failed",

                "message":
                    f"{type(e).__name__}: {e}",

                "atom_idx":
                    None,
            }
        )

    return problems


# ============================================================
# Sanitization-stage diagnosis
# ============================================================

def diagnose_sanitization_stage(
    mol: Chem.Mol,
):
    """
    Try sanitization with catchErrors=True.

    RDKit returns a sanitize flag indicating which stage failed.
    """

    test_mol = Chem.Mol(
        mol
    )

    try:

        flag = Chem.SanitizeMol(
            test_mol,
            catchErrors=True,
        )

        return str(flag)

    except Exception as e:

        return (
            f"EXCEPTION: "
            f"{type(e).__name__}: {e}"
        )


# ============================================================
# Read raw MOL/SDF atom and bond block
# ============================================================

def read_raw_sdf_excerpt(
    sdf_path: Path,
    max_lines: int = 140,
):

    lines = sdf_path.read_text(
        errors="replace"
    ).splitlines()

    excerpt = []

    for i, line in enumerate(
        lines[:max_lines],
        start=1,
    ):

        excerpt.append(
            f"{i:04d}: {line}"
        )

    return "\n".join(
        excerpt
    )


# ============================================================
# Diagnose one ligand
# ============================================================

def diagnose_one(
    pdb_id: str,
    label: float,
    sdf_path: Path,
    report_dir: Path,
):

    report = []

    report.append(
        "=" * 80
    )

    report.append(
        f"PDB ID: {pdb_id}"
    )

    report.append(
        f"Label: {label}"
    )

    report.append(
        f"SDF: {sdf_path}"
    )

    report.append(
        "=" * 80
    )

    # --------------------------------------------------------
    # Strict parse
    # --------------------------------------------------------

    strict_ok = False

    try:

        supplier = Chem.SDMolSupplier(
            str(sdf_path),
            sanitize=True,
            removeHs=False,
        )

        strict_mols = [
            m
            for m in supplier
            if m is not None
        ]

        strict_ok = (
            len(strict_mols) > 0
        )

    except Exception as e:

        strict_mols = []

        report.append(
            f"\nStrict parse exception:\n"
            f"{type(e).__name__}: {e}"
        )

    report.append(
        f"\nStrict parse valid molecules: "
        f"{len(strict_mols)}"
    )

    # --------------------------------------------------------
    # Relaxed parse
    # --------------------------------------------------------

    try:

        supplier = Chem.SDMolSupplier(
            str(sdf_path),
            sanitize=False,
            removeHs=False,
        )

        relaxed_mols = [
            m
            for m in supplier
            if m is not None
        ]

    except Exception as e:

        relaxed_mols = []

        report.append(
            f"\nRelaxed parse exception:\n"
            f"{type(e).__name__}: {e}"
        )

    report.append(
        f"Relaxed parse molecules: "
        f"{len(relaxed_mols)}"
    )

    if not relaxed_mols:

        return {
            "pdb_id": pdb_id,
            "label": label,
            "strict_ok": strict_ok,
            "relaxed_ok": False,
            "n_atoms": None,
            "n_bonds": None,
            "sanitize_flag": None,
            "n_problems": None,
            "problem_atoms": "",
            "problem_types": "",
        }

    mol = relaxed_mols[0]

    report.append(
        f"Atoms: {mol.GetNumAtoms()}"
    )

    report.append(
        f"Bonds: {mol.GetNumBonds()}"
    )

    report.append(
        f"Conformers: "
        f"{mol.GetNumConformers()}"
    )

    # --------------------------------------------------------
    # Sanitization failure stage
    # --------------------------------------------------------

    sanitize_flag = (
        diagnose_sanitization_stage(
            mol
        )
    )

    report.append(
        "\n"
        + "=" * 80
    )

    report.append(
        "SANITIZATION RESULT"
    )

    report.append(
        "=" * 80
    )

    report.append(
        sanitize_flag
    )

    # --------------------------------------------------------
    # RDKit problem detection
    # --------------------------------------------------------

    problems = detect_rdkit_problems(
        mol
    )

    report.append(
        "\n"
        + "=" * 80
    )

    report.append(
        "RDKIT DETECTED PROBLEMS"
    )

    report.append(
        "=" * 80
    )

    if not problems:

        report.append(
            "No explicit problems detected."
        )

    problem_atoms = []

    problem_types = []

    for p in problems:

        report.append(
            f"\nType: "
            f"{p['problem_type']}"
        )

        report.append(
            f"Atom index: "
            f"{p['atom_idx']}"
        )

        report.append(
            f"Message: "
            f"{p['message']}"
        )

        if p[
            "atom_idx"
        ] is not None:

            problem_atoms.append(
                p["atom_idx"]
            )

        problem_types.append(
            p["problem_type"]
        )

    # --------------------------------------------------------
    # Detailed problematic atom environments
    # --------------------------------------------------------

    report.append(
        "\n"
        + "=" * 80
    )

    report.append(
        "PROBLEM ATOM ENVIRONMENTS"
    )

    report.append(
        "=" * 80
    )

    unique_problem_atoms = sorted(
        set(problem_atoms)
    )

    if not unique_problem_atoms:

        report.append(
            "No atom indices supplied by RDKit."
        )

    for idx in unique_problem_atoms:

        if (
            idx < 0
            or idx >= mol.GetNumAtoms()
        ):
            continue

        atom = mol.GetAtomWithIdx(
            idx
        )

        info = describe_atom(
            atom,
            mol,
        )

        report.append(
            f"\n--- Atom #{idx} ---"
        )

        report.append(
            f"Element: "
            f"{info['element']}"
        )

        report.append(
            f"Formal charge: "
            f"{info['formal_charge']}"
        )

        report.append(
            f"Degree: "
            f"{info['degree']}"
        )

        report.append(
            f"Explicit valence: "
            f"{info['explicit_valence']}"
        )

        report.append(
            f"Implicit H: "
            f"{info['implicit_h']}"
        )

        report.append(
            f"Aromatic: "
            f"{info['aromatic']}"
        )

        report.append(
            f"Bond-order sum: "
            f"{info['bond_order_sum']}"
        )

        report.append(
            "Neighbors:"
        )

        for nbr in info[
            "neighbors"
        ]:

            report.append(
                "  "
                f"atom {nbr['neighbor_idx']} "
                f"{nbr['neighbor_element']} "
                f"charge={nbr['neighbor_charge']} "
                f"bond={nbr['bond_type']} "
                f"order={nbr['bond_order']} "
                f"aromatic={nbr['bond_aromatic']}"
            )

    # --------------------------------------------------------
    # Full atom table
    #
    # Useful if fixing one problem exposes another problem.
    # --------------------------------------------------------

    report.append(
        "\n"
        + "=" * 80
    )

    report.append(
        "FULL ATOM TABLE"
    )

    report.append(
        "=" * 80
    )

    for atom in mol.GetAtoms():

        info = describe_atom(
            atom,
            mol,
        )

        report.append(
            f"idx={info['atom_idx']:3d} "
            f"element={info['element']:2s} "
            f"charge={info['formal_charge']:2d} "
            f"degree={info['degree']:2d} "
            f"valence={str(info['explicit_valence']):>4s} "
            f"bond_sum={info['bond_order_sum']:4.1f} "
            f"aromatic={info['aromatic']}"
        )

    # --------------------------------------------------------
    # Full bond table
    # --------------------------------------------------------

    report.append(
        "\n"
        + "=" * 80
    )

    report.append(
        "FULL BOND TABLE"
    )

    report.append(
        "=" * 80
    )

    for bond in mol.GetBonds():

        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()

        report.append(
            f"{a:3d} -- {b:3d} "
            f"type={str(bond.GetBondType()):10s} "
            f"order={bond_order_value(bond):3.1f} "
            f"aromatic={bond.GetIsAromatic()}"
        )

    # --------------------------------------------------------
    # Raw SDF
    # --------------------------------------------------------

    report.append(
        "\n"
        + "=" * 80
    )

    report.append(
        "RAW SDF EXCERPT"
    )

    report.append(
        "=" * 80
    )

    report.append(
        read_raw_sdf_excerpt(
            sdf_path
        )
    )

    # --------------------------------------------------------
    # Write report
    # --------------------------------------------------------

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        report_dir
        / f"{pdb_id}_diagnostic.txt"
    )

    report_path.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print(
        f"[report] {pdb_id} "
        f"-> {report_path}"
    )

    return {
        "pdb_id": pdb_id,
        "label": label,
        "strict_ok": strict_ok,
        "relaxed_ok": True,
        "n_atoms": mol.GetNumAtoms(),
        "n_bonds": mol.GetNumBonds(),
        "sanitize_flag": sanitize_flag,
        "n_problems": len(problems),

        "problem_atoms": ",".join(
            map(
                str,
                unique_problem_atoms,
            )
        ),

        "problem_types": ";".join(
            problem_types
        ),
    }


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
        "--features",
        default=(
            "outputs/strict182/"
            "features.json"
        ),
    )

    parser.add_argument(
        "--out_dir",
        default=(
            "outputs/diagnostics/"
            "failed_ligands"
        ),
    )

    args = parser.parse_args()

    data_dir = Path(
        args.data_dir
    )

    features_path = Path(
        args.features
    )

    out_dir = Path(
        args.out_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Original PDBBind index
    # --------------------------------------------------------

    index_df = pd.read_csv(
        data_dir
        / "pdbbind_v2013_core.csv"
    )

    # --------------------------------------------------------
    # Successfully built features
    # --------------------------------------------------------

    feature_df = pd.read_json(
        features_path
    )

    included_ids = set(
        feature_df[
            "pdb_id"
        ].astype(str)
    )

    # Automatically find excluded IDs.
    failed_df = index_df[
        ~index_df[
            "pdb_id"
        ]
        .astype(str)
        .isin(included_ids)
    ].copy()

    print(
        "Original index:",
        len(index_df)
    )

    print(
        "Included features:",
        len(feature_df)
    )

    print(
        "Excluded:",
        len(failed_df)
    )

    print()

    summary_rows = []

    for _, row in (
        failed_df.iterrows()
    ):

        pdb_id = str(
            row["pdb_id"]
        )

        label = float(
            row["label"]
        )

        sdf_path = (
            data_dir
            / pdb_id
            / f"{pdb_id}_ligand.sdf"
        )

        if not sdf_path.exists():

            print(
                f"[missing] {pdb_id}: "
                f"{sdf_path}"
            )

            summary_rows.append(
                {
                    "pdb_id": pdb_id,
                    "label": label,
                    "strict_ok": False,
                    "relaxed_ok": False,
                    "n_atoms": None,
                    "n_bonds": None,
                    "sanitize_flag":
                        "SDF_MISSING",
                    "n_problems": None,
                    "problem_atoms": "",
                    "problem_types": "",
                }
            )

            continue

        result = diagnose_one(
            pdb_id,
            label,
            sdf_path,
            out_dir,
        )

        summary_rows.append(
            result
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_path = (
        out_dir
        / "failed_ligands_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(
        summary_df[
            [
                "pdb_id",
                "label",
                "sanitize_flag",
                "problem_atoms",
                "problem_types",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        f"Summary CSV: "
        f"{summary_path}"
    )

    print(
        f"Detailed reports: "
        f"{out_dir}"
    )


if __name__ == "__main__":
    main()