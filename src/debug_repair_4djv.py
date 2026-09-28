from pathlib import Path
from rdkit import Chem


pdb_id = "4djv"

sdf = (
    Path("data/v2013-core")
    / pdb_id
    / f"{pdb_id}_ligand.sdf"
)


print("=" * 70)
print("FILE")
print("=" * 70)

print(sdf)


# ============================================================
# 1. relaxed parsing
# ============================================================

supplier = Chem.SDMolSupplier(
    str(sdf),
    sanitize=False,
    removeHs=False,
)

mol = next(
    (
        m
        for m in supplier
        if m is not None
    ),
    None,
)

if mol is None:
    raise RuntimeError(
        "sanitize=False still failed"
    )


print()
print("=" * 70)
print("BEFORE REPAIR")
print("=" * 70)


for atom in mol.GetAtoms():

    if atom.GetSymbol() != "N":
        continue

    idx = atom.GetIdx()

    print()
    print("N atom:", idx)
    print(
        "formal charge:",
        atom.GetFormalCharge()
    )

    print(
        "degree:",
        atom.GetDegree()
    )

    print("neighbors:")

    for nbr in atom.GetNeighbors():

        bond = mol.GetBondBetweenAtoms(
            idx,
            nbr.GetIdx(),
        )

        print(
            " ",
            nbr.GetIdx(),
            nbr.GetSymbol(),
            bond.GetBondType(),
        )


# ============================================================
# 2. reproduce our pattern test explicitly
# ============================================================

print()
print("=" * 70)
print("PATTERN CHECK")
print("=" * 70)


candidate_atoms = []


for atom in mol.GetAtoms():

    if atom.GetSymbol() != "N":
        continue

    print(
        "\nchecking N:",
        atom.GetIdx()
    )

    print(
        "charge == 0:",
        atom.GetFormalCharge() == 0
    )

    print(
        "degree == 3:",
        atom.GetDegree() == 3
    )

    n_h = 0
    n_double_heavy = 0
    other = 0

    for nbr in atom.GetNeighbors():

        bond = mol.GetBondBetweenAtoms(
            atom.GetIdx(),
            nbr.GetIdx(),
        )

        if (
            nbr.GetSymbol() == "H"
            and bond.GetBondType()
            == Chem.BondType.SINGLE
        ):
            n_h += 1

        elif (
            nbr.GetSymbol() != "H"
            and bond.GetBondType()
            == Chem.BondType.DOUBLE
        ):
            n_double_heavy += 1

        else:
            other += 1

    print(
        "single H neighbors:",
        n_h
    )

    print(
        "double heavy neighbors:",
        n_double_heavy
    )

    print(
        "other bonds:",
        other
    )

    match = (
        atom.GetFormalCharge() == 0
        and atom.GetDegree() == 3
        and n_h == 2
        and n_double_heavy == 1
        and other == 0
    )

    print(
        "MATCH:",
        match
    )

    if match:
        candidate_atoms.append(
            atom.GetIdx()
        )


print()
print(
    "candidate atoms:",
    candidate_atoms
)


# ============================================================
# 3. apply proposed repair
# ============================================================

print()
print("=" * 70)
print("APPLY REPAIR")
print("=" * 70)


for idx in candidate_atoms:

    atom = mol.GetAtomWithIdx(
        idx
    )

    print(
        f"N {idx}: charge "
        f"{atom.GetFormalCharge()} -> +1"
    )

    atom.SetFormalCharge(+1)


# ============================================================
# 4. sanitize again
# ============================================================

print()
print("=" * 70)
print("SANITIZE AFTER REPAIR")
print("=" * 70)


try:

    Chem.SanitizeMol(mol)

    print(
        "SANITIZE SUCCESS"
    )

except Exception as e:

    print(
        "SANITIZE FAILED"
    )

    print(
        type(e).__name__,
        repr(e)
    )

    raise


# ============================================================
# 5. test the actual features used by our pipeline
# ============================================================

print()
print("=" * 70)
print("FEATURE TEST")
print("=" * 70)


from rdkit import DataStructs

from rdkit.Chem import (
    AllChem,
    Crippen,
    Descriptors,
    Lipinski,
    rdMolDescriptors,
)


fp = AllChem.GetMorganFingerprintAsBitVect(
    mol,
    radius=2,
    nBits=512,
)

print(
    "Morgan fingerprint:",
    fp.GetNumBits(),
    "bits"
)


print(
    "MolWt:",
    Descriptors.MolWt(mol)
)

print(
    "LogP:",
    Crippen.MolLogP(mol)
)

print(
    "TPSA:",
    rdMolDescriptors.CalcTPSA(mol)
)

print(
    "HBD:",
    Lipinski.NumHDonors(mol)
)

print(
    "HBA:",
    Lipinski.NumHAcceptors(mol)
)

print(
    "RingCount:",
    Lipinski.RingCount(mol)
)


print()
print(
    "ALL TESTS PASSED"
)