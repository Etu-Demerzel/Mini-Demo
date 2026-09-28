from pathlib import Path
from rdkit import Chem


PDB_ID = "4djv"
DATA_DIR = Path("data/v2013-core")

sdf_path = DATA_DIR / PDB_ID / f"{PDB_ID}_ligand.sdf"


print("=" * 70)
print("SDF:", sdf_path)
print("=" * 70)


# --------------------------------------------------
# 1. Relaxed parsing
# --------------------------------------------------

suppl = Chem.SDMolSupplier(
    str(sdf_path),
    sanitize=False,
    removeHs=False,
)

mols = [m for m in suppl if m is not None]

if not mols:
    raise RuntimeError("Relaxed parsing also failed.")

mol = mols[0]

print("Atoms :", mol.GetNumAtoms())
print("Bonds :", mol.GetNumBonds())
print("Conformers :", mol.GetNumConformers())


# --------------------------------------------------
# 2. Print every atom
# --------------------------------------------------

print("\n" + "=" * 70)
print("ATOM TABLE")
print("=" * 70)

for atom in mol.GetAtoms():

    idx = atom.GetIdx()

    print(
        f"idx={idx:2d} "
        f"element={atom.GetSymbol():2s} "
        f"charge={atom.GetFormalCharge():2d} "
        f"degree={atom.GetDegree():2d} "
        f"explicit_valence={atom.GetExplicitValence():2d} "
        f"implicit_H={atom.GetNumImplicitHs():2d} "
        f"aromatic={atom.GetIsAromatic()}"
    )


# --------------------------------------------------
# 3. Detailed inspection of atom #6
# --------------------------------------------------

idx = 6
atom = mol.GetAtomWithIdx(idx)

print("\n" + "=" * 70)
print("PROBLEM ATOM")
print("=" * 70)

print("index:", atom.GetIdx())
print("element:", atom.GetSymbol())
print("formal charge:", atom.GetFormalCharge())
print("degree:", atom.GetDegree())
print("explicit valence:", atom.GetExplicitValence())
print("implicit H:", atom.GetNumImplicitHs())
print("aromatic:", atom.GetIsAromatic())


print("\nNeighbors:")

for nbr in atom.GetNeighbors():

    bond = mol.GetBondBetweenAtoms(
        atom.GetIdx(),
        nbr.GetIdx()
    )

    print(
        f"neighbor={nbr.GetIdx():2d} "
        f"element={nbr.GetSymbol():2s} "
        f"bond_type={bond.GetBondType()} "
        f"bond_aromatic={bond.GetIsAromatic()}"
    )


# --------------------------------------------------
# 4. Raw SDF lines around atom block
# --------------------------------------------------

print("\n" + "=" * 70)
print("RAW SDF")
print("=" * 70)

lines = sdf_path.read_text(errors="replace").splitlines()

# Usually:
# line 4 = counts line
# atom block starts after it
# We print the first 80 lines so we can inspect manually.

for i, line in enumerate(lines[:80], start=1):
    print(f"{i:03d}: {line}")