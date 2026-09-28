from pathlib import Path
from rdkit import Chem

FAILED = [
    "1o3f",
    "1o5b",
    "1oyt",
    "1sln",
    "1sqa",
    "2d1o",
    "3ag9",
    "3ge7",
    "3gy4",
    "3utu",
    "4djv",
]

DATA_DIR = Path("data/v2013-core")


for pdb_id in FAILED:
    sdf = DATA_DIR / pdb_id / f"{pdb_id}_ligand.sdf"

    print("=" * 70)
    print(f"{pdb_id}: {sdf}")

    if not sdf.exists():
        print("SDF MISSING")
        continue

    # 1. Strict parsing
    try:
        suppl = Chem.SDMolSupplier(
            str(sdf),
            sanitize=True,
            removeHs=False,
        )
        mols = [m for m in suppl if m is not None]
        print("strict parse:", len(mols), "valid molecules")
    except Exception as e:
        print("strict parse exception:", repr(e))

    # 2. Relaxed parsing
    try:
        suppl = Chem.SDMolSupplier(
            str(sdf),
            sanitize=False,
            removeHs=False,
        )
        mols = [m for m in suppl if m is not None]

        print("relaxed parse:", len(mols), "molecules")

        if not mols:
            continue

        mol = mols[0]

        print("atoms:", mol.GetNumAtoms())
        print("bonds:", mol.GetNumBonds())
        print("conformers:", mol.GetNumConformers())

        # 3. Try full sanitization afterwards
        try:
            Chem.SanitizeMol(mol)
            print("manual sanitize: SUCCESS")
        except Exception as e:
            print("manual sanitize: FAILED")
            print("reason:", repr(e))

    except Exception as e:
        print("relaxed parse exception:", repr(e))