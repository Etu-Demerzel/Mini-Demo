from rdkit import Chem


def try_repair(mol):

    repaired_atoms = []

    for atom in mol.GetAtoms():

        if atom.GetSymbol() != "N":
            continue

        if atom.GetFormalCharge() != 0:
            continue

        if atom.GetExplicitValence() != 4:
            continue

        h_neighbors = []
        double_heavy_neighbors = []

        for nbr in atom.GetNeighbors():

            bond = mol.GetBondBetweenAtoms(
                atom.GetIdx(),
                nbr.GetIdx()
            )

            if (
                nbr.GetSymbol() == "H"
                and bond.GetBondType() == Chem.BondType.SINGLE
            ):
                h_neighbors.append(nbr)

            if (
                nbr.GetSymbol() != "H"
                and bond.GetBondType() == Chem.BondType.DOUBLE
            ):
                double_heavy_neighbors.append(nbr)

        # Exact pattern:
        # N = heavy_atom + H + H
        if (
            len(h_neighbors) == 2
            and len(double_heavy_neighbors) == 1
        ):
            atom.SetFormalCharge(+1)

            repaired_atoms.append(
                atom.GetIdx()
            )

    try:
        Chem.SanitizeMol(mol)
        return True, repaired_atoms

    except Exception as e:
        return False, repaired_atoms

success, repaired_atoms = try_repair(mol)

print("repair success:", success)
print("repaired atoms:", repaired_atoms)

from rdkit.Chem import AllChem

fp = AllChem.GetMorganFingerprintAsBitVect(
    mol,
    radius=2,
    nBits=512
)

print("fingerprint OK")