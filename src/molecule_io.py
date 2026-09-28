from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple

from rdkit import Chem


@dataclass
class MoleculeLoadInfo:
    """
    保存每个 ligand 的读取/修复信息。

    method:
        strict
            原始 SDF 直接通过 RDKit sanitize

        repair_N_charge_plus1
            strict 失败后，通过 targeted N charge repair 成功

        failed
            最终仍然失败
    """
    success: bool
    method: str
    repaired_atoms: List[int]
    message: str

    def to_dict(self):
        return asdict(self)


def _is_target_neutral_n_pattern(
    atom: Chem.Atom,
    mol: Chem.Mol,
) -> bool:
    """
    检测我们在 4djv 中已经确认过的特定 pattern：

        neutral N
        formal charge = 0
        degree = 3

        一个 double-bonded heavy atom
        两个 single-bonded H

    即类似：

            H
            |
        C = N
            |
            H

    这种结构如果 N 没有 +1 formal charge，
    RDKit 会报 explicit valence 4 错误。

    注意：
    这里故意只修非常窄的 pattern，
    不会对所有 valence 异常的 N 自动改 charge。
    """

    if atom.GetSymbol() != "N":
        return False

    if atom.GetFormalCharge() != 0:
        return False

    if atom.GetDegree() != 3:
        return False

    n_single_h = 0
    n_double_heavy = 0
    other_bonds = 0

    for nbr in atom.GetNeighbors():

        bond = mol.GetBondBetweenAtoms(
            atom.GetIdx(),
            nbr.GetIdx(),
        )

        if bond is None:
            return False

        bond_type = bond.GetBondType()

        # N-H single bond
        if (
            nbr.GetSymbol() == "H"
            and bond_type == Chem.BondType.SINGLE
        ):
            n_single_h += 1
            continue

        # N=heavy atom
        if (
            nbr.GetSymbol() != "H"
            and bond_type == Chem.BondType.DOUBLE
        ):
            n_double_heavy += 1
            continue

        other_bonds += 1

    return (
        n_single_h == 2
        and n_double_heavy == 1
        and other_bonds == 0
    )


def _repair_known_patterns(
    mol: Chem.Mol,
) -> MoleculeLoadInfo:
    """
    对 sanitize=False 得到的 molecule 尝试我们已经确认的
    targeted chemical repair。

    当前只支持：

        neutral N:
        one double bond to heavy atom
        + two H
        -> set formal charge to +1

    修复之后必须重新 Chem.SanitizeMol()。

    如果 sanitize 仍然失败，就认为修复不可靠，
    不允许进入机器学习 workflow。
    """

    repaired_atoms = []

    for atom in mol.GetAtoms():

        if _is_target_neutral_n_pattern(atom, mol):

            idx = atom.GetIdx()

            atom.SetFormalCharge(+1)

            repaired_atoms.append(idx)

    if not repaired_atoms:

        return MoleculeLoadInfo(
            success=False,
            method="failed",
            repaired_atoms=[],
            message=(
                "Strict parsing failed, but no supported "
                "targeted repair pattern was found."
            ),
        )

    try:

        Chem.SanitizeMol(mol)

    except Exception as e:

        return MoleculeLoadInfo(
            success=False,
            method="failed",
            repaired_atoms=repaired_atoms,
            message=(
                "Targeted repair was attempted, but "
                f"SanitizeMol still failed: "
                f"{type(e).__name__}: {e}"
            ),
        )

    return MoleculeLoadInfo(
        success=True,
        method="repair_N_charge_plus1",
        repaired_atoms=repaired_atoms,
        message=(
            "Strict parsing failed. "
            "Neutral valence-4 N pattern was repaired "
            "by assigning formal charge +1, and "
            "SanitizeMol succeeded."
        ),
    )


def load_ligand(
    sdf_path: str | Path,
) -> Tuple[Chem.Mol, MoleculeLoadInfo]:
    """
    统一的 ligand loading 函数。

    Workflow:

        SDF
         |
         v
    strict parse
    sanitize=True
         |
      success
         |
         v
       return

    如果 strict 失败：

        sanitize=False
             |
             v
      relaxed molecule
             |
             v
      targeted repair
             |
             v
       SanitizeMol
             |
        success only
             |
             v
           return

    原始 SDF 永远不会被写回或修改。
    """

    sdf_path = Path(sdf_path)

    if not sdf_path.exists():

        raise FileNotFoundError(
            f"Ligand SDF does not exist: {sdf_path}"
        )

    # =========================================================
    # Stage 1
    # 正常严格读取
    # =========================================================

    try:

        supplier = Chem.SDMolSupplier(
            str(sdf_path),
            sanitize=True,
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

    except Exception:

        mol = None

    if mol is not None:

        if mol.GetNumConformers() == 0:

            raise ValueError(
                f"Strict parsing succeeded but ligand "
                f"has no conformer: {sdf_path}"
            )

        info = MoleculeLoadInfo(
            success=True,
            method="strict",
            repaired_atoms=[],
            message=(
                "Original SDF passed RDKit strict "
                "parsing and sanitization."
            ),
        )

        return mol, info

    # =========================================================
    # Stage 2
    # relaxed read
    # =========================================================

    try:

        supplier = Chem.SDMolSupplier(
            str(sdf_path),
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

    except Exception as e:

        raise ValueError(
            "Both strict and relaxed SDF parsing failed: "
            f"{sdf_path} | {type(e).__name__}: {e}"
        )

    if mol is None:

        raise ValueError(
            f"Relaxed parsing also produced no molecule: "
            f"{sdf_path}"
        )

    if mol.GetNumConformers() == 0:

        raise ValueError(
            f"Relaxed molecule has no 3D conformer: "
            f"{sdf_path}"
        )

    # =========================================================
    # Stage 3
    # targeted repair
    # =========================================================

    info = _repair_known_patterns(mol)

    if not info.success:

        raise ValueError(
            f"Unable to chemically validate ligand: "
            f"{sdf_path} | {info.message}"
        )

    return mol, info