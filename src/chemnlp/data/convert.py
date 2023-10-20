from pymatgen.core import Structure, Molecule
from pymatgen.io.cif import CifWriter
from pymatgen.io.xyz import XYZ
from rdkit import Chem

from givemeconformer.api import _get_conformer

from typing import Union, Optional
from pathlib import Path


def cif_file_to_string(
    path: Union[Path, str],
    primitive: bool = True,
    symprec: float = None,
    significant_figures: int = 3,
) -> str:
    """Write a cif file to a string

    Args:
        path (Union[Path, str]): Path to the cif file
        primitive (bool, optional): Compute primitive cell. Defaults to True.
        symprec (float, optional): If not None, symmetrizes the structure with the given
            symmetry tolerance. In this case, the space group (and other symmetry info)
            will be in the CIF. Defaults to None.
        significant_figures (int, optional): No. of significant figures to write. Defaults to 3.

    Returns:
        str: String representation of the cif file
    """
    s = Structure.from_file(path)
    if primitive:
        s = s.get_primitive_structure()

    return (
        "[CIF]\n"
        + str(
            CifWriter(s, symprec=symprec, significant_figures=significant_figures)
        ).replace("# generated using pymatgen\n", "")
        + "[/CIF]\n"
    )


def xyz_file_to_string(path: Union[Path, str], significant_figures: int = 3) -> str:
    """Write a xyz file to a string

    Args:
        path (Union[Path, str]): Path to the xyz file
        significant_figures (int, optional): No. of significant figures to write. Defaults to 3.

    Returns:
        str: String representation of the xyz file
    """
    s = Molecule.from_file(path)
    return "[XYZ]\n" + str(XYZ(s, coord_precision=significant_figures)) + "[\XYZ]"


def smiles_to_3Dstring(
    smiles, outformat="xyz", conformer_kwargs: Optional[dict] = None
) -> str:
    """Convert a smiles string to a 3D string representation

    Args:
        smiles (str): Smiles string
        outformat (str, optional): Output format. Defaults to 'xyz'.
        conformer_kwargs (Optional[dict], optional): kwargs for conformer generation.
        All kwargs of givemeconformer.api._get_conformer are supported.
            Defaults to None.

    Returns:
        str: String representation of the molecule
    """
    if conformer_kwargs is None:
        conformer_kwargs = {}
    mol, _conformer = _get_conformer(smiles, **conformer_kwargs)
    if outformat == "xyz":
        return "[XYZ]\n" + Chem.MolToXYZBlock(mol, confId=-1) + "[\XYZ]"
    elif outformat == "V2000MolBlock":
        return _write_mol2000(mol)
    elif outformat == "V3000MolBlock":
        return _write_mol3000(mol)
    else:
        raise ValueError(f"outformat {outformat} not supported")


def _write_mol2000(mol):
    return "[V2000]\n" + Chem.MolToMolBlock(mol, confId=-1) + "[\V2000]"


def _write_mol3000(mol):
    return "[V3000]\n" + Chem.MolToV3KMolBlock(mol, confId=-1) + "[\V3000]"


def get_token_count(string):
    from transformers import GPTNeoXTokenizerFast

    tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")
    return len(tokenizer(string))


def is_longer_than_allowed(string, tolerance=0.8, window=2000):
    if get_token_count(string) > tolerance * window:
        return True
    return False
