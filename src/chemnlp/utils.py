from pathlib import Path
from typing import Union
import requests
import tarfile

from io import BytesIO
import yaml
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds


def load_config(path: Union[str, Path]):
    with open(path, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def extract_tarball(url, output_dir):
    # Download the tarball from the URL
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Failed to download tarball from {url}")
    
    # Extract the contents of the tarball to the output directory
    with tarfile.open(fileobj=BytesIO(response.content), mode="r|gz") as tar:
        tar.extractall(output_dir)

    
def xyz_to_mol(xyzfile: Union[Path, str], charge: int = 0, remove_h: bool = False):
    """Read an xyz file into an RDKit molecule object.

    Args:
        xyzfile (Path | str): Filepath
        charge (int, optional): Charge of the molecule. Defaults to 0.
        remove_h (bool, optional): If True, removes hydrogens. Defaults to False.

    Returns:
        Mol: Rdkit molecule object
    """
    raw_mol = Chem.MolFromXYZFile(xyzfile)
    mol = Chem.Mol(raw_mol)
    rdDetermineBonds.DetermineBonds(mol,charge=charge)
    if remove_h:
        mol = Chem.RemoveAllHs(mol)
    return mol