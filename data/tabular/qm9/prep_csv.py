import os
import re
import tarfile
import tempfile
from typing import Dict, List
from urllib import request as request

import numpy as np
import pandas as pd
from tqdm import tqdm

url = "https://ndownloader.figshare.com/files/3195389"
QM9_PROPERTIES = [
    "rotational_constant_a",
    "rotational_constant_b",
    "rotational_constant_c",
    "dipole_moment",
    "polarizability",
    "homo",
    "lumo",
    "gap",
    "r2",
    "zero_point_energy",
    "u0",
    "u298",
    "h298",
    "g298",
    "heat_capacity",
]


def prepare_qm9_files(tar_path: str, raw_path: str, url: str) -> List[str]:
    """
    Downloads and extracts files from a tar archive to a specified directory.

    Args:
        tar_path (str): Path to save the downloaded tar archive.
        raw_path (str): Directory where the files will be extracted.
        url (str): URL to download the tar archive.

    Returns:
        List[str]: List of sorted file names in the raw_path directory.
    """
    # Download the tar archive
    request.urlretrieve(url, tar_path)

    # Extract files from the tar archive to raw_path
    with tarfile.open(tar_path) as tar:
        tar.extractall(raw_path)

    # List and sort the files in raw_path
    ordered_files = sorted(
        os.listdir(raw_path), key=lambda x: (int(re.sub(r"\D", "", x)), x)
    )

    return ordered_files


def parse_qm9_xyz(filename: str, properties: List[str]) -> Dict[str, str]:
    """
    Parse a QM9 XYZ file and extract relevant information.

    Args:
        filename (str): Path to the QM9 XYZ file.
        properties (List[str]): List of property names for QM9 data.

    Returns:
        Dict[str, str]: A dictionary containing extracted information, including 'inchi', 'smiles',
                        and QM9 properties.
    """
    mol = {}
    with open(filename, "r") as f:
        lines = f.readlines()
        mol["inchi"] = lines[-1].split()[-1]
        mol["smiles"] = lines[-2].split()[-2]
        list_properties = lines[1].split()[2:]
        for i, property_name in enumerate(properties):
            mol[property_name] = list_properties[i]

    return mol


def prepare_csv_data(tar_path: str, raw_path: str, url: str, QM9_PROPERTIES: List[str]):
    """
    Prepare QM9 data, parse it, and save it to a CSV file.

    Args:
        tar_path (str): Path to save the downloaded tar archive.
        raw_path (str): Directory where the files will be extracted.
        url (str): URL to download the tar archive.
        QM9_PROPERTIES (List[str]): List of property names for QM9 data.

    Returns:
        None
    """
    mol_dict = []
    ordered_files = prepare_qm9_files(tar_path, raw_path, url)
    file_iter = np.arange(len(ordered_files), dtype=int)

    for i in tqdm(file_iter):
        xyzfile = os.path.join(raw_path, ordered_files[i])
        mol = parse_qm9_xyz(xyzfile, QM9_PROPERTIES)
        mol_dict.append(mol)

    df = pd.DataFrame(mol_dict)
    df.to_csv("qm9_dataset.csv", index=False)


if __name__ == "__main__":
    tmpdir = tempfile.mkdtemp("qm9_data", dir="./")
    tar_path = os.path.join(tmpdir, "gdb9.tar.gz")
    raw_path = os.path.join(tmpdir, "gdb9_xyz")

    print("Preparing QM9 CSV dataset")
    prepare_csv_data(tar_path, raw_path, url, QM9_PROPERTIES)
    print("Finished")
