import os

import modin.pandas as pd
import numpy as np
from fire import Fire
from rdkit import Chem

from chemnlp.utils import extract_tarball, load_config, xyz_to_mol

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_LABEL_FILE = os.path.join(_THIS_DIR, "../denali_labels.csv")
_STRUCTURE_DIR = os.path.join(_THIS_DIR, "../xyz_files")


# perhaps set export MODIN_ENGINE=ray


def _download_files_if_not_existent(config):
    # for some reason the number of files does not equal the number of rows
    if not os.path.exists(_STRUCTURE_DIR):
        print("🚧 Didn't find structure dir")
        structure_link = list(
            filter(lambda x: x["description"] == "structure download", config["links"])
        )
        extract_tarball(structure_link[0]["url"], "..", md5=structure_link[0]["md5"])
        # now, rerun check
        # if not os.path.exists(_STRUCTURE_DIR) or len(xyz_files) != config["num_points"]:
        #     raise ValueError("Download failed!")

    if (
        not os.path.exists(_LABEL_FILE)
        or len(pd.read_csv(_LABEL_FILE)) != config["num_points"]
    ):
        print(f"🚧 Didn't find enough rows, found {len(pd.read_csv(_LABEL_FILE))}")
        csv_link = list(
            filter(lambda x: x["description"] == "label download", config["links"])
        )
        extract_tarball(csv_link[0]["url"], "..", md5=csv_link[0]["md5"])
        # now, rerun check
        if (
            not os.path.exists(_LABEL_FILE)
            or len(pd.read_csv(_LABEL_FILE)) != config["num_points"]
        ):
            raise ValueError("Download failed!")


def _add_smiles_and_filepath(row) -> pd.Series:
    try:
        filepath = os.path.join(
            _STRUCTURE_DIR, row["mol_id"], row["sample_id"] + ".xyz"
        )
        mol = xyz_to_mol(filepath, row["charge"])
        path = os.path.abspath(filepath)

        return pd.Series([Chem.MolToSmiles(mol), str(path)])

    except Exception:
        return pd.Series([np.nan, np.nan])


def _create_smiles_and_filepath(debug: bool = False):
    df = pd.read_csv(_LABEL_FILE)

    if debug:
        df = df.sample(50)

    df[["SMILES", "path"]] = df.apply(_add_smiles_and_filepath, axis=1)

    df.dropna(subset=["SMILES", "path", "charge"], inplace=True)

    df.to_csv("data_clean.csv")


def cli(debug: bool = False):
    config = load_config(os.path.join(_THIS_DIR, "meta.yaml"))

    _download_files_if_not_existent(config=config)

    _create_smiles_and_filepath(debug)


if __name__ == "__main__":
    Fire(cli)
