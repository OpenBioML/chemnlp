import glob
import multiprocessing as mp
import os
import random
import time
from functools import partial

import deepsmiles
import pandas as pd
import pubchempy as pcp
import requests
import selfies
from rdkit import Chem

# tucan needs very likely python 3.10
# from tucan.canonicalization import canonicalize_molecule
# from tucan.io import graph_from_molfile_text
# from tucan.serialization import serialize_molecule
from utils import load_yaml


def augment_smiles(smiles: str, int_aug: int = 50, deduplicate: bool = True) -> str:
    """
    Takes a SMILES (not necessarily canonical) and returns `int_aug` random variations of this SMILES.
    """

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None
    else:
        if int_aug > 0:
            augmented = [
                Chem.MolToSmiles(mol, canonical=False, doRandom=True)
                for _ in range(int_aug)
            ]
            if deduplicate:
                augmented = list(set(augmented))
            return augmented
        else:
            raise ValueError("int_aug must be greater than zero.")


def smiles_to_selfies(smiles: str) -> str:
    """
    Takes a SMILES and return the selfies encoding.
    """

    return selfies.encoder(smiles)


def smiles_to_deepsmiles(smiles: str) -> str:
    """
    Takes a SMILES and return the DeepSMILES encoding.
    """
    converter = deepsmiles.Converter(rings=True, branches=True)
    return converter.encode(smiles)


def smiles_to_canoncial(smiles: str) -> str:
    """
    Takes a SMILES and return the canoncial SMILES.
    """
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol)


def smiles_to_inchi(smiles: str) -> str:
    """
    Takes a SMILES and return the InChI.
    """
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToInchi(mol)


# def smiles_to_tucan(smiles: str) -> str:
#    """
#    Takes a SMILES and return the Tucan encoding.
#    For this, create a molfile as StringIO, read it with graph_from_file,
#    canonicalize it and serialize it.
#    """
#    molfile = Chem.MolToMolBlock(Chem.MolFromSmiles(smiles), forceV3000=True)
#    mol = graph_from_molfile_text(molfile)
#    mol = canonicalize_molecule(mol)
#    return serialize_molecule(mol)


CACTUS = "https://cactus.nci.nih.gov/chemical/structure/{0}/{1}"


def smiles_to_iupac_name(smiles: str) -> str:
    """Use the chemical name resolver https://cactus.nci.nih.gov/chemical/structure.
    If this does not work, use pubchem.
    """
    try:
        time.sleep(0.001)
        rep = "iupac_name"
        url = CACTUS.format(smiles, rep)
        response = requests.get(url, allow_redirects=True, timeout=10)
        response.raise_for_status()
        name = response.text
        if "html" in name:
            return None
        return name
    except Exception:
        try:
            compound = pcp.get_compounds(smiles, "smiles")
            return compound[0].iupac_name
        except Exception:
            return None


def _try_except_none(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception:
        return None


def line_reps_from_smiles(
    smiles: str,
    unique_smiles_processed: list = None,
    df_processed: pd.DataFrame = None,
    path_processed_smiles: str = None,
) -> dict:
    """
    Takes a SMILES and returns a dictionary with the different representations.
    Use None if some representation cannot be computed.
    """

    if smiles in unique_smiles_processed:
        print("SMILES was already previously processed.")
        representations = df_processed[df_processed.SMILES == smiles].to_dict(
            orient="records"
        )[0]
    else:
        print("Process SMILES.")
        representations = {
            "smiles": smiles,
            "selfies": _try_except_none(smiles_to_selfies, smiles),
            "deepsmiles": _try_except_none(smiles_to_deepsmiles, smiles),
            "canonical": _try_except_none(smiles_to_canoncial, smiles),
            "inchi": _try_except_none(smiles_to_inchi, smiles),
            # "tucan": _try_except_none(smiles_to_tucan, smiles),
            "iupac_name": _try_except_none(smiles_to_iupac_name, smiles),
        }

        # Note: This needs proper filelocking to work.
        # if path_processed_smiles:
        #    pd.DataFrame(representations, index=[0]).to_csv(path_processed_smiles, mode="a", header=False, index=False)
        #    print("Added processed SMILES to extend_tabular_processed.csv file.")

    return representations


if __name__ == "__main__":
    path_base = __file__.replace("text_sampling/extend_tabular.py", "")
    path_data_dir = sorted(glob.glob(path_base + "tabular/*"))
    path_processed_smiles = path_base + "text_sampling/extend_tabular_processed.csv"

    if os.path.isfile(path_processed_smiles):
        df_processed = pd.read_csv(path_processed_smiles)
        unique_smiles_processed = df_processed.SMILES.unique().tolist()
        process_func = partial(
            line_reps_from_smiles,
            df_processed=df_processed,
            unique_smiles_processed=unique_smiles_processed,
            path_processed_smiles=path_processed_smiles,
        )
        print("Using preprocessed SMILES.")
    else:
        process_func = line_reps_from_smiles

    for path in path_data_dir:
        if not os.path.isdir(path):
            continue

        print(f"\n###### {path}")

        path_meta = path + "/meta.yaml"
        path_data = path + "/data_clean.csv"
        # check if files are there
        if not os.path.isfile(path_meta):
            print("No meta.yaml file in the dataset directory.")
            continue
        if not os.path.isfile(path_data):
            print("No data_clean.csv file in the dataset directory.")
            continue

        # check if SMILES column is there
        df = pd.read_csv(path_data, index_col=False, nrows=0)  # only get columns
        cols = df.columns.tolist()
        if "SMILES" not in cols:
            print("No SMILES identifier in the data_clean.csv.")
            continue

        # check if SMILES identifier is in the meta.yaml file
        meta = load_yaml(path_meta)
        if not (
            any([identifier["id"] == "SMILES" for identifier in meta["identifiers"]])
        ):
            # if no SMILES identifier in the meta.yaml we continue
            print(
                "No SMILES identifier in the meta.yaml. Please define custom text templates."
            )
            continue

        df = pd.read_csv(path_data)

        # if {
        #    "SMILES",
        #    "selfies",
        #    "deepsmiles",
        #    "canonical",
        #    "inchi",
        #    "iupac_name",
        #    }.issubset(df.columns):
        #    # if they colums are already there we don't need to do anything.
        #    print(
        #        "Extended data is already in the data_clean.csv file."
        #    )
        #    continue

        parsed = []
        n_proc = mp.cpu_count() - 1 or 1
        print(f"{n_proc=}")
        start = time.time()
        with mp.Pool(processes=n_proc) as pool:
            parsed = pool.map(process_func, df.SMILES.tolist())
        end = time.time()
        print(f"processing time: {(end - start)/60:.2f} min")
        print("Random parsing examples:")
        for sample in random.sample(parsed, k=5):
            for key, value in sample.items():
                print(f"{key:<12}{value}")
            print()

        data = {
            "selfies": [],
            "deepsmiles": [],
            "canonical": [],
            "inchi": [],
            # "tucan": [],
            "iupac_name": [],
        }

        for entry in parsed:
            for key in data:
                data[key].append(entry[key])

        df_data = pd.DataFrame(data)

        df_new = pd.concat([df, df_data], axis=1)
        df_new.to_csv(path_data, index=False)
