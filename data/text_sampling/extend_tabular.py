import glob
import multiprocessing as mp
import os
import random
import time
from functools import partial

import pandas as pd
from utils import load_yaml

from chemnlp.data.reprs import (
    smiles_to_canoncial,
    smiles_to_deepsmiles,
    smiles_to_inchi,
    smiles_to_iupac_name,
    smiles_to_safe,
    smiles_to_selfies,
)


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
        # print("SMILES was already previously processed.")
        representations = df_processed[df_processed.SMILES == smiles].to_dict(
            orient="records"
        )[0]
    else:
        # print("Process SMILES.")
        representations = {
            "smiles": smiles,
            "selfies": _try_except_none(smiles_to_selfies, smiles),
            "deepsmiles": _try_except_none(smiles_to_deepsmiles, smiles),
            "canonical": _try_except_none(smiles_to_canoncial, smiles),
            "inchi": _try_except_none(smiles_to_inchi, smiles),
            "iupac_name": _try_except_none(smiles_to_iupac_name, smiles),
            "safe": _try_except_none(smiles_to_safe, smiles),
        }

        # Note: This needs proper filelocking to work.
        # if path_processed_smiles:
        #    pd.DataFrame(representations, index=[0]).to_csv(path_processed_smiles, mode="a", header=False, index=False)
        #    print("Added processed SMILES to extend_tabular_processed.csv file.")

    return representations


if __name__ == "__main__":
    path_base = __file__.replace("text_sampling/extend_tabular.py", "")
    path_data_dir = sorted(glob.glob(path_base + "tabular/*"))
    path_data_dir += sorted(
        [p for p in glob.glob(path_base + "kg/*") if os.path.isdir(p)]
    )
    path_processed_smiles = path_base + "text_sampling/extend_tabular_processed.csv"

    if os.path.isfile(path_processed_smiles):
        df_processed = pd.read_csv(path_processed_smiles, low_memory=False)
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
        # subselect one path
        # if path.find("data/kg/compound_protein_compound") == -1: continue
        # if path.find("data/tabular/h2_storage_materials") == -1: continue
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

        df = pd.read_csv(path_data, low_memory=False)

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
            "safe": [],
        }

        for entry in parsed:
            for key in data:
                data[key].append(entry[key])

        df_data = pd.DataFrame(data)

        df_new = pd.concat([df, df_data], axis=1)
        df_new.to_csv(path_data, index=False)
