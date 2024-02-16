"""This script checks again for leakage and also attempts to fix potential leakage
by assigning all "test" SMILES to the test fold in the `csv`.

It also merges files that have been created by `dask` if they are chunks of one large dataset.

This script needs to be run after the splitting script.
"""
import os
from glob import glob
from pathlib import Path
from typing import List, Literal, Union

import dask.dataframe as dd
import fire
import pandas as pd
import yaml
from pandas.errors import ParserError
from tqdm import tqdm


def merge_files(dir):
    fns = sorted(glob(os.path.join(dir, "data_clean-*.csv")))
    fn_merged = os.path.join(dir, "data_clean.csv")
    if os.path.exists(fn_merged):
        os.remove(fn_merged)
    for fn in fns:
        df = pd.read_csv(fn, index_col=False, low_memory=False)
        df.to_csv(
            fn_merged, mode="a", index=False, header=not os.path.exists(fn_merged)
        )
        os.remove(fn)
        del df


def get_columns_of_type(
    yaml_file: Union[str, Path],
    column_type: Literal["SMILES", "AS_SEQUENCE"] = "SMILES",
) -> List[str]:
    """Returns the id for all columns with type SMILES"""
    with open(yaml_file, "r") as f:
        meta = yaml.safe_load(f)

    smiles_columns = []
    if "targets" in meta:
        for target in meta["targets"]:
            if target["type"] == column_type:
                if target["id"] not in non_relevant_ids:
                    smiles_columns.append(target["id"])
    if "identifiers" in meta:
        for identifier in meta["identifiers"]:
            if identifier["type"] == column_type:
                if identifier["id"] not in non_relevant_ids:
                    smiles_columns.append(identifier["id"])

    return smiles_columns


relevant_ids = ["info.mofid.mofid"]  # deduplication based on this makes sense

# these are not relevant for deduplication
non_relevant_ids = [
    "info.mofid.smiles_nodes",
    "info.mofid.smiles_linkers",
    "info.mofid.smiles",
    "odd_one_out_mol",
    "biggest_sim_0",
    "biggest_sim_1",
    "most_diff_0",
    "most_diff_1",
]


def get_all_identifier_columns(yaml_file: Union[str, Path]) -> List[str]:
    """Returns the id for all columns with type SMILES"""
    with open(yaml_file, "r") as f:
        meta = yaml.safe_load(f)

    identifier_columns = []
    if "identifiers" in meta:
        for identifier in meta["identifiers"]:
            if (
                identifier["type"]
                in ["SMILES", "AS_SEQUENCE", "COMPOSITION", "RXNSMILES", "PSMILES"]
                or identifier["id"] in relevant_ids
            ):
                if identifier["id"] not in non_relevant_ids:
                    identifier_columns.append(identifier["id"])

    # if there are no identifiers, we simply append
    if len(identifier_columns) == 0:
        if "identifiers" in meta:
            for identifier in meta["identifiers"]:
                identifier_columns.append(identifier["id"])
                break
    return identifier_columns


def read_ddf(file):
    try:
        ddf = dd.read_csv(
            os.path.join(os.path.dirname(file), "data_clean.csv"),
            low_memory=False,
        )
    except ParserError:
        print(f"Could not parse {file}. Using blocksize=None.")
        ddf = dd.read_csv(
            os.path.join(os.path.dirname(file), "data_clean.csv"),
            low_memory=False,
            blocksize=None,
        )
    except ValueError as e:
        if "Mismatched dtypes" in str(e):
            print(f"Could not parse {file}. Inferring dtypes via pandas.")
            chunk = pd.read_csv(
                os.path.join(os.path.dirname(file), "data_clean.csv"),
                low_memory=False,
                nrows=10000,
            )
            d = dict(zip(chunk.dtypes.index, [str(t) for t in chunk.dtypes.values]))
            ddf = dd.read_csv(
                os.path.join(os.path.dirname(file), "data_clean.csv"),
                low_memory=False,
                dtype=d,
                assume_missing=True,
            )
    return ddf


def process_file(file: Union[str, Path], id_cols):
    # if file does not fit in memory, use dask to drop duplicates, based on identifiers
    dir = Path(file).parent
    # check if there is any csv file
    if not glob(os.path.join(dir, "*.csv")):
        return
    if len(glob(os.path.join(dir, "data_clean-0*.csv"))) >= 1:
        merge_files(dir)
    df_file = os.path.join(dir, "data_clean.csv")
    size = os.path.getsize(df_file) / (1024**3)
    if size > 30:  # 120 GB pandas memory assuming factor 4
        # note that this assumes that we know that the data in the large files
        # is such that it does not make sense for the same identifier to
        # appear multiple times
        ddf = read_ddf(file)
        ddf = ddf.drop_duplicates(subset=id_cols)
        ddf.to_csv(os.path.join(dir, "data_clean-{*}.csv"), index=False)
        merge_files(dir)

    else:
        df = pd.read_csv(df_file, index_col=False, low_memory=False)
        test_smiles = []
        val_smiles = []

        for id in id_cols:
            test_smiles.extend(df[df["split"] == "test"][id].to_list())
            val_smiles.extend(df[df["split"] == "valid"][id].to_list())

        test_smiles = set(test_smiles)
        val_smiles = set(val_smiles)

        is_in_test = df.apply(
            lambda x: any([s in test_smiles for s in x[id_cols]]), axis=1
        )

        is_in_val = df.apply(
            lambda x: any([s in val_smiles for s in x[id_cols]]), axis=1
        )
        df.loc[is_in_test, "split"] = "test"
        df.loc[is_in_val, "split"] = "valid"

        print(df["split"].value_counts())

        for id in id_cols:
            this_test_smiles = set(df[df["split"] == "test"][id].to_list())
            this_val_smiles = set(df[df["split"] == "valid"][id].to_list())
            this_train_smiles = set(df[df["split"] == "train"][id].to_list())
            assert (
                len(this_test_smiles.intersection(this_train_smiles)) == 0
            ), f"Smiles in test and train for {id}"
            assert (
                len(this_val_smiles.intersection(this_train_smiles)) == 0
            ), f"Smiles in valid and train for {id}"
            assert (
                len(this_test_smiles.intersection(this_val_smiles)) == 0
            ), f"Smiles in test and valid for {id}"

        df.to_csv(os.path.join(dir,"data_clean.csv"), index=False)


def process_all_files(data_dir):
    all_yaml_files = sorted(glob(os.path.join(data_dir, "**", "**", "meta.yaml")))
    index = [
        i for i, x in enumerate(all_yaml_files) if str(x).find("orbnet_denali") != -1
    ][0]
    all_yaml_files = [all_yaml_files[index]]
    for yaml_file in tqdm(all_yaml_files):
        print(f"Processing {yaml_file}")
        try:
            id_cols = get_all_identifier_columns(yaml_file)
            smiles_columns = get_columns_of_type(yaml_file)
            if smiles_columns:
                id_cols = smiles_columns
            process_file(yaml_file, id_cols)
        except Exception as e:
            print(f"Could not process {yaml_file}: {e}")


if __name__ == "__main__":
    fire.Fire(process_all_files)
