"""Perform scaffold split on all datasets and rewrite data_clean.csv files.

Scaffold split is a method of splitting data that ensures that the same scaffold
is not present in both the train and test sets. This is important for evaluating
the generalizability of a model.

For more information, see:
    - Wu, Z.; Ramsundar, B.; Feinberg, E. N.; Gomes, J.; Geniesse, C.; Pappu, A. S.;
        Leswing, K.; Pande, V. MoleculeNet: A Benchmark for Molecular Machine Learning.
        Chemical Science 2018, 9 (2), 513–530. https://doi.org/10.1039/c7sc02664a.
    - Jablonka, K. M.; Rosen, A. S.; Krishnapriyan, A. S.; Smit, B.
        An Ecosystem for Digital Reticular Chemistry. ACS Central Science 2023, 9 (4), 563–581.
        https://doi.org/10.1021/acscentsci.2c01177.

"""
import os
from glob import glob
from typing import List

import fire
import numpy as np
import pandas as pd
from rdkit import RDLogger
from tqdm import tqdm

from chemnlp.data.split import create_scaffold_split

RDLogger.DisableLog("rdApp.*")


REPRESENTATION_LIST = [
    "SMILES",
    "PSMILES",
    "SELFIES",
    "RXNSMILES",
    "RXNSMILESWAdd",
    "IUPAC",
    "InChI",
    "InChIKey",
    "COMPOSITION",
    "Sentence",
    "AS_SEQUENCE",
]


def rewrite_data_with_splits(
    csv_paths: List[str],
    repr_col: str,
    train_test_df: pd.DataFrame,
    override: bool = False,
    check: bool = True,
) -> None:
    """Rewrite dataframes with the correct split column

    Args:
        csv_paths (List[str]): list of files to merge (data_clean.csv)
        train_test_df (pd.DataFrame): dataframe containing merged SMILES representations
            from all datasets uniquely split into train and test
        override (bool): whether to override the existing data_clean.csv files
            defaults to False
        check (bool): whether to check if the split was successful
            defaults to True. Can be turned off to save memory
        repr_col (str): the column name for where SMILES representation is stored
            defaults to "SMILES"
    """

    for path in csv_paths:
        read_dataset = pd.read_csv(path)

    if check:
        train_smiles = set(train_test_df.query("split == 'train'")[repr_col].to_list())

    for path in csv_paths:
        read_dataset = pd.read_csv(path)
        if repr_col in read_dataset.columns:
            try:
                read_dataset = read_dataset.drop("split", axis=1)
                message = f"Split column found in {path}."
                if override:
                    message += " Overriding..."
                print(message)
            except KeyError:
                print(f"No split column in {path}")

            merged_data = pd.merge(read_dataset, train_test_df, on=repr_col, how="left")
            # merged_data = merged_data.dropna()
            if override:
                merged_data.to_csv(path, index=False)
            else:
                # rename the old data_clean.csv file to data_clean_old.csv
                os.rename(path, path.replace(".csv", "_old.csv"))
                # write the new data_clean.csv file
                merged_data.to_csv(path, index=False)

            if len(merged_data.query("split == 'train'")) == 0:
                raise ValueError("Split failed, no train data")
            if len(merged_data.query("split == 'test'")) == 0:
                raise ValueError("Split failed, no test data")
            if check:
                test_split_smiles = set(
                    merged_data.query("split == 'test'")[repr_col].to_list()
                )
                if len(train_smiles.intersection(test_split_smiles)) > 0:
                    raise ValueError("Split failed, train and test overlap")


TRACKED_DATASETS = []


def per_repr(
    repr_col: str,
    seed: int = 42,
    train_size: float = 0.8,
    val_size: float = 0.0,
    test_size: float = 0.2,
    path: str = "*/data_clean.csv",
    override: bool = False,
    check: bool = True,
):
    paths_to_data = glob(path)

    # uncomment the following lines for debugging on a subset of data
    # filtered_paths = []
    # for path in paths_to_data:
    #     if "flashpoint" in path:
    #         filtered_paths.append(path)
    #     elif "BBBP" in path:
    #         filtered_paths.append(path)
    #     elif "peptide" in path:
    #         filtered_paths.append(path)
    #     elif "bicerano_dataset" in path:
    #         filtered_paths.append(path)
    # paths_to_data = filtered_paths

    representations = []

    for path in paths_to_data:
        df = pd.read_csv(path)

        if repr_col in df.columns and path not in TRACKED_DATASETS:
            print("Processing", path.split("/")[0])
            TRACKED_DATASETS.append(path)
            representations.extend(df[repr_col].to_list())

    repr_df = pd.DataFrame()
    repr_df[repr_col] = list(set(representations))

    if "SMILES" in repr_df.columns:
        split = create_scaffold_split(
            repr_df, seed=seed, frac=[train_size, val_size, test_size]
        )
    else:
        split_ = pd.DataFrame(
            np.random.choice(
                ["train", "test", "valid"],
                size=len(repr_df),
                p=[1 - val_size - test_size, test_size, val_size],
            )
        )

        repr_df["split"] = split_
        train = repr_df.query("split == 'train'").reset_index(drop=True)
        test = repr_df.query("split == 'test'").reset_index(drop=True)
        valid = repr_df.query("split == 'valid'").reset_index(drop=True)

        split = {"train": train, "test": test, "valid": valid}

    # create train and test dataframes
    train_df = split["train"]
    test_df = split["test"]
    valid_df = split["valid"]
    # add split columns to train and test dataframes
    train_df["split"] = len(train_df) * ["train"]
    test_df["split"] = len(test_df) * ["test"]
    valid_df["split"] = len(valid_df) * ["valid"]

    # merge train and test across all datasets
    merge = pd.concat([train_df, test_df, valid_df], axis=0)
    # rewrite data_clean.csv for each dataset
    rewrite_data_with_splits(
        csv_paths=paths_to_data,
        repr_col=repr_col,
        train_test_df=merge,
        override=override,
        check=check,
    )


def cli(
    seed: int = 42,
    train_size: float = 0.75,
    val_size: float = 0.125,
    test_size: float = 0.125,
    path: str = "*/data_clean.csv",
    override: bool = False,
    check: bool = True,
):
    for representation in tqdm(REPRESENTATION_LIST):
        if representation == "SMILES":
            print("Processing priority representation: SMILES")
        else:
            print("Processing datasets with the representation column:", representation)
        per_repr(
            representation, seed, train_size, val_size, test_size, path, override, check
        )


if __name__ == "__main__":
    fire.Fire(cli)
