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
import sys
from collections import defaultdict
from glob import glob
from random import Random
from typing import Dict, List

import fire
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")


def print_sys(s):
    """system print

    Args:
        s (str): the string to print
    """
    print(s, flush=True, file=sys.stderr)


def create_scaffold_split(
    df: pd.DataFrame, seed: int, frac: List[float], entity: str = "SMILES"
) -> Dict[str, pd.DataFrame]:
    """create scaffold split. it first generates molecular scaffold for each molecule
    and then split based on scaffolds
    adapted from: https://github.com/mims-harvard/TDC/tdc/utils/split.py

    Args:
        df (pd.DataFrame): dataset dataframe
        fold_seed (int): the random seed
        frac (list): a list of train/valid/test fractions
        entity (str): the column name for where molecule stores

    Returns:
        dict: a dictionary of splitted dataframes, where keys are train/valid/test
        and values correspond to each dataframe
    """
    random = Random(seed)

    s = df[entity].values
    scaffolds = defaultdict(set)

    error_smiles = 0
    for i, smiles in tqdm(enumerate(s), total=len(s)):
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                mol=Chem.MolFromSmiles(smiles), includeChirality=False
            )
            scaffolds[scaffold].add(i)
        except Exception:
            print_sys(smiles + " returns RDKit error and is thus omitted...")
            error_smiles += 1

    train, val, test = [], [], []
    train_size = int((len(df) - error_smiles) * frac[0])
    val_size = int((len(df) - error_smiles) * frac[1])
    test_size = (len(df) - error_smiles) - train_size - val_size
    train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0

    # index_sets = sorted(list(scaffolds.values()), key=lambda i: len(i), reverse=True)
    index_sets = list(scaffolds.values())
    big_index_sets = []
    small_index_sets = []
    for index_set in index_sets:
        if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
            big_index_sets.append(index_set)
        else:
            small_index_sets.append(index_set)
    random.seed(seed)
    random.shuffle(big_index_sets)
    random.shuffle(small_index_sets)
    index_sets = big_index_sets + small_index_sets

    if frac[2] == 0:
        for index_set in index_sets:
            if len(train) + len(index_set) <= train_size:
                train += index_set
                train_scaffold_count += 1
            else:
                val += index_set
                val_scaffold_count += 1
    else:
        for index_set in index_sets:
            if len(train) + len(index_set) <= train_size:
                train += index_set
                train_scaffold_count += 1
            elif len(val) + len(index_set) <= val_size:
                val += index_set
                val_scaffold_count += 1
            else:
                test += index_set
                test_scaffold_count += 1

    return {
        "train": df.iloc[train].reset_index(drop=True),
        "valid": df.iloc[val].reset_index(drop=True),
        "test": df.iloc[test].reset_index(drop=True),
    }


def rewrite_data_with_splits(
    csv_paths: List[str],
    train_test_df: pd.DataFrame,
    override: bool = False,
    check: bool = True,
    repr_col: str = "SMILES",
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
    if check:
        train_smiles = set(train_test_df.query("split == 'train'")["SMILES"].to_list())

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

            col_to_merge = "SMILES"
            merged_data = pd.merge(
                read_dataset, train_test_df, on=col_to_merge, how="left"
            )
            merged_data = merged_data.dropna()
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
                    merged_data.query("split == 'test'")["SMILES"].to_list()
                )
                if len(train_smiles.intersection(test_split_smiles)) > 0:
                    raise ValueError("Split failed, train and test overlap")
        else:
            print(f"Skipping {path} as it does not contain {repr_col} column")


def cli(
    seed: int = 42,
    train_size: float = 0.8,
    val_size: float = 0.0,
    test_size: float = 0.2,
    path: str = "*/data_clean.csv",
    override: bool = False,
    check: bool = True,
    repr_col: str = "SMILES",
):
    paths_to_data = glob(path)

    # uncomment the following lines for debugging on a subset of data
    # filtered_paths = []
    # for path in paths_to_data:
    #     if "flashpoint" in path:
    #         filtered_paths.append(path)
    #     elif "freesolv" in path:
    #         filtered_paths.append(path)
    #     elif "peptide" in path:
    #         filtered_paths.append(path)
    #     elif "bicerano_dataset" in path:
    #         filtered_paths.append(path)
    # paths_to_data = filtered_paths

    REPRESENTATION_LIST = []

    for path in tqdm(paths_to_data):
        df = pd.read_csv(path)
        if repr_col in df.columns:
            REPRESENTATION_LIST.extend(df[repr_col].to_list())
        else:
            df["split"] = np.random.choice(
                ["train", "test", "valid"],
                size=len(df),
                p=[1 - val_size - test_size, test_size, val_size],
            )

            if override:
                df.to_csv(path, index=False)
            else:
                # rename the old data_clean.csv file to data_clean_old.csv
                os.rename(path, path.replace(".csv", "_old.csv"))
                # write the new data_clean.csv file
                df.to_csv(path, index=False)

            if len(df.query("split == 'train'")) == 0:
                raise ValueError("Split failed, no train data")
            if len(df.query("split == 'test'")) == 0:
                raise ValueError("Split failed, no test data")

    REPR_DF = pd.DataFrame()
    REPR_DF["SMILES"] = list(set(REPRESENTATION_LIST))

    scaffold_split = create_scaffold_split(
        REPR_DF, seed=seed, frac=[train_size, val_size, test_size]
    )

    # create train and test dataframes
    train_df = scaffold_split["train"]
    test_df = scaffold_split["test"]
    # add split columns to train and test dataframes
    train_df["split"] = len(train_df) * ["train"]
    test_df["split"] = len(test_df) * ["test"]

    # merge train and test across all datasets
    merge = pd.concat([train_df, test_df], axis=0)
    # rewrite data_clean.csv for each dataset
    rewrite_data_with_splits(
        paths_to_data, merge, override=override, check=check, repr_col=repr_col
    )


if __name__ == "__main__":
    fire.Fire(cli)
