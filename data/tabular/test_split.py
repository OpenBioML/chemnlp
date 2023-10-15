# from ..train_test_split import create_scaffold_split, rewrite_data_with_splits
import pandas as pd
import importlib
import os
from glob import glob
import sys
from train_test_split import create_scaffold_split, rewrite_data_with_splits

def test_split(seed):
    paths_to_data = glob("tests/csvs_no_split_column/*.csv")

    REPRESENTATION_LIST = []

    for path in paths_to_data:
        df = pd.read_csv(path)
        if "SMILES" in df.columns:
            REPRESENTATION_LIST.extend(df["SMILES"].to_list())

    REPR_DF = pd.DataFrame()
    REPR_DF["SMILES"] = list(set(REPRESENTATION_LIST))

    scaffold_split = create_scaffold_split(REPR_DF, seed=seed, frac=[0.8, 0, 0.2])

    # create train and test dataframes
    train_df = scaffold_split["train"]
    test_df = scaffold_split["test"]
    # add split columns to train and test dataframes
    train_df["split"] = len(train_df) * ["train"]
    test_df["split"] = len(test_df) * ["test"]

    # merge train and test across all datasets
    merge = pd.concat([train_df, test_df], axis=0)
    # rewrite data_clean.csv for each dataset
    rewrite_data_with_splits(paths_to_data, merge)


if __name__ == "__main__":
    test_split(42)