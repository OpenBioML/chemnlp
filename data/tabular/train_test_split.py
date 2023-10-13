import glob
import os

import numpy as np
import pandas as pd


def transform(dataframe: pd.DataFrame):
    pass


def check_split(dataframe: pd.DataFrame):
    cols = dataframe.columns
    return "split" in cols


if __name__ == "__main__":
    data_clean_csvs = glob.glob("*/data_clean.csv")
    datasets = [dataset.split("/")[0] for dataset in data_clean_csvs]
    dataframes = [pd.read_csv(csv) for csv in data_clean_csvs]

    check_split = [
        {datasets[dataset_idx]: check_split(df)}
        for dataset_idx, df in enumerate(dataframes)
    ]

    print(check_split)
