from typing import List

import yaml
from typing import Any


import numpy as np

def add_random_split_column(df):
    # Calculate the number of rows for each split
    n_rows = len(df)
    n_train = int(0.7 * n_rows)
    n_test = int(0.15 * n_rows)
    n_valid = n_rows - n_train - n_test

    # Create the split column
    split = ['train'] * n_train + ['test'] * n_test + ['valid'] * n_valid

    # Shuffle the split column
    np.random.shuffle(split)

    # Add the split column to the dataframe
    df['split'] = split

    return df

def oxford_comma_join(items: List[str]) -> str:
    """Join a list of items with Oxford comma"""
    if len(items) == 1:
        return items[0]
    elif len(items) == 2:
        return f"{items[0]} and {items[1]}"
    else:
        return ", ".join(items[:-1]) + f", and {items[-1]}"


def load_yaml(file_path: str) -> Any:
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def save_yaml(data: Any, file_path: str) -> None:
    with open(file_path, "w") as file:
        yaml.dump(data, file, sort_keys=False)


def str_presenter(dumper, data):
    if len(data.splitlines()) > 1:  # check for multiline string
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


yaml.add_representer(str, str_presenter)
yaml.representer.SafeRepresenter.add_representer(str, str_presenter)
