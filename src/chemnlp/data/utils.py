from typing import List

import yaml
from typing import Any

import fire
import numpy as np
import pandas as pd


from pathlib import Path
import fire

def get_all_datasets(root_dir):
    return [d.name for d in Path(root_dir).iterdir() if d.is_dir()]

def concatenate_jsonl_files(root_dir, output_file, datasets=None, file_type='train'):
    root_dir = Path(root_dir)

    if datasets is None:
        datasets = get_all_datasets(root_dir)
    elif isinstance(datasets, str):
        datasets = [datasets]

    print(f"Processing datasets: {', '.join(datasets)}")
    print(f"File type: {file_type}.jsonl")

    with open(output_file, 'w') as outfile:
        for dataset in datasets:
            dataset_path = root_dir / dataset
            if not dataset_path.is_dir():
                print(f"Warning: Dataset '{dataset}' not found. Skipping.")
                continue

            for chunk_dir in dataset_path.glob('chunk_*'):
                for template_dir in chunk_dir.glob('template_*'):
                    jsonl_file = template_dir / f'{file_type}.jsonl'
                    if jsonl_file.is_file():
                        with open(jsonl_file, 'r') as infile:
                            for line in infile:
                                outfile.write(line)

    print(f"Concatenated {file_type}.jsonl files have been saved to {output_file}")

def concatenate_jsonl_files_cli():
    fire.Fire(concatenate_jsonl_files)



def add_random_split_column(df):
    # Calculate the number of rows for each split
    n_rows = len(df)
    n_train = int(0.7 * n_rows)
    n_test = int(0.15 * n_rows)
    n_valid = n_rows - n_train - n_test

    # Create the split column
    split = ["train"] * n_train + ["test"] * n_test + ["valid"] * n_valid

    # Shuffle the split column
    np.random.shuffle(split)

    # Add the split column to the dataframe
    df["split"] = split

    return df


def _add_random_split_column(file):
    df = pd.read_csv(file)
    df = add_random_split_column(df)
    df.to_csv(file, index=False)


def add_random_split_column_cli():
    fire.Fire(_add_random_split_column)


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
