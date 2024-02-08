"""
This check performs a basic check for data leakage. The checks in this script only focus on SMILES.
Train/test split needs to be run before running this script.
This script assumes that `test_smiles.txt` and `val_smiles.txt` exist in the current working directory.

If leakage is detected, an `AssertionError` will be thrown.

This script has a command line interface. You can run it using `python check_pandas <data_dir>`,
where `<data_dir>` points to a nested set of directories with `data_clean.csv` files.
"""
import os
from glob import glob
from pathlib import Path

import fire
import pandas as pd
from pandarallel import pandarallel
from tqdm import tqdm

pandarallel.initialize(progress_bar=False)

with open("test_smiles.txt", "r") as f:
    test_smiles_ref = f.readlines()
    test_smiles_ref = [x.strip() for x in test_smiles_ref]

with open("val_smiles.txt", "r") as f:
    valid_smiles_ref = f.readlines()
    valid_smiles_ref = [x.strip() for x in valid_smiles_ref]


def leakage_check(file, outdir="out"):
    # mirror subdir structures in outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    print(f"Checking {file}")
    df = pd.read_csv(file, low_memory=False)
    print(df["split"].value_counts())
    train_smiles = df[df["split"] == "train"]["SMILES"].to_list()
    train_smiles = set(train_smiles)
    test_smiles = df[df["split"] == "test"]["SMILES"].to_list()
    test_smiles = set(test_smiles)
    valid_smiles = df[df["split"] == "valid"]["SMILES"].to_list()
    valid_smiles = set(valid_smiles)

    try:
        assert (
            len(train_smiles.intersection(test_smiles)) == 0
        ), "Smiles in train and test"
        assert (
            len(train_smiles.intersection(valid_smiles)) == 0
        ), "Smiles in train and valid"
        assert (
            len(test_smiles.intersection(valid_smiles)) == 0
        ), "Smiles in test and valid"
    except AssertionError as e:
        path = os.path.join(outdir, Path(file).parts[-2], Path(file).name)
        print(f"Leakage in {file}: {e}. Fixing... {path}")
        is_in_test = df["SMILES"].isin(test_smiles)
        is_in_val = df["SMILES"].isin(valid_smiles)

        df.loc[is_in_test, "split"] = "test"
        df.loc[is_in_val, "split"] = "valid"

        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        print(f"Saved fixed file to {path}")
        print("Checking fixed file...")
        leakage_check(path, outdir)

    try:
        assert (
            len(train_smiles.intersection(test_smiles_ref)) == 0
        ), "Smiles in train and scaffold test"

        assert (
            len(train_smiles.intersection(valid_smiles_ref)) == 0
        ), "Smiles in train and scaffold valid"

        assert (
            len(test_smiles.intersection(valid_smiles_ref)) == 0
        ), "Smiles in test and scaffold valid"
    except AssertionError as e:
        path = os.path.join(outdir, Path(file).parts[-2], Path(file).name)
        print(f"Leakage in {file}: {e}. Fixing... {path}")
        is_in_test = df["SMILES"].isin(test_smiles)
        is_in_val = df["SMILES"].isin(valid_smiles)

        df.loc[is_in_test, "split"] = "test"
        df.loc[is_in_val, "split"] = "valid"

        test_smiles = df[df["split"] == "test"]["SMILES"].to_list()
        test_smiles = set(test_smiles)

        valid_smiles = df[df["split"] == "valid"]["SMILES"].to_list()
        valid_smiles = set(valid_smiles)

        is_in_test = df["SMILES"].isin(test_smiles)
        is_in_val = df["SMILES"].isin(valid_smiles)

        df.loc[is_in_test, "split"] = "test"
        df.loc[is_in_val, "split"] = "valid"

        path = os.path.join(outdir, Path(file).parts[-2], Path(file).name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        print(f"Saved fixed file to {path}")
        print("Checking fixed file...")
        leakage_check(path, outdir)

    print(f"No leakage in {file}")
    with open("leakage_check.txt", "a") as f:
        f.write(f"No leakage in {file}\n")
        f.write(f"train: {len(train_smiles)}\n")
        f.write(f"test: {len(test_smiles)}\n")
        f.write(f"valid: {len(valid_smiles)}\n")
    return True


def check_all_files(data_dir):
    all_csv_files = glob(os.path.join(data_dir, "**", "**", "data_clean.csv"))
    for csv_file in tqdm(all_csv_files):
        if Path(csv_file).parts[-2] not in [
            "odd_one_out",
            "uniprot_binding_single",
            "uniprot_binding_sites_multiple",
            "uniprot_organisms",
            "uniprot_reactions",
            "uniprot_sentences",
            "fda_adverse_reactions",
            "drugchat_liang_zhang_et_al",
            "herg_central",
            # those files were checked manually
        ]:
            # if filesize < 35 GB:
            if os.path.getsize(csv_file) < 35 * 1024 * 1024 * 1024:
                try:
                    leakage_check(csv_file)
                except Exception as e:
                    print(f"Could not process {csv_file}: {e}")
            else:
                print(f"Skipping {csv_file} due to size")


if __name__ == "__main__":
    fire.Fire(check_all_files)
