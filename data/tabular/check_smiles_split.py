import dask.dataframe as dd
import fire
from glob import glob
from pathlib import Path
import yaml
from typing import List, Union, Literal


def get_all_yamls(data_dir):
    return glob(str(Path(data_dir) / "**" / "*.yaml"), recursive=True)


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
                smiles_columns.append(target["id"])
    if "identifiers" in meta:
        for identifier in meta["identifiers"]:
            if identifier["type"] == column_type:
                smiles_columns.append(identifier["id"])

    return smiles_columns


def get_all_identifier_columns(yaml_file: Union[str, Path]) -> List[str]:
    """Returns the id for all columns with type SMILES"""
    with open(yaml_file, "r") as f:
        meta = yaml.safe_load(f)

    identifier_columns = []
    if "identifiers" in meta:
        for identifier in meta["identifiers"]:
            identifier_columns.append(identifier["id"])

    return identifier_columns


def check_general_data_leakage(
    data_path,
):
    identifier_columns = get_all_identifier_columns(data_path)
    # check if train/valid/test splits have the same identifiers
    data = dd.read_csv(data_path)
    train = data[data["split"] == "train"]
    valid = data[data["split"] == "valid"]
    test = data[data["split"] == "test"]

    train_valid = dd.merge(
        train[identifier_columns], valid[identifier_columns], how="inner"
    ).compute()

    train_test = dd.merge(
        train[identifier_columns], test[identifier_columns], how="inner"
    ).compute()

    valid_test = dd.merge(
        valid[identifier_columns], test[identifier_columns], how="inner"
    ).compute()

    if not train_valid.empty:
        raise ValueError(
            "Data leakage detected: There are identifiers that appear in both train and valid splits."
        )
    if not train_test.empty:
        raise ValueError(
            "Data leakage detected: There are identifiers that appear in both train and test splits."
        )
    if not valid_test.empty:
        raise ValueError(
            "Data leakage detected: There are identifiers that appear in both valid and test splits."
        )


def check_data_leakage(
    data_path,
    test_smiles_path="test_smiles.txt",
    val_smiles_path="val_smiles.txt",
    test_as_path="test_as.txt",
    val_as_path="val_as.txt",
    col_type="SMILES",
):
    # Load the data with Dask
    data = dd.read_csv(data_path)

    columns = get_columns_of_type(data_path, col_type)

    if col_type == "SMILES":
        # Load predefined SMILES lists
        with open(test_smiles_path) as f:
            test_smiles_list = f.read().splitlines()

        with open(val_smiles_path) as f:
            val_smiles_list = f.read().splitlines()

    elif col_type == "AS_SEQUENCE":
        # Load predefined SMILES lists
        with open(test_as_path) as f:
            test_smiles_list = f.read().splitlines()

        with open(val_as_path) as f:
            val_smiles_list = f.read().splitlines()

    # Compute split counts (optional, just to have an overview)
    split_counts = data["split"].value_counts().compute()
    print("Split counts:")
    print(split_counts)

    for col in columns:
        # Check that all predefined test SMILES are only in the test set
        test_smiles_in_data = data[data[col].isin(test_smiles_list)].compute()
        val_smiles_in_data = data[data[col].isin(val_smiles_list)].compute()

        # Check for overlaps between predefined SMILES and splits
        test_in_val_or_train = test_smiles_in_data["split"] != "test"
        val_in_test_or_train = val_smiles_in_data["split"] != "valid"

        if test_in_val_or_train.any():
            raise ValueError(
                "Data leakage detected: Some test SMILES are in validation or train splits."
            )
        if val_in_test_or_train.any():
            raise ValueError(
                "Data leakage detected: Some validation SMILES are in test or train splits."
            )

        # Check for overlaps between splits by merging on SMILES and checking for multiple split assignments
        merged_splits = dd.merge(
            data[data["split"] == "train"][[col]],
            data[data["split"] == "valid"][[col]],
            on=col,
            how="inner",
        )
        merged_splits = dd.merge(
            merged_splits,
            data[data["split"] == "test"][[col]],
            on=col,
            how="inner",
        ).compute()

        if not merged_splits.empty:
            raise ValueError(
                "Data leakage detected: There are SMILES that appear in multiple splits."
            )

        print("No data leakage detected.")


if __name__ == "__main__":
    fire.Fire(check_data_leakage)
