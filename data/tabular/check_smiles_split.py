import dask.dataframe as dd
import fire


def check_data_leakage(
    data_path, test_smiles_path="test_smiles.txt", val_smiles_path="val_smiles.txt"
):
    # Load the data with Dask
    data = dd.read_csv(data_path)

    # Load predefined SMILES lists
    with open(test_smiles_path) as f:
        test_smiles_list = f.read().splitlines()

    with open(val_smiles_path) as f:
        val_smiles_list = f.read().splitlines()

    # Compute split counts (optional, just to have an overview)
    split_counts = data["split"].value_counts().compute()
    print("Split counts:")
    print(split_counts)

    # Check that all predefined test SMILES are only in the test set
    test_smiles_in_data = data[data["SMILES"].isin(test_smiles_list)].compute()
    val_smiles_in_data = data[data["SMILES"].isin(val_smiles_list)].compute()

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
        data[data["split"] == "train"][["SMILES"]],
        data[data["split"] == "valid"][["SMILES"]],
        on="SMILES",
        how="inner",
    )
    merged_splits = dd.merge(
        merged_splits,
        data[data["split"] == "test"][["SMILES"]],
        on="SMILES",
        how="inner",
    ).compute()

    if not merged_splits.empty:
        raise ValueError(
            "Data leakage detected: There are SMILES that appear in multiple splits."
        )

    print("No data leakage detected.")


if __name__ == "__main__":
    fire.Fire(check_data_leakage)
