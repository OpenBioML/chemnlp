import datasets
import pandas as pd

DATASET_NAME = "qm9"


def prepare_data():
    dataset_name = "n0w0f/qm9-csv"
    split_name = "train"  # data without any split @ hf
    filename_to_save = "data_clean.csv"

    # Load the dataset from Hugging Face
    dataset = datasets.load_dataset(dataset_name, split=split_name)

    df = pd.DataFrame(dataset)

    # assert column names
    fields_orig = df.columns.tolist()
    assert fields_orig == [
        "inchi",
        "smiles",
        "rotational_constant_a",
        "rotational_constant_b",
        "rotational_constant_c",
        "dipole_moment",
        "polarizability",
        "homo",
        "lumo",
        "gap",
        "r2",
        "zero_point_energy",
        "u0",
        "u298",
        "h298",
        "g298",
        "heat_capacity",
    ]
    assert not df.duplicated().sum()

    # remove duplicates if any
    df = df.drop_duplicates()

    datapoints = len(df)
    # some parts of the code assume that "SMILES" is in upper case, rename this column
    df.rename(columns={"smiles": "SMILES"}, inplace=True)
    df.to_csv(filename_to_save, index=False)
    return datapoints


if __name__ == "__main__":
    print(f" Preparing  clean tabular {DATASET_NAME} datatset")
    datapoints = prepare_data()
    print(
        f" Finished Preparing  clean tabular {DATASET_NAME} datatset with {datapoints} datapoints"
    )
