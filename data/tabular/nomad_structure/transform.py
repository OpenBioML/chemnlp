import datasets
import pandas as pd

DATASET_NAME = "nomad-structure"


def prepare_data():
    dataset_name = "n0w0f/nomad-structure-csv"
    split_name = "train"  # data without any split @ hf
    filename_to_save = "data_clean.csv"

    # Load the dataset from Hugging Face
    dataset = datasets.load_dataset(dataset_name, split=split_name)

    df = pd.DataFrame(dataset)

    # assert column names
    fields_orig = df.columns.tolist()
    assert fields_orig == [
        "cif",
        "formula",
        "spacegroup",
        "spacegroup_number",
        "crystal_system",
        "pointgroup",
        "density",
        "is_longer_than_allowed",
    ]

    # remove duplicates if any
    df = df.drop_duplicates()

    df.to_csv(filename_to_save, index=False)
    datapoints = len(df)
    return datapoints


if __name__ == "__main__":
    print(f" Preparing  clean tabular {DATASET_NAME} datatset")
    datapoints = prepare_data()
    print(
        f" Finished Preparing  clean tabular {DATASET_NAME} datatset with {datapoints} datapoints"
    )
