import pandas as pd


def load_dataset() -> pd.DataFrame:
    bace = pd.read_csv(
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv"
    )
    return bace


def transform_data():
    bace = load_dataset()
    bace = bace.rename(columns={"mol": "SMILES", "Class": "BACE_inhibition"})
    bace = bace.drop(columns=["CID", "Model", "canvasUID"])

    # Keeping only qualitative and quantitative pIC50 values
    # Removing all the RDKit computed descriptors
    cols_to_keep = ["SMILES", "pIC50", "BACE_inhibition"]
    bace = bace[cols_to_keep]
    bace.to_csv("data_clean.csv", index=False)


if __name__ == "__main__":
    transform_data()
