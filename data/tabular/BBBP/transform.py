import pandas as pd


def load_dataset():
    BBBP = pd.read_csv(
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv"
    )
    return BBBP


def transform_data():
    BBBP = load_dataset()

    cols_to_keep = [
        "smiles",
        "p_np",
    ]

    BBBP = BBBP[cols_to_keep]
    BBBP = BBBP.rename(columns={"smiles": "SMILES"})
    BBBP.to_csv("data_clean.csv", index=False)


if __name__ == "__main__":
    transform_data()
