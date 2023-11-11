import pandas as pd


def process():
    df = pd.read_csv(
        "http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/thermosol.csv"
    )
    df.rename(columns={"smile": "SMILES"}, inplace=True)
    df.dropna(inplace=True)
    print(len(df))
    df[["SMILES", "target"]].to_csv("data_clean.csv", index=False)


if __name__ == "__main__":
    process()
