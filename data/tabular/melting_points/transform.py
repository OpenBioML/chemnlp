import pandas as pd


def preprocess():
    df = pd.read_csv(
        "https://www.dropbox.com/scl/fi/op8hf1zcl8cin4zb3qj0s/ochem_clean.csv?rlkey=j41m2z1jk7o9hupec19gaxov9&dl=1"
    )
    df = df.rename(columns={"Melting Point": "mp_range"})
    df.dropna(subset=["mp", "NAME", "SMILES", "mp_range"], inplace=True)
    df.to_csv("data_clean.csv", index=False)


if __name__ == "__main__":
    preprocess()
