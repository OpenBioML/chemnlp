import os.path

import fire
from datasets import load_dataset


def clean_df(df):
    df.dropna(inplace=True)
    df[
        [
            "NumHDonors",
            "NumHAcceptors",
            "NumHeteroatoms",
            "RingCount",
            "NumRotatableBonds",
            "NumAromaticBonds",
            "NumAcidGroups",
            "NumBasicGroups",
        ]
    ] = df[
        [
            "NumHDonors",
            "NumHAcceptors",
            "NumHeteroatoms",
            "RingCount",
            "NumRotatableBonds",
            "NumAromaticBonds",
            "NumAcidGroups",
            "NumBasicGroups",
        ]
    ].astype(
        int
    )
    df["MolLogP"] = df["MolLogP"].astype(float)
    df["Apol"] = df["Apol"].astype(float)
    df.rename(columns={"text": "SMILES"}, inplace=True)
    return df


def process():
    if not (os.path.isfile("data_clean.csv")):
        df = load_dataset(
            "maykcaldas/smiles-transformers", split="validation"
        ).to_pandas()
        df = clean_df(df)
        df["split"] = "valid"
        df.to_csv("data_clean.csv", index=False)
        del df

        df = load_dataset("maykcaldas/smiles-transformers", split="test").to_pandas()
        df = clean_df(df)
        df["split"] = "test"
        df.to_csv("data_clean.csv", index=False, mode="a", header=False)
        del df

        splits = [f"train[{k}%:{k+5}%]" for k in range(0, 100, 5)]
        for s in splits:
            df = load_dataset("maykcaldas/smiles-transformers", split=s).to_pandas()
            df = clean_df(df)
            df["split"] = "train"
            df.to_csv("data_clean.csv", index=False, mode="a", header=False)
    else:
        print("Reusing present data_clean.csv.")


if __name__ == "__main__":
    fire.Fire(process)
