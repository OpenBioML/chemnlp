import fire
import pandas as pd
from datasets import load_dataset


def process(debug=False):
    if debug:
        dataset = load_dataset("maykcaldas/smiles-transformers", split="train[:100]")
        train_pandas = dataset.to_pandas()
        test_pandas = dataset.to_pandas()
        valid_pandas = dataset.to_pandas()
    else:
        dataset = load_dataset("maykcaldas/smiles-transformers")
        train_pandas = dataset["train"].to_pandas()
        test_pandas = dataset["test"].to_pandas()
        valid_pandas = dataset["validation"].to_pandas()
    train_pandas["split"] = "train"
    test_pandas["split"] = "test"
    valid_pandas["split"] = "valid"
    df = pd.concat([train_pandas, test_pandas, valid_pandas])
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

    print(df.columns)
    df.rename(columns={"text": "SMILES"}, inplace=True)

    print(len(df))

    df.to_csv("data_clean.csv", index=False)


if __name__ == "__main__":
    fire.Fire(process)
