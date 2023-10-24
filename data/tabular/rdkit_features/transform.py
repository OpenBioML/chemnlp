from datasets import load_dataset
import pandas as pd
import fire


def process(debug=False):
    dataset = load_dataset("maykcaldas/smiles-transformers")
    if debug:
        dataset = dataset.select(range(100))
    train_pandas = dataset["train"].to_pandas()
    test_pandas = dataset["test"].to_pandas()
    valid_pandas = dataset["validation"].to_pandas()
    train_pandas["split"] = "train"
    test_pandas["split"] = "test"
    valid_pandas["split"] = "valid"
    df = pd.concat([train_pandas, test_pandas, valid_pandas])
    df.rename(columns={"text": "SMILES"}, inplace=True)

    print(len(df))

    df.to_csv("data_clean.csv", index=False)


if __name__ == "__main__":
    fire.Fire(process)
