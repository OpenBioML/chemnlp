import os

import fire
import pandas as pd
from datasets import load_dataset


def process(debug=False):
    if not os.path.exists("combined_json.jsonl"):
        dataset = load_dataset("kjappelbaum/chemnlp_iupac_smiles")
        df = pd.DataFrame(dataset["train"])
    else:
        file = "combined_json.jsonl"
        df = pd.read_json(file, lines=True)

    df.drop_duplicates(subset=["SMILES"], inplace=True)
    print(len(df))

    if debug:
        df = df.sample(1000)

    df.to_csv("data_clean.csv", index=False)


if __name__ == "__main__":
    fire.Fire(process)
