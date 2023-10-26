from datasets import load_dataset
import pandas as pd


def process():
    df = pd.read_json(
        "https://huggingface.co/datasets/kjappelbaum/chemnlp-rhea-db/resolve/main/rhea-reaction-smiles_prompts.json"
    )
    df.dropna(subset=["masked_rxn_smiles", "missing_component"], inplace=True)
    print(len(df))
    df.to_csv("data_clean.csv", index=False)


if __name__ == "__main__":
    process()
