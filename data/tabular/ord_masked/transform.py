import pandas as pd
from huggingface_hub import hf_hub_download


def process():
    file = hf_hub_download(
        repo_id="kjappelbaum/chemnlp-ord",
        filename="ord_rxn.json",
        repo_type="dataset",
    )
    df = pd.read_json(file)
    df.dropna(subset=["masked_rxn_smiles", "missing_component"], inplace=True)
    print(len(df))
    df.to_csv("data_clean.csv", index=False)


if __name__ == "__main__":
    process()
