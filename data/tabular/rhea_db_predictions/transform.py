from datasets import load_dataset
import pandas as pd


def oxford_comma_join(l):
    if len(l) == 1:
        return l[0]
    elif len(l) == 2:
        return " and ".join(l)
    else:
        return ", ".join(l[:-1]) + ", and " + l[-1]


def process():
    df = pd.read_json(
        "https://huggingface.co/datasets/kjappelbaum/chemnlp-rhea-db/resolve/main/rhea-reaction-smiles_prompts.json"
    )
    df["educt_string"] = df["educts"].apply(oxford_comma_join)
    df["product_string"] = df["products"].apply(oxford_comma_join)
    df.rename(columns={"canonical_rxn_smiles": "RXNSMILES"}, inplace=True)
    df.dropna(subset=["educt_string", "product_string"], inplace=True)
    print(len(df))
    df[["RXNSMILES", "educt_string", "product_string"]].to_csv(
        "data_clean.csv", index=False
    )


if __name__ == "__main__":
    process()
