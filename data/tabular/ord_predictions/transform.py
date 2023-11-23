import pandas as pd
from huggingface_hub import hf_hub_download


def oxford_comma_join(elements):
    try:
        if len(elements) == 1:
            return elements[0]
        elif len(elements) == 2:
            return " and ".join(elements)
        else:
            return ", ".join(elements[:-1]) + ", and " + elements[-1]
    except Exception:
        return None


def process():
    file = hf_hub_download(
        repo_id="kjappelbaum/chemnlp-ord",
        filename="ord_rxn.json",
        repo_type="dataset",
    )
    df = pd.read_json(file)
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
