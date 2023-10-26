import pandas as pd


def oxford_comma_join(elements):
    if len(elements) == 1:
        return elements[0]
    elif len(elements) == 2:
        return " and ".join(elements)
    else:
        return ", ".join(elements[:-1]) + ", and " + elements[-1]


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
