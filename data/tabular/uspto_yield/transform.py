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
    file_train = hf_hub_download(
        repo_id="kjappelbaum/chemnlp-uspto",
        filename="US_patents_1976-Sep2016_1product_reactions_test_prompts.json",
        repo_type="dataset",
    )
    df_train = pd.read_json(file_train)
    df_train["split"] = "train"

    file_test = hf_hub_download(
        repo_id="kjappelbaum/chemnlp-uspto",
        filename="US_patents_1976-Sep2016_1product_reactions_test_prompts.json",
        repo_type="dataset",
    )
    df_test = pd.read_json(file_test)
    df_test["split"] = "test"

    file_valid = hf_hub_download(
        repo_id="kjappelbaum/chemnlp-uspto",
        filename="US_patents_1976-Sep2016_1product_reactions_test_prompts.json",
        repo_type="dataset",
    )

    df_valid = pd.read_json(file_valid)
    df_valid["split"] = "valid"

    df = pd.concat([df_train, df_test, df_valid])
    df = df.query("WithinTolerance == True")
    df["yield"] = df["MeanYield"]
    df["educt_string"] = df["educts"].apply(oxford_comma_join)
    df["product_string"] = df["products"].apply(oxford_comma_join)
    df["RXNSMILES"] = df["canonical_rxn_smiles"]
    print(len(df))
    df.to_csv("data_clean.csv", index=False)


if __name__ == "__main__":
    process()
