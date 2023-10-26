import pandas as pd
from huggingface_hub import hf_hub_download


def process():
    file = hf_hub_download(
        repo_id="kjappelbaum/chemnlp-ord",
        filename="ord_data_compiled.json",
        repo_type="dataset",
    )
    df = pd.read_json(file)
    df.rename(columns={"rxn_smiles": "RXNSMILES"}, inplace=True)
    df = df.dropna(subset=["RXNSMILES", "procedure"])
    df = df.query("RXNSMILES != 'None'")
    # make sure RXNSMILES values have at least 10 characters
    df = df[df["RXNSMILES"].str.len() > 10]
    df = df.query("procedure != 'None'")
    df.query(
        "steps_string != 'None'", inplace=True
    )  # this removes cases in which is just says "follow the procedure above"
    df = df.query("procedure != ''")
    df = df[["RXNSMILES", "procedure"]]
    print(len(df))
    df.to_csv("data_clean.csv", index=False)


if __name__ == "__main__":
    process()
