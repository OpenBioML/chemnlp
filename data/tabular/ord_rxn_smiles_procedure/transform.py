import pandas as pd
from huggingface_hub import hf_hub_download
from rxn.chemutils.reaction_equation import rxn_standardization
from rxn.chemutils.reaction_smiles import parse_any_reaction_smiles


def canoncialize_rxn_smiles(rxn_smiles):
    try:
        return rxn_standardization(parse_any_reaction_smiles(rxn_smiles)).to_string()
    except Exception:
        return None


def process():
    file = hf_hub_download(
        repo_id="kjappelbaum/chemnlp-ord",
        filename="ord_data_compiled.json",
        repo_type="dataset",
    )
    df = pd.read_json(file)
    df["canonical_rxn_smiles"] = df["rxn_smiles"].apply(canoncialize_rxn_smiles)
    df.rename(columns={"canonical_rxn_smiles": "RXNSMILES"}, inplace=True)
    df = df.dropna(subset=["RXNSMILES", "procedure"])
    df = df.query("RXNSMILES != 'None'")
    # make sure RXNSMILES values have at least 10 characters
    df = df[df["RXNSMILES"].str.len() > 10]
    # there must be > in the reaction SMILES
    df = df[df["RXNSMILES"].str.contains(">")]
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
