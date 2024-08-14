import pandas as pd
from canonicalize_psmiles.canonicalize import canonicalize
from huggingface_hub import hf_hub_download


def transform_data():
    file = hf_hub_download(
        repo_id="chemNLP/bicerano_polymers",
        filename="HT_MD_polymer_properties.csv",
        repo_type="dataset",
    )
    original_data = pd.read_csv(file)
    clean_data = original_data.drop("sl_num", axis=1)

    assert not clean_data.duplicated().sum()

    clean_columns = [
        "compound_name",
        "PSMILES",
        "Tg_exp",
        "Tg_calc",
        "Tg_calc_std",
        "rho_300K_exp",
        "rho_300K_calc",
        "rho_300K_calc_std",
        "glass_CTE_calc",
        "glass_CTE_calc_std",
        "rubber_CTE_calc",
        "rubber_CTE_calc_std",
    ]

    clean_data.columns = clean_columns

    clean_data["compound_name"] = clean_data["compound_name"].str.strip()

    clean_data["PSMILES"] = clean_data["PSMILES"].str.replace(
        "[Ce]", "[*]", regex=False
    )
    clean_data["PSMILES"] = clean_data["PSMILES"].str.replace(
        "[Th]", "[*]", regex=False
    )
    clean_data["PSMILES"] = clean_data["PSMILES"].str.replace(
        "[[*]]", "[*]", regex=False
    )

    clean_data["PSMILES"] = clean_data["PSMILES"].apply(
        lambda smiles: canonicalize(smiles)
    )

    clean_data.to_csv("data_clean.csv")


if __name__ == "__main__":
    transform_data()
