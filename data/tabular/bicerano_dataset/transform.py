import pandas as pd
from canonicalize_psmiles.canonicalize import canonicalize


def transform_data():
    url = "https://huggingface.co/datasets/chemNLP/bicerano_polymers/raw/main/HT_MD_polymer_properties.csv"
    original_data = pd.read_csv(url)
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
