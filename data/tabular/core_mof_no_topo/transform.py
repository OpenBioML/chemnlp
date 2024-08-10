import pandas as pd
from huggingface_hub import hf_hub_download


def process():
    file = hf_hub_download(
        repo_id="kjappelbaum/chemnlp-core-mof",
        filename="core_mofid.json",
        repo_type="dataset",
    )
    df = pd.read_json(file)
    df = df.query("is_longer_than_allowed==False").dropna(
        subset=[
            "outputs.pure_CO2_kH",
            "outputs.pure_CO2_widomHOA",
            "outputs.pure_methane_kH",
            "outputs.pure_methane_widomHOA",
            "outputs.pure_uptake_CO2_298.00_15000",
            "outputs.pure_uptake_CO2_298.00_1600000",
            "outputs.pure_uptake_methane_298.00_580000",
            "outputs.pure_uptake_methane_298.00_6500000",
            "outputs.logKH_CO2",
            "outputs.logKH_CH4",
            "outputs.CH4DC",
            "outputs.CH4HPSTP",
            "outputs.CH4LPSTP",
            "smiles_linkers",
            "smiles_nodes",
        ]
    )

    print(len(df))

    df["smiles_linkers"] = df["smiles_linkers"].apply(lambda x: ", ".join(x))
    df["smiles_nodes"] = df["smiles_nodes"].apply(lambda x: ", ".join(x))

    df[
        [
            "outputs.pure_CO2_kH",
            "outputs.pure_CO2_widomHOA",
            "outputs.pure_methane_kH",
            "outputs.pure_methane_widomHOA",
            "outputs.pure_uptake_CO2_298.00_15000",
            "outputs.pure_uptake_CO2_298.00_1600000",
            "outputs.pure_uptake_methane_298.00_580000",
            "outputs.pure_uptake_methane_298.00_6500000",
            "outputs.logKH_CO2",
            "outputs.logKH_CH4",
            "outputs.CH4DC",
            "outputs.CH4HPSTP",
            "outputs.CH4LPSTP",
            "smiles_linkers",
            "smiles_nodes",
            "cif",
        ]
    ].to_csv("data_clean.csv", index=False)


if __name__ == "__main__":
    process()
