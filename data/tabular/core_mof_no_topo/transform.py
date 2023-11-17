import pandas as pd


def process():
    df = pd.read_json("core_mofid.json")
    # df = pd.read_json(
    #     "https://huggingface.co/datasets/kjappelbaum/chemnlp-core-mof/resolve/main/core_mofid.json"
    # )
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
