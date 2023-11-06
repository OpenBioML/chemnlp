import ast

import numpy as np
import pandas as pd


def process():
    df = pd.read_json(
        "https://huggingface.co/datasets/kjappelbaum/chemnlp-qmof-data/resolve/main/qmof_data.json"
    )
    df.dropna(
        subset=[
            "outputs.CO2-henry_coefficient-mol--kg--Pa",
            "outputs.CO2-adsorption_energy-kJ--mol",
            "outputs.N2-henry_coefficient-mol--kg--Pa",
            "outputs.N2-adsorption_energy-kJ--mol",
            "outputs.CH4-henry_coefficient-mol--kg--Pa",
            "outputs.CH4-adsorption_energy-kJ--mol",
            "outputs.CH4-enthalphy_of_adsorption_58_bar_298_K-kJ--mol",
            "outputs.CH4-enthalphy_of_adsorption_65_bar_298_K-kJ--mol",
            "outputs.CH4-working_capacity_vol_58_to_65_bar_298_K-cm3_STP--cm3",
            "outputs.CH4-working_capacity_mol_58_to_65_bar_298_K-mol--kg",
            "outputs.CH4-working_capacity_fract_58_to_65_bar_298_K-",
            "outputs.CH4-working_capacity_wt%_58_to_65_bar_298_K-g--g*100",
            "outputs.O2-henry_coefficient-mol--kg--Pa",
            "outputs.O2-adsorption_energy-kJ--mol",
            "outputs.O2-enthalphy_of_adsorption_5_bar_298_K-kJ--mol",
            "outputs.O2-enthalphy_of_adsorption_140_bar_298_K-kJ--mol",
            "outputs.O2-working_capacity_vol_5_to_140_bar_298_K-cm3_STP--cm3",
            "outputs.O2-working_capacity_mol_5_to_140_bar_298_K-mol--kg",
            "outputs.O2-working_capacity_fract_5_to_140_bar_298_K-",
            "outputs.O2-working_capacity_wt%_5_to_140_bar_298_K-g--g*100",
            "outputs.Xe-henry_coefficient-mol--kg--Pa",
            "outputs.Xe-adsorption_energy-kJ--mol",
            "outputs.Kr-henry_coefficient-mol--kg--Pa",
            "outputs.Kr-adsorption_energy-kJ--mol",
            "outputs.Xe--Kr-selectivity_298_K-",
            "outputs.H2-working_capacity_5_to_100_bar_298_to_198_K-g--L",
            "outputs.H2-working_capacity_5_to_100_bar_77_K-g--L",
            "outputs.H2-working_capacity_1_to_100_bar_77_K-g--L",
            "outputs.H2-working_capacity_wt%_5_to_100_bar_298_to_198_K-g--g100",
            "outputs.H2-working_capacity_wt%_5_to_100_bar_77_K-g--g100",
            "outputs.H2-working_capacity_wt%_1_to_100_bar_77_K-g--g100",
            "outputs.H2S-henry_coefficient-mol--kg--Pa",
            "outputs.H2S-adsorption_energy-kJ--mol",
            "outputs.H2O-henry_coefficient-mol--kg--Pa",
            "outputs.H2O-adsorption_energy-kJ--mol",
            "outputs.H2S--H2O-selectivity_298_K-",
            "outputs.CH4--N2-selectivity_298_K-",
            "info.pld",
            "info.lcd",
            "info.density",
            "info.mofid.mofid",
            "info.mofid.smiles_nodes",
            "info.mofid.smiles_linkers",
            "info.mofid.topology",
            "info.symmetry.spacegroup_number",
        ],
        inplace=True,
    )

    df["lg10_CO2_Henry"] = np.log10(df["outputs.CO2-henry_coefficient-mol--kg--Pa"])
    df["lg10_N2_Henry"] = np.log10(df["outputs.N2-henry_coefficient-mol--kg--Pa"])
    df["lg10_CH4_Henry"] = np.log10(df["outputs.CH4-henry_coefficient-mol--kg--Pa"])
    df["lg10_O2_Henry"] = np.log10(df["outputs.O2-henry_coefficient-mol--kg--Pa"])
    df["lg10_Xe_Henry"] = np.log10(df["outputs.Xe-henry_coefficient-mol--kg--Pa"])
    df["lg10_Kr_Henry"] = np.log10(df["outputs.Kr-henry_coefficient-mol--kg--Pa"])
    df["lg10_H2S_Henry"] = np.log(df["outputs.H2S-henry_coefficient-mol--kg--Pa"])
    df["lg10_H20_Henry"] = np.log(df["outputs.H2O-henry_coefficient-mol--kg--Pa"])

    df["info.mofid.smiles_nodes"] = df["info.mofid.smiles_nodes"].apply(
        lambda x: ", ".join(ast.literal_eval(x))
    )

    df["info.mofid.smiles_linkers"] = df["info.mofid.smiles_linkers"].apply(
        lambda x: ", ".join(ast.literal_eval(x))
    )

    print(len(df))

    df.to_csv("data_clean.csv", index=False)


if __name__ == "__main__":
    process()
