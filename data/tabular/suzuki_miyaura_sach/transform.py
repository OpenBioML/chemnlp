import random

import pandas as pd
from rdkit import Chem  # 2022.9.5
from rxn.chemutils.reaction_equation import rxn_standardization
from rxn.chemutils.reaction_smiles import parse_any_reaction_smiles


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


def extract_reaction_info(equation_string):
    # First, either pick from reactants or products
    # then pick a random element from the chosen side
    # then replace it with a mask and return the new equation as
    # well as the element that was replaced
    # return equation, replaced_element
    equation = rxn_standardization(equation_string)
    std = rxn_standardization(equation)
    type_of_list = ["educts", "products"]
    type_of_list = random.choice(type_of_list)
    if type_of_list == "educts":
        list_of_elements = equation.reactants
    else:
        list_of_elements = equation.products
    element = random.choice(list_of_elements)
    replaced_element = element
    # remove the element from the list
    list_of_elements.remove(element)
    # replace it with a mask
    list_of_elements.append("MASK")
    # return the new equation as well as the element that was replaced
    if type_of_list == "educts":
        equation.reactants = list_of_elements
    else:
        equation.products = list_of_elements

    reaction = {
        "educts": std.reactants,
        "products": std.products,
        "canonical_rxn_smiles": std.to_string(),
        "masked_rxn_smiles": equation.to_string(),
        "missing_component": replaced_element,
        "rxn_smiles": equation_string,
        "educt_string": oxford_comma_join([str(x) for x in std.reactants]),
        "product_string": oxford_comma_join([str(x) for x in std.products]),
    }

    return reaction


reactant_1_smiles_dict = {
    "6-chloroquinoline": "C1=C(Cl)C=CC2=NC=CC=C12.CCC1=CC(=CC=C1)CC",
    "6-Bromoquinoline": "C1=C(Br)C=CC2=NC=CC=C12.CCC1=CC(=CC=C1)CC",
    "6-triflatequinoline": "C1C2C(=NC=CC=2)C=CC=1OS(C(F)(F)F)(=O)=O.CCC1=CC(=CC=C1)CC",
    "6-Iodoquinoline": "C1=C(I)C=CC2=NC=CC=C12.CCC1=CC(=CC=C1)CC",
    "6-quinoline-boronic acid hydrochloride": "C1C(B(O)O)=CC=C2N=CC=CC=12.Cl.O",
    "Potassium quinoline-6-trifluoroborate": "[B-](C1=CC2=C(C=C1)N=CC=C2)(F)(F)F.[K+].O",
    "6-Quinolineboronic acid pinacol ester": "B1(OC(C(O1)(C)C)(C)C)C2=CC3=C(C=C2)N=CC=C3.O",
}

reactant_2_smiles_dict = {
    "2a, Boronic Acid": "CC1=CC=C2C(C=NN2C3OCCCC3)=C1B(O)O",
    "2b, Boronic Ester": "CC1=CC=C2C(C=NN2C3OCCCC3)=C1B4OC(C)(C)C(C)(C)O4",
    "2c, Trifluoroborate": "CC1=CC=C2C(C=NN2C3OCCCC3)=C1[B-](F)(F)F.[K+]",
    "2d, Bromide": "CC1=CC=C2C(C=NN2C3OCCCC3)=C1Br",
}

catalyst_smiles_dict = {"Pd(OAc)2": "CC(=O)O~CC(=O)O~[Pd]"}

ligand_smiles_dict = {
    "P(tBu)3": "CC(C)(C)P(C(C)(C)C)C(C)(C)C",
    "P(Ph)3 ": "c3c(P(c1ccccc1)c2ccccc2)cccc3",
    "AmPhos": "CC(C)(C)P(C1=CC=C(C=C1)N(C)C)C(C)(C)C",
    "P(Cy)3": "C1(CCCCC1)P(C2CCCCC2)C3CCCCC3",
    "P(o-Tol)3": "CC1=CC=CC=C1P(C2=CC=CC=C2C)C3=CC=CC=C3C",
    "CataCXium A": "CCCCP(C12CC3CC(C1)CC(C3)C2)C45CC6CC(C4)CC(C6)C5",
    "SPhos": "COc1cccc(c1c2ccccc2P(C3CCCCC3)C4CCCCC4)OC",
    "dtbpf": "CC(C)(C)P(C1=CC=C[CH]1)C(C)(C)C.CC(C)(C)P(C1=CC=C[CH]1)C(C)(C)C.[Fe]",
    "XPhos": "P(c2ccccc2c1c(cc(cc1C(C)C)C(C)C)C(C)C)(C3CCCCC3)C4CCCCC4",
    "dppf": "C1=CC=C(C=C1)P([C-]2C=CC=C2)C3=CC=CC=C3.C1=CC=C(C=C1)P([C-]2C=CC=C2)C3=CC=CC=C3.[Fe+2]",
    "Xantphos": "O6c1c(cccc1P(c2ccccc2)c3ccccc3)C(c7cccc(P(c4ccccc4)c5ccccc5)c67)(C)C",
    "None": "",
    "nan": "",
}

reagent_1_smiles_dict = {
    "NaOH": "[OH-].[Na+]",
    "NaHCO3": "[Na+].OC([O-])=O",
    "CsF": "[F-].[Cs+]",
    "K3PO4": "[K+].[K+].[K+].[O-]P([O-])([O-])=O",
    "KOH": "[K+].[OH-]",
    "LiOtBu": "[Li+].[O-]C(C)(C)C",
    "Et3N": "CCN(CC)CC",
    "None": "",
    "nan": "",
}

solvent_1_smiles_dict = {
    "MeCN": "CC#N.O",
    "THF": "C1CCOC1.O",
    "DMF": "CN(C)C=O.O",
    "MeOH": "CO.O",
    "MeOH/H2O_V2 9:1": "CO.O",
    "THF_V2": "C1CCOC1.O",
}


def canonicalize_smiles(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        return Chem.MolToSmiles(mol)
    return ""


def make_reaction_smiles(row):
    precursors = f" {row['reactant1_SMILES']}.{row['reactant2_SMILES']}.{row['catalyst_SMILES']}.{row['ligand_SMILES']}.{row['reagent_SMILES']}.{row['solvent_SMILES']} "  # noqa
    product = "C1=C(C2=C(C)C=CC3N(C4OCCCC4)N=CC2=3)C=CC2=NC=CC=C12"
    #     print(precursors, product)
    can_precursors = Chem.MolToSmiles(
        Chem.MolFromSmiles(
            precursors.replace("...", ".")
            .replace("..", ".")
            .replace(" .", "")
            .replace(". ", "")
            .replace(" ", "")
        )
    )
    can_product = Chem.MolToSmiles(Chem.MolFromSmiles(product))

    return f"{can_precursors}>>{can_product}"


def add_molecules_and_rxn_smiles_to_df(df):
    df["reactant1_SMILES"] = df.Reactant_1_Name.apply(
        lambda molecule: canonicalize_smiles(reactant_1_smiles_dict[molecule])
    )
    df["reactant2_SMILES"] = df.Reactant_2_Name.apply(
        lambda molecule: canonicalize_smiles(reactant_2_smiles_dict[molecule])
    )
    df["catalyst_SMILES"] = df.Catalyst_1_Short_Hand.apply(
        lambda molecule: canonicalize_smiles(catalyst_smiles_dict[molecule])
    )
    df["ligand_SMILES"] = df.Ligand_Short_Hand.apply(
        lambda molecule: canonicalize_smiles(ligand_smiles_dict[molecule])
    )
    df["reagent_SMILES"] = df.Reagent_1_Short_Hand.apply(
        lambda molecule: canonicalize_smiles(reagent_1_smiles_dict[molecule])
    )
    df["solvent_SMILES"] = df.Solvent_1_Short_Hand.apply(
        lambda molecule: canonicalize_smiles(solvent_1_smiles_dict[molecule])
    )

    df["RXNSMILES"] = df.apply(lambda row: make_reaction_smiles(row), axis=1)

    equation = df["RXNSMILES"].apply(parse_any_reaction_smiles)
    results_from_rxn_equation = equation.apply(extract_reaction_info).tolist()
    df_results_from_rxn_equation = pd.DataFrame(results_from_rxn_equation)
    df = pd.concat([df, df_results_from_rxn_equation], axis=1)

    return df


def get_and_transform_data():
    # get raw data
    fn_data_original = "Richardson_and_Sach_input_data.csv"
    data = pd.read_excel(
        "https://github.com/reymond-group/drfp/raw/main/data/Suzuki-Miyaura/aap9112_Data_File_S1.xlsx"
    )
    data.to_csv(fn_data_original, index=False)
    df = pd.read_csv(fn_data_original, delimiter=",")

    # check if fields are the same
    fields_orig = df.columns.tolist()
    assert fields_orig == [
        "Reaction_No",
        "Reactant_1_Name",
        "Reactant_1_Short_Hand",
        "Reactant_1_eq",
        "Reactant_1_mmol",
        "Reactant_2_Name",
        "Reactant_2_eq",
        "Catalyst_1_Short_Hand",
        "Catalyst_1_eq",
        "Ligand_Short_Hand",
        "Ligand_eq",
        "Reagent_1_Short_Hand",
        "Reagent_1_eq",
        "Solvent_1_Short_Hand",
        "Product_Yield_PCT_Area_UV",
        "Product_Yield_Mass_Ion_Count",
    ]

    # data cleaning
    # make the columns we look up to str
    df[
        [
            "Reactant_1_Name",
            "Reactant_2_Name",
            "Catalyst_1_Short_Hand",
            "Ligand_Short_Hand",
            "Reagent_1_Short_Hand",
            "Solvent_1_Short_Hand",
        ]
    ] = df[
        [
            "Reactant_1_Name",
            "Reactant_2_Name",
            "Catalyst_1_Short_Hand",
            "Ligand_Short_Hand",
            "Reagent_1_Short_Hand",
            "Solvent_1_Short_Hand",
        ]
    ].astype(
        str
    )
    df = add_molecules_and_rxn_smiles_to_df(df)

    # save to csv
    fn_data_csv = "data_clean.csv"
    df.dropna(
        subset=[
            "RXNSMILES",
            "educt_string",
            "product_string",
            "missing_component",
            "masked_rxn_smiles",
        ],
        inplace=True,
    )
    df = df[df["masked_rxn_smiles"].str.contains("MASK")]
    df.rename(columns={"Product_Yield_PCT_Area_UV": "yield"}, inplace=True)
    assert len(df["RXNSMILES"].unique()) == len(df)
    print(len(df))
    df.to_csv(fn_data_csv, index=False)


if __name__ == "__main__":
    get_and_transform_data()
