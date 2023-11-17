import random
from copy import deepcopy

import pandas as pd
from rdkit import Chem  # 2022.9.5
from rdkit.Chem import rdChemReactions
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
    try:
        # First, either pick from reactants or products
        # then pick a random element from the chosen side
        # then replace it with a mask and return the new equation as
        # well as the element that was replaced
        # return equation, replaced_element
        equation = equation_string
        std = deepcopy(equation)
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
    except Exception as e:
        print(e)
        return {
            "educts": None,
            "products": None,
            "canonical_rxn_smiles": None,
            "masked_rxn_smiles": None,
            "missing_component": None,
            "rxn_smiles": equation_string,
            "educt_string": None,
            "product_string": None,
        }
    return reaction


def generate_buchwald_hartwig_rxns(df):
    """
    Converts the entries in the excel files to reaction SMILES.
    From: https://github.com/reymond-group/drfp/blob/main/scripts/encoding/encode_buchwald_hartwig_reactions.py
    and https://github.com/rxn4chemistry/rxn_yields/blob/master/rxn_yields/data.py
    """
    df = df.copy()
    fwd_template = "[F,Cl,Br,I]-[c;H0;D3;+0:1](:[c,n:2]):[c,n:3].[NH2;D1;+0:4]-[c:5]>>[c,n:2]:[c;H0;D3;+0:1](:[c,n:3])-[NH;D2;+0:4]-[c:5]"  # noqa
    methylaniline = "Cc1ccc(N)cc1"
    pd_catalyst = "O=S(=O)(O[Pd]1~[NH2]C2C=CC=CC=2C2C=CC=CC1=2)C(F)(F)F"
    methylaniline_mol = Chem.MolFromSmiles(methylaniline)
    rxn = rdChemReactions.ReactionFromSmarts(fwd_template)
    products = []

    for _i, row in df.iterrows():
        reacts = (Chem.MolFromSmiles(row["aryl_halide"]), methylaniline_mol)
        rxn_products = rxn.RunReactants(reacts)

        rxn_products_smiles = set([Chem.MolToSmiles(mol[0]) for mol in rxn_products])
        assert len(rxn_products_smiles) == 1
        products.append(list(rxn_products_smiles)[0])

    df["product"] = products
    rxns = []

    for _i, row in df.iterrows():
        reactants = Chem.MolToSmiles(
            Chem.MolFromSmiles(
                f"{row['aryl_halide']}.{methylaniline}.{pd_catalyst}.{row['ligand']}.{row['base']}.{row['additive']}"
            )
        )
        rxns.append(f"{reactants.replace('N~', '[NH2]')}>>{row['product']}")

    return rxns


def get_and_transform_data():
    # get raw data
    fn_data_original = "Dreher_and_Doyle_input_data.csv"
    data = pd.read_excel(
        "https://github.com/reymond-group/drfp/raw/main/data/Dreher_and_Doyle_input_data.xlsx"
    )
    data.to_csv(fn_data_original, index=False)

    # create dataframe
    df = pd.read_csv(
        fn_data_original,
        delimiter=",",
    )  # not necessary but ensure we can load the saved data

    # check if fields are the same
    fields_orig = df.columns.tolist()
    assert fields_orig == ["Ligand", "Additive", "Base", "Aryl halide", "Output"]

    # overwrite column names = fields
    fields_clean = ["ligand", "additive", "base", "aryl_halide", "yield"]
    df.columns = fields_clean

    # data cleaning
    reaction_SMILES = generate_buchwald_hartwig_rxns(df)  # compile reactions
    df.insert(4, "reaction_SMILES", reaction_SMILES)  # add reaction SMILES column

    df["RXNSMILES"] = df["reaction_SMILES"]
    equation = df["RXNSMILES"].apply(parse_any_reaction_smiles)
    results_from_rxn_equation = equation.apply(extract_reaction_info).tolist()
    df_results_from_rxn_equation = pd.DataFrame(results_from_rxn_equation)
    df = pd.concat([df, df_results_from_rxn_equation], axis=1)

    # save to csv
    fn_data_csv = "data_clean.csv"
    df.to_csv(fn_data_csv, index=False)


if __name__ == "__main__":
    get_and_transform_data()
