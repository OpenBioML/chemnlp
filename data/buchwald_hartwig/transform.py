import pandas as pd
import yaml
from rdkit import Chem # 2022.9.5
from rdkit.Chem import rdChemReactions

def generate_buchwald_hartwig_rxns(df):
    """
    Converts the entries in the excel files to reaction SMILES.
    From: https://github.com/reymond-group/drfp/blob/main/scripts/encoding/encode_buchwald_hartwig_reactions.py
    and https://github.com/rxn4chemistry/rxn_yields/blob/master/rxn_yields/data.py
    """
    df = df.copy()
    fwd_template = "[F,Cl,Br,I]-[c;H0;D3;+0:1](:[c,n:2]):[c,n:3].[NH2;D1;+0:4]-[c:5]>>[c,n:2]:[c;H0;D3;+0:1](:[c,n:3])-[NH;D2;+0:4]-[c:5]"
    methylaniline = "Cc1ccc(N)cc1"
    pd_catalyst = "O=S(=O)(O[Pd]1~[NH2]C2C=CC=CC=2C2C=CC=CC1=2)C(F)(F)F"
    methylaniline_mol = Chem.MolFromSmiles(methylaniline)
    rxn = rdChemReactions.ReactionFromSmarts(fwd_template)
    products = []

    for i, row in df.iterrows():
        reacts = (Chem.MolFromSmiles(row["aryl_halide"]), methylaniline_mol)
        rxn_products = rxn.RunReactants(reacts)

        rxn_products_smiles = set([Chem.MolToSmiles(mol[0]) for mol in rxn_products])
        assert len(rxn_products_smiles) == 1
        products.append(list(rxn_products_smiles)[0])

    df["product"] = products
    rxns = []

    for i, row in df.iterrows():
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
    data = pd.read_excel('https://github.com/reymond-group/drfp/raw/main/data/Dreher_and_Doyle_input_data.xlsx')
    data.to_csv(fn_data_original, index=False)
    
    # create dataframe
    df = pd.read_csv(
        fn_data_original,
        delimiter=",",
    )  # not necessary but ensure we can load the saved data

    # check if fields are the same
    fields_orig = df.columns.tolist()
    assert fields_orig == [
        'Ligand',
        'Additive', 
        'Base', 
        'Aryl halide', 
        'Output'
    ]

    # overwrite column names = fields
    fields_clean = [
    "ligand",
    "additive",
    "base",
    "aryl_halide",
    'yield'
    ]
    df.columns = fields_clean

    # data cleaning
    reaction_SMILES = generate_buchwald_hartwig_rxns(df) # compile reactions
    df.insert(4, 'reaction_SMILES', reaction_SMILES) # add reaction SMILES column

    assert not df.duplicated().sum()

    # save to csv
    fn_data_csv = "data_clean.csv"
    df.to_csv(fn_data_csv, index=False)

    # create meta yaml
    meta = {
    "name": "buchwald_hartwig_doyle",  # unique identifier, we will also use this for directory names
    "description": """High-throughput experimentation palladium-catalyzed Buchwald Hardwig C-N cross-coupling data set with yields.""",
    "targets": [
        {
            "id": "yield",  # name of the column in a tabular dataset
            "description": "Reaction yields analyzed by UPLC",  # description of what this column means
            "units": "%",  # units of the values in this column (leave empty if unitless)
            "type": "continuous",  # can be "categorical", "ordinal", "continuous"
            "names": [  # names for the property (to sample from for building the prompts)
                "Reaction yield",
                "yield",
            ],
        },
    ],
    "identifiers": [
        {
            "id": "reaction_SMILES",  # column name
            "type": "RXN-SMILES",  # can be "SMILES", "SELFIES", "IUPAC", "Other"
            "description": "RXN-SMILES",  # description (optional, except for "Other")
        },
        {
            "id": "ligand",  # column name
            "type": "SMILES",  # can be "SMILES", "SELFIES", "IUPAC", "Other"
            "description": "ligand SMILES",  # description (optional, except for "Other")
        },
        {
            "id": "additive",  # column name
            "type": "SMILES",  # can be "SMILES", "SELFIES", "IUPAC", "Other"
            "description": "additive SMILES",  # description (optional, except for "Other")
        },
        {
            "id": "base",  # column name
            "type": "SMILES",  # can be "SMILES", "SELFIES", "IUPAC", "Other"
            "description": "base SMILES",  # description (optional, except for "Other")
        },
        {
            "id": "aryl_halide",  # column name
            "type": "SMILES",  # can be "SMILES", "SELFIES", "IUPAC", "Other"
            "description": "aryl halide SMILES",  # description (optional, except for "Other")
        },
    ],
    "license": "MIT",  # license under which the original dataset was published
    "links": [  # list of relevant links (original dataset, other uses, etc.)
        {
            "url": "https://doi.org/10.1126/science.aar5169",
            "description": "corresponding publication",
        },
        {
            "url": "https://www.sciencedirect.com/science/article/pii/S2451929420300851",
            "description": "publication with data processing",
        },
        {
            "url": "https://github.com/rxn4chemistry/rxn_yields/blob/master/rxn_yields/data.py",
            "description": "preprocessing",
        },
        {
            "url": "https://github.com/reymond-group/drfp/tree/main/data",
            "description": "dataset",
        }
    ],
        "num_points": len(df),  # number of datapoints in this dataset
        "url": "https://doi.org/10.1126/science.aar5169",
        "bibtex": [
            """@article{ahneman2018predicting,
title={Predicting reaction performance in C--N cross-coupling using machine learning},
author={Ahneman, Derek T and Estrada, Jes{\'u}s G and Lin, Shishi and Dreher, Spencer D and Doyle, Abigail G},
journal={Science},
volume={360},
number={6385},
pages={186--190},
year={2018},
publisher={American Association for the Advancement of Science},
}""",
    ],
    }

    def str_presenter(dumper, data):
        """configures yaml for dumping multiline strings
        Ref: https://stackoverflow.com/questions/8640959/how-can-i-control-what-scalar-form-pyyaml-uses-for-my-data
        """
        if data.count("\n") > 0:  # check for multiline string
            return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
        return dumper.represent_scalar("tag:yaml.org,2002:str", data)

    yaml.add_representer(str, str_presenter)
    yaml.representer.SafeRepresenter.add_representer(
        str, str_presenter
    )  # to use with safe_dum
    fn_meta = "meta.yaml"
    with open(fn_meta, "w") as f:
        yaml.dump(meta, f, sort_keys=False)

    print(f"Finished processing {meta['name']} dataset!")


if __name__ == "__main__":
    get_and_transform_data()
