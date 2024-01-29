import os

import pandas as pd
import yaml
from rdkit import Chem

# create meta yaml
meta_template = {
    "name": None,
    "description": "Molecule representation translation task data.",
    "identifiers": None,
    "targets": None,
    "benchmarks": [
        {
            "name": None,
            "link": None,
            "split_column": "split",  # name of the column that contains the split information
        },
    ],
    "license": "Please see source material.",
    "links": [
        {
            "url": None,
            "description": None,
        },
    ],
    "num_points": None,
    "bibtex": ["Please see source material."],
    "templates": [
        "The molecule with the {IDENTIFIER__names__noun} {#representation of |!}{IDENTIFIER#} can also be represented with the {TARGET__names__noun} {#representation |!}{TARGET#}.",  # noqa: E501
        "The molecule with the {TARGET__names__noun} {#representation of |!}{TARGET#} can also be represented with the {IDENTIFIER__names__noun} {#representation |!}{IDENTIFIER#}.",  # noqa: E501
        # Instruction tuning text templates
        """Task: Please {#create|generate!} a molecule representation based on {#the input molecule representation and |!}the description.
Description: {#Generate|Create!} the {TARGET__names__noun} from the {IDENTIFIER__names__noun}.
{#Molecule |!}{IDENTIFIER__names__noun}: {IDENTIFIER#}
Constraint: Even if you are {#uncertain|not sure!}, you must answer with a representation without using any {#other|additional!} words.
Result: {TARGET#}""",  # noqa: E501
        """Task: Please {#create|generate!} a molecule representation based on {#the input molecule representation and |!}the description.
Description: {#Generate|Create!} the {IDENTIFIER__names__noun} from the {TARGET__names__noun}.
{#Molecule |!}{TARGET__names__noun}: {TARGET#}
Constraint: Even if you are {#uncertain|not sure!}, you must answer with a representation without using any {#other|additional!} words.
Result: {IDENTIFIER#}""",  # noqa: E501
        # Conversational text templates
        """User: Can you {#tell me|create|generate!} the {TARGET__names__noun} of the molecule with the {IDENTIFIER__names__noun} {IDENTIFIER#}?
Assistant: {#Yes|Of course|Sure|Yes, I'm happy to help!}, this molecule has a {TARGET__names__noun} of {TARGET#}.""",  # noqa: E501
        """User: Can you {#tell me|create|generate!} the {IDENTIFIER__names__noun} of the molecule with the {TARGET__names__noun} {TARGET#}?
Assistant: {#Yes|Of course|Sure|Yes, I'm happy to help!}, this molecule has a {IDENTIFIER__names__noun} of {IDENTIFIER#}.""",  # noqa: E501
        # Benchmarking text templates
        "The molecule with the {IDENTIFIER__names__noun} {#representation of |!}{IDENTIFIER#} can also be represented with the {TARGET__names__noun}{# representation|!}:<EOI> {TARGET#}.",  # noqa: E501
        "The molecule with the {TARGET__names__noun} {#representation of |!}{TARGET#} can also be represented with the {IDENTIFIER__names__noun}{# representation|!}:<EOI> {IDENTIFIER#}.",  # noqa: E501
        """Task: Please {#create|generate!} a molecule representation based on {#the input molecule representation and |!}the description.
Description: {#Generate|Create!} the {TARGET__names__noun} from the {IDENTIFIER__names__noun}.
{#Molecule |!}{IDENTIFIER__names__noun}: {IDENTIFIER#}
Constraint: Even if you are {#uncertain|not sure!}, you must answer with a representation without using any {#other|additional!} words.
Result:<EOI> {TARGET#}""",  # noqa: E501
        """Task: Please {#create|generate!} a molecule representation based on {#the input molecule representation and |!}the description.
Description: {#Generate|Create!} the {IDENTIFIER__names__noun} from the {TARGET__names__noun}.
{#Molecule |!}{TARGET__names__noun}: {TARGET#}
Constraint: Even if you are {#uncertain|not sure!}, you must answer with a representation without using any {#other|additional!} words.
Result:<EOI> {IDENTIFIER#}""",  # noqa: E501
    ],
}


def str_presenter(dumper, data: str):
    """configures yaml for dumping multiline strings
    Ref: https://stackoverflow.com/questions/8640959/how-can-i-control-what-scalar-form-pyyaml-uses-for-my-data
    """
    if data.count("\n") > 0:  # check for multiline string
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


def smiles_with_hydrogens(smiles):
    """Add hydrogens to smiles string"""

    try:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        return Chem.MolToSmiles(mol)
    except:
        return pd.NA


def get_and_transform_data():
    path_base = os.getcwd()
    path_csv = (
        "/".join(path_base.split("/")[:-2])
        + "/text_sampling/extend_tabular_processed.csv"
    )

    df = pd.read_csv(
        path_csv,
        delimiter=",",
    )
    #df["SMILES_with_H"] = df["SMILES"].apply(smiles_with_hydrogens)

    #if "split" in df.columns:
    assert df.columns[-1] == "split", "Split column needs to be the last column."
    col_len = len(df.columns) - 1
    #else:
    #    print(
    #        "CAUTION: No split information found, maybe you need to rerun the train_test_split.py script over extend_tabular_processed.csv?"  # noqa: E501
    #    )
    #    col_len = len(df.columns)

    for i in range(col_len):
        for j in range(i + 1, col_len):
            subset_cols = [df.columns[i], df.columns[j]]
            print(subset_cols)
            dataset_name = "mol_repr_transl_" + "_".join(subset_cols)
            dataset_name = dataset_name.lower()

            path_export = "/".join(path_base.split("/")[:-1]) + "/" + dataset_name
            os.makedirs(path_export, exist_ok=True)

            # df export
            col_suffix = "_text"  # to exclude from other preprocessing steps
            #if "split" in df.columns:
            df_subset = df[subset_cols + ["split"]].dropna()
            #elif "split" not in df.columns:
            #    df_subset = df[subset_cols].dropna()
            df_subset.columns = [
                x + col_suffix if x != "split" else x for x in subset_cols
            ] + ["split"]
            df_subset.to_csv(path_export + "/data_clean.csv", index=False)

            # meta yaml export
            names = {
                "SMILES": "SMILES",
                "selfies": "SELFIES",
                "deepsmiles": "DeepSMILES",
                "canonical": "canonical SMILES",
                "inchi": "InChI string",
                "iupac_name": "IUPAC name",
                "SMILES_with_H": "SMILES with hydrogens",
            }
            meta_copy = meta_template.copy()
            meta_copy["name"] = dataset_name
            meta_copy["num_points"] = len(df_subset)
            meta_copy["identifiers"] = [
                {
                    "id": subset_cols[0] + col_suffix,
                    "description": subset_cols[0],
                    "type": "Other",
                    "names": [{"noun": names[subset_cols[0]]}],
                }
            ]
            meta_copy["targets"] = [
                {
                    "id": subset_cols[1] + col_suffix,
                    "description": subset_cols[1],
                    "type": "Other",
                    "names": [{"noun": names[subset_cols[1]]}],
                }
            ]

            meta_copy["templates"] = [
                t.replace("TARGET", subset_cols[0] + col_suffix).replace(
                    "IDENTIFIER", subset_cols[1] + col_suffix
                )
                for t in meta_copy["templates"]
            ]

            yaml.add_representer(str, str_presenter)
            yaml.representer.SafeRepresenter.add_representer(str, str_presenter)
            fn_meta = path_export + "/meta.yaml"
            with open(fn_meta, "w") as f:
                yaml.dump(meta_copy, f, sort_keys=False)

            print(dataset_name)


if __name__ == "__main__":
    get_and_transform_data()
