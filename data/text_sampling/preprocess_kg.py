import glob
import os
import shutil

import pandas as pd
import yaml

templates = {
    "chebi_chebi": [
        # todo: needs to be checked in detail
    ],
    "compound_chebi": [
        """The {node1_type#} SMILES {node1_smiles#} {rel1_type#} {node2_name#}.""",
    ],
    "compound_chebi_chebi": [
        """The {node1_type#} SMILES {node1_smiles#} {rel1_type#} {node2_name#} and {rel2_type#} {node3_name#}.""",  # noqa E501
    ],
    "compound_protein": [
        """The {node1_type#} SMILES {node1_smiles#} {rel1_type#} the {node2_type#} {node2_protein_names#}.""",
        """The {node2_type#} {node2_protein_names#} is targeted by the drugs SMILES {node1_smiles#}.""",
        """User: Can you give me an example for a {node1_type#} SMILES that {rel1_type#} the {node2_type#} {node2_protein_names#}?
Assistant: Yes, The {node1_type#} SMILES {node1_smiles#} {rel1_type#} the {node2_type#} {node2_protein_names#}.""",  # noqa E501
    ],
    "compound_protein_compound": [
        """The {node1_type#} SMILES {node1_smiles#} {rel1_type#} the {node2_type#} {node2_protein_names#} and {rel2_type#} the {node3_type#} {node3_name#}.""",  # noqa E501
        """The {node2_type#} {node2_protein_names#} is targeted by the drugs SMILES {node1_smiles#} and {node3_name#}.""",  # noqa E501
        """User: Can you give me an example for a {node1_type#} SMILES that {rel1_type#} the {node2_type#} {node2_protein_names#}?
Assistant: Yes, The {node1_type#} SMILES {node1_smiles#} {rel1_type#} the {node2_type#} {node2_protein_names#}.
User: Can you tell me another {node1_type#} SMILES that {rel1_type#} the {node2_type#} {node2_protein_names#}?
Assistant: Of course, the {node1_type#} SMILES {node1_smiles#} {rel1_type#} the {node3_type#} SMILES {node3_smiles#}.""",  # noqa E501
    ],
    "compound_protein_disease": [
        """The {node1_type#} {node1_smiles#} {rel1_type#} the {node2_type#} {node2_protein_names#} which {rel2_type#} the {node3_type#} {node3_name#}.""",  # noqa E501
        """The {node1_type#} {node1_smiles#} {rel1_type#} the {node2_type#} {node2_protein_names#}. The {node2_type#} {node2_protein_names#} {rel2_type#} the {node3_type#} {node3_name#}.""",  # noqa E501
        """User: Give me an example for a protein that is targeted by the {node1_type#} {node1_smiles#}?
Assistant: Sure, the {node2_type#} {node2_protein_names#} is targeted by the {node1_type#} {node1_smiles#}.
User: Can you tell me which disease the {node2_type#} {node2_protein_names#} {rel2_type#}?
Assistant: The {node2_type#} {node2_protein_names#} {rel2_type#} the {node3_name#} {node3_type#}.""",  # noqa E501
    ],
    "compound_protein_domain": [
        """{node1_smiles#} {rel1_type#} the {node2_type#} {node2_protein_names#} which {rel2_type#} a {node3_name#}.""",  # noqa E501
        """The {node1_type#} {node1_smiles#} {rel1_type#} the {node2_type#} {node2_protein_names#} which {rel2_type#} a {node3_name#}.""",  # noqa E501
        """User: Can you give me an example for a protein that binds the {node1_type#} {node1_smiles#}?
Assistant: Yes, the {node1_type#} {node1_smiles#} {rel1_type#} for example the {node2_type#} {node2_protein_names#}.
User: Can you tell me a domain of the {node2_type#} {node2_protein_names#}?
Assistant: The {node2_type#} {node2_protein_names#} {rel2_type#} a {node3_name#}.""",  # noqa E501
    ],
    "compound_protein_ec_number": [
        """The {node1_type#} {node1_smiles#} {rel1_type#} the {node2_type#} {node2_protein_names#} and {rel2_type#} the {node3_name#} (EC {node3_id#}).""",  # noqa E501
        """The {node1_type#} {node1_smiles#} {rel1_type#} the {node2_type#} {node2_protein_names#}. Furthermore, the {node1_type#} {node1_smiles#} {rel2_type#} the {node3_name#} (EC {node3_id#}).""",  # noqa E501
        """User: Can you give me an example for a protein that binds the {node1_type#} {node1_smiles#}?
Assistant: Of course, the {node1_type#} {node1_smiles#} {rel1_type#} the {node2_type#} {node2_protein_names#}.
User: Can you tell me which enzyme the {node2_type#} {node2_protein_names#} {rel2_type#}?
Assistant: The {node2_type#} {node2_protein_names#} {rel2_type#} a {node3_name#} (EC {node3_id#}).""",  # noqa E501
    ],
    # todo: There are some entries that have the EC number under node3_name and node3_id
    # and this is not handled yet properly.
    "compound_protein_go_term": [
        """The {node1_type#} {node1_smiles#} {rel1_type#} the {node2_type#} {node2_protein_names#} which {rel2_type#} the {node3_name#}.""",  # noqa E501
        """The {node1_type#} {node1_smiles#} {rel1_type#} the {node2_type#} {node2_protein_names#}. The {node2_type#} {node2_protein_names#} {rel2_type#} the {node3_name#}.""",  # noqa E501
    ],
    "compound_protein_hpo": [
        """The {node1_type#} {node1_smiles#} {rel1_type#} the {node2_type#} {node2_protein_names#} and {rel2_type#} {node3_name#}.""",  # noqa E501
        """The {node1_type#} {node1_smiles#} {rel1_type#} the {node2_type#} {node2_protein_names#}. The {node2_type#} {node2_protein_names#} {rel2_type#} {node3_name#}.""",  # noqa E501
        """The {node1_type#} {node1_smiles#} {rel1_type#} the {node2_type#} {node2_protein_names#}. The {node2_type#} {node2_protein_names#} {rel2_type#} the human phenotype represented by {node3_name#}.""",  # noqa E501
        """The {node1_type#} {node1_smiles#} {rel1_type#} the {node2_type#} {node2_protein_names#}. The {node2_type#} {rel2_type#} the {node3_name#}.""",  # noqa E501
    ],
    "compound_protein_pathway": [
        """The {node1_type#} {node1_smiles#} {rel1_type#} the {node2_type#} {node2_protein_names#} which {rel2_type#} the {node3_name#}.""",  # noqa E501
        """The {node1_type#} {node1_smiles#} {rel1_type#} the {node2_type#} {node2_protein_names#}. The {node2_type#} {node2_protein_names#} {rel2_type#} the {node3_name#}.""",  # noqa E501
    ],
    "compound_protein_protein": [
        """The {node1_type#} {node1_smiles#} {rel1_type#} the {node2_type#} {node2_protein_names#} which {rel2_type#} the {node3_type#} {node3_protein_names#}.""",  # noqa E501
        """The {node2_type#} {node2_protein_names#} is targeted by {node1_smiles#}. The {node2_type#} {node2_protein_names#} {rel2_type#} {node3_protein_names#}.""",  # noqa E501
        """User: Can you give me an example for a protein that binds the {node1_type#} {node1_smiles#}?
Assistant: The {node1_type#} {node1_smiles#} {rel1_type#} for example the {node2_type#} {node2_protein_names#}.
User: Can you tell me a {node3_type#} that {rel2_type#} {node2_type#} {node2_protein_names#}?
Assistant: Yes, the {node2_type#} {node2_protein_names#} {rel2_type#} {node3_protein_names#}.""",  # noqa E501
    ],
    "drug_chebi": [
        """The {node1_type#} {node1_name#|node1_smiles#} {rel1_type#} {node2_name#}.""",
    ],
    "drug_chebi_chebi": [
        """The {node1_type#} {node1_name#|node1_smiles#} {rel1_type#} {node2_name#} and {rel2_type#} {node3_name#}.""",  # noqa E501
    ],
    "drug_disease_pathway": [
        """The {node1_type#} {node1_name#|node1_smiles#} is indicated for the {node2_name#} {node2_type#} and {rel2_type#} the {node3_name#} {node3_type#}.""",  # noqa E501
    ],
    "drug_protein": [
        """The {node1_type#} {node1_name#|node1_smiles#} {rel1_type#} the {node2_type#} {node2_protein_names#}.""",
        """User: Give me an example for a protein that is targeted by the {node1_type#} {node1_name#|node1_smiles#}?
Assistant: Sure, the {node2_type#} {node2_protein_names#} is targeted by the {node1_type#} {node1_name#|node1_smiles#}.""",  # noqa E501
    ],
    "drug_protein_disease": [
        """The {node1_type#} {node1_name#|node1_smiles#} {rel1_type#} the {node2_type#} {node2_protein_names#} which {rel2_type#} the {node3_type#} {node3_name#}.""",  # noqa E501
        """The {node1_type#} {node1_name#|node1_smiles#} {rel1_type#} the {node2_type#} {node2_protein_names#}. The {node2_type#} {node2_protein_names#} {rel2_type#} the {node3_type#} {node3_name#}.""",  # noqa E501
        """User: Give me an example for a protein that is targeted by the {node1_type#} {node1_name#|node1_smiles#}?
Assistant: Sure, the {node2_type#} {node2_protein_names#} is targeted by the {node1_type#} {node1_name#|node1_smiles#}.
User: Can you tell me which disease the {node2_type#} {node2_protein_names#} {rel2_type#}?
Assistant: The {node2_type#} {node2_protein_names#} {rel2_type#} the {node3_name#} {node3_type#}.""",  # noqa E501
    ],
    "drug_protein_domain": [
        """{node1_name#|node1_smiles#} {rel1_type#} the {node2_type#} {node2_protein_names#} which {rel2_type#} a {node3_name#}.""",  # noqa E501
        """The {node1_type#} {node1_name#|node1_smiles#} {rel1_type#} the {node2_type#} {node2_protein_names#} which {rel2_type#} a {node3_name#}.""",  # noqa E501
        """User: Can you give me an example for a protein that binds the {node1_type#} {node1_name#|node1_smiles#}?
Assistant: Yes, the {node1_type#} {node1_name#|node1_smiles#} {rel1_type#} for example the {node2_type#} {node2_protein_names#}.
User: Can you tell me a domain of the {node2_type#} {node2_protein_names#}?
Assistant: The {node2_type#} {node2_protein_names#} {rel2_type#} a {node3_name#}.""",  # noqa E501
    ],
    "drug_protein_drug": [
        """The {node1_type#} {node1_name#|node1_smiles#} {rel1_type#} the {node2_type#} {node2_protein_names#} and {rel2_type#} the {node3_type#} {node3_name#|node3_smiles#}.""",  # noqa E501
        """The {node2_type#} {node2_protein_names#} is targeted by the drugs {node1_name#|node1_smiles#} and {node3_name#|node3_smiles#}.""",  # noqa E501
        """User: Can you give me an example for a {node1_type#} that {rel1_type#} the {node2_type#} {node2_protein_names#}?
Assistant: Yes, The {node1_type#} {node1_name#|node1_smiles#} {rel1_type#} the {node2_type#} {node2_protein_names#}.
User: Can you tell me another {node1_type#} that {rel1_type#} the {node2_type#} {node2_protein_names#}?
Assistant: Of course, the {node1_type#} {node1_name#|node1_smiles#} {rel1_type#} the {node3_type#} {node3_name#|node3_smiles#}.""",  # noqa E501
    ],
    "drug_protein_ec_number": [
        """The {node1_type#} {node1_name#|node1_smiles#} {rel1_type#} the {node2_type#} {node2_name#} and {rel2_type#} the {node3_name#} (EC {node3_id#}).""",  # noqa E501
        """The {node1_type#} {node1_name#|node1_smiles#} {rel1_type#} the {node2_type#} {node2_name#}. Furthermore, the {node1_type#} {node1_name#|node1_smiles#} {rel2_type#} the {node3_name#} (EC {node3_id#}).""",  # noqa E501
        """User: Can you give me an example for a protein that binds the {node1_type#} {node1_name#|node1_smiles#}?
Assistant: Of course, the {node1_type#} {node1_name#|node1_smiles#} {rel1_type#} the {node2_type#} {node2_name#}.
User: Can you tell me which enzyme the {node2_type#} {node2_name#} {rel2_type#}?
Assistant: The {node2_type#} {node2_name#} {rel2_type#} a {node3_name#} (EC {node3_id#}).""",  # noqa E501
    ],
    "drug_protein_go_term": [
        """The {node1_type#} {node1_name#|node1_smiles#} {rel1_type#} the {node2_type#} {node2_name#} which {rel2_type#} the {node3_name#}.""",  # noqa E501
        """The {node1_type#} {node1_name#|node1_smiles#} {rel1_type#} the {node2_type#} {node2_name#}. The {node2_type#} {node2_name#} {rel2_type#} the {node3_name#}.""",  # noqa E501
    ],
    "drug_protein_hpo": [
        """The {node1_type#} {node1_name#|node1_smiles#} {rel1_type#} the {node2_type#} {node2_protein_names#} and {rel2_type#} {node3_name#}.""",  # noqa E501
        """The {node1_type#} {node1_name#|node1_smiles#} {rel1_type#} the {node2_type#} {node2_protein_names#}. The {node2_type#} {node2_protein_names#} {rel2_type#} {node3_name#}.""",  # noqa E501
        """The {node1_type#} {node1_name#|node1_smiles#} {rel1_type#} the {node2_type#} {node2_protein_names#}. The {node2_type#} {node2_protein_names#} {rel2_type#} the human phenotype represented by {node3_name#}.""",  # noqa E501
        """The {node1_type#} {node1_name#|node1_smiles#} {rel1_type#} the {node2_type#} {node2_protein_names#}. The {node2_type#} {rel2_type#} the {node3_name#}.""",  # noqa E501
    ],
    "drug_protein_pathway": [
        """The {node1_type#} {node1_name#|node1_smiles#} {rel1_type#} the {node2_type#} {node2_protein_names#} which {rel2_type#} the {node3_name#}.""",  # noqa E501
        """The {node1_type#} {node1_name#|node1_smiles#} {rel1_type#} the {node2_type#} {node2_protein_names#}. The {node2_type#} {node2_protein_names#} {rel2_type#} the {node3_name#}.""",  # noqa E501
    ],
    "drug_protein_protein": [
        """The {node1_type#} {node1_name#|node1_smiles#} {rel1_type#} the {node2_type#} {node2_protein_names#} which {rel2_type#} the {node3_type#} {node3_name#}.""",  # noqa E501
        """The {node2_type#} {node2_protein_names#} is targeted by {node1_name#|node1_smiles#}. The {node2_type#} {node2_protein_names#} {rel2_type#} {node3_name#}.""",  # noqa E501
        """User: Can you give me an example for a protein that binds the {node1_type#} {node1_name#|node1_smiles#}?
Assistant: The {node1_type#} {node1_name#|node1_smiles#} {rel1_type#} for example the {node2_type#} {node2_protein_names#}.
User: Can you tell me a {node3_type#} that {rel2_type#} {node2_type#} {node2_protein_names#}?
Assistant: Yes, the {node2_type#} {node2_name#} {rel2_type#} {node3_name#}.""",  # noqa E501
    ],
}

# create meta yaml
meta_template = {
    "name": None,
    "description": "Knowledgegraph data samples.",
    "targets": None,
    "identifiers": None,
    "license": "VERIFY WITH AUTHORS",
    "links": [
        {
            "url": "https://crossbar.kansil.org",
            "description": "original knowledge graph link",
        }
    ],
    "num_points": None,
    "bibtex": [
        """@article{10.1093/nar/gkab543,
author = {Doğan, Tunca and Atas, Heval and Joshi, Vishal and Atakan, Ahmet and Rifaioglu, Ahmet Sureyya and Nalbat, Esra and Nightingale, Andrew and Saidi, Rabie and Volynkin, Vladimir and Zellner, Hermann and Cetin-Atalay, Rengul and Martin, Maria and Atalay, Volkan},
title = "{CROssBAR: comprehensive resource of biomedical relations with knowledge graph representations}",
journal = {Nucleic Acids Research},
volume = {49},
number = {16},
pages = {e96-e96},
year = {2021},
month = {06},
issn = {0305-1048},
doi = {10.1093/nar/gkab543},
url = {https://doi.org/10.1093/nar/gkab543},
}"""  # noqa E501
    ],
    "templates": None,
}


recode = {
    "Drug": "drug",
    "Protein": "protein",
    "Pathway": "pathway",
    "has_functional_parent": "has the functional parent",
    "has_parent_hydride": "has the parent hydride",
    "has_part": "has the part",
    "has_role": "has the role",
    "interacts_with": "interacts with",
    "involved_in": "is involved in",
    "is_a": "is a",
    "is_associated_with": "is associated with",
    "is_conjugate_acid_of": "is the conjugate acid of",
    "is_conjugate_base_of": "is the conjugate based of",
    "is_enantiomer_of": "is a enantiomer of",
    "is_involved_in": "is involved in",
    "is_ortholog_to": "is ortholog to",
    "is_related_to": "is related to",
    "is_substituent_group_from": "is the substituent group from",
    "is_tautomer_of": "is a tautomer of",
    "located_in": "is located in",
    "Drug metabolism": "drug metabolism",
    "Calcium": "calcium",
    "Rectal fistula": "rectal fistula",
}


def create_yamls(dirs):
    for path in dirs:
        df = pd.read_csv(path + "data_original.csv", nrows=0)  # only get columns
        cols = df.columns.tolist()

        dataset_name = path.split("/")[-2]
        meta_copy = meta_template.copy()
        meta_copy["name"] = dataset_name
        meta_copy["num_points"] = len(df)
        meta_copy["identifiers"] = [
            {"id": c, "description": c, "type": "Other"} for c in cols if "1" in c
        ]
        meta_copy["targets"] = [
            {
                "id": c,
                "description": c,
                "type": "Other",
                "units": c,
                "names": [{"noun": c}],
            }
            for c in cols
            if "1" not in c
        ]
        meta_copy["templates"] = templates[dataset_name]

        fn_meta = path + "meta.yaml"
        with open(fn_meta, "w") as f:
            yaml.dump(meta_copy, f, sort_keys=False)


def format_kg_df(df):
    df.drop_duplicates(inplace=True)

    # recode based on lookup dict
    df.replace(recode, inplace=True)

    # relations to lower caser
    df.node2_type = df.node2_type.apply(lambda x: x.lower())
    if "node3_type" in df.columns:
        df.node3_type = df.node3_type.apply(lambda x: x.lower())

    # for drug_protein_drug
    if "rel2_type" in df.columns:
        df.rel2_type.replace({"targets": "which is also targeted by"}, inplace=True)

    if "node3_type" in df.columns:
        if df.node3_type.unique().tolist() == ["Domain"]:
            df.node3_name = df.node3_name.apply(lambda x: x.replace(",", ""))

            def check_and_add_domain(x):
                if "domain" in x.lower():
                    return x
                else:
                    return x + " domain"

            df.node3_name = df.node3_name.apply(check_and_add_domain)

    # for drug_chebi_chebi
    if "node3_type" in df.columns:
        if (df.node2_type.unique().tolist() == ["chebi"]) and (
            df.node3_type.unique().tolist() == ["chebi"]
        ):
            df.drop(
                labels=df[
                    df.node1_name.apply(lambda x: x.lower())
                    == df.node2_name.apply(lambda x: x.lower())
                ].index.tolist(),
                inplace=True,
            )

    return df


def str_presenter(dumper, data):
    """configures yaml for dumping multiline strings
    Ref: https://stackoverflow.com/questions/8640959/how-can-i-control-what-scalar-form-pyyaml-uses-for-my-data
    """
    if data.count("\n") > 0:  # check for multiline string
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


def preprocess_kg_data(path_data_dir):
    # create separate dirs, move csv files there, and save cleaned data
    fns_data_raw = sorted(glob.glob(path_data_dir + "*csv"))

    for fn in fns_data_raw:
        if fn.endswith("_mappings.csv") or fn.endswith("_full.csv"):
            continue
        dir_new = fn.split("/")[-1].split(".csv")[0].lower()
        path_new = path_data_dir + dir_new
        print(fn, path_new)
        os.makedirs(path_new, exist_ok=True)
        path_data_original = path_new + "/data_original.csv"
        shutil.copyfile(fn, path_data_original)
        df = pd.read_csv(path_data_original)
        df = format_kg_df(df)
        path_data_clean = path_data_original.replace("_original.csv", "_clean.csv")
        df.to_csv(path_data_clean, index=False)

    # set up yamls
    dirs = sorted(glob.glob(path_data_dir + "*/", recursive=False))

    yaml.add_representer(str, str_presenter)
    yaml.representer.SafeRepresenter.add_representer(str, str_presenter)

    create_yamls(dirs)


if __name__ == "__main__":
    path_data_dir = __file__.replace("text_sampling/preprocess_kg.py", "kg/")
    preprocess_kg_data(path_data_dir)
