import pandas as pd
import yaml


def get_and_transform_data():
    target_folder = "papyrus_protein_targets"
    data_path = "https://data.4tu.nl/file/ca10bf7d-f508-4d54-9c9a-5a9e9c1adef9/e5863d58-c613-418b-8393-012eb6c9a04a"
    fn_data_original = "data_original.csv"
    df = pd.read_csv(data_path, compression="gzip", sep="\t")
    df.to_csv(fn_data_original, index=None)
    df = df.fillna("unkown")
    df["organism_common_name"] = df["Organism"].apply(
        lambda s: s[s.index("(") + 1 : -1] if "(" in s else "unknown"
    )
    df["target_id_without_mutation"] = df["target_id"].apply(
        lambda s: s.split("_")[0] if "_" in s else s
    )
    df["UniProtID"] = df["UniProtID"].apply(
        lambda s: s.split("_")[0] if "_" in s else s
    )
    df = df.drop_duplicates(subset="target_id")
    fields_orig = df.columns.tolist()
    assert fields_orig == [
        "target_id",
        "HGNC_symbol",
        "UniProtID",
        "Status",
        "Organism",
        "Classification",
        "Length",
        "Sequence",
        "organism_common_name",
        "target_id_without_mutation",
    ]

    fields_clean = [
        "target_id",
        "target_id_without_mutation",
        "HGNC_symbol",
        "UniProtID",
        "Status",
        "Organism",
        "organism_common_name",
        "Classification",
        "Length",
        "Sequence",
    ]
    df = df[fields_clean]
    fields_clean = [
        "target_id",
        "target_id_without_mutation",
        "HGNC_symbol",
        "UniProtID",
        "Status",
        "Organism",
        "organism_common_name",
        "Classification",
        "seq_length",
        "Sequence",
    ]

    assert fields_orig != fields_clean
    assert not df.duplicated().sum()
    fn_data_csv = "data_clean.csv"
    df.to_csv(fn_data_csv, index=False)

    # create meta yaml
    meta = {
        "name": f"{target_folder}",  # unique identifier, we will also use this for directory names
        "description": """Papyrus is an aggregated dataset of small molecule bioactivities. File contains data about proteins (e.g. sequence, organism,classification).""",  # noqa: E501
        "targets": [
            {
                "id": "Organism",  # name of the column in a tabular dataset
                "description": "Organism of the protein",  # description of what this column means
                "units": "",  # units of the values in this column (leave empty if unitless)
                "type": "text",  # can be "categorical", "ordinal", "continuous"
                "names": [  # names for the property (to sample from for building the prompts).
                    {"noun": "The organism that the protein extracted from"},
                    {"noun": "For which organism protein related to"},
                    {"noun": "living that the protein extract from"},
                ],
                "uris": [
                    "http://purl.bioontology.org/ontology/CCON",  # organism
                ],
            },
            {
                "id": "organism_common_name",  # name of the column in a tabular dataset
                "description": "common name of the organism that protein extract from.",
                "units": "",  # units of the values in this column (leave empty if unitless)
                "type": "text",  # can be "categorical", "ordinal", "continuous"
                "names": [  # names for the property (to sample from for building the prompts).
                    {
                        "noun": "common name of the organism that the protein extracted from"
                    },
                    {
                        "noun": "common name of the organism for which protein related to"
                    },
                    {
                        "noun": "common name of the living that the protein extracted from"
                    },
                ],
                "uris": [
                    "http://purl.bioontology.org/ontology/CCON",  # organism
                ],
            },
            {
                "id": "Classification",  # name of the column in a tabular dataset
                "description": "Protein classification as given by ChEMBL(version 29). Levels are separated by '->'. Multiple classifications are separated by a semilcolon ';'",  # noqa: E501
                "units": "",  # units of the values in this column (leave empty if unitless)
                "type": "text",  # can be "categorical", "ordinal", "continuous"
                "names": [  # names for the property (to sample from for building the prompts).
                    {"noun": "Protein classification"},
                    {"noun": "protein classification by levels"},
                    {"noun": "Levels for which protein classify"},
                ],
            },
            {
                "id": "seq_length",  # name of the column in a tabular dataset
                "description": "Length of the protein sequence",  # description of what this column means
                "units": "",  # units of the values in this column (leave empty if unitless)
                "type": "continuous",  # can be "categorical", "ordinal", "continuous"
                "names": [  # names for the property (to sample from for building the prompts).
                    {"noun": "Protein sequence length"},
                    {"noun": "Length for protein string"},
                ],
            },
            {
                "id": "Sequence",  # name of the column in a tabular dataset
                "description": "Protein sequence including mutations",  # description of what this column means
                "units": "",  # units of the values in this column (leave empty if unitless)
                "type": "string",  # can be "categorical", "ordinal", "continuous"
                "names": [  # names for the property (to sample from for building the prompts).
                    {"noun": "Protein sequence character"},
                    {"noun": "FASTQ of the protein"},
                    {"noun": "protein string"},
                ],
                "uris": [
                    "http://purl.bioontology.org/ontology/MESH/D009154"  # mutation
                ],
            },
        ],
        "identifiers": [
            {
                "id": "target_id",  # column name
                "type": "Other",
                "names": [
                    {"noun": "protein identifier wtih mutation"},
                    {"noun": "target id plus mutation"},
                    {"noun": "protein target combined with mutation"},
                ],
                "description": "A unique Papyrus protein identifier. It results from the concatenation of accessions and mutations(e.g. P47747_WT or P10721_V559D_T670I)",  # noqa: E501
            },
            {
                "id": "target_id_without_mutation",  # column name
                "type": "Other",
                "names": [
                    {"noun": "protein identifier"},
                    {"noun": "target id"},
                    {"noun": "protein target"},
                ],
                "description": "A unique protein identifier",  # description (optional, except for "Other")
            },
            {
                "id": "UniProtID",  # column name
                "type": "Other",
                "names": [
                    {"noun": "UniProt identifier"},
                    {"noun": "UniProtID"},
                ],
                "description": "The UniProt identifier of the sequence",  # description (optional, except for "Other")
            },
        ],
        "license": "CC BY-SA 4.0",  # license under which the original dataset was published
        "links": [  # list of relevant links (original dataset, other uses, etc.)
            {
                "url": "https://doi.org/10.1186/s13321-022-00672-x",
                "description": "corresponding publication",
            },
            {
                "url": "https://doi.org/10.4121/16896406.v3",
                "description": "data source",
            },
            {
                "url": "https://data.4tu.nl/articles/_/16896406/3",
                "description": "data source",
            },
        ],
        "num_points": len(df),  # number of datapoints in this dataset
        "bibtex": [
            """@article{B_quignon_2023,
          doi = {10.1186/s13321-022-00672-x},
          url = {https://doi.org/10.1186%2Fs13321-022-00672-x},
          year = {2023},
          month = jan,
          publisher = {Springer Science and Business Media LLC},
          volume = {15},
          number = {1},
          author = {O. J. M. Bequignon and B. J. Bongers and W. Jespers and A. P. IJzerman and B. van der Water and G. J. P. van Westen},
          title = {Papyrus: a large-scale curated dataset aimed at bioactivity predictions},
          journal = {Journal of Cheminformatics}""",  # noqa: E501
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
