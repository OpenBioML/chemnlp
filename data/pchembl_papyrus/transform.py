import pandas as pd
import yaml


def get_and_transform_data():
    target_folder = "pchembl_papyrus"
    url = "https://huggingface.co/datasets/phalem/awesome_chem_clean_data/resolve/main/pchembl_papyrus.csv.gz"
    df = pd.read_csv(url)
    df = df.drop_duplicates()
    df = df.dropna(subset='pchembl_value')
    df = df.fillna('Unknown')
    fields_orig = df.columns.tolist()
    assert fields_orig == ['Activity_ID', 'CID', 'SMILES', 'target_id', 'accession',
          'Protein_Type', 'AID', 'doc_id', 'Year', 'activity_type', 'relation',
          'pchembl_value', 'pchembl_value_Mean', 'pchembl_value_StdDev',
          'pchembl_value_N', 'class']
    fields_clean = ['Activity_ID', 'CID', 'SMILES', 'target_id', 'accession',
          'Protein_Type', 'AID', 'doc_id', 'activity_type', #'relation',
          'pchembl_value_Mean','class']
    df =df[fields_clean]
    assert fields_orig != fields_clean
    assert not df.duplicated().sum()
    fn_data_csv = f"data_clean.csv"
    df.to_csv(fn_data_csv, index=False)

    # create meta yaml
    meta = {
    "name": f"{target_folder}",  # unique identifier, we will also use this for directory names
    "description": """Papyrus is an aggregated dataset of small molecule bioactivities. This file contains the highly standardized and normalized data. Although not containing stereochemistry, it provides the highest quality of the data.""",
    "targets": [
        {
            "id": "pchembl_value",  # name of the column in a tabular dataset
            "description": "Ensemble of log-transformed activity values available for then compound-protein pair at the highest quality possible.",  # description of what this column means
            "units": "",  # units of the values in this column (leave empty if unitless)
            "type": "continuous",  # can be "categorical", "ordinal", "continuous"
            "names": [  # names for the property (to sample from for building the prompts).
            {"noun": "Activity value of the compound against target"},
            {"noun": "Compound potency"},
            {"noun": "Activity against target"},
            {"noun": "Ensemble of log transformed activity values"},

            ],
        },
        {
            "id": "pchembl_value_Mean",  # name of the column in a tabular dataset
            "description": "Mean average of 'pchembl_value' activities..",  # description of what this column means
            "units": "",  # units of the values in this column (leave empty if unitless)
            "type": "continuous",  # can be "categorical", "ordinal", "continuous"
            "names": [  # names for the property (to sample from for building the prompts).
            {"noun": "Mean ctivity value of the compound against target"},
            {"noun": "Compound potency mean value"},
            {"noun": "Mean of ensemble log transformed activity values"},
            ],
        },
        {
            "id": "activity_type",  # name of the column in a tabular dataset
            "description": "Types of the value activity reported in like IC50, EC50, KD, Ki or other.",  # description of what this column means
            "units": "",  # units of the values in this column (leave empty if unitless)
            "type": "categorical",  # can be "categorical", "ordinal", "continuous"
            "names": [  # names for the property (to sample from for building the prompts).
            {"noun": "Type of activity measure that activity reported in"},
            {"noun": "Measurement value used"},
            ],
            "uris":[
            "http://purl.obolibrary.org/obo/MI_0640",
            ]
        },

        {
            "id": "class",  # name of the column in a tabular dataset
            "description": "what is the class of the protein",  # description of what this column means
            "units": "",  # units of the values in this column (leave empty if unitless)
            "type": "categorical",  # can be "categorical", "ordinal", "continuous"
            "names": [  # names for the property (to sample from for building the prompts).
            {"noun": "Which category protein related to"},
            {"noun": "The family that protein related"},
            {"noun": "Function of the protein"},
            ],
        },
       {
            "id": "doc_id",  # name of the column in a tabular dataset
            "description": "First published or filed document characterizing the compound-protein interation. This does not always correspond to the source of the reported activities",  # description of what this column means
            "units": "",  # units of the values in this column (leave empty if unitless)
            "type": "text",  # can be "categorical", "ordinal", "continuous"
            "names": [  # names for the property (to sample from for building the prompts).
            {"noun": "Reference"},
            {"noun": "published or filed document characterizing the compound-protein interation"},
            {"noun": "Reference for compound experiment"},
            ],
        },
        {
            "id": "AID",  # name of the column in a tabular dataset
            "description": "Assay identifiers (only original identifiers for ExCAPE-DB and ChEMBL data)",  # description of what this column means
            "units": "",  # units of the values in this column (leave empty if unitless)
            "type": "string",  # can be "categorical", "ordinal", "continuous"
            "names": [  # names for the property (to sample from for building the prompts).
            {"noun": "where the compound reported"},
            {"noun": "Assay which compound are reported at"},
            ],
        },
        {
            "id": "Protein_Type",  # name of the column in a tabular dataset
            "description": "Either 'WT' for wild-type sequences or a sequence of the mutations underscore separated.",  # description of what this column means
            "units": "",  # units of the values in this column (leave empty if unitless)
            "type": "text",  # can be "categorical", "ordinal", "continuous"
            "names": [  # names for the property (to sample from for building the prompts).
            {"noun": "Type of protein for which compound are active on"},
            {"noun": "varaition of protein which may include mutation"},
            {"noun": "absence of a presence of mutation"},
            ],
        },
        {
            "id": "target_id",  # name of the column in a tabular dataset
            "description": "Unique protein identifier ",  # description of what this column means
            "units": "",  # units of the values in this column (leave empty if unitless)
            "type": "text",  # can be "categorical", "ordinal", "continuous"
            "names": [  # names for the property (to sample from for building the prompts).
            {"noun": "Protein id"},
            {"noun": "Protein which compound have activity on"},
            {"noun": "Unique protein identifier"},
            ],
        }
    ],
    "identifiers": [
        {
                "id": "Activity_ID",  # column name
                "type": "Other",
                "names": [
                    {"noun": "Concatenation of the  molcule InChI and protein unique identifier"},
                    {"noun": "Molcule InChI plus protein unique identifier"},

                ],
                "description": "A unique Papyrus compound-protein identifier.It results from the concatenation of the  molcule InChI and protein unique identifier.",  # description (optional, except for "Other")
            },
        {
                "id": "SMILES",  # column name
                "type": "SMILES",
                "description": "Standardized SMILES notation of the molecule",  # description (optional, except for "Other")
            },
        {
                "id": "accession",  # column name
                "type": "Other",
                "names": [
                    {"noun": "UniProt identifier"},
                    {"noun": "UniProtID"},
                ],
                "description": "The UniProt identifier of the target protein",  # description (optional, except for "Other")
            },
        {
                "id": "CID",  # column name
                "type": "Other",
                "names": [
                    {"noun": "Compound identifiers"},
                    {"noun": "CID"},
                ],
                "description": "Compound identifiers (only original identifiers for ExCAPE-DB and ChEMBL data)",  # description (optional, except for "Other")
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

        }
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
      journal = {Journal of Cheminformatics}""",
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
