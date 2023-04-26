import pandas as pd
import yaml
from tdc.single_pred import Tox


def get_and_transform_data():
    # get raw data
    splits = Tox(name="hERG_Karim").get_split()
    df_train = splits["train"]
    df_valid = splits["valid"]
    df_test = splits["test"]
    df_train["split"] = "train"
    df_valid["split"] = "valid"
    df_test["split"] = "test"
    df = pd.concat([df_train, df_valid, df_test], axis=0)

    fn_data_original = "data_original.csv"
    df.to_csv(fn_data_original, index=False)
    del df

    # create dataframe
    df = pd.read_csv(
        fn_data_original,
        delimiter=",",
    )  # not necessary but ensure we can load the saved data

    # check if fields are the same
    fields_orig = df.columns.tolist()
    assert fields_orig == [
        "Drug_ID",
        "Drug",
        "Y",
        "split",
    ]

    # overwrite column names = fields
    fields_clean = [
        "compound_id",
        "SMILES",
        "herg_blocker",
        "split",
    ]
    df.columns = fields_clean

    assert not df.duplicated().sum()

    # save to csv
    fn_data_csv = "data_clean.csv"
    df.to_csv(fn_data_csv, index=False)

    # create meta yaml
    meta = {
        "name": "herg_karim_et_al",  # unique identifier, we will also use this for directory names
        "description": """A integrated Ether-à-go-go-related gene (hERG) dataset consisting
of molecular structures labelled as hERG (<10uM) and non-hERG (>=10uM) blockers in
the form of SMILES strings was obtained from the DeepHIT, the BindingDB database,
ChEMBL bioactivity database, and other literature.""",
        "targets": [
            {
                "id": "herg_blocker",  # name of the column in a tabular dataset
                "description": "whether it blocks hERG (1, <10uM) or not (0, >=10uM)",
                "units": None,  # units of the values in this column (leave empty if unitless)
                "type": "boolean",  # can be "categorical", "ordinal", "continuous"
                "names": [  # names for the property (to sample from for building the prompts)
                    {"noun": "hERG blocker"},
                    {"noun": "hERG active compound"},
                    {"noun": "hERG active compound (<10uM)"},
                    {"noun": "human ether-à-go-go related gene (hERG) blocker"},
                    {"verb": "blocks hERG"},
                    {"verb": "blocks human ether-à-go-go related gene (hERG)"},
                    {"verb": "active against hERG (<10uM)"},
                    {"verb": "active against human ether-à-go-go related gene (hERG)"},
                ],
                "uris": [
                    "http://purl.obolibrary.org/obo/MI_2136",
                ],
            },
        ],
        "benchmarks": [
            {
                "name": "TDC",  # unique benchmark name
                "link": "https://tdcommons.ai/",  # benchmark URL
                "split_column": "split",  # name of the column that contains the split information
            },
        ],
        "identifiers": [
            {
                "id": "SMILES",  # column name
                "type": "SMILES",  # can be "SMILES", "SELFIES", "IUPAC", "Other"
                "description": "SMILES",  # description (optional, except for "Other")
            },
        ],
        "license": "CC BY 4.0",  # license under which the original dataset was published
        "links": [  # list of relevant links (original dataset, other uses, etc.)
            {
                "url": "https://doi.org/10.1186/s13321-021-00541-z",
                "description": "corresponding publication",
            },
            {
                "url": "https://tdcommons.ai/single_pred_tasks/tox/#herg-karim-et-al",
                "description": "Data source",
            },
        ],
        "num_points": len(df),  # number of datapoints in this dataset
        "bibtex": [
            """@article{Karim2021,
doi = {10.1186/s13321-021-00541-z},
url = {https://doi.org/10.1186/s13321-021-00541-z},
year = {2021},
month = aug,
publisher = {Springer Science and Business Media LLC},
volume = {13},
number = {1},
author = {Abdul Karim and Matthew Lee and Thomas Balle and Abdul Sattar},
title = {CardioTox net: a robust predictor for hERG channel blockade
based on deep learning meta-feature ensembles},
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
