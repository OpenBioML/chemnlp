import pandas as pd
import yaml
from tdc.single_pred import ADME


def get_and_transform_data():
    # get raw data
    target_subfolder = "CYP2C9_Veith"
    splits = ADME(name=target_subfolder).get_split()
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
    assert fields_orig == ["Drug_ID", "Drug", "Y", "split"]

    # overwrite column names = fields
    fields_clean = [
        "chembl_id",
        "SMILES",
        f"{target_subfolder.split('_')[0]}_inhibition",
        "split",
    ]

    df.columns = fields_clean

    # data cleaning
    df = df.dropna()
    assert not df.duplicated().sum()

    # save to csv
    fn_data_csv = "data_clean.csv"
    df.to_csv(fn_data_csv, index=False)
    meta = {
        "name": "cyp_p450_2c9_inhibition_veith_et_al",  # unique identifier, we will also use this for directory names
        "description": """The CYP P450 genes are involved in the formation and breakdown (metabolism)
of various molecules and chemicals within cells. Specifically, the CYP P450
2C9 plays a major role in the oxidation of both xenobiotic and endogenous compounds.""",
        "targets": [
            {
                "id": f"{target_subfolder.split('_')[0]}_inhibition",  # name of the column in a tabular dataset
                "description": "ability of the drug to inhibit CYP P450 2C9 (1) or not (0)",
                "units": None,  # units of the values in this column (leave empty if unitless)
                "type": "boolean",  # can be "categorical", "ordinal", "continuous"
                "names": [  # names for the property (to sample from for building the prompts)
                    {"noun": "CYP P450 2C9 inhibition"},
                    {"noun": "CYP 2C9 inhibition"},
                    {"verb": "inhibits CYP P450 2C9"},
                    {"gerund": "inhibiting CYP P450 2C9"},
                ],
                "uris": None,
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
                "url": "https://doi.org/10.1038/nbt.1581",
                "description": "corresponding publication",
            },
            {
                "url": "https://tdcommons.ai/single_pred_tasks/adme/#cyp-p450-2c9-inhibition-veith-et-al",
                "description": "data source",
            },
        ],
        "num_points": len(df),  # number of datapoints in this dataset
        "bibtex": [
            """@article{Veith2009,
doi = {10.1038/nbt.1581},
url = {https://doi.org/10.1038/nbt.1581},
year = {2009},
month = oct,
publisher = {Springer Science and Business Media LLC},
volume = {27},
number = {11},
pages = {1050--1055},
author = {Henrike Veith and Noel Southall and Ruili Huang and Tim James
and Darren Fayne and Natalia Artemenko and Min Shen and James Inglese
and Christopher P Austin and David G Lloyd and Douglas S Auld},
title = {Comprehensive characterization of cytochrome P450 isozyme selectivity
across chemical libraries},
journal = {Nature Biotechnology}""",
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
