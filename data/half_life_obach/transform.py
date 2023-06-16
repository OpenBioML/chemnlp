import pandas as pd
import yaml
from tdc.single_pred import ADME


def get_and_transform_data():
    # get raw data
    target_subfolder = "Half_Life_Obach"
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
    fields_clean = ["chembl_id", "SMILES", "half_life_duration", "split"]
    df.columns = fields_clean

    # data cleaning
    # remove leading and trailing white space characters
    df.chembl_id = df.chembl_id.str.strip()
    df = df.dropna()
    assert not df.duplicated().sum()

    # save to csv
    fn_data_csv = "data_clean.csv"
    df.to_csv(fn_data_csv, index=False)
    meta = {
        "name": "half_life_obach",  # unique identifier, we will also use this for directory names
        "description": """Half life of a drug is the duration for the concentration of the drug
in the body to be reduced by half. It measures the duration of actions of a drug.
This dataset deposited version under CHEMBL assay 1614674.""",
        "targets": [
            {
                "id": "half_life_duration",  # name of the column in a tabular dataset
                "description": "the time it takes for the plasma concentration of a drug in the body to be reduced by half",  # noqa: E501
                "units": "hours",  # units of the values in this column (leave empty if unitless)
                "type": "continuous",
                "names": [  # names for the property (to sample from for building the prompts)
                    {"noun": "half life"},
                    {"noun": "drug half life time"},
                    {
                        "noun": "the duration by which the concentration of the drug in the body is reduced by half"
                    },
                ],
                "uris": [
                    "http://purl.bioontology.org/ontology/MESH/D006207",
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
            {
                "id": "chembl_id",  # column name
                "type": "Other",  # can be "SMILES", "SELFIES", "IUPAC", "Other"
                "names": ["ChEMBL database id", "ChEMBL identifier number"],
                "description": "ChEMBL ids",  # description (optional, except for "Other")
                "sample": False,
            },
        ],
        "license": "CC BY 4.0",  # license under which the original dataset was published
        "links": [  # list of relevant links (original dataset, other uses, etc.)
            {
                "url": "https://doi.org/10.1124/dmd.108.020479",
                "description": "corresponding publication",
            },
            {
                "url": "https://tdcommons.ai/single_pred_tasks/adme/#half-life-obach-et-al",
                "description": "data source",
            },
        ],
        "num_points": len(df),  # number of datapoints in this dataset
        "bibtex": [
            """@article{Obach2008,
doi = {10.1124/dmd.108.020479},
url = {https://doi.org/10.1124/dmd.108.020479},
year = {2008},
month = apr,
publisher = {American Society for Pharmacology and Experimental Therapeutics (ASPET)},
volume = {36},
number = {7},
pages = {1385--1405},
author = {R. Scott Obach and Franco Lombardo and Nigel J. Waters},
title = {Trend Analysis of a Database of Intravenous Pharmacokinetic
Parameters in Humans for 670 Drug Compounds},
journal = {Drug Metabolism and Disposition}""",
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
