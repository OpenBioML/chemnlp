import pandas as pd
import yaml
from tdc.single_pred import Tox
from tdc.utils import retrieve_label_name_list


def get_and_transform_data():
    # get raw data
    name = "herg_central"
    label_names = retrieve_label_name_list("herg_central")
    dfs = []
    for ln in label_names:
        # get raw data
        splits = Tox(name=name, label_name=ln).get_split()
        df_train = splits["train"]
        df_valid = splits["valid"]
        df_test = splits["test"]
        df_train["split"] = "train"
        df_valid["split"] = "valid"
        df_test["split"] = "test"
        df = pd.concat([df_train, df_valid, df_test], axis=0)
        df["task"] = ln
        dfs.append(df)

    df = pd.concat(dfs, axis=0)

    fn_data_original = "data_original.csv"
    df.to_csv(fn_data_original, index=False)
    del df, dfs, df_train, df_valid, df_test

    # create dataframe
    df = pd.read_csv(
        fn_data_original,
        delimiter=",",
    )  # not necessary but ensure we can load the saved data

    # check if fields are the same
    fields_orig = df.columns.tolist()
    assert fields_orig == ["Drug_ID", "Drug", "Y", "split", "task"]

    # overwrite column names = fields
    fields_clean = ["compound_id", "SMILES", "result", "split", "task"]
    df.columns = fields_clean

    assert not df.duplicated().sum()

    # save to csv
    fn_data_csv = "data_clean.csv"
    df.to_csv(fn_data_csv, index=False)

    # create meta yaml
    meta = {
        "name": "herg_central",  # unique identifier, we will also use this for directory names
        "description": """Human ether-à-go-go related gene (hERG) is crucial for the coordination
of the heart's beating. Thus, if a drug blocks the hERG, it could lead to severe
adverse effects. Therefore, reliable prediction of hERG liability in the early
stages of drug design is quite important to reduce the risk of cardiotoxicity-related
attritions in the later development stages. There are three targets: hERG_at_1microM,
hERG_at_10microM, and hERG_inhib.""",
        "targets": [
            {
                "id": "hERG_at_1uM",
                "description": "the percent inhibition of hERG at a 1uM concentration",
                "units": "uM",
                "type": "continuous",
                "names": [
                    {"noun": "hERG percent inhibition at a 1uM concentration"},
                    {"noun": "hERG percent inhibition at 1uM"},
                    {
                        "noun": "human ether-à-go-go related gene (hERG) percent inhibition at a 1uM concentration"
                    },
                    {
                        "noun": "human ether-à-go-go related gene (hERG) percent inhibition at 1uM"
                    },
                    {"verb": "is active against hERG at a 1uM concentration"},
                    {
                        "verb": "is active against human ether-à-go-go related gene (hERG) at a 1uM concentration"
                    },
                ],
                "uris": [
                    "http://purl.obolibrary.org/obo/MI_2136",
                ],
            },
            {
                "id": "hERG_at_10uM",
                "description": "the percent inhibition of hERG at a 10uM concentration",
                "units": "uM",
                "type": "continuous",
                "names": [
                    {"noun": "hERG percent inhibition at a 10uM concentration"},
                    {"noun": "hERG percent inhibition at 1uM"},
                    {
                        "noun": "human ether-à-go-go related gene (hERG) percent inhibition at a 10uM concentration"
                    },
                    {
                        "noun": "human ether-à-go-go related gene (hERG) percent inhibition at 10uM"
                    },
                    {"verb": "is active against hERG at a 10uM concentration"},
                    {
                        "verb": "is active against human ether-à-go-go related gene (hERG) at a 10uM concentration"
                    },
                ],
                "uris": ["http://purl.obolibrary.org/obo/MI_2136"],
            },
            {
                "id": "hERG_inhib",
                "description": """whether it blocks (1) or not blocks (0) hERG
(This is equivalent to whether hERG_at_10microM < -50, i.e.,
whether the compound has an IC50 of less than 10microM.)""",
                "units": None,
                "type": "boolean",
                "names": [
                    {"noun": "hERG percent inhibitor"},
                    {"verb": "is active against hERG"},
                    {"verb": "inhbits human ether-à-go-go related gene (hERG)"},
                ],
                "uris": [
                    "http://purl.obolibrary.org/obo/MI_2136",
                ],
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
                "url": "https://doi.org/10.1089/adt.2011.0425",
                "description": "corresponding publication",
            },
            {
                "url": "https://bbirnbaum.com/",
                "description": "TDC Contributer",
            },
            {
                "url": "https://tdcommons.ai/single_pred_tasks/tox/#herg-central",
                "description": "Data source",
            },
        ],
        "num_points": len(df),  # number of datapoints in this dataset
        "bibtex": [
            """@article{Du2011,
doi = {10.1089/adt.2011.0425},
url = {https://doi.org/10.1089/adt.2011.0425},
year = {2011},
month = dec,
publisher = {Mary Ann Liebert Inc},
volume = {9},
number = {6},
pages = {580--588},
author = {Fang Du and Haibo Yu and Beiyan Zou and Joseph Babcock
and Shunyou Long and Min Li},
title = {hERGCentral: A Large Database to Store,  Retrieve,  and Analyze Compound Human
Ether-à-go-go Related Gene Channel Interactions to Facilitate Cardiotoxicity Assessment in Drug Development},
journal = {ASSAY and Drug Development Technologies}""",
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
