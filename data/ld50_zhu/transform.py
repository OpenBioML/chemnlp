import pandas as pd
import yaml
from tdc.single_pred import Tox


def get_and_transform_data():
    # get raw data
    splits = Tox(name="LD50_Zhu").get_split()
    df_train = splits["train"]
    df_valid = splits["valid"]
    df_test = splits["test"]
    df_train["split"] = "train"
    df_valid["split"] = "valid"
    df_test["split"] = "test"

    df = pd.concat([df_train, df_valid, df_test], axis=0)

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
        "compound_name",
        "SMILES",
        "acute_toxicity",
        "split",
    ]
    df.columns = fields_clean

    # data cleaning
    df.compound_name = (
        df.compound_name.str.strip()
    )  # remove leading and trailing white space characters

    assert not df.duplicated().sum()

    # save to csv
    fn_data_csv = "data_clean.csv"
    df.to_csv(fn_data_csv, index=False)

    # create meta yaml
    meta = {
        "name": "ld50_zhu",  # unique identifier, we will also use this for directory names
        "description": """Acute toxicity LD50 measures
the most conservative dose that can lead to lethal adverse effects.
The higher the dose, the more lethal of a drug.""",
        "targets": [
            {
                "id": "acute_toxicity",  # name of the column in a tabular dataset
                "description": "Acute Toxicity LD50.",  # description of what this column means
                "units": "log(1/(mol/kg))",  # units of the values in this column (leave empty if unitless)
                "type": "continuous",  # can be "categorical", "ordinal", "continuous"
                "names": [
                    "acute toxicity rat LD50",
                    "rat ld50",
                ],
                "uri": ["http://www.bioassayontology.org/bao#BAO_0002117"],
            },
        ],
        "identifiers": [
            {
                "id": "SMILES",  # column name
                "type": "SMILES",  # can be "SMILES", "SELFIES", "IUPAC", "Other"
                "description": "SMILES",  # description (optional, except for "Other")
            },
            {
                "id": "compound_name",
                "type": "Other",
                "description": "compound name",
                "names": [
                    "compound",
                    "compound name",
                    "drug",
                ],
            },
        ],
        "license": "CC BY 4.0",  # license under which the original dataset was published
        "links": [  # list of relevant links (original dataset, other uses, etc.)
            {
                "url": "https://doi.org/10.1021/tx900189p",
                "description": "corresponding publication",
            },
        ],
        "benchmarks": [
            {
                "name": "TDC",
                "link": "https://tdcommons.ai/",
                "split_column": "split",
            }
        ],
        "num_points": len(df),  # number of datapoints in this dataset
        "url": "https://tdcommons.ai/single_pred_tasks/tox/#acute-toxicity-ld50",
        "bibtex": [
            """@article{Zhu2009,
doi = {10.1021/tx900189p},
url = {https://doi.org/10.1021/tx900189p},
year = {2009},
month = oct,
publisher = {American Chemical Society ({ACS})},
volume = {22},
number = {12},
pages = {1913--1921},
author = {Hao Zhu and Todd M. Martin and Lin Ye and Alexander
Sedykh and Douglas M. Young and Alexander Tropsha},
title = {Quantitative Structure-Activity Relationship Modeling
of Rat Acute Toxicity by Oral Exposure},
journal = {Chemical Research in Toxicology}}""",
        ],
    }

    def str_presenter(dumper, data):
        """configures yaml for dumping multiline strings
        Ref:
        https://stackoverflow.com/questions/8640959/how-can-i-control-what-scalar-form-pyyaml-uses-for-my-data
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
