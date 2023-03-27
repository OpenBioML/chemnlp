import pandas as pd
import yaml
from tdc.single_pred import Tox


def get_and_transform_data():
    # get raw data
    splits = Tox(name="ClinTox").get_split()
    df_train = splits["train"]
    df_valid = splits["valid"]
    df_test = splits["test"]
    df_train["split"] = "train"
    df_valid["split"] = "valid"
    df_test["split"] = "test"

    df = pd.concat([df_train, df_valid, df_test], axis=0)

    # check if fields are the same
    fields_orig = df.columns.tolist()
    assert fields_orig == ["Drug_ID", "Drug", "Y", "split"]

    # overwrite column names = fields
    fields_clean = ["compound_id", "SMILES", "clinical_toxicity", "split"]
    df.columns = fields_clean

    # data cleaning
    df.compound_id = (
        df.compound_id.str.strip()
    )  # remove leading and trailing white space characters

    assert not df.duplicated().sum()

    # save to csv
    fn_data_csv = "data_clean.csv"
    df.to_csv(fn_data_csv, index=False)

    # create meta yaml
    meta = {
        "name": "clintox",  # unique identifier, we will also use this for directory names
        "description": """The ClinTox dataset includes drugs that have failed
clinical trials for toxicity reasons and also drugs that are associated
with successful trials.""",
        "targets": [
            {
                "id": "clinical_toxicity",  # name of the column in a tabular dataset
                "description": "whether it can cause clinical toxicity (1) or not (0).",
                "units": None,  # units of the values in this column (leave empty if unitless)
                "type": "boolean",
                "names": [  # names for the property (to sample from for building the prompts)
                    "clinically toxic",
                    "displaying clinical toxicity",
                    "toxic",
                ],
                "uris": [
                    "http://purl.bioontology.org/ontology/MESH/Q000633",
                    "https://ncit.nci.nih.gov/ncitbrowser/ConceptReport.jsp?dictionary=NCI_Thesaurus&ns=ncit&code=C27990",  # noqa: E501
                    "https://ncit.nci.nih.gov/ncitbrowser/ConceptReport.jsp?dictionary=NCI_Thesaurus&ns=ncit&code=C27955",  # noqa: E501
                ],
            },
        ],
        "benchmarks": [
            {
                "name": "TDC",
                "link": "https://tdcommons.ai/",
                "split_column": "split",
            }
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
                "url": "https://doi.org/10.1016/j.chembiol.2016.07.023",
                "description": "corresponding publication",
            },
        ],
        "num_points": len(df),  # number of datapoints in this dataset
        "url": "https://tdcommons.ai/single_pred_tasks/tox/#clintox",
        "bibtex": [
            """@article{Gayvert2016,
doi = {10.1016/j.chembiol.2016.07.023},
url = {https://doi.org/10.1016/j.chembiol.2016.07.023},
year = {2016},
month = oct,
publisher = {Elsevier {BV}},
volume = {23},
number = {10},
pages = {1294--1301},
author = {Kaitlyn~M. Gayvert and Neel~S. Madhukar and Olivier Elemento},
title = {A Data-Driven Approach to Predicting Successes and Failures of Clinical Trials},
journal = {Cell Chemical Biology}}""",
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
