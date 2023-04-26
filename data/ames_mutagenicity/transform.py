import pandas as pd
import yaml
from tdc.single_pred import Tox


def get_and_transform_data():
    # get raw data
    splits = Tox(name="AMES").get_split()
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
        "mutagenic",
        "split",
    ]
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
        "name": "ames_mutagenicity",  # unique identifier, we will also use this for directory names
        "description": """Mutagenicity means the ability of a drug to induce genetic alterations.
Drugs that can cause damage to the DNA can result in cell death or other severe
adverse effects. Nowadays, the most widely used assay for testing the mutagenicity
of compounds is the Ames experiment which was invented by a professor named
Ames. The Ames test is a short term bacterial reverse mutation assay detecting
a large number of compounds which can induce genetic damage and frameshift mutations.
The dataset is aggregated from four papers.""",
        "targets": [
            {
                "id": "mutagenic",  # name of the column in a tabular dataset
                "description": "whether it is mutagenic (1) or not mutagenic (0)",
                "units": None,  # units of the values in this column (leave empty if unitless)
                "type": "boolean",  # can be "categorical", "ordinal", "continuous"
                "names": [  # names for the property (to sample from for building the prompts)
                    {"noun": "mutagenicity"},
                    {"noun": "Ames mutagenicity"},
                    {"adjective": "mutagenic"},
                    {"adjective": "Ames mutagenic"},
                    {"verb": "has the ability to induce genetic alterations"},
                    {"gerund": "having the potential to cause mutations"},
                    {"gerund": "having the potential to induce genetic alterations"},
                ],
                "uris": [
                    "http://purl.enanomapper.org/onto/ENM_0000042",
                ],
                "pubchem_aids": [
                    "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C16235",
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
                "url": "https://doi.org/10.1021/ci300400a",
                "description": "corresponding publication",
            },
            {
                "url": "https://tdcommons.ai/single_pred_tasks/tox/#ames-mutagenicity",
                "description": "Data source",
            },
        ],
        "num_points": len(df),  # number of datapoints in this dataset
        "bibtex": [
            """@article{Xu2012,
doi = {10.1021/ci300400a},
url = {https://doi.org/10.1021/ci300400a},
year = {2012},
month = oct,
publisher = {American Chemical Society (ACS)},
volume = {52},
number = {11},
pages = {2840--2847},
author = {Congying Xu and Feixiong Cheng and Lei Chen and
Zheng Du and Weihua Li and Guixia Liu and Philip W. Lee and Yun Tang},
title = {In silico Prediction of Chemical Ames Mutagenicity},
journal = {Journal of Chemical Information and Modeling}""",
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
