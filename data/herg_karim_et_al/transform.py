import pandas as pd
import yaml
from tdc.single_pred import Tox


def get_and_transform_data():
    # get raw data
    data = Tox(name="hERG_Karim")
    fn_data_original = "data_original.csv"
    data.get_data().to_csv(fn_data_original, index=False)

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
    ]

    # overwrite column names = fields
    fields_clean = [
        "compound_id",
        "SMILES",
        "hERG_blocker",
    ]
    df.columns = fields_clean

    # data cleaning
    #     df.compound_name = (
    #         df.compound_name.str.strip()
    #     )
    # remove leading and trailing white space characters

    assert not df.duplicated().sum()

    # save to csv
    fn_data_csv = "data_clean.csv"
    df.to_csv(fn_data_csv, index=False)

    # create meta yaml
    meta = {
        "name": "herg_karim_et_al",  # unique identifier, we will also use this for directory names
        "description": """A integrated Ether-a-go-go-related gene (hERG) dataset consisting
of molecular structures labelled as hERG (<10uM) and non-hERG (>=10uM) blockers
in the form of SMILES strings was obtained from the DeepHIT, the BindingDB database,
ChEMBL bioactivity database, and other literature.""",
        "targets": [
            {
                "id": "hERG_blocker",  # name of the column in a tabular dataset
                "description": "whether it blocks (1, <10uM) or not blocks (0, >=10uM)",  # description of what this column means
                "units": "activity",  # units of the values in this column (leave empty if unitless)
                "type": "categorical",  # can be "categorical", "ordinal", "continuous"
                "names": [  # names for the property (to sample from for building the prompts)
                    "hERG blocker",
                    "hERG active compound",
                    "hERG blocker",
                    "hERG active compound <10uM",
                    "Human ether-a-go-go related gene (hERG) blocker",
                    "Activity against Human ether-a-go-go related gene (hERG)",
                ],
                "uris": [
                    "https://bioportal.bioontology.org/ontologies/MI?p=classes&conceptid=http%3A%2F%2Fpurl.obolibrary.org%2Fobo%2FMI_2136",
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
