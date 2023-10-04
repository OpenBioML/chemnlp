import pandas as pd
import yaml
from tdc.single_pred import ADME


def get_and_transform_data():
    # get raw data
    target_subfolder = "VDss_Lombardo"
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
    fields_clean = ["compound_name", "SMILES", "VDss_Lombardo", "split"]
    df.columns = fields_clean

    # data cleaning
    # remove leading and trailing white space characters
    df.compound_name = df.compound_name.str.strip()

    df = df.dropna()
    assert not df.duplicated().sum()

    # save to csv
    fn_data_csv = "data_clean.csv"
    df.to_csv(fn_data_csv, index=False)
    meta = {
        "name": "volume_of_distribution_at_steady_state_lombardo_et_al",
        "description": """The volume of distribution at steady state (VDss) measures the degree
of a drug's concentration in the body tissue compared to concentration in the blood.
Higher VD indicates a higher distribution in the tissue and usually indicates
the drug with high lipid solubility, low plasma protein binidng rate.""",
        "targets": [
            {
                "id": "VDss_Lombardo",  # name of the column in a tabular dataset
                "description": "volume of distribution at steady state (VDss)",
                "units": "L/kg",  # units of the values in this column (leave empty if unitless)
                "type": "continuous",
                "names": [  # names for the property (to sample from for building the prompts)
                    {"noun": "volume of distribution at steady state (VDss)"},
                    {"noun": "VDss"},
                ],
                "uris": [
                    "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C85538",
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
                "id": "compound_name",  # column name
                "type": "Other",  # can be "SMILES", "SELFIES", "IUPAC", "Other"
                "names": [
                    {"noun": "compound name"},
                    {"noun": "drug name"},
                    {"noun": "generic drug name"},
                ],
                "description": "mix of drug name and ids",  # description (optional, except for "Other")
            },
        ],
        "license": "CC BY 4.0",  # license under which the original dataset was published
        "links": [  # list of relevant links (original dataset, other uses, etc.)
            {
                "url": "https://doi.org/10.1021/acs.jcim.6b00044",
                "description": "corresponding publication",
            },
            {
                "url": "https://tdcommons.ai/single_pred_tasks/adme/#vdss-volumn-of-distribution-at-steady-state-lombardo-et-al",  # noqa: E501
                "description": "data source",
            },
        ],
        "num_points": len(df),  # number of datapoints in this dataset
        "bibtex": [
            """@article{Lombardo2016,
doi = {10.1021/acs.jcim.6b00044},
url = {https://doi.org/10.1021/acs.jcim.6b00044},
year = {2016},
month = sep,
publisher = {merican Chemical Society (ACS)},
volume = {56},
number = {10},
pages = {2042--2052},
author = {Franco Lombardo and Yankang Jing},
title = {In Silico Prediction of Volume of Distribution in Humans. Extensive Data Set and the
Exploration of Linear and Nonlinear Methods Coupled with Molecular Interaction Fields Descriptors},
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
