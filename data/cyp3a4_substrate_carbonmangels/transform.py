import pandas as pd
import yaml
from tdc.single_pred import ADME


def get_and_transform_data():
    # get raw data
    target_subfolder = "CYP3A4_Substrate_CarbonMangels"
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
        "compound_name",
        "SMILES",
        f"{'_'.join(target_subfolder.split('_')[:2])}",
        "split",
    ]

    df.columns = fields_clean

    # data cleaning
    df[fields_clean[0]] = df[fields_clean[0]].str.strip()
    # remove leading and trailing white space characters
    df = df.dropna()
    assert not df.duplicated().sum()

    # save to csv
    fn_data_csv = "data_clean.csv"
    df.to_csv(fn_data_csv, index=False)
    meta = {
        "name": "cyp3a4_substrate_carbonmangels",  # unique identifier, we will also use this for directory names
        "description": """CYP3A4 is an important enzyme in the body, mainly found in the liver
and in the intestine. It oxidizes small foreign organic molecules(xenobiotics),
such as toxins or drugs, so that they can be removed from the body. TDC used
a dataset from Carbon Mangels et al, which merged information on substrates
and nonsubstrates from six publications.""",
        "targets": [
            {
                "id": f"{'_'.join(target_subfolder.split('_')[:2])}",  # name of the column in a tabular dataset
                "description": "The drugs that are metabolized by the CYP P450 3A4(1) or not (0)",
                "units": "substrate",  # units of the values in this column (leave empty if unitless)
                "type": "categorical",  # can be "categorical", "ordinal", "continuous"
                "names": [  # names for the property (to sample from for building the prompts)
                    "CYP P450 3A4 Substrate",
                    "CYP3A4 Substrate",
                    "ADME Drug metabolism",
                    "Pharmacokinetics metabolism",
                    "Substrate toward CYP3A4",
                ],
                "uris": [
                    "https://bioportal.bioontology.org/ontologies/NCIT?p=classes&conceptid=http%3A%2F%2Fncicb.nci.nih.gov%2Fxml%2Fowl%2FEVS%2FThesaurus.owl%23C26633",  # noqa E501
                    "https://bioportal.bioontology.org/ontologies/NCIT?p=classes&conceptid=http%3A%2F%2Fncicb.nci.nih.gov%2Fxml%2Fowl%2FEVS%2FThesaurus.owl%23C17573",  # noqa E501
                    "https://bioportal.bioontology.org/ontologies/NCIT?p=classes&conceptid=http%3A%2F%2Fncicb.nci.nih.gov%2Fxml%2Fowl%2FEVS%2FThesaurus.owl%23C120264",  # noqa E501
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
                    "drug bank name",
                    "drug name pubchem",
                    "drug generic name",
                    "drug chemical (generic) name",
                    "chemical name",
                ],
                "description": "drug name",  # description (optional, except for "Other")
            },
        ],
        "license": "CC BY 4.0",  # license under which the original dataset was published
        "links": [  # list of relevant links (original dataset, other uses, etc.)
            {
                "url": "https://doi.org/10.1002/minf.201100069",
                "description": "corresponding publication",
            },
            {
                "url": "https://doi.org/10.1021/ci300367a",
                "description": "corresponding publication",
            },
            {
                "url": "https://tdcommons.ai/single_pred_tasks/adme/#cyp3a4-substrate-carbon-mangels-et-al",
                "description": "data source",
            },
        ],
        "num_points": len(df),  # number of datapoints in this dataset
        "bibtex": [
            """@article{CarbonMangels2011,
doi = {10.1002/minf.201100069},
url = {https://doi.org/10.1002/minf.201100069},
year = {2011},
month = sep,
publisher = {Wiley},
volume = {30},
number = {10},
pages = {885--895},
author = {Miriam Carbon-Mangels and Michael C. Hutter},
title = {Selecting Relevant Descriptors for Classification by Bayesian Estimates:
A Comparison with Decision Trees and Support Vector Machines Approaches for Disparate Data Sets},
journal = {Molecular Informatics}""",
            """@article{Cheng2012,
doi = {10.1021/ci300367a},
url = {https://doi.org/10.1021/ci300367a},
year = {2012},
month = nov,
publisher = {American Chemical Society (ACS)},
volume = {52},
number = {11},
pages = {3099--3105},
author = {Feixiong Cheng and Weihua Li and Yadi Zhou and Jie Shen
and Zengrui Wu and Guixia Liu and Philip W. Lee and Yun Tang},
title = {admetSAR: A Comprehensive Source and Free Tool for
Assessment of Chemical ADMET Properties},
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
