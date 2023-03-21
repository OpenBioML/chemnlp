import pandas as pd
import requests
import yaml


def get_and_transform_data():
    # get raw data
    data_path = (
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv"
    )
    fn_data_original = "data_original.txt"
    data = requests.get(data_path)
    with open(fn_data_original, "wb") as f:
        f.write(data.content)

    # create dataframe
    df = pd.read_csv(fn_data_original, delimiter=",")

    # check if fields are the same
    assert df.columns.tolist() == ["CMPD_CHEMBLID", "exp", "smiles"]

    # check if no duplicated
    assert not df.duplicated().sum()

    # overwrite column names = fields
    fields_clean = [
        "CMPD_CHEMBLID",
        "exp",
        "SMILES",
    ]
    df.columns = fields_clean

    # save to csv
    fn_data_csv = "data_clean.csv"
    df.to_csv(fn_data_csv, index=False)

    # create meta yaml
    meta = {
        "name": "lipophilicity",  # unique identifier, we will also use this for directory names
        "description": "Experimental results of octanol/water distribution coefficient (logD at pH 7.4).",
        "targets": [
            {
                "id": "exp",  # name of the column in a tabular dataset
                "description": "experimental results of octanol/water distribution coefficient (logD at pH 7.4)",
                "units": None,
                "type": "continuous",  # can be "categorical", "ordinal", "continuous"
                "names": [  # names for the property (to sample from for building the prompts)
                    "octanol/water distribution coefficient (logD at pH 7.4)",
                    "octanol/water distribution coefficient",
                ],
                "uris": [
                    "http://www.bioassayontology.org/bao#BAO_0002129",
                    "http://purl.obolibrary.org/obo/MI_2107",
                ],
            },
        ],
        "identifiers": [
            {
                "id": "SMILES",  # column name
                "type": "SMILES",  # can be "SMILES", "SELFIES", "IUPAC", "OTHER"
                "description": "SMILES",  # description (optional, except for "OTHER")
            },
        ],
        "license": "CC BY-SA 3.0",  # license under which the original dataset was published
        "links": [  # list of relevant links (original dataset, other uses, etc.)
            {
                "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv",
                "description": "original dataset link",
            },
            {
                "url": "https://github.com/cheminfo/molecule-features/blob/main/data/lipophilicity/meta.yaml",
                "description": "original meta data",
            },
            {
                "url": "https://deepchem.readthedocs.io/en/latest/api_reference/moleculenet.html#lipo-datasets",
                "description": "original dataset link from moleculenet",
            },
            {
                "url": "https://www.ebi.ac.uk/chembl/document_report_card/CHEMBL3301361/",
                "description": "original report card",
            },
            {
                "url": "https://chembl.gitbook.io/chembl-interface-documentation/about#data-licensing",
                "description": "original dataset license from chembl",
            },
            {
                "url": "https://creativecommons.org/licenses/by-sa/3.0/",
                "description": "used dataset license",
            },
        ],
        "num_points": len(df),  # number of datapoints in this dataset
        "bibtex": [
            """@techreport{hersey2015chembl,
title={ChEMBL Deposited Data Set-AZ dataset},
author={Hersey, Anne},
year={2015},
institution={Technical Report, Technical report, EMBL-EBI, 2015. https://www. ebi. ac. uk}
}""",
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
