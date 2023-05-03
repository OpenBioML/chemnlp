import pandas as pd
import yaml
from tdc.single_pred import ADME


def get_and_transform_data():
    # get raw data
    target_subfolder = "PPBR_AZ"
    data = ADME(name=target_subfolder)
    dfs = []
    for species in [
        "Canis lupus familiaris",
        "Cavia porcellus",
        "Homo sapiens",
        "Mus musculus",
        "Rattus norvegicus",
    ]:
        data.get_other_species(species)
        splits = data.get_split()
        df_train = splits["train"]
        df_valid = splits["valid"]
        df_test = splits["test"]
        df_train["split"] = "train"
        df_valid["split"] = "valid"
        df_test["split"] = "test"
        df_tmp = pd.concat([df_train, df_valid, df_test], axis=0)
        df_tmp["Species"] = species
        dfs.append(df_tmp)
    df = pd.concat(dfs, axis=0)

    fn_data_original = "data_original.csv"
    df.to_csv(fn_data_original, index=False)
    del df

    # create dataframe
    df = pd.read_csv(
        fn_data_original,
        delimiter=",",
    )  # not necessary but ensure we can load the saved data
    # create dataframe
    df = pd.read_csv(fn_data_original, sep=",")
    # check if fields are the same
    fields_orig = df.columns.tolist()
    assert fields_orig == ["Drug_ID", "Drug", "Y", "split", "Species"]

    # overwrite column names = fields
    fields_clean = ["chembl_id", "SMILES", "rate_of_PPBR", "split", "Species"]
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
        "name": "plasma_protein_binding_rate_astrazeneca",
        "description": """The human plasma protein binding rate (PPBR) is expressed as the percentage
of a drug bound to plasma proteins in the blood. This rate strongly affect a drug's
efficiency of delivery. The less bound a drug is, the more efficiently it can
traverse and diffuse to the site of actions.""",
        "targets": [
            {
                "id": "rate_of_PPBR",  # name of the column in a tabular dataset
                "description": "percentage of a drug bound to plasma proteins in the blood",
                "units": "percentage",  # units of the values in this column (leave empty if unitless)
                "type": "continuous",  # can be "categorical", "ordinal", "continuous"
                "names": [  # names for the property (to sample from for building the prompts)
                    {"noun": "human plasma protein binding rate (PPBR)"},
                    {"noun": "human plasma protein binding rate"},
                    {"noun": "PPBR"},
                    {
                        "noun": "percentage of a drug bound to plasma proteins in the blood"
                    },
                ],
                "uris": [
                    "http://purl.jp/bio/4/id/201306028362680450",
                    "http://www.bioassayontology.org/bao#BAO_0010135",
                    # todo: this is the assay but is the number after BAO_ the PubChem assay id?
                ],
            },
        ],
        "identifiers": [
            {
                "id": "SMILES",  # column name
                "type": "SMILES",  # can be "SMILES", "SELFIES", "IUPAC", "Other"
                "description": "SMILES",  # description (optional, except for "Other")
            },
            {
                "id": "Species",  # column name
                "type": "Other",  # can be "SMILES", "SELFIES", "IUPAC", "Other"
                "names": ["species"],
                "description": "species in which the measurement was carried out",
            },
            {
                "id": "chembl_id",  # column name
                "type": "Other",  # can be "SMILES", "SELFIES", "IUPAC", "Other"
                "names": ["ChEMBL id", "ChEMBL identifier number"],
                "description": "chembl ids",  # description (optional, except for "Other")
                "sample": False,
            },
        ],
        "license": "CC BY 4.0",  # license under which the original dataset was published
        "links": [  # list of relevant links (original dataset, other uses, etc.)
            {
                "url": "http://dx.doi.org/10.6019/CHEMBL3301361",
                "description": "corresponding publication",
            },
            {
                "url": "https://tdcommons.ai/single_pred_tasks/adme/#ppbr-plasma-protein-binding-rate-astrazeneca",
                "description": "data source",
            },
        ],
        "num_points": len(df),  # number of datapoints in this dataset
        "bibtex": [
            """@techreport{Hersey2015,
doi = {10.6019/chembl3301361},
url = {https://doi.org/10.6019/chembl3301361},
year = {2015},
month = feb,
publisher = {EMBL-EBI},
author = {Anne Hersey},
title = {ChEMBL Deposited Data Set - AZ dataset}""",
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
