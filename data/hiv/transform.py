import pandas as pd
import yaml
from tdc.single_pred import HTS


def get_and_transform_data():
    # get raw data
    data = HTS(name="HIV")
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
        "activity_HIV",
    ]
    df.columns = fields_clean

    #     # data cleaning
    #     df.compound_id = (
    #         df.compound_id.str.strip()
    #     )  # remove leading and trailing white space characters

    assert not df.duplicated().sum()

    # save to csv
    fn_data_csv = "data_clean.csv"
    df.to_csv(fn_data_csv, index=False)

    # create meta yaml
    meta = {
        "name": "hiv",  # unique identifier, we will also use this for directory names
        "description": """The HIV dataset was introduced by the Drug Therapeutics Program (DTP)
AIDS Antiviral Screen, which tested the ability to inhibit HIV replication for
over 40,000 compounds. From MoleculeNet.""",
        "targets": [
            {
                "id": "activity_HIV",  # name of the column in a tabular dataset
                "description": "whether it active against HIV virus (1) or not (0).",
                "units": "activity",  # units of the values in this column (leave empty if unitless)
                "type": "categorical",  # can be "categorical", "ordinal", "continuous"
                "names": [  # names for the property (to sample from for building the prompts)
                    "HIV activity",
                    "HIV Inhibitor",
                    "activity against HIV",
                    "HIV disease",
                ],
                "uris": [
                    "https://bioportal.bioontology.org/ontologies/MESH?p=classes&conceptid=http%3A%2F%2Fpurl.bioontology.org%2Fontology%2FMESH%2FD006678",
                    "https://bioportal.bioontology.org/ontologies/OCHV?p=classes&conceptid=http%3A%2F%2Fsbmi.uth.tmc.edu%2Fontology%2Fochv%236185",
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
                "url": "https://wiki.nci.nih.gov/display/NCIDTPdata/AIDS+Antiviral+Screen+Data",
                "description": "data source",
            },
            {
                "url": "https://doi.org/10.1039/C7SC02664A",
                "description": "corresponding publication",
            },
            {
                "url": "https://tdcommons.ai/single_pred_tasks/hts/#hiv",
                "description": "data source",
            },
        ],
        "num_points": len(df),  # number of datapoints in this dataset
        "bibtex": [
            """@article{Wu2018,
doi = {10.1039/c7sc02664a},
url = {https://doi.org/10.1039/c7sc02664a},
year = {2018},
publisher = {Royal Society of Chemistry (RSC)},
volume = {9},
number = {2},
pages = {513--530},
author = {Zhenqin Wu and Bharath Ramsundar and Evan~N. Feinberg and Joseph Gomes
and Caleb Geniesse and Aneesh S. Pappu and Karl Leswing and Vijay Pande},
title = {MoleculeNet: a benchmark for molecular machine learning},
journal = {Chemical Science}""",
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
