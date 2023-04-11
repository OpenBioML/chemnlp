import pandas as pd
import yaml
from tdc.generation import MolGen

DATASET_NAME = "chembl_29"


def get_and_transform_data():
    # get raw data per dataset
    def get_single_dataset(dataset_name):
        splits = MolGen(name=dataset_name).get_split()
        df_train = splits["train"]
        df_valid = splits["valid"]
        df_test = splits["test"]
        df_train["split"] = "train"
        df_valid["split"] = "valid"
        df_test["split"] = "test"
        df = pd.concat([df_train, df_valid, df_test], axis=0)
        df["dataset"] = dataset_name
        return df

    # get raw data
    df = get_single_dataset(DATASET_NAME)
    df = df.drop_duplicates()

    # check if fields are the same
    fields_orig = df.columns.tolist()
    assert fields_orig == [
        "smiles",
        "split",
        "dataset",
    ]
    assert not df.duplicated().sum()

    # save to csv
    fn_data_csv = "data_clean.csv"
    df.to_csv(fn_data_csv, index=False)

    # create meta yaml
    meta = {
        "name": DATASET_NAME,  # unique identifier, we will also use this for directory names
        "description": """ChEMBL is a manually curated database of bioactive molecules with drug-like properties.
It brings together chemical, bioactivity and genomic data
to aid the translation of genomic information into effective new drugs.""",
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
                "type": "SMILES",  # can be "SMILES", "SELFIES", "IUPAC", "OTHER"
                "description": "SMILES",  # description (optional, except for "OTHER")
            },
        ],
        "license": "CC BY-SA 3.0",  # license under which the original dataset was published
        "links": [  # list of relevant links (original dataset, other uses, etc.)
            {
                "url": "https://academic.oup.com/nar/article/47/D1/D930/5162468",
                "description": "Article about original dataset",
            },
            {
                "url": "https://academic.oup.com/nar/article/43/W1/W612/2467881",
                "description": "Exemplary related article shown in tdc's website",
            },
        ],
        "num_points": len(df),  # number of datapoints in this dataset
        "bibtex": [
            """@article{10.1093/nar/gky1075,
author = {Mendez, David and Gaulton, Anna and Bento, A Patricia and Chambers, Jon and De Veij,
Marleen and Felix, Eloy and Magarinos, Maria Paula and Mosquera,
Juan F and Mutowo, Prudence and Nowotka, Michal and Gordillo-Maranon,
Maria and Hunter, Fiona and Junco, Laura and Mugumbate, Grace and Rodriguez-Lopez, Milagros and Atkinson,
Francis and Bosc, Nicolas and Radoux, Chris J and Segura-Cabrera, Aldo and Hersey, Anne and Leach, Andrew R},
title = {ChEMBL: towards direct deposition of bioassay data},
journal = {Nucleic Acids Research},
volume = {47},
number = {D1},
pages = {D930-D940},
year = {2018},
month = {11},
abstract = "{ChEMBL is a large, open-access bioactivity database
(https://www.ebi.ac.uk/chembl), previously described in the 2012,
2014 and 2017 Nucleic Acids Research Database Issues.
In the last two years, several important improvements have been made to the database and are described here.
These include more robust capture and representation of assay details;
a new data deposition system, allowing updating of data sets and deposition of supplementary data;
and a completely redesigned web interface, with enhanced search and filtering capabilities.}",
issn = {0305-1048},
doi = {10.1093/nar/gky1075},
url = {https://doi.org/10.1093/nar/gky1075},
eprint = {https://academic.oup.com/nar/article-pdf/47/D1/D930/27437436/gky1075.pdf},
}""",
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
