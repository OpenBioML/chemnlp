import pandas as pd
import yaml
from tdc.single_pred import ADME

def get_and_transform_data():
    # get raw data
    target_subfolder = 'Bioavailability_Ma'
    splits = ADME(name = target_subfolder).get_split()
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
        "split"
    ]


    # overwrite column names = fields
    fields_clean =['compound_name', 
                   'SMILES', 
                   "bioavailable",
                    'split']
    df.columns = fields_clean

    # data cleaning
#     df.compound_name = (
#         df.compound_name.str.strip()
#     )  
    # remove leading and trailing white space characters
    df = df.dropna()
    assert not df.duplicated().sum()
    
    # save to csv
    fn_data_csv = "data_clean.csv"
    df.to_csv(fn_data_csv, index=False)
    
    # create meta yaml
    meta =  {
        "name": "bioavailability_ma_et_al",  # unique identifier, we will also use this for directory names
        "description": """Oral bioavailability is defined as the rate and extent to which the
active ingredient or active moiety is absorbed from a drug product and becomes
available at the site of action.""",
        "targets": [
            {
        "id": "bioavailable",  # name of the column in a tabular dataset
        "description": "whether it available at the site of action (1) or not (0)",  # description of what this column means
        "units": "bioavailable",  # units of the values in this column (leave empty if unitless)
        "type": "categorical",  # can be "categorical", "ordinal", "continuous"
        "names": [  # names for the property (to sample from for building the prompts)
            "Oral bioavailability",
            "bioavailability",
            "ADME bioavailability",
            "Drug delivery"
                ],
                "uris":[
                "https://bioportal.bioontology.org/ontologies/NCIT?p=classes&conceptid=http%3A%2F%2Fncicb.nci.nih.gov%2Fxml%2Fowl%2FEVS%2FThesaurus.owl%23C70913",
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
                    "names":[
                    "drug bank name",
                    "drug name pubchem",
                    "drug generic name",
                    "drug chemical (generic) name",
                    "chemical name"
                    ],
                    "description": "Drug name",  # description (optional, except for "Other")
            },
        ],
        "license": "CC BY 4.0",  # license under which the original dataset was published
        "links": [  # list of relevant links (original dataset, other uses, etc.)
            {
                "url": "https://doi.org/10.1016/j.jpba.2008.03.023",
                "description": "corresponding publication",
            },
            {
                "url": "https://tdcommons.ai/single_pred_tasks/adme/#bioavailability-ma-et-al",
                "description": "data source",

            }
        ],
        "num_points": len(df),  # number of datapoints in this dataset
        "bibtex": [
            """@article{Ma2008,
doi = {10.1016/j.jpba.2008.03.023},
url = {https://doi.org/10.1016/j.jpba.2008.03.023},
year = {2008},
month = aug,
publisher = {Elsevier BV},
volume = {47},
number = {4-5},
author = {Chang-Ying Ma and Sheng-Yong Yang and Hui Zhang
and Ming-Li Xiang and Qi Huang and Yu-Quan Wei},
title = {Prediction models of human plasma protein binding rate and
oral bioavailability derived by using GA-CG-SVM method},
journal = {Journal of Pharmaceutical and Biomedical Analysis}""",
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
