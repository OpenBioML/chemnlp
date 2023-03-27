import pandas as pd
import yaml
from tdc.single_pred import ADME

def get_and_transform_data():
    # get raw data
    target_folder = 'P_glycoprotein_Inhibition_Broccatelli_et_al'
    target_subfolder = 'Pgp_Broccatelli'
    data = ADME(name = target_subfolder)
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
    fields_clean =['compound_name', 'SMILES', f"Pgp_inhibition"]
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
        "name": "p_glycoprotein_inhibition_broccatelli_et_al",  # unique identifier, we will also use this for directory names
        "description": """P-glycoprotein (Pgp) is an ABC transporter protein involved in intestinal
absorption, drug metabolism, and brain penetration, and its inhibition can seriously
alter a drug's bioavailability and safety. In addition, inhibitors of Pgp can
be used to overcome multidrug resistance.""",
        "targets": [
            {
                "id": "Pgp_inhibition",  # name of the column in a tabular dataset
                "description": "whether it active toward Pgp inhibition (1) or not (0)",  # description of what this column means
                "units": "Pgp_inhibition",  # units of the values in this column (leave empty if unitless)
                "type": "categorical",  # can be "categorical", "ordinal", "continuous"
                "names": [  # names for the property (to sample from for building the prompts)
                    "P-glycoprotein",
                    "Pgp Inhibition",
                    "Pgp",
                    "ADME absorption Pgp",
                    "Pgp activity",
                    "Pgp Inhibition activity"
                ],
                "uris":[
                "https://bioportal.bioontology.org/ontologies/CRISP?p=classes&conceptid=http%3A%2F%2Fpurl.bioontology.org%2Fontology%2FCSP%2F4000-0278",
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
                    "iupac like name",
                    "Synonyms",
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
                "url": "https://doi.org/10.1021/jm101421d",
                "description": "corresponding publication",
            },
            {
                "url": "https://tdcommons.ai/single_pred_tasks/adme/#pgp-p-glycoprotein-inhibition-broccatelli-et-al",
                "description": "data source",

            }
        ],
        "num_points": len(df),  # number of datapoints in this dataset
        "bibtex": [
            """@article{Broccatelli2011,
doi = {10.1021/jm101421d},
url = {https://doi.org/10.1021/jm101421d},
year = {2011},
month = feb,
publisher = {American Chemical Society (ACS)},
volume = {54},
number = {6},
author = {Fabio Broccatelli and Emanuele Carosati and Annalisa Neri and
Maria Frosini and Laura Goracci and Tudor I. Oprea and Gabriele Cruciani},
title = {A Novel Approach for Predicting P-Glycoprotein (ABCB1) Inhibition
Using Molecular Interaction Fields},
journal = {Journal of Medicinal Chemistry}""",
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
