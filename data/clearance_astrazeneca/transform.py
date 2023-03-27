import pandas as pd
import yaml
from tdc.single_pred import ADME

def get_and_transform_data():
    # get raw data
    target_folder = 'Clearance_AstraZeneca'
    target_subfolder = 'Clearance_Hepatocyte_AZ'
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
    assert fields_orig == ['Drug_ID', 'Drug', 'Y']

    # overwrite column names = fields
    fields_clean = ['chembl_id', 'SMILES', 'drug_clearance']
    df.columns = fields_clean

    # data cleaning
    df[fields_clean[0]] = (
        df[fields_clean[0]].str.strip()
    )  
    # remove leading and trailing white space characters
    df = df.dropna()
    assert not df.duplicated().sum()
    
    # save to csv
    fn_data_csv = "data_clean.csv"
    df.to_csv(fn_data_csv, index=False)
    meta = {
        "name": "clearance_astrazeneca",  # unique identifier, we will also use this for directory names
        "description": """Drug clearance is defined as the volume of plasma cleared of a drug
over a specified time period and it measures the rate at which the active drug
is removed from the body. This is a dataset curated from ChEMBL database containing
experimental results on intrinsic clearance, deposited from AstraZeneca. It
contains clearance measures from two experiments types, hepatocyte and microsomes.""",
        "targets": [
            {
        "id": "drug_clearance",  # name of the column in a tabular dataset
        "description": "The volume of plasma cleared of a drug over a specified time period and it measures the rate at which the active drug is removed from the body",  # description of what this column means
        "units": "mL.min-1.g-1",  # units of the values in this column (leave empty if unitless)
        "type": "continuous",  # can be "categorical", "ordinal", "continuous"
        "names": [  # names for the property (to sample from for building the prompts)
            "Drug clearance",
            "The rate at which the active drug is removed from the body",
            "ADME Drug clearance",
            "Pharmacokinetics drug clearance",
            "Drug Excretion",
            "microsome/hepatocyte clearance",
                ],
                "uris":[
                    "https://bioportal.bioontology.org/ontologies/MESH?p=classes&conceptid=http%3A%2F%2Fpurl.bioontology.org%2Fontology%2FMESH%2FD006207",
                    "https://bioportal.bioontology.org/ontologies/NCIT?p=classes&conceptid=http%3A%2F%2Fncicb.nci.nih.gov%2Fxml%2Fowl%2FEVS%2FThesaurus.owl%23C94618",
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
                "id": "chembl_id",  # column name
                "type": "Other",  # can be "SMILES", "SELFIES", "IUPAC", "Other"
                "names": [
                "ChEMBL database id",
                "ChEMBL identifier number"
                ],  
                "description": "ChEMBL ids",
            },

        ],
        "license": "CC BY 4.0",  # license under which the original dataset was published
        "links": [  # list of relevant links (original dataset, other uses, etc.)
            {
                "url": "http://dx.doi.org/10.6019/CHEMBL3301361",
                "description": "corresponding publication",
            },
            {
                "url": "https://doi.org/10.1016/j.ejmech.2012.06.043",
                "description": "corresponding publication",
            },
            {
                "url": "https://tdcommons.ai/single_pred_tasks/adme/#clearance-astrazeneca",
                "description": "data source",
            }
        ],
        "num_points": len(df),  # number of datapoints in this dataset
        "bibtex": [
            """@techreport{Hersey2015,
doi = {10.6019/chembl3301361},
url = {https://doi.org/10.6019/chembl3301361},
year = {2015},
month = feb,
publisher = {{EMBL}-{EBI}},
author = {Anne Hersey},
title = {{ChEMBL} Deposited Data Set - {AZ dataset}}""",
            """@article{Di2012,
doi = {10.1016/j.ejmech.2012.06.043},
url = {https://doi.org/10.1016/j.ejmech.2012.06.043},
year = {2012},
month = nov,
publisher = {Elsevier BV},
volume = {57},
pages = {441--448},
author = {Li Di and Christopher Keefer and Dennis O. Scott and Timothy J. Strelevitz
and George Chang and Yi-An Bi and Yurong Lai and Jonathon Duckworth and
Katherine Fenner and Matthew D. Troutman and R. Scott Obach},
title = {Mechanistic insights from comparing intrinsic clearance values between
human liver microsomes and hepatocytes to guide drug design},
journal = {European Journal of Medicinal Chemistry}""",
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
