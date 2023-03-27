import pandas as pd
import yaml
from tdc.single_pred import ADME

def get_and_transform_data():
    # get raw data
    target_folder = 'Plasma_Protein_Binding_Rate_AstraZeneca'
    target_subfolder = 'PPBR_AZ'
    data = ADME(name = target_subfolder)
    fn_data_original = 'data/ppbr_az.tab'
    # create dataframe
    df = pd.read_csv(fn_data_original, sep='\t')
    # check if fields are the same
    fields_orig = df.columns.tolist()
    assert fields_orig == ['Drug_ID', 'Drug', 'Y', 'Species']

    # overwrite column names = fields
    fields_clean = ['chembl_id', 'SMILES', 'rate_of_PPBR', 'Species']
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
        "name": "plasma_protein_binding_rate_astrazeneca",  # unique identifier, we will also use this for directory names
        "description": """The human plasma protein binding rate (PPBR) is expressed as the percentage
of a drug bound to plasma proteins in the blood. This rate strongly affect a drug's
efficiency of delivery. The less bound a drug is, the more efficiently it can
traverse and diffuse to the site of actions. From a ChEMBL assay deposited
by AstraZeneca.""",
        "targets": [
            {
                "id": "rate_of_PPBR",  # name of the column in a tabular dataset
                "description": "The percentage of a drug bound to plasma proteins in the blood",  # description of what this column means
                "units": "percentage",  # units of the values in this column (leave empty if unitless)
                "type": "continuous",  # can be "categorical", "ordinal", "continuous"
                "names": [  # names for the property (to sample from for building the prompts)
                    "a drug bound to plasma proteins",
                    "ADME plasma proteins binding",
                    "The percentage of a drug bound to plasma proteins in the blood",
                    "drug bound plasma proteins in the blood",
                    "drug body interaction",
                    "Pharmacokinetics plasma protein binding rate",
                    "ability of a drug to bind to bound to plasma proteins in the blood",
                ],
                "uris":[
                    "https://bioportal.bioontology.org/ontologies/IOBC?p=classes&conceptid=http%3A%2F%2Fpurl.jp%2Fbio%2F4%2Fid%2F201306028362680450",
                    "https://bioportal.bioontology.org/ontologies/BAO?p=classes&conceptid=http%3A%2F%2Fwww.bioassayontology.org%2Fbao%23BAO_0010135",
                    "https://bioportal.bioontology.org/ontologies/MESH?p=classes&conceptid=http%3A%2F%2Fpurl.bioontology.org%2Fontology%2FMESH%2FD010599",
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
                "id": "Species",  # column name
                "type": "Other",  # can be "SMILES", "SELFIES", "IUPAC", "Other"
                "names": [
                "Species",
                "Species that drug measure in"
                ], 
                "description": "Species which drug have a percentage bound to plasma proteins in blood", 
            },
            {
                "id": "chembl_id",  # column name
                "type": "Other",  # can be "SMILES", "SELFIES", "IUPAC", "Other"
                "names": [
                "ChEMBL database id",
                "ChEMBL identifier number"
                ], 
                "description": "chembl ids",  # description (optional, except for "Other")
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
            }
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
