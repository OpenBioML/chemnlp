import pandas as pd
import yaml
from tdc.single_pred import ADME

def get_and_transform_data():
    # get raw data
    target_subfolder = 'HydrationFreeEnergy_FreeSolv'
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
    fields_clean = ['compound_name', 
                    'SMILES',
                    'hydration_free_energy',
                    'split'
                      ]
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
    
    # create meta yaml
    meta =  {
        "name": "hydrationfreeenergy_freesolv",  # unique identifier, we will also use this for directory names
        "description": """The Free Solvation Database, FreeSolv(SAMPL), provides experimental
and calculated hydration free energy of small molecules in water. The calculated
values are derived from alchemical free energy calculations using molecular
dynamics simulations. From MoleculeNet.""",
        "targets": [
            {
                "id": "hydration_free_energy",  # name of the column in a tabular dataset
                "description": "alchemical free energy calculations using molecular dynamics simulations",  # description of what this column means
                "units": "hydration free energy(kJ mol-1)",  # units of the values in this column (leave empty if unitless)
                "type": "continuous",  # can be "categorical", "ordinal", "continuous"
                "names": [  # names for the property (to sample from for building the prompts)
                    "hydration free energy",
                    "free energy",
                    "free energy calculations using molecular dynamics simulations",
                    "alchemical free energy",
                ],
                "uris":[
                    "https://bioportal.bioontology.org/ontologies/NCIT?p=classes&conceptid=http%3A%2F%2Fncicb.nci.nih.gov%2Fxml%2Fowl%2FEVS%2FThesaurus.owl%23C18218",
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
                    "type": "IUPAC",  # can be "SMILES", "SELFIES", "IUPAC", "Other"
                    "description": "compound name",  # description (optional, except for "Other")
            },
        ],
        "license": "CC BY-NC-SA 4.0",  # license under which the original dataset was published
        "links": [  # list of relevant links (original dataset, other uses, etc.)
            {
                "url": "https://doi.org/10.1007/s10822-014-9747-x",
                "description": "corresponding publication",
            },
            {
                "url": "https://rb.gy/0xx91v",
                "description": "corresponding publication",
            },

            {
                "url": "https://tdcommons.ai/single_pred_tasks/adme/#hydration-free-energy-freesolv",
                "description": "data source",
            }
        ],
        "num_points": len(df),  # number of datapoints in this dataset
        "bibtex": [
            """@article{Mobley2014,
doi = {10.1007/s10822-014-9747-x},
url = {https://doi.org/10.1007/s10822-014-9747-x},
year = {2014},
month = jun,
publisher = {Springer Science and Business Media LLC},
volume = {28},
number = {7},
pages = {711--720},
author = {David L. Mobley and J. Peter Guthrie},
title = {FreeSolv: a database of experimental and calculated hydration
free energies,  with input files},
journal = {Journal of Computer Aided Molecular Design}""",
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
