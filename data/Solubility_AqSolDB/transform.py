import pandas as pd
import yaml
from tdc.single_pred import ADME

def get_and_transform_data():
    # get raw data
    target_folder = 'Solubility_AqSolDB'
    target_subfolder = 'Solubility_AqSolDB'
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
    fields_clean = ['compound_name', 'SMILES', 'aqeuous_solubility']
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
        "name": f"{target_folder}",  # unique identifier, we will also use this for directory names
        "description": """Aqeuous solubility measures a drug's ability to dissolve in water. Poor water solubility could lead to slow drug absorptions, inadequate bioavailablity and even induce toxicity. More than 40% of new chemical entities are not soluble.""",
        "targets": [
            {
                "id": "aqeuous_solubility",  # name of the column in a tabular dataset
                "description": "drug's ability to dissolve in water it can be in mg/L or ppm",  # description of what this column means
                "units": "aqueous solubility",  # units of the values in this column (leave empty if unitless)
                "type": "continuous",  # can be "categorical", "ordinal", "continuous"
                "names": [  # names for the property (to sample from for building the prompts)
                    "aqeuous solubility",
                    "dissolve in a water",
                    "ADME aqeuous solubility",
                    "Drug Absorption",
                    "solubility",
                    "ability of a drug to dissolve in a water"
                ],
                "uris":[
                "https://bioportal.bioontology.org/ontologies/IOBC?p=classes&conceptid=http%3A%2F%2Fpurl.jp%2Fbio%2F4%2Fid%2F200906006880450101",
                "https://bioportal.bioontology.org/ontologies/NCIT?p=classes&conceptid=http%3A%2F%2Fncicb.nci.nih.gov%2Fxml%2Fowl%2FEVS%2FThesaurus.owl%23C60821",
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
                    "id": "compound_name",  # column name
                    "type": "Other",  # can be "SMILES", "SELFIES", "IUPAC", "Other"
                    "description": "compound name",  # description (optional, except for "Other")
            },
        ],
        "license": "CC BY 4.0",  # license under which the original dataset was published
        "links": [  # list of relevant links (original dataset, other uses, etc.)
            {
                "url": "https://doi.org/10.1038/s41597-019-0151-1",
                "description": "corresponding publication",
            },
            {
                "url": "https://tdcommons.ai/single_pred_tasks/adme/#solubility-aqsoldb",
                "description": "data source",

            }
        ],
        "num_points": len(df),  # number of datapoints in this dataset
        "bibtex": [
            """@article{Sorkun_2019,
            doi = {10.1038/s41597-019-0151-1},
            url = {https://doi.org/10.1038%2Fs41597-019-0151-1},
            year = 2019,
            month = {aug},
            publisher = {Springer Science and Business Media {LLC}},
            volume = {6},
            number = {1},
            author = {Murat Cihan Sorkun and Abhishek Khetan and Süleyman Er},
            title = {{AqSolDB}, a curated reference set of aqueous solubility and 2D descriptors for a diverse set of compounds},
            journal = {Scientific Data}}""",
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
