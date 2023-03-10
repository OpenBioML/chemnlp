import pandas as pd
import yaml
from tdc.single_pred import Tox


def get_and_transform_data():
    # get raw data
    df1 = pd.read_csv('https://github.com/reymond-group/drfp/raw/main/data/uspto_yields_above.csv')
    df2 = pd.read_csv('https://github.com/reymond-group/drfp/raw/main/data/uspto_yields_below.csv')
    data = pd.concat([df1,df2])
    data = data[['rxn','yield']]
    data= data.drop_duplicates(subset='rxn')
    fn_data_original = "uptso.csv"
    data.to_csv(fn_data_original, index=False)
    
    # create dataframe
    df = pd.read_csv(fn_data_original, 
                     delimiter=","
        )# not necessary but ensure we can load the saved data
    
    # check if fields are the same
    fields_orig = df.columns.tolist()
    assert fields_orig == ['rxn', 'yield']
    fields_clean = [
        "reaction_SMILES",
        "yield"
    ]
    
    # overwrite column names = fields
    df.columns = fields_clean
    assert fields_orig != fields_clean
    
    # remove leading and trailing white space characters
    assert not df.duplicated().sum()
    
    # save to csv
    fn_data_csv = "data_clean.csv"
    df.to_csv(fn_data_csv, index=False)

    
    # create meta yaml
    meta =  {
        "name": "USPTO_500k",  # unique identifier, we will also use this for directory names
        "description": """United States Patent and Trademark Office reaction dataset with yields.""",
        "targets": [
            {
                "id": "yield",  # name of the column in a tabular dataset
                "description": "Reaction yields analyzed by UPLC",  # description of what this column means
                "units": "%",  # units of the values in this column (leave empty if unitless)
                "type": "continuous",  # can be "categorical", "ordinal", "continuous"
                "names": [  # names for the property (to sample from for building the prompts)
                    "Reaction yield",
                    "yield",
                ],
                "uris":[
                    "https://bioportal.bioontology.org/ontologies/AFO?p=classes&conceptid=http%3A%2F%2Fpurl.allotrope.org%2Fontologies%2Fquality%23AFQ_0000227",
                    "https://en.wikipedia.org/wiki/Yield_(chemistry)",
                ],
            },
        ],
        "identifiers": [
            {
                "id": "reaction_SMILES",  # column name
                "type": "SMILES",  # can be "SMILES", "SELFIES", "IUPAC", "Other"
                "description": "reaction SMILES",  # description (optional, except for "Other")
            },
        ],
        "license": "CC0",  # license under which the original dataset was published
        "links": [  # list of relevant links (original dataset, other uses, etc.)
            {
                "url": "https://doi.org/10.17863/CAM.16293",
                "description": "corresponding publication",
            },
            {
                "url": "https://github.com/reymond-group/drfp/blob/main/data/uspto_yields_below.csv",
                "description": "data source",
            },
            {
                "url": "https://github.com/reymond-group/drfp/blob/main/data/uspto_yields_above.csv",
                "description": "data source",
            },
            {
                "url": "https://tdcommons.ai/single_pred_tasks/yields/#uspto",
                "description": "other source",
            }
        ],
        "num_points": len(df),  # number of datapoints in this dataset
        "bibtex": [
            """@article{https://doi.org/10.17863/cam.16293,
              doi = {10.17863/CAM.16293},
              url = {https://www.repository.cam.ac.uk/handle/1810/244727},
              author = {Lowe,  Daniel Mark},
              keywords = {Name to structure,  OPSIN,  Chemical text mining,  Text mining,  Patent reaction extraction,  Reaction mining,  Patents},
              language = {en},
              title = {Extraction of chemical structures and reactions from the literature},
              publisher = {Apollo - University of Cambridge Repository},
              year = {2012},
              copyright = {All Rights Reserved}}""",
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
