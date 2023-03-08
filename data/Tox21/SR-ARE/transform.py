import pandas as pd
import yaml
from tdc.single_pred import Tox
from tdc.utils import retrieve_label_name_list
import os

def get_and_transform_data():
    # get raw data
    target_folder = 'Tox21'
    label_list = retrieve_label_name_list(f'{target_folder}')
    target_subfolder = f'{label_list[7]}'
    data = Tox(name = f'{target_folder}', label_name = target_subfolder)
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
    fields_clean =['compound_id', 'SMILES', f'toxicity_{target_subfolder}']
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
        "name": f"{target_subfolder}",  # unique identifier, we will also use this for directory names
        "description": """Tox21 is a data challenge which contains qualitative toxicity measurements for 7,831 compounds on 12 different targets, such as nuclear receptors and stree response pathways.""",
        "targets": [
            {
                "id": f"toxicity_{target_subfolder}",  # name of the column in a tabular dataset
                "description": "whether it toxic in a specific assay (1) or not toxic (0)",  # description of what this column means
                "units": "toxicity",  # units of the values in this column (leave empty if unitless)
                "type": "categorical",  # can be "categorical", "ordinal", "continuous"
                "names": [  # names for the property (to sample from for building the prompts)
                    "SR-ARE toxicity",
                    "SR-ARE",
                    "Tox21",
                    "Tox21 SR-Antioxidant response element",
                    "SR-Antioxidant response element",
                    "Antioxidant response element assay",
                    "Antioxidant response element toxicity",

                ],
            },
        ],
        "uris":[
                "https://bioportal.bioontology.org/ontologies/IOBC?p=classes&conceptid=http%3A%2F%2Fpurl.jp%2Fbio%2F4%2Fid%2F201306084823705580",
                "https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-018-2523-5/tables/3",
        ]
        ,
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
                "url": "http://dx.doi.org/10.3389/fenvs.2017.00003",
                "description": "corresponding publication",
            },
            {
                "url": "https://tdcommons.ai/single_pred_tasks/tox/#herg-karim-et-al",
                "description": "data source",

            },
            {
                "url":"https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-018-2523-5/tables/3",
                "description": "Assay name",
            }
        ],
        "num_points": len(df),  # number of datapoints in this dataset
        "bibtex": [
            """@article{Huang2017,
          doi = {10.3389/fenvs.2017.00003},
          url = {https://doi.org/10.3389/fenvs.2017.00003},
          year = {2017},
          month = jan,
          publisher = {Frontiers Media {SA}},
          volume = {5},
          author = {Ruili Huang and Menghang Xia},
          title = {Editorial: Tox21 Challenge to Build Predictive Models of Nuclear Receptor and Stress Response Pathways As Mediated by Exposure to Environmental Toxicants and Drugs},
          journal = {Frontiers in Environmental Science}}""",
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
