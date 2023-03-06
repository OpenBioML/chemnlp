import pandas as pd
import yaml
from tdc.single_pred import Tox
from tdc.utils import retrieve_label_name_list

def get_and_transform_data():
    # get raw data
    label = 'herg_central'
    df_all = [Tox(name = label, label_name = x).get_data(format = 'df') for x in retrieve_label_name_list(label)]    
    df_path = f'data/{label}.tab'
    df = pd.read_csv(df_path, sep = '\t')
    fn_data_original = "data_original.csv"
    df.to_csv(fn_data_original, index=None)

    # create dataframe
    df = pd.read_csv(
        fn_data_original,
        delimiter=",",
    )  # not necessary but ensure we can load the saved data

    # check if fields are the same
    fields_orig = df.columns.tolist()
    assert fields_orig == [
        'ID',
        'X', 
        'hERG_at_1uM',
        'hERG_at_10uM', 
        'hERG_inhib']

    # overwrite column names = fields
    fields_clean =  [
        "compound_id",
        "SMILES",
        "hERG_at_1uM",
        "hERG_at_10uM",
        "hERG_inhib",
    ]
    df.columns = fields_clean

#     # data cleaning
#     df.compound_name = (
#         df.compound_name.str.strip()
#     )  # remove leading and trailing white space characters

    assert not df.duplicated().sum()

    # save to csv
    fn_data_csv = "data_clean.csv"
    df.to_csv(fn_data_csv, index=False)

    # create meta yaml
    meta = {
            "name": "hERG Central",  # unique identifier, we will also use this for directory names
            "description": """Human ether-a-go-go related gene (hERG) is crucial for the coordination of the heart's beating. Thus, if a drug blocks the hERG, it could lead to severe adverse effects. Therefore, reliable prediction of hERG liability in the early stages of drug design is quite important to reduce the risk of cardiotoxicity-related attritions in the later development stages. There are three targets: hERG_at_1microM, hERG_at_10microM, and hERG_inhib.""",
            "targets": [
                {
                    "id": "hERG_at_1uM",  # name of the column in a tabular dataset
                    "description": "The percent inhibition at a 1microM concentration",  # description of what this column means
                    "units": "1microM concentration",  # units of the values in this column (leave empty if unitless)
                    "type": "continuous",  # can be "categorical", "ordinal", "continuous"
                    "names": [  # names for the property (to sample from for building the prompts)
                        "hERG activity",
                        "hERG active compound",
                        "hERG blocker",
                        "Human ether-a-go-go related gene (hERG) blocker",
                        "Activity against Human ether-a-go-go related gene (hERG)",
                        "hERG at 1microM",
                        "hERG activity 1microM",
                        "The percent inhibition at a 1microM concentration",
                        "Compound percent activity at 1microM"
                        ,
                    ],
                },
                {
                    "id": "hERG_at_10uM",  # name of the column in a tabular dataset
                    "description": "The percent inhibition at a 10microM concentration",  # description of what this column means
                    "units": "1microM concentration",  # units of the values in this column (leave empty if unitless)
                    "type": "continuous",  # can be "categorical", "ordinal", "continuous"
                    "names": [  # names for the property (to sample from for building the prompts)
                        "hERG activity",
                        "hERG active compound",
                        "hERG blocker",
                        "Human ether-a-go-go related gene (hERG) blocker",
                        "Activity against Human ether-a-go-go related gene (hERG)",
                        "hERG at 10microM",
                        "hERG activity 10microM",
                        "The percent inhibition at a 10microM concentration",
                        "Compound percent activity at 10microM"
                    ],
                },
                {
                    "id": "hERG_inhib",  # name of the column in a tabular dataset
                    "description": "whether it blocks (1) or not blocks (0). This is equivalent to whether hERG_at_10microM < -50, i.e. whether the compound has an IC50 of less than 10microM.",  # description of what this column means
                    "units": "1microM concentration",  # units of the values in this column (leave empty if unitless)
                    "type": "categorical",  # can be "categorical", "ordinal", "continuous"
                    "names": [  # names for the property (to sample from for building the prompts)
                        "hERG activity",
                        "hERG active compound",
                        "hERG blocker",
                        "Human ether-a-go-go related gene (hERG) blocker",
                        "Activity against Human ether-a-go-go related gene (hERG)",
                    ],
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
                    "url": "https://doi.org/10.1089/adt.2011.0425",
                    "description": "corresponding publication",
                },
                {
                    "url": "https://bbirnbaum.com/",
                    "description": "TDC Contributer",
                },
                {
                    "url": "https://tdcommons.ai/single_pred_tasks/tox/#herg-central",
                    "description": "Data source",
                },
            ],
            "num_points": len(df),  # number of datapoints in this dataset
            "bibtex": [
                """@article{Du2011,
                  doi = {10.1089/adt.2011.0425},
                  url = {https://doi.org/10.1089/adt.2011.0425},
                  year = {2011},
                  month = dec,
                  publisher = {Mary Ann Liebert Inc},
                  volume = {9},
                  number = {6},
                  pages = {580--588},
                  author = {Fang Du and Haibo Yu and Beiyan Zou and Joseph Babcock and Shunyou Long and Min Li},
                  title = {{hERGCentral}: A Large Database to Store,  Retrieve,  and Analyze Compound-Human $\less$i$\greater$Ether-{\`{a}}-go-go$\less$/i$\greater$ Related Gene Channel Interactions to Facilitate Cardiotoxicity Assessment in Drug Development},
                  journal = {{ASSAY} and Drug Development Technologies}}""",
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
